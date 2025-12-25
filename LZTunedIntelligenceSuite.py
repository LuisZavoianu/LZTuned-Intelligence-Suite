import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xml.etree.ElementTree as ET
from scipy.interpolate import interp2d
import io
import re
from typing import Dict, List, Tuple, Optional

# ============================================================================
# CONFIGURARE PAGINĂ
# ============================================================================
st.set_page_config(
    page_title="LZTuned Intelligence Suite",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# 1. MODUL XML PARSER (THE MAP GUIDE)
# ============================================================================
@st.cache_data
def parse_romraider_xml(xml_content: bytes) -> Dict:
    """
    Parsează XML-ul RomRaider și extrage definițiile hărților ECU.
    
    Returns:
        Dict cu structura: {
            'software_id_address': str,
            'tables': {
                'table_name': {
                    'address': int,
                    'type': str,
                    'scaling': str,
                    'x_axis': {...},
                    'y_axis': {...}
                }
            }
        }
    """
    try:
        root = ET.fromstring(xml_content)
        result = {
            'software_id_address': None,
            'tables': {}
        }
        
        # Caută internal ID address
        for rom in root.findall('.//rom'):
            id_addr = rom.get('internalidaddress')
            if id_addr:
                result['software_id_address'] = id_addr
                break
        
        # Parsează toate tabelele
        for table in root.findall('.//table'):
            table_name = table.get('name', 'Unknown')
            storage_addr = table.get('storageaddress')
            table_type = table.get('type', '3D')
            
            if not storage_addr:
                continue
            
            table_data = {
                'address': int(storage_addr, 16),
                'type': table_type,
                'scaling': None,
                'x_axis': None,
                'y_axis': None,
                'data_format': table.get('storagetype', 'uint8')
            }
            
            # Extrage scaling formula
            scaling = table.find('.//scaling')
            if scaling is not None:
                table_data['scaling'] = scaling.get('expression', 'x')
            
            # Extrage axele
            for axis in table.findall('.//table'):
                axis_type = axis.get('type')
                axis_addr = axis.get('storageaddress')
                
                if axis_type and axis_addr:
                    axis_data = {
                        'address': int(axis_addr, 16),
                        'length': int(axis.get('length', '0')),
                        'scaling': None
                    }
                    
                    axis_scaling = axis.find('.//scaling')
                    if axis_scaling is not None:
                        axis_data['scaling'] = axis_scaling.get('expression', 'x')
                    
                    if 'X' in axis_type:
                        table_data['x_axis'] = axis_data
                    elif 'Y' in axis_type:
                        table_data['y_axis'] = axis_data
            
            result['tables'][table_name] = table_data
        
        return result
    
    except Exception as e:
        st.error(f"Eroare parsare XML: {e}")
        return {'software_id_address': None, 'tables': {}}

# ============================================================================
# 2. MODUL BIN READER (THE SOURCE)
# ============================================================================
@st.cache_data
def read_bin_file(bin_content: bytes, xml_data: Dict) -> Dict:
    result = {'valid': False, 'software_id': None, 'maps': {}, 'offset': 0}
    
    # Lista de offset-uri comune pentru MS42 (Full Dump 512KB)
    possible_offsets = [0x0, 0x38000, 0x40000]
    target_id = "0110C7"
    
    # Încercăm să găsim ID-ul la adresa din XML aplicând diverse offset-uri
    xml_id_addr = int(xml_data['software_id_address'])
    
    found_offset = None
    for off in possible_offsets:
        check_addr = xml_id_addr + off
        if check_addr + 6 <= len(bin_content):
            extracted_id = bin_content[check_addr:check_addr+6].decode('ascii', errors='ignore')
            if extracted_id == target_id:
                found_offset = off
                break
    
    # Dacă nu l-am găsit la adresele standard, scanăm brut tot fișierul
    if found_offset is None:
        if bin_content.find(target_id.encode('ascii')) != -1:
            actual_pos = bin_content.find(target_id.encode('ascii'))
            found_offset = actual_pos - xml_id_addr
        else:
            # Forțăm offset-ul de 512KB dacă fișierul are această mărime
            if len(bin_content) > 100000:
                found_offset = 0x38000
                st.sidebar.warning("⚠️ ID-ul nu a fost găsit clar, dar aplicăm offset-ul standard de 512KB.")

    if found_offset is not None:
        result['offset'] = found_offset
        result['software_id'] = target_id
        result['valid'] = True
        st.sidebar.success(f"✅ Offset activat: {hex(found_offset)}")
    else:
        st.error("❌ Nu s-a putut valida fișierul BIN. Verificați dacă este un dump de MS42.")
        return result

    # Extragere hărți cu offset-ul confirmat
    for table_name, table_info in xml_data['tables'].items():
        try:
            # Ajustăm toate adresele (tabel + axe) cu offset-ul găsit
            adjusted_info = table_info.copy()
            adjusted_info['address'] += result['offset']
            
            if adjusted_info.get('x_axis'):
                adjusted_info['x_axis'] = adjusted_info['x_axis'].copy()
                adjusted_info['x_axis']['address'] += result['offset']
            if adjusted_info.get('y_axis'):
                adjusted_info['y_axis'] = adjusted_info['y_axis'].copy()
                adjusted_info['y_axis']['address'] += result['offset']
                
            map_data = extract_map_data(bin_content, adjusted_info)
            if map_data['z_data'] is not None:
                result['maps'][table_name] = map_data
        except:
            continue
            
    return result
    
def extract_map_data(bin_content: bytes, table_info: Dict) -> Dict:
    """Extrage datele unei hărți din fișierul BIN."""
    addr = table_info['address']
    
    # Extrage axele
    x_axis = None
    y_axis = None
    
    if table_info['x_axis']:
        x_info = table_info['x_axis']
        x_addr = x_info['address']
        x_len = x_info['length']
        x_raw = np.frombuffer(bin_content[x_addr:x_addr+x_len*2], dtype=np.uint16)
        x_axis = apply_scaling(x_raw, x_info['scaling'])
    
    if table_info['y_axis']:
        y_info = table_info['y_axis']
        y_addr = y_info['address']
        y_len = y_info['length']
        y_raw = np.frombuffer(bin_content[y_addr:y_addr+y_len*2], dtype=np.uint16)
        y_axis = apply_scaling(y_raw, y_info['scaling'])
    
    # Extrage datele principale
    if table_info['type'] == '3D' and x_axis is not None and y_axis is not None:
        data_size = len(x_axis) * len(y_axis)
        if table_info['data_format'] == 'uint16':
            raw_data = np.frombuffer(bin_content[addr:addr+data_size*2], dtype=np.uint16)
        else:
            raw_data = np.frombuffer(bin_content[addr:addr+data_size], dtype=np.uint8)
        
        z_data = apply_scaling(raw_data.reshape(len(y_axis), len(x_axis)), table_info['scaling'])
    else:
        z_data = None
    
    return {
        'x_axis': x_axis,
        'y_axis': y_axis,
        'z_data': z_data
    }

def apply_scaling(data: np.ndarray, formula: str) -> np.ndarray:
    """Aplică formula de scaling din XML."""
    if formula is None or formula == 'x':
        return data
    
    try:
        # Înlocuiește 'x' cu datele și evaluează
        formula_safe = formula.replace('x', 'data')
        return eval(formula_safe)
    except:
        return data

# ============================================================================
# 3. MODUL CSV PARSER (THE REALITY)
# ============================================================================
@st.cache_data
def parse_telemetry_csv(csv_content: bytes) -> pd.DataFrame:
    """
    Parsează logul CSV și curăță headerele de unități de măsură.
    """
    try:
        df = pd.read_csv(io.StringIO(csv_content.decode('utf-8')))
        
        # Curățare headere
        column_mapping = {}
        for col in df.columns:
            # Extrage numele fără unități
            clean_name = re.sub(r'\s*\[.*?\]', '', col).strip().lower().replace(' ', '_')
            column_mapping[col] = clean_name
        
        df.rename(columns=column_mapping, inplace=True)
        
        # Convertește la numeric unde e posibil
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass
        
        return df
    
    except Exception as e:
        st.error(f"Eroare parsare CSV: {e}")
        return pd.DataFrame()

# ============================================================================
# 4. NUCLEUL DE CALCUL (THE INTELLIGENCE ENGINE)
# ============================================================================
def bilinear_interpolation(x: float, y: float, x_axis: np.ndarray, 
                          y_axis: np.ndarray, z_data: np.ndarray) -> float:
    """
    Interpolare bilineară pentru a calcula valoarea la un punct (x, y).
    """
    try:
        # Găsește indicii pentru interpolare
        x_idx = np.searchsorted(x_axis, x)
        y_idx = np.searchsorted(y_axis, y)
        
        # Boundary check
        if x_idx == 0: x_idx = 1
        if x_idx >= len(x_axis): x_idx = len(x_axis) - 1
        if y_idx == 0: y_idx = 1
        if y_idx >= len(y_axis): y_idx = len(y_axis) - 1
        
        # Punctele de referință
        x1, x2 = x_axis[x_idx-1], x_axis[x_idx]
        y1, y2 = y_axis[y_idx-1], y_axis[y_idx]
        
        # Valorile din grid
        q11 = z_data[y_idx-1, x_idx-1]
        q12 = z_data[y_idx, x_idx-1]
        q21 = z_data[y_idx-1, x_idx]
        q22 = z_data[y_idx, x_idx]
        
        # Interpolare
        if x2 == x1 or y2 == y1:
            return q11
        
        r1 = ((x2 - x) / (x2 - x1)) * q11 + ((x - x1) / (x2 - x1)) * q21
        r2 = ((x2 - x) / (x2 - x1)) * q12 + ((x - x1) / (x2 - x1)) * q22
        p = ((y2 - y) / (y2 - y1)) * r1 + ((y - y1) / (y2 - y1)) * r2
        
        return p
    
    except:
        return 0.0

def calculate_trace(log_df: pd.DataFrame, map_data: Dict, 
                   x_col: str, y_col: str) -> np.ndarray:
    """
    Calculează valorile Target din hartă pentru fiecare punct din log.
    """
    if map_data['x_axis'] is None or map_data['z_data'] is None:
        return np.zeros(len(log_df))
    
    target_values = []
    for _, row in log_df.iterrows():
        x_val = row.get(x_col, 0)
        y_val = row.get(y_col, 0)
        
        target = bilinear_interpolation(
            x_val, y_val,
            map_data['x_axis'],
            map_data['y_axis'],
            map_data['z_data']
        )
        target_values.append(target)
    
    return np.array(target_values)

def calculate_residence_time(log_df: pd.DataFrame, x_col: str, y_col: str,
                            x_bins: np.ndarray, y_bins: np.ndarray) -> np.ndarray:
    """
    Calculează timpul petrecut în fiecare celulă (frecvență de utilizare).
    """
    hist, _, _ = np.histogram2d(
        log_df[x_col], log_df[y_col],
        bins=[x_bins, y_bins]
    )
    return hist.T

# ============================================================================
# 5. INTERFAȚA DE VIZUALIZARE (DASHBOARD PRO)
# ============================================================================
def create_heatmap_with_trace(map_data: Dict, log_df: pd.DataFrame, 
                              current_idx: int, x_col: str, y_col: str,
                              knock_col: Optional[str] = None) -> go.Figure:
    """
    Creează heatmap 2D cu punct de urmărire (fantoma).
    """
    fig = go.Figure()
    
    # Heatmap principal
    fig.add_trace(go.Heatmap(
        x=map_data['x_axis'],
        y=map_data['y_axis'],
        z=map_data['z_data'],
        colorscale='RdYlGn_r',
        showscale=True,
        hovertemplate='X: %{x}<br>Y: %{y}<br>Z: %{z:.2f}<extra></extra>'
    ))
    
    # Punct curent (fantoma)
    if current_idx < len(log_df):
        x_current = log_df.iloc[current_idx][x_col]
        y_current = log_df.iloc[current_idx][y_col]
        
        # Culoare punct: roșu dacă knock, alb altfel
        point_color = 'red'
        point_size = 20
        
        if knock_col and knock_col in log_df.columns:
            knock_val = log_df.iloc[current_idx][knock_col]
            if knock_val > 0:
                point_color = 'red'
                point_size = 30  # Pulsează
            else:
                point_color = 'white'
        else:
            point_color = 'white'
        
        fig.add_trace(go.Scatter(
            x=[x_current],
            y=[y_current],
            mode='markers',
            marker=dict(
                size=point_size,
                color=point_color,
                symbol='circle',
                line=dict(color='black', width=2)
            ),
            showlegend=False,
            hovertemplate=f'Current Position<br>X: {x_current:.0f}<br>Y: {y_current:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title='ECU Map with Live Position',
        xaxis_title='RPM',
        yaxis_title='Load',
        height=600
    )
    
    return fig

def create_gauge(value: float, title: str, max_val: float, 
                 unit: str = '', threshold_color: Optional[Dict] = None) -> go.Figure:
    """
    Creează un gauge (ceas de bord) pentru telemetrie.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': f"{title} ({unit})"},
        gauge={
            'axis': {'range': [0, max_val]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, max_val*0.6], 'color': "lightgray"},
                {'range': [max_val*0.6, max_val*0.8], 'color': "yellow"},
                {'range': [max_val*0.8, max_val], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_val * 0.9
            }
        }
    ))
    
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def create_vanos_comparison(log_df: pd.DataFrame, current_idx: int,
                           target_col: str, actual_col: str) -> go.Figure:
    """
    Grafic dual pentru VANOS: cerut vs real.
    """
    window_size = 100
    start_idx = max(0, current_idx - window_size)
    end_idx = min(len(log_df), current_idx + window_size)
    
    subset = log_df.iloc[start_idx:end_idx]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        y=subset[target_col],
        mode='lines',
        name='Target (Cerut)',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        y=subset[actual_col],
        mode='lines',
        name='Actual (Real)',
        line=dict(color='red', width=2, dash='dot')
    ))
    
    # Marker pentru poziția curentă
    fig.add_vline(x=current_idx-start_idx, line_dash="dash", 
                  line_color="green", annotation_text="NOW")
    
    fig.update_layout(
        title='VANOS Drift Analysis',
        yaxis_title='Cam Angle (degrees)',
        xaxis_title='Time Steps',
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_residence_heatmap(residence_data: np.ndarray, x_bins: np.ndarray,
                            y_bins: np.ndarray) -> go.Figure:
    """
    Heatmap gri pentru frecvența de utilizare.
    """
    fig = go.Figure(go.Heatmap(
        x=x_bins[:-1],
        y=y_bins[:-1],
        z=residence_data,
        colorscale='Greys',
        showscale=True,
        colorbar=dict(title="Time (samples)")
    ))
    
    fig.update_layout(
        title='Engine Operating Range (Residence Time)',
        xaxis_title='RPM',
        yaxis_title='Load',
        height=500
    )
    
    return fig

# ============================================================================
# 6. EXPERT DIAGNOSTIC (POST-SESSION REPORT)
# ============================================================================
def generate_diagnostic_report(log_df: pd.DataFrame) -> List[str]:
    """
    Scanează anomalii și generează raport diagnostic.
    """
    alerts = []
    
    # Check IAT
    if 'iat' in log_df.columns or 'intake_air_temp' in log_df.columns:
        iat_col = 'iat' if 'iat' in log_df.columns else 'intake_air_temp'
        max_iat = log_df[iat_col].max()
        avg_iat = log_df[iat_col].mean()
        
        if max_iat > 50:
            alerts.append(f"⚠️ IAT High: Peak {max_iat:.1f}°C - Pierdere de densitate a aerului observată!")
        if avg_iat > 40:
            alerts.append(f"⚠️ IAT Average High: {avg_iat:.1f}°C - Considerați intercooler upgrade")
    
    # Check AFR
    afr_cols = [c for c in log_df.columns if 'afr' in c.lower()]
    load_cols = [c for c in log_df.columns if 'load' in c.lower()]
    
    if afr_cols and load_cols:
        afr_col = afr_cols[0]
        load_col = load_cols[0]
        
        high_load = log_df[log_df[load_col] > 80]
        if len(high_load) > 0:
            lean_events = high_load[high_load[afr_col] > 13.5]
            if len(lean_events) > 0:
                alerts.append(f"🚨 ATENȚIE: Amestec sărac (Lean) în sarcină mare detectat! "
                            f"{len(lean_events)} evenimente AFR > 13.5")
    
    # Check Knock
    knock_cols = [c for c in log_df.columns if 'knock' in c.lower()]
    if knock_cols:
        knock_col = knock_cols[0]
        total_knock = log_df[knock_col].sum()
        max_knock = log_df[knock_col].max()
        
        if total_knock > 0:
            alerts.append(f"🔥 Knock Retard Detectat: Total {total_knock:.1f}°, "
                        f"Max {max_knock:.1f}° - Verificați calitatea combustibilului!")
    
    # Check Coolant
    coolant_cols = [c for c in log_df.columns if 'coolant' in c.lower() or 'temp' in c.lower()]
    if coolant_cols:
        coolant_col = coolant_cols[0]
        max_temp = log_df[coolant_col].max()
        
        if max_temp > 100:
            alerts.append(f"🌡️ Temperatură Coolant Ridicată: {max_temp:.1f}°C - Verificați sistemul de răcire!")
    
    if not alerts:
        alerts.append("✅ Toate parametrii în limite normale - Sesiune Clean!")
    
    return alerts

# ============================================================================
# APLICAȚIA PRINCIPALĂ
# ============================================================================
def main():
    st.title("🏎️ LZTuned Intelligence Suite (MS42 Expert)")
    st.markdown("---")
    
    # ========== SIDEBAR: UPLOAD ==========
    st.sidebar.header("📁 Data Upload")
    
    xml_file = st.sidebar.file_uploader("XML Definition (RomRaider)", type=['xml'])
    bin_file = st.sidebar.file_uploader("BIN File (ECU Dump)", type=['bin'])
    csv_file = st.sidebar.file_uploader("CSV Log (Telemetry)", type=['csv'])
    
    st.sidebar.markdown("---")
    
    # Verificare fișiere încărcate
    if not all([xml_file, bin_file, csv_file]):
        st.info("👆 Vă rugăm încărcați toate cele 3 fișiere în sidebar pentru a începe analiza.")
        return
    
    # ========== PROCESARE DATE ==========
    with st.spinner("🔄 Procesare XML..."):
        xml_data = parse_romraider_xml(xml_file.read())
    
    with st.spinner("🔄 Citire BIN..."):
        bin_data = read_bin_file(bin_file.read(), xml_data)
    
    with st.spinner("🔄 Parsare CSV..."):
        log_df = parse_telemetry_csv(csv_file.read())
    
    # Verificare validitate
    if not bin_data['valid']:
        st.error("❌ Fișierul BIN nu corespunde definițiilor XML!")
        return
    
    st.sidebar.success(f"✅ Software ID: {bin_data['software_id']}")
    st.sidebar.success(f"✅ Maps detectate: {len(bin_data['maps'])}")
    st.sidebar.success(f"✅ Log samples: {len(log_df)}")
    
    # ========== SELECTARE HĂRȚI ==========
    st.sidebar.markdown("---")
    st.sidebar.header("🗺️ Map Selection")
    
    available_maps = list(bin_data['maps'].keys())
    selected_map = st.sidebar.selectbox("Selectează hartă pentru analiză:", available_maps)
    
    # ========== SLIDER PRINCIPAL ==========
    st.sidebar.markdown("---")
    st.sidebar.header("⏱️ Timeline Control")
    
    current_idx = st.sidebar.slider(
        "Poziție în log:",
        min_value=0,
        max_value=len(log_df)-1,
        value=0,
        step=1
    )
    
    # Detectare coloane relevante
    rpm_cols = [c for c in log_df.columns if 'rpm' in c.lower()]
    load_cols = [c for c in log_df.columns if 'load' in c.lower()]
    knock_cols = [c for c in log_df.columns if 'knock' in c.lower()]
    iat_cols = [c for c in log_df.columns if 'iat' in c.lower() or 'intake' in c.lower()]
    coolant_cols = [c for c in log_df.columns if 'coolant' in c.lower() or 'temp' in c.lower()]
    
    rpm_col = rpm_cols[0] if rpm_cols else None
    load_col = load_cols[0] if load_cols else None
    knock_col = knock_cols[0] if knock_cols else None
    iat_col = iat_cols[0] if iat_cols else None
    coolant_col = coolant_cols[0] if coolant_cols else None
    
    # ========== TABS ==========
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔥 Ignition & Knock",
        "⚙️ VANOS Drift Spec",
        "📊 Telemetrie Sincronizată",
        "📋 Expert Diagnostic"
    ])
    
    # ========== TAB 1: IGNITION & KNOCK ==========
    with tab1:
        st.header("Ignition Map Analysis")
        
        if selected_map and rpm_col and load_col:
            map_data = bin_data['maps'][selected_map]
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = create_heatmap_with_trace(
                    map_data, log_df, current_idx,
                    rpm_col, load_col, knock_col
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📍 Current Point")
                
                if current_idx < len(log_df):
                    row = log_df.iloc[current_idx]
                    st.metric("RPM", f"{row[rpm_col]:.0f}")
                    st.metric("Load", f"{row[load_col]:.2f}")
                    
                    if knock_col:
                        knock_val = row[knock_col]
                        st.metric("Knock Retard", f"{knock_val:.2f}°",
                                delta=f"{'⚠️ ACTIVE' if knock_val > 0 else '✅ OK'}")
                    
                    # Calculează target din hartă
                    target = bilinear_interpolation(
                        row[rpm_col], row[load_col],
                        map_data['x_axis'], map_data['y_axis'], map_data['z_data']
                    )
                    st.metric("Target (din hartă)", f"{target:.2f}°")
        else:
            st.warning("Lipsesc coloanele RPM sau Load în log!")
    
    # ========== TAB 2: VANOS DRIFT ==========
    with tab2:
        st.header("VANOS Position Analysis")
        
        vanos_target_cols = [c for c in log_df.columns if 'vanos' in c.lower() and 'target' in c.lower()]
        vanos_actual_cols = [c for c in log_df.columns if 'vanos' in c.lower() and 'actual' in c.lower()]
        
        if vanos_target_cols and vanos_actual_cols:
            fig_vanos = create_vanos_comparison(
                log_df, current_idx,
                vanos_target_cols[0], vanos_actual_cols[0]
            )
            st.plotly_chart(fig_vanos, use_container_width=True)
            
            # Calculează latență
            if current_idx < len(log_df):
                target_val = log_df.iloc[current_idx][vanos_target_cols[0]]
                actual_val = log_df.iloc[current_idx][vanos_actual_cols[0]]
                delta = abs(target_val - actual_val)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Target", f"{target_val:.2f}°")
                col2.metric("Actual", f"{actual_val:.2f}°")
                col3.metric("Delta", f"{delta:.2f}°", 
                          delta_color="inverse")
        else:
            st.warning("Nu s-au găsit coloane VANOS în log!")
    
    # ========== TAB 3: TELEMETRIE ==========
    with tab3:
        st.header("Live Telemetry Dashboard")
        
        if current_idx < len(log_df):
            row = log_df.iloc[current_idx]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if rpm_col:
                    fig_rpm = create_gauge(row[rpm_col], "RPM", 8000, "rpm")
                    st.plotly_chart(fig_rpm, use_container_width=True)
            
            with col2:
                if load_col:
                    fig_load = create_gauge(row[load_col], "Load", 100, "%")
                    st.plotly_chart(fig_load, use_container_width=True)
            
            with col3:
                if iat_col:
                    fig_iat = create_gauge(row[iat_col], "IAT", 100, "°C")
                    st.plotly_chart(fig_iat, use_container_width=True)
            
            with col4:
                if coolant_col:
                    fig_coolant = create_gauge(row[coolant_col], "Coolant", 120, "°C")
                    st.plotly_chart(fig_coolant, use_container_width=True)
            
            # Time series
            st.subheader("📈 Historical Trends")
            
            window = 200
            start = max(0, current_idx - window)
            end = min(len(log_df), current_idx + window)
            subset = log_df.iloc[start:end]
            
            fig_trends = make_subplots(
                rows=2, cols=1,
                subplot_titles=('RPM & Load', 'Temperatures')
            )
            
            if rpm_col:
                fig_trends.add_trace(
                    go.Scatter(y=subset[rpm_col], name='RPM', line=dict(color='blue')),
                    row=1, col=1
                )
            
            if load_col:
                fig_trends.add_trace(
                    go.Scatter(y=subset[load_col], name='Load', line=dict(color='red')),
                    row=1, col=1
                )
            
            if iat_col:
                fig_trends.add_trace(
                    go.Scatter(y=subset[iat_col], name='IAT', line=dict(color='orange')),
                    row=2, col=1
                )
            
            if coolant_col:
                fig_trends.add_trace(
                    go.Scatter(y=subset[coolant_col], name='Coolant', line=dict(color='green')),
                    row=2, col=1
                )
            
            fig_trends.add_vline(x=current_idx-start, line_dash="dash", line_color="black")
            fig_trends.update_layout(height=600, hovermode='x unified')
            st.plotly_chart(fig_trends, use_container_width=True)
    
    # ========== TAB 4: DIAGNOSTIC ==========
    with tab4:
        st.header("🔬 Expert Diagnostic Report")
        
        # Residence Time
        st.subheader("Operating Range Analysis")
        
        if rpm_col and load_col:
            x_bins = np.linspace(log_df[rpm_col].min(), log_df[rpm_col].max(), 20)
            y_bins = np.linspace(log_df[load_col].min(), log_df[load_col].max(), 20)
            
            residence = calculate_residence_time(log_df, rpm_col, load_col, x_bins, y_bins)
            
            fig_residence = create_residence_heatmap(residence, x_bins, y_bins)
            st.plotly_chart(fig_residence, use_container_width=True)
        
        # Alerte automate
        st.subheader("⚠️ Automated Alerts")
        alerts = generate_diagnostic_report(log_df)
        
        for alert in alerts:
            if "✅" in alert:
                st.success(alert)
            elif "⚠️" in alert:
                st.warning(alert)
            elif "🚨" in alert or "🔥" in alert:
                st.error(alert)
            else:
                st.info(alert)
        
        # Statistici generale
        st.subheader("📊 Session Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Samples", len(log_df))
            if rpm_col:
                st.metric("Avg RPM", f"{log_df[rpm_col].mean():.0f}")
        
        with col2:
            if load_col:
                st.metric("Avg Load", f"{log_df[load_col].mean():.1f}%")
                st.metric("Max Load", f"{log_df[load_col].max():.1f}%")
        
        with col3:
            if knock_col:
                total_knock = log_df[knock_col].sum()
                st.metric("Total Knock Retard", f"{total_knock:.1f}°")
                st.metric("Knock Events", len(log_df[log_df[knock_col] > 0]))

if __name__ == "__main__":
    main()
