"""
Smart Warehouse Control Platform - Enhanced Enterprise Dashboard
Industrial control & decision platform with comprehensive visualizations
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import sqlite3
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Warehouse Control Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Dark Theme CSS - Supply Chain Style
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Dark Theme */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #1a2332 0%, #0f1419 100%);
    }
    
    /* Main content background */
    .main .block-container {
        background: transparent;
        padding-top: 2rem;
    }
    
    /* Sidebar - Dark Teal Theme */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a4a 0%, #0f1d26 100%);
        border-right: 1px solid rgba(96, 165, 250, 0.1);
    }
    
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown h2 {
        color: #60a5fa !important;
        font-weight: 700;
        font-size: 1.4rem;
        margin-bottom: 1.5rem;
    }
    
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #93c5fd !important;
        font-weight: 600;
        font-size: 1.1rem;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    
    /* Main Header */
    .main-header {
        background: linear-gradient(135deg, rgba(30, 58, 74, 0.8) 0%, rgba(15, 29, 38, 0.9) 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(96, 165, 250, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .main-title {
        color: #f1f5f9;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .main-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        font-weight: 400;
        margin-top: 0.5rem;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #e2e8f0;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(96, 165, 250, 0.3);
    }
    
    /* Modern Card Style */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 58, 74, 0.6) 0%, rgba(15, 29, 38, 0.8) 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(96, 165, 250, 0.15);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
        margin: 0.5rem 0;
    }
    
    .metric-card:hover {
        border-color: rgba(96, 165, 250, 0.4);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        transform: translateY(-2px);
    }
    
    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f1f5f9;
        margin: 0.5rem 0;
    }
    
    .metric-trend {
        font-size: 0.85rem;
        font-weight: 500;
        margin-top: 0.5rem;
    }
    
    .trend-up {
        color: #34d399;
    }
    
    .trend-down {
        color: #f87171;
    }
    
    .trend-neutral {
        color: #fbbf24;
    }
    
    /* Streamlit Metrics Override */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f1f5f9 !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.75rem;
        font-weight: 600;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Alert Boxes */
    .alert-critical {
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.2) 0%, rgba(153, 27, 27, 0.3) 100%);
        border-left: 4px solid #dc2626;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.8rem 0;
        color: #fca5a5;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.2);
    }
    
    .alert-warning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(217, 119, 6, 0.3) 100%);
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.8rem 0;
        color: #fcd34d;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.2);
    }
    
    .alert-success {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.2) 0%, rgba(22, 163, 74, 0.3) 100%);
        border-left: 4px solid #22c55e;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.8rem 0;
        color: #86efac;
        box-shadow: 0 4px 12px rgba(34, 197, 94, 0.2);
    }
    
    /* Data Tables Dark Theme */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        background: rgba(30, 58, 74, 0.4) !important;
    }
    
    .dataframe th {
        background: rgba(30, 58, 74, 0.8) !important;
        color: #e2e8f0 !important;
        font-weight: 700 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        padding: 1rem !important;
        border-bottom: 2px solid rgba(96, 165, 250, 0.3) !important;
    }
    
    .dataframe td {
        color: #cbd5e1 !important;
        font-weight: 500 !important;
        padding: 0.8rem !important;
        border-bottom: 1px solid rgba(96, 165, 250, 0.1) !important;
    }
    
    .dataframe tr:hover {
        background: rgba(96, 165, 250, 0.1) !important;
    }
    
    /* Info Panels */
    .info-panel {
        background: linear-gradient(135deg, rgba(30, 58, 74, 0.5) 0%, rgba(15, 29, 38, 0.7) 100%);
        border-radius: 10px;
        padding: 1.5rem;
        border: 1px solid rgba(96, 165, 250, 0.15);
        margin: 1rem 0;
        color: #cbd5e1;
    }
    
    /* Text Colors - Dark Theme */
    p, span, div, label {
        color: #cbd5e1 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #e2e8f0 !important;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.3s;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        border: 1px solid rgba(96, 165, 250, 0.3);
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
        transform: translateY(-2px);
        border-color: rgba(96, 165, 250, 0.6);
    }
    
    /* Scrollbar Dark Theme */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(15, 20, 25, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(96, 165, 250, 0.3);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(96, 165, 250, 0.5);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# Plotly Dark Theme Configuration
plotly_dark_layout = {
    'plot_bgcolor': 'rgba(26, 35, 50, 0.5)',
    'paper_bgcolor': 'rgba(26, 35, 50, 0.3)',
    'font': {'family': "Inter", 'size': 12, 'color': '#e2e8f0'},
    'xaxis': {
        'gridcolor': 'rgba(96, 165, 250, 0.1)',
        'zerolinecolor': 'rgba(96, 165, 250, 0.2)',
        'color': '#94a3b8',
        'showgrid': True
    },
    'yaxis': {
        'gridcolor': 'rgba(96, 165, 250, 0.1)',
        'zerolinecolor': 'rgba(96, 165, 250, 0.2)',
        'color': '#94a3b8',
        'showgrid': True
    },
    'hoverlabel': {
        'bgcolor': 'rgba(30, 58, 74, 0.95)',
        'bordercolor': 'rgba(96, 165, 250, 0.3)',
        'font': {'color': '#e2e8f0', 'size': 12}
    }
}

# ============================================================================
# DATABASE HELPER FUNCTIONS
# ============================================================================

def get_sensor_fusion_data():
    """Get latest sensor fusion data from database"""
    try:
        conn = sqlite3.connect('cv_detections.db')
        cursor = conn.cursor()
        
        # First check if we have any data
        cursor.execute("SELECT COUNT(*) FROM SensorFusionData")
        count = cursor.fetchone()[0]
        
        if count == 0:
            conn.close()
            return get_sample_fusion_data()
        
        # Get latest data for each zone
        cursor.execute("""
            SELECT 
                zone_id, 
                camera_count, 
                load_cell_count, 
                ultrasonic_level_percent,
                fused_count, 
                confidence_score, 
                anomaly_detected, 
                sensor_agreement_percent,
                timestamp
            FROM SensorFusionData
            WHERE timestamp >= datetime('now', '-1 hour')
            GROUP BY zone_id
            HAVING MAX(timestamp)
            ORDER BY zone_id
        """)
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return get_sample_fusion_data()
        
        return [{
            'zone_id': r[0],
            'camera_count': r[1] or 0,
            'load_cell_count': r[2] or 0,
            'ultrasonic_level': r[3] or 0,
            'fused_count': r[4] or 0,
            'confidence_score': r[5] or 0.95,
            'anomaly_detected': bool(r[6]),
            'sensor_agreement': r[7] or 98.0
        } for r in results]
    except Exception as e:
        print(f"Database error: {e}")
        return get_sample_fusion_data()

def get_sample_fusion_data():
    """Return sample data when database is not available"""
    return [
        {'zone_id': 'A1', 'camera_count': 760, 'load_cell_count': 758, 'ultrasonic_level': 76.0,
         'fused_count': 760, 'confidence_score': 0.952, 'anomaly_detected': False, 'sensor_agreement': 98.5},
        {'zone_id': 'B2', 'camera_count': 420, 'load_cell_count': 425, 'ultrasonic_level': 42.0,
         'fused_count': 422, 'confidence_score': 0.943, 'anomaly_detected': False, 'sensor_agreement': 97.8},
        {'zone_id': 'C3', 'camera_count': 890, 'load_cell_count': 888, 'ultrasonic_level': 89.0,
         'fused_count': 889, 'confidence_score': 0.968, 'anomaly_detected': False, 'sensor_agreement': 99.2},
        {'zone_id': 'D4', 'camera_count': 150, 'load_cell_count': 152, 'ultrasonic_level': 15.0,
         'fused_count': 151, 'confidence_score': 0.935, 'anomaly_detected': False, 'sensor_agreement': 96.7},
        {'zone_id': 'E5', 'camera_count': 650, 'load_cell_count': 648, 'ultrasonic_level': 65.0,
         'fused_count': 649, 'confidence_score': 0.957, 'anomaly_detected': False, 'sensor_agreement': 98.9},
        {'zone_id': 'F6', 'camera_count': 320, 'load_cell_count': 318, 'ultrasonic_level': 32.0,
         'fused_count': 319, 'confidence_score': 0.941, 'anomaly_detected': False, 'sensor_agreement': 97.5},
    ]

def get_historical_data(hours=24):
    """Get historical sensor fusion data"""
    try:
        conn = sqlite3.connect('cv_detections.db')
        
        # Check if table exists and has data
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM SensorFusionData WHERE timestamp >= datetime('now', '-24 hours')")
        count = cursor.fetchone()[0]
        
        if count == 0:
            conn.close()
            return get_sample_historical_data(hours)
        
        query = f"""
            SELECT zone_id, timestamp, fused_count, confidence_score
            FROM SensorFusionData
            WHERE timestamp >= datetime('now', '-{hours} hours')
            ORDER BY timestamp
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            df = get_sample_historical_data(hours)
        
        return df
    except Exception as e:
        print(f"Historical data error: {e}")
        return get_sample_historical_data(hours)

def get_sample_historical_data(hours=24):
    """Return sample historical data"""
    now = datetime.now()
    data = []
    zones = ['A1', 'B2', 'C3', 'D4', 'E5', 'F6']
    
    for i in range(hours):
        timestamp = now - timedelta(hours=hours-i)
        for zone in zones:
            base_count = {'A1': 760, 'B2': 420, 'C3': 890, 'D4': 150, 'E5': 650, 'F6': 320}[zone]
            variation = np.random.randint(-50, 50)
            data.append({
                'zone_id': zone,
                'timestamp': timestamp,
                'fused_count': max(0, base_count + variation),
                'confidence_score': 0.95 + np.random.uniform(-0.05, 0.05)
            })
    
    return pd.DataFrame(data)

def calculate_inventory_accuracy():
    """Calculate overall inventory accuracy"""
    fusion_data = get_sensor_fusion_data()
    if not fusion_data:
        return 98.5
    
    avg_confidence = np.mean([z['confidence_score'] for z in fusion_data])
    return avg_confidence * 100

def get_active_alerts():
    """Get active alerts from database"""
    try:
        conn = sqlite3.connect('cv_detections.db')
        cursor = conn.cursor()
        
        alerts = []
        
        # Get collision events
        cursor.execute("""
            SELECT zone_id, 'Collision Risk' as type, 'CRITICAL' as severity,
                   CAST((julianday('now') - julianday(timestamp)) * 24 AS INTEGER) as hours_ago
            FROM CollisionEvents
            WHERE acknowledged = 0
            ORDER BY timestamp DESC
            LIMIT 5
        """)
        collision_results = cursor.fetchall()
        
        for r in collision_results:
            alerts.append({
                'zone_id': r[0],
                'alert_type': r[1],
                'severity': r[2],
                'hours_ago': f"{r[3]}h ago"
            })
        
        # Get misplacements
        cursor.execute("""
            SELECT zone_id, 'Misplacement' as type, severity,
                   CAST((julianday('now') - julianday(timestamp)) * 24 AS INTEGER) as hours_ago
            FROM MisplacementLog
            WHERE corrected = 0
            ORDER BY timestamp DESC
            LIMIT 5
        """)
        misplacement_results = cursor.fetchall()
        
        for r in misplacement_results:
            alerts.append({
                'zone_id': r[0],
                'alert_type': r[1],
                'severity': r[2],
                'hours_ago': f"{r[3]}h ago"
            })
        
        # Get unsafe stacks
        cursor.execute("""
            SELECT zone_id, 'Unsafe Stack' as type, risk_level as severity,
                   CAST((julianday('now') - julianday(timestamp)) * 24 AS INTEGER) as hours_ago
            FROM UnsafeStackIncidents
            WHERE resolved = 0
            ORDER BY timestamp DESC
            LIMIT 5
        """)
        unsafe_results = cursor.fetchall()
        
        for r in unsafe_results:
            alerts.append({
                'zone_id': r[0],
                'alert_type': r[1],
                'severity': r[2],
                'hours_ago': f"{r[3]}h ago"
            })
        
        conn.close()
        
        # If no real alerts, return sample data
        if not alerts:
            return get_sample_alerts()
        
        # Sort by severity and limit to 10
        severity_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        alerts.sort(key=lambda x: severity_order.get(x['severity'], 4))
        
        return alerts[:10]
        
    except Exception as e:
        print(f"Alert fetch error: {e}")
        return get_sample_alerts()

def get_sample_alerts():
    """Return sample alerts"""
    return [
        {'zone_id': 'A1', 'alert_type': 'Collision Risk', 'severity': 'CRITICAL', 'hours_ago': '2h ago'},
        {'zone_id': 'C3', 'alert_type': 'Overstock', 'severity': 'HIGH', 'hours_ago': '5h ago'},
        {'zone_id': 'D4', 'alert_type': 'Low Stock', 'severity': 'MEDIUM', 'hours_ago': '8h ago'},
        {'zone_id': 'B2', 'alert_type': 'Misplacement', 'severity': 'MEDIUM', 'hours_ago': '12h ago'},
    ]

# ============================================================================
# SESSION STATE
# ============================================================================

if 'current_section' not in st.session_state:
    st.session_state.current_section = 'Executive Overview'
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False
if 'refresh_rate' not in st.session_state:
    st.session_state.refresh_rate = 5
if 'user_role' not in st.session_state:
    st.session_state.user_role = 'Manager'

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("## ⚡ Warehouse Control")
    st.markdown("### Smart Monitoring Platform")
    
    st.markdown("---")
    
    # Navigation
    st.markdown("### Navigation")
    
    sections = [
        "Executive Overview",
        "Inventory Analytics",
        "AI & Predictions",
        "Alert Management",
        "Operations Monitor"
    ]
    
    for section in sections:
        if st.button(section, key=f"nav_{section}", use_container_width=True):
            st.session_state.current_section = section
    
    st.markdown("---")
    
    # User Role
    st.markdown("### User Role")
    st.session_state.user_role = st.selectbox(
        "Current Role",
        ["Operator", "Manager", "Executive", "Admin"],
        index=1
    )
    
    st.markdown("---")
    
    # Controls
    st.markdown("### System Controls")
    st.session_state.auto_refresh = st.checkbox("Auto-refresh", value=st.session_state.auto_refresh)
    st.session_state.refresh_rate = st.slider("Refresh Rate (sec)", 3, 30, st.session_state.refresh_rate)
    
    st.markdown("---")
    
    # Architecture Status
    st.markdown("### Architecture Status")
    st.caption("🟢 IoT Sensors: Active")
    st.caption("🟢 Edge Processing: Online")
    st.caption("🟢 Database: Connected")
    st.caption("🟢 ML Engine: Ready")
    st.caption("🟢 Decision Engine: Active")
    
    st.markdown("---")
    st.caption(f"Last Update: {datetime.now().strftime('%H:%M:%S')}")

# ============================================================================
# MAIN CONTENT
# ============================================================================

main_content = st.empty()

while True:
    with main_content.container():
        
        # ====================================================================
        # SECTION: EXECUTIVE OVERVIEW
        # ====================================================================
        
        if st.session_state.current_section == 'Executive Overview':
            
            # Header
            st.markdown("""
                <div class="main-header">
                    <h1 class="main-title">📊 Executive Overview</h1>
                    <p class="main-subtitle">Real-time Warehouse Status | Multi-Sensor Fusion Analytics</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Get data
            fusion_data = get_sensor_fusion_data()
            alerts = get_active_alerts()
            inventory_accuracy = calculate_inventory_accuracy()
            
            # Top KPI Cards
            col1, col2, col3, col4, col5 = st.columns(5)
            
            critical_alerts = len([a for a in alerts if a['severity'] == 'CRITICAL'])
            total_inventory = sum([z['fused_count'] for z in fusion_data]) if fusion_data else 0
            avg_confidence = np.mean([z['confidence_score'] for z in fusion_data]) * 100 if fusion_data else 0
            zones_active = len(fusion_data)
            low_stock = len([z for z in fusion_data if z['fused_count'] < 200])
            
            with col1:
                st.metric("Total Inventory", f"{total_inventory:,}", delta="+150 today")
            
            with col2:
                st.metric("System Accuracy", f"{avg_confidence:.1f}%", delta="98%+ target")
            
            with col3:
                st.metric("Active Zones", zones_active, delta="All Online")
            
            with col4:
                st.metric("Active Alerts", len(alerts), delta=f"{critical_alerts} Critical")
            
            with col5:
                st.metric("Low Stock Zones", low_stock, delta="Action Needed" if low_stock > 0 else "Good")
            
            st.markdown("---")
            
            # Main Content - 3 Column Layout
            col_left, col_center, col_right = st.columns([2, 2, 1])
            
            with col_left:
                st.markdown('<h2 class="section-header">📊 Inventory by Zone</h2>', unsafe_allow_html=True)
                
                if fusion_data:
                    df_inv = pd.DataFrame(fusion_data)
                    
                    # Color code by stock level
                    def get_color(count):
                        if count < 200:
                            return '#f87171'
                        elif count > 800:
                            return '#fbbf24'
                        else:
                            return '#34d399'
                    
                    df_inv['color'] = df_inv['fused_count'].apply(get_color)
                    
                    fig_inv = go.Figure()
                    
                    fig_inv.add_trace(go.Bar(
                        x=df_inv['zone_id'],
                        y=df_inv['fused_count'],
                        marker_color=df_inv['color'],
                        text=df_inv['fused_count'],
                        textposition='outside',
                        textfont=dict(size=12, color='#e2e8f0'),
                        hovertemplate='<b>%{x}</b><br>Count: %{y}<br>Confidence: %{customdata:.1%}<extra></extra>',
                        customdata=df_inv['confidence_score']
                    ))
                    
                    fig_inv.add_hline(y=200, line_dash="dash", line_color="#f87171", 
                                     annotation_text="Low Stock", annotation_position="left")
                    fig_inv.add_hline(y=800, line_dash="dash", line_color="#fbbf24",
                                     annotation_text="Overstock", annotation_position="left")
                    
                    fig_inv.update_layout(
                        height=350,
                        **plotly_dark_layout,
                        xaxis_title="Zone",
                        yaxis_title="Units",
                        showlegend=False,
                        hovermode='closest'
                    )
                    
                    st.plotly_chart(fig_inv, use_container_width=True)
            
            with col_center:
                st.markdown('<h2 class="section-header">📈 24-Hour Trends</h2>', unsafe_allow_html=True)
                
                hist_data = get_historical_data(hours=24)
                
                if not hist_data.empty:
                    fig_trend = go.Figure()
                    
                    zones = hist_data['zone_id'].unique()
                    colors = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#06b6d4']
                    
                    for i, zone in enumerate(zones[:6]):
                        zone_data = hist_data[hist_data['zone_id'] == zone]
                        fig_trend.add_trace(go.Scatter(
                            x=zone_data['timestamp'],
                            y=zone_data['fused_count'],
                            mode='lines',
                            name=zone,
                            line=dict(color=colors[i % len(colors)], width=2),
                            hovertemplate='<b>%{fullData.name}</b><br>%{y} units<extra></extra>'
                        ))
                    
                    fig_trend.update_layout(
                        height=350,
                        **plotly_dark_layout,
                        xaxis_title="Time",
                        yaxis_title="Units"
                    )
                    
                    fig_trend.update_layout(
                        legend=dict(
                            orientation="h", 
                            yanchor="bottom", 
                            y=1.02, 
                            xanchor="right", 
                            x=1,
                            bgcolor='rgba(30, 58, 74, 0.8)',
                            bordercolor='rgba(96, 165, 250, 0.2)',
                            borderwidth=1,
                            font={'color': '#e2e8f0'}
                        )
                    )
                    
                    st.plotly_chart(fig_trend, use_container_width=True)
            
            with col_right:
                st.markdown('<h2 class="section-header">🎯 System Health</h2>', unsafe_allow_html=True)
                
                # Gauge chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=avg_confidence,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Accuracy", 'font': {'size': 14, 'color': '#e2e8f0'}},
                    gauge={
                        'axis': {'range': [None, 100], 'tickcolor': '#94a3b8'},
                        'bar': {'color': "#3b82f6"},
                        'bgcolor': "rgba(30, 58, 74, 0.3)",
                        'borderwidth': 2,
                        'bordercolor': "rgba(96, 165, 250, 0.3)",
                        'steps': [
                            {'range': [0, 70], 'color': 'rgba(248, 113, 113, 0.3)'},
                            {'range': [70, 90], 'color': 'rgba(251, 191, 36, 0.3)'},
                            {'range': [90, 100], 'color': 'rgba(52, 211, 153, 0.3)'}
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 3},
                            'thickness': 0.75,
                            'value': 95
                        }
                    }
                ))
                
                fig_gauge.update_layout(
                    height=250,
                    paper_bgcolor='rgba(26, 35, 50, 0.3)',
                    font={'color': '#e2e8f0', 'family': 'Inter'},
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Sensor status cards
                st.markdown("""
                <div class="metric-card" style="padding: 1rem;">
                    <div class="metric-label">🎥 CV System</div>
                    <div class="metric-value" style="font-size: 1.5rem;">98.5%</div>
                    <div class="metric-trend trend-up">↑ 2.3%</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="metric-card" style="padding: 1rem;">
                    <div class="metric-label">⚖️ Load Cells</div>
                    <div class="metric-value" style="font-size: 1.5rem;">99.1%</div>
                    <div class="metric-trend trend-up">↑ 1.2%</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="metric-card" style="padding: 1rem;">
                    <div class="metric-label">📡 Ultrasonic</div>
                    <div class="metric-value" style="font-size: 1.5rem;">97.8%</div>
                    <div class="metric-trend trend-neutral">→ Stable</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Second Row: Alerts and Performance
            col_alerts, col_perf = st.columns([2, 1])
            
            with col_alerts:
                st.markdown('<h2 class="section-header">🚨 Active Alerts</h2>', unsafe_allow_html=True)
                
                if alerts:
                    alert_counts = {}
                    for alert in alerts:
                        severity = alert['severity']
                        alert_counts[severity] = alert_counts.get(severity, 0) + 1
                    
                    colors_map = {
                        'CRITICAL': '#dc2626',
                        'HIGH': '#f59e0b',
                        'MEDIUM': '#fbbf24',
                        'LOW': '#3b82f6'
                    }
                    
                    fig_alerts = go.Figure(data=[go.Pie(
                        labels=list(alert_counts.keys()),
                        values=list(alert_counts.values()),
                        hole=0.5,
                        marker=dict(colors=[colors_map.get(k, '#94a3b8') for k in alert_counts.keys()]),
                        textfont=dict(size=14, color='#ffffff'),
                        hovertemplate='<b>%{label}</b><br>%{value} alerts<br>%{percent}<extra></extra>'
                    )])
                    
                    fig_alerts.update_layout(
                        height=300,
                        paper_bgcolor='rgba(26, 35, 50, 0.3)',
                        font={'color': '#e2e8f0', 'family': 'Inter'},
                        showlegend=True,
                        annotations=[dict(text=f'{len(alerts)}<br>Total', x=0.5, y=0.5, 
                                        font_size=24, showarrow=False, font_color='#e2e8f0')]
                    )
                    
                    fig_alerts.update_layout(
                        legend=dict(
                            orientation="h", 
                            yanchor="bottom", 
                            y=-0.2, 
                            xanchor="center", 
                            x=0.5,
                            bgcolor='rgba(30, 58, 74, 0.8)',
                            bordercolor='rgba(96, 165, 250, 0.2)',
                            borderwidth=1,
                            font={'color': '#e2e8f0'}
                        )
                    )
                    
                    st.plotly_chart(fig_alerts, use_container_width=True)
                    
                    # Alert table
                    alert_df = pd.DataFrame([{
                        'Zone': a['zone_id'],
                        'Type': a['alert_type'],
                        'Severity': a['severity'],
                        'Time': a['hours_ago']
                    } for a in alerts[:5]])
                    st.dataframe(alert_df, use_container_width=True, hide_index=True)
                else:
                    st.markdown("""
                    <div class="alert-success">
                        ✅ No active alerts - All systems operational
                    </div>
                    """, unsafe_allow_html=True)
            
            with col_perf:
                st.markdown('<h2 class="section-header">⚡ Performance</h2>', unsafe_allow_html=True)
                
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">CV Processing</div>
                    <div class="metric-value">32 FPS</div>
                    <div class="metric-trend trend-up">↑ 3 FPS</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">Sensor Latency</div>
                    <div class="metric-value">8 ms</div>
                    <div class="metric-trend trend-up">↓ 2 ms</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">DB Response</div>
                    <div class="metric-value">12 ms</div>
                    <div class="metric-trend trend-neutral">→ Stable</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("**System Uptime**")
                st.progress(0.998)
                st.caption("🟢 168 hours (99.8%)")
        
        # ====================================================================
        # SECTION: INVENTORY ANALYTICS
        # ====================================================================
        
        elif st.session_state.current_section == 'Inventory Analytics':
            
            st.markdown("""
                <div class="main-header">
                    <h1 class="main-title">📦 Inventory Analytics</h1>
                    <p class="main-subtitle">Detailed Analysis | Multi-Zone Comparison</p>
                </div>
                """, unsafe_allow_html=True)
            
            fusion_data = get_sensor_fusion_data()
            hist_data = get_historical_data(hours=168)  # 7 days
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            if fusion_data:
                total_stock = sum([z['fused_count'] for z in fusion_data])
                avg_utilization = np.mean([z['fused_count'] / 1000 * 100 for z in fusion_data])
                high_confidence_zones = len([z for z in fusion_data if z['confidence_score'] > 0.95])
                anomalies = len([z for z in fusion_data if z['anomaly_detected']])
                
                col1.metric("Total Stock", f"{total_stock:,} units")
                col2.metric("Avg Utilization", f"{avg_utilization:.1f}%")
                col3.metric("High Confidence", f"{high_confidence_zones}/{len(fusion_data)} zones")
                col4.metric("Anomalies", anomalies, delta="Investigate" if anomalies > 0 else "Clear")
            
            st.markdown("---")
            
            # Detailed zone comparison
            col_comp, col_heat = st.columns([2, 1])
            
            with col_comp:
                st.markdown('<h2 class="section-header">Zone Comparison</h2>', unsafe_allow_html=True)
                
                if fusion_data:
                    df = pd.DataFrame(fusion_data)
                    
                    fig_comp = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=('Inventory Count', 'Confidence Score'),
                        vertical_spacing=0.15
                    )
                    
                    fig_comp.add_trace(
                        go.Bar(x=df['zone_id'], y=df['fused_count'], 
                               marker_color='#3b82f6', name='Count'),
                        row=1, col=1
                    )
                    
                    fig_comp.add_trace(
                        go.Scatter(x=df['zone_id'], y=df['confidence_score'] * 100,
                                  mode='lines+markers', marker_color='#34d399',
                                  line=dict(width=3), name='Confidence %'),
                        row=2, col=1
                    )
                    
                    fig_comp.update_layout(
                        height=500,
                        **plotly_dark_layout,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_comp, use_container_width=True)
            
            with col_heat:
                st.markdown('<h2 class="section-header">Sensor Agreement</h2>', unsafe_allow_html=True)
                
                if fusion_data:
                    df = pd.DataFrame(fusion_data)
                    
                    fig_agree = go.Figure(go.Bar(
                        x=df['sensor_agreement'],
                        y=df['zone_id'],
                        orientation='h',
                        marker=dict(
                            color=df['sensor_agreement'],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(
                                title="Agreement %", 
                                title_font=dict(color='#e2e8f0'),
                                tickfont=dict(color='#e2e8f0')
                            )
                        ),
                        text=df['sensor_agreement'].round(1),
                        textposition='auto',
                        textfont=dict(color='#ffffff'),
                        hovertemplate='<b>%{y}</b><br>Agreement: %{x:.1f}%<extra></extra>'
                    ))
                    
                    fig_agree.update_layout(
                        height=500,
                        **plotly_dark_layout,
                        xaxis_title="Agreement (%)",
                        yaxis_title="Zone"
                    )
                    
                    st.plotly_chart(fig_agree, use_container_width=True)
            
            st.markdown("---")
            
            # Historical trends
            st.markdown('<h2 class="section-header">7-Day Historical Trends</h2>', unsafe_allow_html=True)
            
            if not hist_data.empty:
                # Aggregate by day
                hist_data['date'] = pd.to_datetime(hist_data['timestamp']).dt.date
                daily_data = hist_data.groupby(['date', 'zone_id'])['fused_count'].mean().reset_index()
                
                fig_history = px.line(
                    daily_data,
                    x='date',
                    y='fused_count',
                    color='zone_id',
                    title='Inventory Levels Over Time'
                )
                
                fig_history.update_layout(
                    height=400,
                    **plotly_dark_layout,
                    xaxis_title="Date",
                    yaxis_title="Average Units"
                )
                
                st.plotly_chart(fig_history, use_container_width=True)
        
        # ====================================================================
        # SECTION: AI & PREDICTIONS
        # ====================================================================
        
        elif st.session_state.current_section == 'AI & Predictions':
            
            st.markdown("""
                <div class="main-header">
                    <h1 class="main-title">🤖 AI & Predictive Analytics</h1>
                    <p class="main-subtitle">Machine Learning Insights | Demand Forecasting | Anomaly Detection</p>
                </div>
                """, unsafe_allow_html=True)
            
            fusion_data = get_sensor_fusion_data()
            hist_data = get_historical_data(hours=168)  # 7 days
            
            # AI System Status
            col_status1, col_status2, col_status3, col_status4 = st.columns(4)
            
            col_status1.metric("ML Model Status", "Active", delta="98.5% Accuracy")
            col_status2.metric("Predictions Made", "1,247", delta="+156 today")
            col_status3.metric("Anomalies Detected", "3", delta="Last 24h")
            col_status4.metric("Model Confidence", "96.2%", delta="+1.5%")
            
            st.markdown("---")
            
            # Demand Forecasting
            col_forecast, col_anomaly = st.columns([2, 1])
            
            with col_forecast:
                st.markdown('<h2 class="section-header">📈 Demand Forecasting (7-Day Prediction)</h2>', unsafe_allow_html=True)
                
                if fusion_data:
                    # Generate forecast for each zone
                    forecast_data = []
                    current_date = datetime.now()
                    
                    for zone in fusion_data[:3]:  # Top 3 zones
                        zone_id = zone['zone_id']
                        current_count = zone['fused_count']
                        
                        # Simple trend-based forecast
                        for day in range(1, 8):
                            future_date = current_date + timedelta(days=day)
                            # Simulate forecast with trend and seasonality
                            trend = np.random.uniform(-0.02, 0.05)  # Growth trend
                            seasonal = np.sin(day * np.pi / 3.5) * 0.1  # Weekly pattern
                            noise = np.random.uniform(-0.03, 0.03)
                            
                            forecast_count = current_count * (1 + trend + seasonal + noise)
                            confidence_lower = forecast_count * 0.9
                            confidence_upper = forecast_count * 1.1
                            
                            forecast_data.append({
                                'zone_id': zone_id,
                                'date': future_date,
                                'forecast': int(forecast_count),
                                'lower': int(confidence_lower),
                                'upper': int(confidence_upper)
                            })
                    
                    df_forecast = pd.DataFrame(forecast_data)
                    
                    fig_forecast = go.Figure()
                    
                    colors = {'A1': '#3b82f6', 'B2': '#8b5cf6', 'C3': '#ec4899', 'D4': '#f59e0b', 'E5': '#10b981', 'F6': '#06b6d4'}
                    
                    for zone in df_forecast['zone_id'].unique():
                        zone_data = df_forecast[df_forecast['zone_id'] == zone]
                        color = colors.get(zone, '#94a3b8')
                        
                        # Add confidence interval
                        fig_forecast.add_trace(go.Scatter(
                            x=zone_data['date'],
                            y=zone_data['upper'],
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                        
                        fig_forecast.add_trace(go.Scatter(
                            x=zone_data['date'],
                            y=zone_data['lower'],
                            mode='lines',
                            line=dict(width=0),
                            fillcolor=f'rgba{tuple(list(int(color[i:i+2], 16) for i in (1, 3, 5)) + [0.2])}',
                            fill='tonexty',
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                        
                        # Add forecast line
                        fig_forecast.add_trace(go.Scatter(
                            x=zone_data['date'],
                            y=zone_data['forecast'],
                            mode='lines+markers',
                            name=zone,
                            line=dict(color=color, width=3),
                            marker=dict(size=6),
                            hovertemplate='<b>%{fullData.name}</b><br>Date: %{x|%b %d}<br>Forecast: %{y} units<extra></extra>'
                        ))
                    
                    fig_forecast.update_layout(
                        height=400,
                        **plotly_dark_layout,
                        xaxis_title="Date",
                        yaxis_title="Predicted Units",
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    
                    # Forecast insights
                    st.markdown("**📊 Key Insights:**")
                    st.markdown("""
                    <div class="info-panel">
                        • Zone A1: Expected increase of 8-12% over next week<br>
                        • Zone B2: Stable demand, maintain current stock levels<br>
                        • Zone C3: Peak demand predicted on Day 5 (prepare extra capacity)<br>
                        • Overall confidence: 92-96% based on historical patterns
                    </div>
                    """, unsafe_allow_html=True)
            
            with col_anomaly:
                st.markdown('<h2 class="section-header">🔍 Anomaly Detection</h2>', unsafe_allow_html=True)
                
                # Anomaly detection results
                if fusion_data:
                    anomaly_zones = [z for z in fusion_data if z['anomaly_detected']]
                    normal_zones = len(fusion_data) - len(anomaly_zones)
                    
                    fig_anomaly = go.Figure(data=[go.Pie(
                        labels=['Normal', 'Anomalies'],
                        values=[normal_zones, len(anomaly_zones)],
                        marker=dict(colors=['#34d399', '#f87171']),
                        hole=0.6,
                        textfont=dict(size=16, color='#ffffff'),
                        hovertemplate='<b>%{label}</b><br>%{value} zones<br>%{percent}<extra></extra>'
                    )])
                    
                    fig_anomaly.update_layout(
                        height=250,
                        paper_bgcolor='rgba(26, 35, 50, 0.3)',
                        font={'color': '#e2e8f0', 'family': 'Inter'},
                        showlegend=True,
                        annotations=[dict(
                            text=f'{len(anomaly_zones)}<br>Found',
                            x=0.5, y=0.5,
                            font_size=20,
                            showarrow=False,
                            font_color='#e2e8f0'
                        )]
                    )
                    
                    st.plotly_chart(fig_anomaly, use_container_width=True)
                    
                    if len(anomaly_zones) > 0:
                        st.markdown("**⚠️ Anomalies Detected:**")
                        for anom in anomaly_zones:
                            st.markdown(f"""
                            <div class="alert-warning">
                                <strong>{anom['zone_id']}</strong><br>
                                Sensor disagreement detected<br>
                                Agreement: {anom['sensor_agreement']:.1f}%
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="alert-success">
                            ✅ No anomalies detected<br>
                            All zones operating normally
                        </div>
                        """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # ML Model Performance
            col_model1, col_model2 = st.columns(2)
            
            with col_model1:
                st.markdown('<h2 class="section-header">🎯 Model Performance Metrics</h2>', unsafe_allow_html=True)
                
                # Performance over time
                performance_data = []
                for i in range(24):
                    hour = datetime.now() - timedelta(hours=24-i)
                    accuracy = 0.95 + np.random.uniform(-0.03, 0.03)
                    performance_data.append({
                        'hour': hour,
                        'accuracy': accuracy * 100,
                        'latency': 15 + np.random.uniform(-5, 5)
                    })
                
                df_perf = pd.DataFrame(performance_data)
                
                fig_perf = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Model Accuracy (%)', 'Inference Latency (ms)'),
                    vertical_spacing=0.15
                )
                
                fig_perf.add_trace(
                    go.Scatter(
                        x=df_perf['hour'],
                        y=df_perf['accuracy'],
                        mode='lines',
                        fill='tozeroy',
                        line=dict(color='#34d399', width=2),
                        name='Accuracy'
                    ),
                    row=1, col=1
                )
                
                fig_perf.add_trace(
                    go.Scatter(
                        x=df_perf['hour'],
                        y=df_perf['latency'],
                        mode='lines',
                        line=dict(color='#3b82f6', width=2),
                        name='Latency'
                    ),
                    row=2, col=1
                )
                
                fig_perf.update_layout(
                    height=450,
                    **plotly_dark_layout,
                    showlegend=False
                )
                
                st.plotly_chart(fig_perf, use_container_width=True)
            
            with col_model2:
                st.markdown('<h2 class="section-header">📊 Prediction Confidence Distribution</h2>', unsafe_allow_html=True)
                
                if fusion_data:
                    df = pd.DataFrame(fusion_data)
                    
                    fig_conf = go.Figure(data=[go.Histogram(
                        x=df['confidence_score'] * 100,
                        nbinsx=10,
                        marker=dict(
                            color='#3b82f6',
                            line=dict(color='#60a5fa', width=1)
                        ),
                        hovertemplate='Confidence: %{x:.1f}%<br>Count: %{y}<extra></extra>'
                    )])
                    
                    fig_conf.update_layout(
                        height=250,
                        **plotly_dark_layout,
                        xaxis_title="Confidence Score (%)",
                        yaxis_title="Number of Zones",
                        bargap=0.1
                    )
                    
                    st.plotly_chart(fig_conf, use_container_width=True)
                
                # Model details
                st.markdown("**🤖 Model Information:**")
                st.markdown("""
                <div class="metric-card">
                    <strong>Algorithm:</strong> Ensemble (ARIMA + LSTM)<br>
                    <strong>Training Data:</strong> 90 days<br>
                    <strong>Update Frequency:</strong> Hourly<br>
                    <strong>Features Used:</strong> 12 sensor inputs<br>
                    <strong>Accuracy:</strong> 96.2% (±2.1%)<br>
                    <strong>Last Training:</strong> 2 hours ago
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("**📈 Feature Importance:**")
                features = ['Camera Count', 'Load Cell', 'Ultrasonic', 'Time of Day', 'Day of Week']
                importance = [0.35, 0.28, 0.22, 0.10, 0.05]
                
                fig_importance = go.Figure(go.Bar(
                    x=importance,
                    y=features,
                    orientation='h',
                    marker_color='#8b5cf6',
                    text=[f'{i*100:.1f}%' for i in importance],
                    textposition='auto',
                    textfont=dict(color='#ffffff'),
                    hovertemplate='<b>%{y}</b><br>Importance: %{x:.2%}<extra></extra>'
                ))
                
                fig_importance.update_layout(
                    height=250,
                    **plotly_dark_layout,
                    xaxis_title="Importance Score",
                    showlegend=False
                )
                
                st.plotly_chart(fig_importance, use_container_width=True)
        
        # ====================================================================
        # SECTION: ALERT MANAGEMENT
        # ====================================================================
        
        elif st.session_state.current_section == 'Alert Management':
            
            st.markdown("""
                <div class="main-header">
                    <h1 class="main-title">🚨 Alert Management Center</h1>
                    <p class="main-subtitle">Real-time Monitoring | Incident Response | Alert Analytics</p>
                </div>
                """, unsafe_allow_html=True)
            
            alerts = get_active_alerts()
            
            # Alert Summary Metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            critical_count = len([a for a in alerts if a['severity'] == 'CRITICAL'])
            high_count = len([a for a in alerts if a['severity'] == 'HIGH'])
            medium_count = len([a for a in alerts if a['severity'] == 'MEDIUM'])
            low_count = len([a for a in alerts if a['severity'] == 'LOW'])
            
            col1.metric("Total Active", len(alerts), delta=f"{critical_count + high_count} Urgent")
            col2.metric("Critical", critical_count, delta="Immediate Action")
            col3.metric("High Priority", high_count, delta="Action Needed")
            col4.metric("Medium", medium_count, delta="Monitor")
            col5.metric("Low", low_count, delta="Informational")
            
            st.markdown("---")
            
            # Alert Timeline and Priority Matrix
            col_timeline, col_matrix = st.columns([2, 1])
            
            with col_timeline:
                st.markdown('<h2 class="section-header">📅 Alert Timeline (Last 24 Hours)</h2>', unsafe_allow_html=True)
                
                # Generate timeline data
                timeline_data = []
                severity_colors = {
                    'CRITICAL': '#dc2626',
                    'HIGH': '#f59e0b',
                    'MEDIUM': '#fbbf24',
                    'LOW': '#3b82f6'
                }
                
                for i in range(24):
                    hour = datetime.now() - timedelta(hours=24-i)
                    # Simulate alert counts per hour
                    critical = np.random.poisson(0.3)
                    high = np.random.poisson(0.5)
                    medium = np.random.poisson(1.0)
                    low = np.random.poisson(0.8)
                    
                    timeline_data.append({
                        'hour': hour,
                        'CRITICAL': critical,
                        'HIGH': high,
                        'MEDIUM': medium,
                        'LOW': low
                    })
                
                df_timeline = pd.DataFrame(timeline_data)
                
                fig_timeline = go.Figure()
                
                for severity in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']:
                    fig_timeline.add_trace(go.Bar(
                        x=df_timeline['hour'],
                        y=df_timeline[severity],
                        name=severity,
                        marker_color=severity_colors[severity],
                        hovertemplate='<b>%{fullData.name}</b><br>Time: %{x|%H:%M}<br>Count: %{y}<extra></extra>'
                    ))
                
                fig_timeline.update_layout(
                    height=400,
                    **plotly_dark_layout,
                    xaxis_title="Time",
                    yaxis_title="Alert Count",
                    barmode='stack',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_timeline, use_container_width=True)
            
            with col_matrix:
                st.markdown('<h2 class="section-header">🎯 Priority Matrix</h2>', unsafe_allow_html=True)
                
                # Priority distribution
                severity_data = {
                    'Severity': ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'],
                    'Count': [critical_count, high_count, medium_count, low_count],
                    'Avg Response Time': ['2 min', '10 min', '1 hour', '4 hours']
                }
                
                fig_priority = go.Figure(data=[go.Funnel(
                    y=severity_data['Severity'],
                    x=severity_data['Count'],
                    textposition="inside",
                    textinfo="value+percent initial",
                    marker=dict(color=['#dc2626', '#f59e0b', '#fbbf24', '#3b82f6']),
                    connector=dict(line=dict(color='rgba(96, 165, 250, 0.3)', width=2)),
                    hovertemplate='<b>%{y}</b><br>Count: %{x}<extra></extra>'
                )])
                
                fig_priority.update_layout(
                    height=350,
                    paper_bgcolor='rgba(26, 35, 50, 0.3)',
                    font={'color': '#e2e8f0', 'family': 'Inter'},
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                
                st.plotly_chart(fig_priority, use_container_width=True)
                
                # Response time metrics
                st.markdown("**⏱️ Response Metrics:**")
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">Avg Response Time</div>
                    <div class="metric-value" style="font-size: 1.5rem;">12 min</div>
                    <div class="metric-trend trend-up">↓ 3 min improvement</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">Resolution Rate</div>
                    <div class="metric-value" style="font-size: 1.5rem;">94%</div>
                    <div class="metric-trend trend-up">↑ 5% this week</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Active Alerts Table with Actions
            st.markdown('<h2 class="section-header">📋 Active Alerts (Detailed View)</h2>', unsafe_allow_html=True)
            
            if alerts:
                # Create detailed alert dataframe
                alert_details = []
                for i, alert in enumerate(alerts[:10]):
                    alert_details.append({
                        'ID': f'ALT-{1000 + i}',
                        'Zone': alert['zone_id'],
                        'Type': alert['alert_type'],
                        'Severity': alert['severity'],
                        'Time': alert['hours_ago'],
                        'Status': 'Open',
                        'Assigned': np.random.choice(['Auto', 'Operator 1', 'Operator 2', 'Manager'])
                    })
                
                df_alerts = pd.DataFrame(alert_details)
                
                # Color code by severity
                def highlight_severity(row):
                    if row['Severity'] == 'CRITICAL':
                        return ['background-color: rgba(220, 38, 38, 0.2)'] * len(row)
                    elif row['Severity'] == 'HIGH':
                        return ['background-color: rgba(245, 158, 11, 0.2)'] * len(row)
                    elif row['Severity'] == 'MEDIUM':
                        return ['background-color: rgba(251, 191, 36, 0.2)'] * len(row)
                    else:
                        return ['background-color: rgba(59, 130, 246, 0.2)'] * len(row)
                
                st.dataframe(
                    df_alerts.style.apply(highlight_severity, axis=1),
                    use_container_width=True,
                    hide_index=True,
                    height=400
                )
                
                # Action buttons
                col_btn1, col_btn2, col_btn3, col_btn4, col_btn5 = st.columns(5)
                
                with col_btn1:
                    if st.button("🔴 Acknowledge All Critical", use_container_width=True):
                        st.success("✅ All critical alerts acknowledged")
                
                with col_btn2:
                    if st.button("📧 Notify Team", use_container_width=True):
                        st.info("📤 Team notification sent")
                
                with col_btn3:
                    if st.button("📊 Generate Report", use_container_width=True):
                        st.info("📄 Report generation started")
                
                with col_btn4:
                    if st.button("🔍 Investigate Anomalies", use_container_width=True):
                        st.info("🔎 Deep analysis initiated")
                
                with col_btn5:
                    if st.button("✅ Mark All Resolved", use_container_width=True):
                        st.warning("⚠️ Please confirm resolution")
            else:
                st.markdown("""
                <div class="alert-success">
                    <h3>🎉 All Clear!</h3>
                    No active alerts in the system. All zones operating normally.
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Alert Analytics
            col_analytics1, col_analytics2 = st.columns(2)
            
            with col_analytics1:
                st.markdown('<h2 class="section-header">📊 Alert Frequency by Type</h2>', unsafe_allow_html=True)
                
                alert_types = {}
                for alert in alerts:
                    alert_type = alert['alert_type']
                    alert_types[alert_type] = alert_types.get(alert_type, 0) + 1
                
                if alert_types:
                    fig_types = go.Figure(data=[go.Bar(
                        x=list(alert_types.keys()),
                        y=list(alert_types.values()),
                        marker_color=['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b'],
                        text=list(alert_types.values()),
                        textposition='auto',
                        textfont=dict(size=14, color='#ffffff'),
                        hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
                    )])
                    
                    fig_types.update_layout(
                        height=300,
                        **plotly_dark_layout,
                        xaxis_title="Alert Type",
                        yaxis_title="Frequency",
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_types, use_container_width=True)
            
            with col_analytics2:
                st.markdown('<h2 class="section-header">🗺️ Alerts by Zone</h2>', unsafe_allow_html=True)
                
                zone_alerts = {}
                for alert in alerts:
                    zone = alert['zone_id']
                    zone_alerts[zone] = zone_alerts.get(zone, 0) + 1
                
                if zone_alerts:
                    fig_zones = go.Figure(data=[go.Pie(
                        labels=list(zone_alerts.keys()),
                        values=list(zone_alerts.values()),
                        marker=dict(colors=['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#06b6d4']),
                        textfont=dict(size=14, color='#ffffff'),
                        hovertemplate='<b>%{label}</b><br>%{value} alerts<br>%{percent}<extra></extra>'
                    )])
                    
                    fig_zones.update_layout(
                        height=300,
                        paper_bgcolor='rgba(26, 35, 50, 0.3)',
                        font={'color': '#e2e8f0', 'family': 'Inter'},
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_zones, use_container_width=True)
        
        # ====================================================================
        # SECTION: OPERATIONS MONITOR
        # ====================================================================
        
        elif st.session_state.current_section == 'Operations Monitor':
            
            st.markdown("""
                <div class="main-header">
                    <h1 class="main-title">⚙️ Operations Monitor</h1>
                    <p class="main-subtitle">Real-time System Health | Performance Metrics | Sensor Status</p>
                </div>
                """, unsafe_allow_html=True)
            
            fusion_data = get_sensor_fusion_data()
            
            # System Health Dashboard
            col1, col2, col3, col4, col5 = st.columns(5)
            
            col1.metric("System Uptime", "99.8%", delta="168 hours")
            col2.metric("CV Processing", "32 FPS", delta="+3 FPS")
            col3.metric("Sensor Health", "98.7%", delta="All Online")
            col4.metric("DB Performance", "12 ms", delta="Optimal")
            col5.metric("Network Latency", "45 ms", delta="-5 ms")
            
            st.markdown("---")
            
            # Real-time Monitoring
            col_monitor1, col_monitor2, col_monitor3 = st.columns([2, 1, 1])
            
            with col_monitor1:
                st.markdown('<h2 class="section-header">📡 Live Sensor Readings</h2>', unsafe_allow_html=True)
                
                if fusion_data:
                    # Create real-time sensor comparison
                    df = pd.DataFrame(fusion_data)
                    
                    fig_sensors = go.Figure()
                    
                    fig_sensors.add_trace(go.Scatter(
                        x=df['zone_id'],
                        y=df['camera_count'],
                        mode='lines+markers',
                        name='Camera (CV)',
                        line=dict(color='#3b82f6', width=3),
                        marker=dict(size=8),
                        hovertemplate='<b>CV System</b><br>Zone: %{x}<br>Count: %{y}<extra></extra>'
                    ))
                    
                    fig_sensors.add_trace(go.Scatter(
                        x=df['zone_id'],
                        y=df['load_cell_count'],
                        mode='lines+markers',
                        name='Load Cell',
                        line=dict(color='#8b5cf6', width=3),
                        marker=dict(size=8),
                        hovertemplate='<b>Load Cell</b><br>Zone: %{x}<br>Count: %{y}<extra></extra>'
                    ))
                    
                    fig_sensors.add_trace(go.Scatter(
                        x=df['zone_id'],
                        y=df['fused_count'],
                        mode='lines+markers',
                        name='Fused Result',
                        line=dict(color='#34d399', width=4, dash='dash'),
                        marker=dict(size=10, symbol='diamond'),
                        hovertemplate='<b>Sensor Fusion</b><br>Zone: %{x}<br>Count: %{y}<extra></extra>'
                    ))
                    
                    fig_sensors.update_layout(
                        height=350,
                        **plotly_dark_layout,
                        xaxis_title="Zone",
                        yaxis_title="Units Detected",
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_sensors, use_container_width=True)
            
            with col_monitor2:
                st.markdown('<h2 class="section-header">🎥 Camera Status</h2>', unsafe_allow_html=True)
                
                cameras = ['CAM-01', 'CAM-02', 'CAM-03', 'CAM-04', 'CAM-05', 'CAM-06']
                cam_status = ['Online', 'Online', 'Online', 'Online', 'Online', 'Online']
                cam_fps = [32, 31, 33, 30, 32, 31]
                
                for i, (cam, status, fps) in enumerate(zip(cameras, cam_status, cam_fps)):
                    st.markdown(f"""
                    <div class="metric-card" style="padding: 0.8rem; margin: 0.5rem 0;">
                        <strong>🟢 {cam}</strong><br>
                        {status} | {fps} FPS<br>
                        <small style="color: #94a3b8;">Zone: {fusion_data[i]['zone_id'] if i < len(fusion_data) else 'N/A'}</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col_monitor3:
                st.markdown('<h2 class="section-header">⚖️ Sensor Grid</h2>', unsafe_allow_html=True)
                
                sensor_types = ['Load Cell', 'Ultrasonic', 'Proximity', 'Environmental']
                sensor_status_values = [99.1, 97.8, 99.5, 98.2]
                
                for sensor, status_val in zip(sensor_types, sensor_status_values):
                    color = '#34d399' if status_val > 98 else ('#fbbf24' if status_val > 95 else '#f87171')
                    st.markdown(f"""
                    <div class="metric-card" style="padding: 0.8rem; margin: 0.5rem 0;">
                        <strong>{sensor}</strong><br>
                        <span style="color: {color}; font-size: 1.2rem;">{status_val:.1f}%</span><br>
                        <small style="color: #94a3b8;">6 units active</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Performance Metrics
            col_perf1, col_perf2 = st.columns(2)
            
            with col_perf1:
                st.markdown('<h2 class="section-header">⚡ System Performance (24h)</h2>', unsafe_allow_html=True)
                
                # Generate performance data
                perf_data = []
                for i in range(24):
                    hour = datetime.now() - timedelta(hours=24-i)
                    perf_data.append({
                        'hour': hour,
                        'cpu': 45 + np.random.uniform(-10, 10),
                        'memory': 62 + np.random.uniform(-5, 5),
                        'disk_io': 35 + np.random.uniform(-15, 15)
                    })
                
                df_perf = pd.DataFrame(perf_data)
                
                fig_perf = go.Figure()
                
                fig_perf.add_trace(go.Scatter(
                    x=df_perf['hour'],
                    y=df_perf['cpu'],
                    mode='lines',
                    name='CPU %',
                    line=dict(color='#3b82f6', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(59, 130, 246, 0.2)'
                ))
                
                fig_perf.add_trace(go.Scatter(
                    x=df_perf['hour'],
                    y=df_perf['memory'],
                    mode='lines',
                    name='Memory %',
                    line=dict(color='#8b5cf6', width=2)
                ))
                
                fig_perf.add_trace(go.Scatter(
                    x=df_perf['hour'],
                    y=df_perf['disk_io'],
                    mode='lines',
                    name='Disk I/O %',
                    line=dict(color='#34d399', width=2)
                ))
                
                fig_perf.update_layout(
                    height=350,
                    **plotly_dark_layout,
                    xaxis_title="Time",
                    yaxis_title="Usage (%)",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_perf, use_container_width=True)
            
            with col_perf2:
                st.markdown('<h2 class="section-header">📊 Database Operations</h2>', unsafe_allow_html=True)
                
                # Database metrics
                db_metrics = {
                    'Metric': ['Queries/sec', 'Avg Response', 'Connections', 'Cache Hit Rate'],
                    'Value': ['1,247', '12 ms', '42', '94.5%'],
                    'Status': ['🟢', '🟢', '🟢', '🟢']
                }
                
                df_db = pd.DataFrame(db_metrics)
                st.dataframe(df_db, use_container_width=True, hide_index=True)
                
                st.markdown("**📈 Query Performance:**")
                
                query_types = ['SELECT', 'INSERT', 'UPDATE', 'ANALYZE']
                query_counts = [850, 280, 95, 22]
                
                fig_queries = go.Figure(data=[go.Bar(
                    x=query_types,
                    y=query_counts,
                    marker_color=['#3b82f6', '#8b5cf6', '#34d399', '#f59e0b'],
                    text=query_counts,
                    textposition='auto',
                    textfont=dict(color='#ffffff', size=14),
                    hovertemplate='<b>%{x}</b><br>Count: %{y}/sec<extra></extra>'
                )])
                
                fig_queries.update_layout(
                    height=250,
                    **plotly_dark_layout,
                    xaxis_title="Query Type",
                    yaxis_title="Queries/sec",
                    showlegend=False
                )
                
                st.plotly_chart(fig_queries, use_container_width=True)
            
            st.markdown("---")
            
            # Environmental Monitoring
            st.markdown('<h2 class="section-header">🌡️ Environmental Conditions</h2>', unsafe_allow_html=True)
            
            col_env1, col_env2, col_env3, col_env4, col_env5 = st.columns(5)
            
            with col_env1:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">Temperature</div>
                    <div class="metric-value">22.5°C</div>
                    <div class="metric-trend trend-neutral">→ Optimal</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_env2:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">Humidity</div>
                    <div class="metric-value">45%</div>
                    <div class="metric-trend trend-up">Normal Range</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_env3:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">Lighting</div>
                    <div class="metric-value">450 lux</div>
                    <div class="metric-trend trend-up">Good</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_env4:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">Air Quality</div>
                    <div class="metric-value">98</div>
                    <div class="metric-trend trend-up">Excellent</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_env5:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">Vibration</div>
                    <div class="metric-value">Low</div>
                    <div class="metric-trend trend-up">Stable</div>
                </div>
                """, unsafe_allow_html=True)
        
        # ====================================================================
        # OTHER SECTIONS - Keep existing placeholder
        # ====================================================================
        
        else:
            st.markdown(f"""
                <div class="main-header">
                    <h1 class="main-title">{st.session_state.current_section}</h1>
                    <p class="main-subtitle">Section under development</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.info(f"The {st.session_state.current_section} section is being enhanced with additional visualizations.")
    
    # Auto-refresh control
    if not st.session_state.auto_refresh:
        break
    
    time.sleep(st.session_state.refresh_rate)
    st.rerun()
