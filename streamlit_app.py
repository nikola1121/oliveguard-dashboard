"""
═══════════════════════════════════════════════════════════════
🫒 OliveGuard Dashboard — Streamlit Cloud

   ლოკალურად:  streamlit run dashboard.py
   ონლაინ:     Streamlit Cloud → ლინკი → გაგზავნა
═══════════════════════════════════════════════════════════════
"""
import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta

# ═══════════════════════════════════════════
# კონფიგურაცია
# ═══════════════════════════════════════════

st.set_page_config(
    page_title="🫒 OliveGuard",
    page_icon="🫒",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_db_config():
    try:
        return {
            "host": st.secrets["database"]["host"],
            "port": st.secrets["database"]["port"],
            "database": st.secrets["database"]["database"],
            "user": st.secrets["database"]["user"],
            "password": st.secrets["database"]["password"],
        }
    except Exception:
        return {
            "host": "localhost",
            "port": 5432,
            "database": "gis_db",
            "user": "postgres",
            "password": "dolomiti1121"
        }

def get_connection():
    return psycopg2.connect(**get_db_config())


# ═══════════════════════════════════════════
# მონაცემების ჩატვირთვა
# ═══════════════════════════════════════════

@st.cache_data(ttl=300)
def load_parcels():
    conn = get_connection()
    try:
        return pd.read_sql("""
            SELECT 
                f.id AS field_id, f.name, f.cadastral_code, f.parcel_nr,
                f.latitude, f.longitude, f.area_ha, f.variety,
                ds.date AS summary_date,
                ds.temp_max, ds.temp_min, ds.temp_avg,
                ds.precipitation, ds.humidity_avg, ds.wind_max,
                ds.et0, ds.vpd,
                ds.gdd_daily, ds.gdd_cumulative,
                ds.phase_code, ds.phase_name,
                ds.latest_ndvi, ds.latest_ndwi,
                ds.wsi, ds.vhi,
                ds.max_disease_risk, ds.max_disease_name,
                CAST(ds.frost_risk AS integer) AS frost_risk,
                CAST(ds.heat_stress AS integer) AS heat_stress,
                ds.critical_alerts_count, ds.high_alerts_count,
                si.ndvi_trend_7d, si.ndvi_trend_14d, si.ndvi_trend_30d,
                dr.peacock_risk_score, dr.verticillium_risk_score,
                dr.anthracnose_risk_score, dr.olive_fly_risk_score,
                dr.olive_moth_risk_score, dr.leaf_wetness_hours,
                sat.ndvi_mean AS sat_ndvi, sat.ndwi_mean AS sat_ndwi,
                sat.evi_mean AS sat_evi, sat.acquisition_date AS sat_date,
                gdd.cumulative_gdd, gdd.chilling_hours
            FROM fields f
            LEFT JOIN LATERAL (SELECT * FROM daily_summary WHERE field_id = f.id ORDER BY date DESC LIMIT 1) ds ON true
            LEFT JOIN LATERAL (SELECT * FROM stress_indicators WHERE field_id = f.id ORDER BY date DESC LIMIT 1) si ON true
            LEFT JOIN LATERAL (SELECT * FROM disease_risk WHERE field_id = f.id ORDER BY date DESC LIMIT 1) dr ON true
            LEFT JOIN LATERAL (SELECT * FROM satellite_indices WHERE field_id = f.id ORDER BY acquisition_date DESC LIMIT 1) sat ON true
            LEFT JOIN LATERAL (SELECT * FROM gdd_tracking WHERE field_id = f.id ORDER BY date DESC LIMIT 1) gdd ON true
            WHERE f.is_active = TRUE
            ORDER BY f.id
        """, conn)
    finally:
        conn.close()

@st.cache_data(ttl=300)
def load_forecast():
    conn = get_connection()
    try:
        return pd.read_sql("""
            SELECT DISTINCT ON (forecast_date)
                forecast_date, temp_max, temp_min, precipitation_sum,
                humidity_max, wind_speed_max, et0_fao, gdd
            FROM weather_forecast
            WHERE forecast_date >= CURRENT_DATE
            ORDER BY forecast_date, fetched_at DESC
            LIMIT 14
        """, conn)
    finally:
        conn.close()

@st.cache_data(ttl=300)
def load_history(days=30):
    conn = get_connection()
    try:
        return pd.read_sql("""
            SELECT ds.field_id, f.cadastral_code, ds.date,
                   ds.latest_ndvi, ds.wsi, ds.max_disease_risk,
                   ds.temp_max, ds.temp_min, ds.precipitation
            FROM daily_summary ds
            JOIN fields f ON f.id = ds.field_id
            WHERE ds.date >= CURRENT_DATE - %s
            ORDER BY ds.date, ds.field_id
        """, conn, params=[days])
    finally:
        conn.close()

@st.cache_data(ttl=300)
def load_alerts():
    conn = get_connection()
    try:
        return pd.read_sql("""
            SELECT a.*, f.cadastral_code, f.name AS field_name
            FROM alerts a
            JOIN fields f ON f.id = a.field_id
            WHERE a.alert_date >= CURRENT_DATE - 7
            ORDER BY a.alert_date DESC, a.priority_score DESC
            LIMIT 50
        """, conn)
    finally:
        conn.close()


# ═══ სტილი ═══
st.markdown("""<style>
    .block-container { padding-top: 1rem; }
    [data-testid="stMetricValue"] { font-size: 26px !important; }
</style>""", unsafe_allow_html=True)


# ═══ სათაური ═══
col_title, col_refresh = st.columns([5, 1])
with col_title:
    st.title("🫒 OliveGuard Dashboard")
    st.caption(f"📅 {date.today()} | დედოფლისწყარო, კახეთი")
with col_refresh:
    if st.button("🔄 განახლება", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

try:
    df = load_parcels()
    forecast = load_forecast()
    alerts = load_alerts()
except Exception as e:
    st.error(f"❌ ბაზასთან კავშირი ვერ მოხერხდა: {e}")
    st.stop()

if df.empty:
    st.warning("📊 მონაცემები ცარიელია")
    st.stop()


# ═══ Sidebar ═══
with st.sidebar:
    st.header("⚙️ ფილტრები")
    if df['phase_name'].notna().any():
        st.info(f"🌱 **{df['phase_name'].iloc[0]}**")
    c1, c2 = st.columns(2)
    with c1:
        if df['cumulative_gdd'].notna().any():
            st.metric("GDD", f"{df['cumulative_gdd'].iloc[0]:.0f}")
    with c2:
        if df['chilling_hours'].notna().any():
            st.metric("❄️ CH", f"{df['chilling_hours'].iloc[0]:.0f}/700")
    st.divider()
    all_codes = ['ყველა'] + sorted(df['cadastral_code'].dropna().tolist())
    selected = st.selectbox("📍 ნაკვეთი", all_codes)
    if selected != 'ყველა':
        df = df[df['cadastral_code'] == selected]
    history_days = st.slider("📅 ისტორია (დღე)", 7, 90, 30)
    st.divider()
    st.caption(f"📊 {len(df)} ნაკვეთი")


# ═══ KPI ═══
st.markdown("### 📊 ძირითადი მაჩვენებლები")
k1, k2, k3, k4, k5, k6 = st.columns(6)
ndvi_vals = df['sat_ndvi'].dropna()
wsi_vals = df['wsi'].dropna()

with k1:
    st.metric("🛰 NDVI", f"{ndvi_vals.mean():.3f}" if not ndvi_vals.empty else "—",
              f"min: {ndvi_vals.min():.3f}" if not ndvi_vals.empty else None)
with k2:
    st.metric("💧 WSI", f"{wsi_vals.mean():.2f}" if not wsi_vals.empty else "—",
              f"max: {wsi_vals.max():.2f}" if not wsi_vals.empty else None, delta_color="inverse")
with k3:
    n_d = int((df['max_disease_risk'] >= 2).sum())
    st.metric("🦠 დაავადება", f"{n_d}/{len(df)}")
with k4:
    if df['temp_max'].notna().any():
        st.metric("🌡 ტემპ.", f"{df['temp_min'].iloc[0]:.0f}°/{df['temp_max'].iloc[0]:.0f}°C")
    else: st.metric("🌡 ტემპ.", "—")
with k5:
    if df['precipitation'].notna().any():
        st.metric("🌧 ნალექი", f"{df['precipitation'].iloc[0]:.1f} მმ")
    else: st.metric("🌧 ნალექი", "—")
with k6:
    st.metric("🚨 ალერტები", int(df['critical_alerts_count'].fillna(0).sum()))


# ═══ ტაბები ═══
st.markdown("---")
tab_map, tab_ndvi, tab_disease, tab_forecast, tab_history, tab_alerts, tab_table = st.tabs([
    "🗺️ რუკა", "🛰 NDVI/WSI", "🦠 დაავადებები", "⛅ პროგნოზი", "📈 ისტორია", "🚨 ალერტები", "📋 ცხრილი"
])

# ═══ რუკა ═══
with tab_map:
    c1, c2 = st.columns(2)
    map_df = df[df['latitude'].notna() & df['sat_ndvi'].notna()].copy()
    with c1:
        st.subheader("NDVI რუკა")
        if not map_df.empty:
            map_df['size'] = 300
            fig = px.scatter_mapbox(map_df, lat='latitude', lon='longitude',
                color='sat_ndvi', color_continuous_scale='RdYlGn', range_color=[0, 0.7],
                size='size', hover_name='cadastral_code',
                hover_data={'sat_ndvi':':.3f','wsi':':.2f','area_ha':':.1f','size':False},
                zoom=11, height=500, mapbox_style='open-street-map')
            fig.update_layout(margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("WSI რუკა")
        map_df2 = df[df['latitude'].notna() & df['wsi'].notna()].copy()
        if not map_df2.empty:
            map_df2['size'] = 300
            fig2 = px.scatter_mapbox(map_df2, lat='latitude', lon='longitude',
                color='wsi', color_continuous_scale='RdYlGn_r', range_color=[0, 1],
                size='size', hover_name='cadastral_code',
                hover_data={'wsi':':.2f','sat_ndvi':':.3f','size':False},
                zoom=11, height=500, mapbox_style='open-street-map')
            fig2.update_layout(margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig2, use_container_width=True)

# ═══ NDVI / WSI ═══
with tab_ndvi:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("NDVI — ნაკვეთები")
        nd = df[df['sat_ndvi'].notna()].sort_values('sat_ndvi').copy()
        if not nd.empty:
            nd['color'] = nd['sat_ndvi'].apply(lambda v: '#4CAF50' if v>0.5 else '#FFC107' if v>0.3 else '#FF9800' if v>0.2 else '#F44336')
            nd['label'] = nd['cadastral_code'].str[-6:]
            fig = go.Figure(go.Bar(y=nd['label'], x=nd['sat_ndvi'], orientation='h',
                marker_color=nd['color'], text=nd['sat_ndvi'].round(3), textposition='outside', textfont_size=9))
            fig.add_vline(x=0.5, line_dash='dash', line_color='green', opacity=0.4)
            fig.update_layout(height=max(400, len(nd)*22), xaxis_title='NDVI')
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("WSI — წყლის სტრესი")
        wd = df[df['wsi'].notna()].sort_values('wsi', ascending=False).copy()
        if not wd.empty:
            wd['color'] = wd['wsi'].apply(lambda v: '#F44336' if v>=0.7 else '#FF9800' if v>=0.5 else '#FFC107' if v>=0.3 else '#4CAF50')
            wd['label'] = wd['cadastral_code'].str[-6:]
            fig = go.Figure(go.Bar(y=wd['label'], x=wd['wsi'], orientation='h',
                marker_color=wd['color'], text=wd['wsi'].round(2), textposition='outside', textfont_size=9))
            fig.add_vline(x=0.5, line_dash='dash', line_color='red', opacity=0.4)
            fig.update_layout(height=max(400, len(wd)*22), xaxis_title='WSI', xaxis_range=[0,1])
            st.plotly_chart(fig, use_container_width=True)

# ═══ დაავადებები ═══
with tab_disease:
    st.subheader("🦠 დაავადებების რისკი")
    dcols = ['peacock_risk_score','verticillium_risk_score','anthracnose_risk_score','olive_fly_risk_score','olive_moth_risk_score']
    dnames = ['Peacock Spot','Verticillium','Anthracnose','Olive Fly','Olive Moth']
    ac = [c for c in dcols if c in df.columns and df[c].notna().any()]
    if ac:
        ddf = df[['cadastral_code']+ac].set_index('cadastral_code')
        ddf.columns = [dnames[dcols.index(c)] for c in ac]
        fig = px.imshow(ddf.T.values, x=ddf.index.tolist(), y=ddf.columns.tolist(),
            color_continuous_scale=['#4CAF50','#FFC107','#FF9800','#F44336'],
            zmin=0, zmax=3, aspect='auto', text_auto=True)
        fig.update_layout(height=300, xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

# ═══ პროგნოზი ═══
with tab_forecast:
    st.subheader("⛅ 14-დღიანი პროგნოზი")
    if not forecast.empty:
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=forecast['forecast_date'], y=forecast['temp_max'],
                name='Max °C', line=dict(color='red', width=2), mode='lines+markers'))
            fig.add_trace(go.Scatter(x=forecast['forecast_date'], y=forecast['temp_min'],
                name='Min °C', line=dict(color='blue', width=2), mode='lines+markers'))
            fig.add_hline(y=0, line_dash='solid', line_color='cyan', opacity=0.5)
            fig.update_layout(title='ტემპერატურა', height=400, yaxis_title='°C')
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=forecast['forecast_date'], y=forecast['precipitation_sum'],
                name='ნალექი (მმ)', marker_color='#2196F3', opacity=0.7), secondary_y=False)
            if 'gdd' in forecast.columns and df['gdd_cumulative'].notna().any():
                base = df['gdd_cumulative'].iloc[0] or 0
                fig.add_trace(go.Scatter(x=forecast['forecast_date'],
                    y=forecast['gdd'].fillna(0).cumsum() + base,
                    name='GDD', line=dict(color='green', width=2), mode='lines+markers'), secondary_y=True)
            fig.update_layout(title='ნალექი + GDD', height=400)
            st.plotly_chart(fig, use_container_width=True)

# ═══ ისტორია ═══
with tab_history:
    st.subheader(f"📈 ბოლო {history_days} დღე")
    hist = load_history(history_days)
    if not hist.empty:
        fig = px.line(hist, x='date', y='latest_ndvi', color='cadastral_code', title='NDVI ტრენდი')
        fig.add_hline(y=0.3, line_dash='dash', line_color='red', opacity=0.3)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.line(hist, x='date', y='wsi', color='cadastral_code', title='WSI ტრენდი')
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.line(hist, x='date', y='temp_max', color='cadastral_code', title='ტემპერატურა Max')
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# ═══ ალერტები ═══
with tab_alerts:
    st.subheader("🚨 ალერტები")
    if not alerts.empty:
        for _, a in alerts.head(15).iterrows():
            icon = "🔴" if a['priority'] == 'CRITICAL' else "🟡"
            with st.expander(f"{icon} {a.get('cadastral_code','')} — {a['alert_title'][:60]}"):
                st.write(f"📅 {a['alert_date']}")
                if pd.notna(a.get('alert_description')): st.write(a['alert_description'])
                if pd.notna(a.get('recommendations')): st.success(f"💡 {a['recommendations']}")
    else:
        st.success("✅ ალერტები არ არის")

# ═══ ცხრილი ═══
with tab_table:
    st.subheader("📋 მონაცემები")
    cols = ['cadastral_code','area_ha','sat_ndvi','sat_ndwi','wsi','max_disease_risk',
            'max_disease_name','temp_max','temp_min','precipitation','gdd_cumulative','phase_name']
    avail = [c for c in cols if c in df.columns]
    st.dataframe(df[avail], use_container_width=True, height=600)
    st.download_button("📥 CSV", df[avail].to_csv(index=False), "oliveguard.csv", "text/csv")

st.markdown("---")
st.caption("🫒 OliveGuard | დედოფლისწყარო, კახეთი")
