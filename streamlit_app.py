"""
🫒 OliveGuard Dashboard v3
   - პოლიგონები რუკაზე (folium)
   - წერტილოვანი ფენა
   - პრობლემური ნაკვეთების გამოყოფა
   - ისტორია თარიღებით
"""
import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta

try:
    import folium
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

# ═══════════════════════════════════════════
# Config
# ═══════════════════════════════════════════

st.set_page_config(page_title="OliveGuard", page_icon="🫒", layout="wide", initial_sidebar_state="expanded")

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
        return {"host": "localhost", "port": 5432, "database": "gis_db",
                "user": "postgres", "password": "dolomiti1121"}

def get_connection():
    return psycopg2.connect(**get_db_config())


# ═══════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════

@st.cache_data(ttl=300)
def load_parcels():
    conn = get_connection()
    try:
        return pd.read_sql("""
            SELECT DISTINCT ON (f.cadastral_code)
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
            ORDER BY f.cadastral_code, f.id
        """, conn)
    finally:
        conn.close()


@st.cache_data(ttl=300)
def load_polygons():
    """PostGIS-დან პოლიგონების GeoJSON"""
    conn = get_connection()
    try:
        return pd.read_sql("""
            SELECT DISTINCT ON (f.cadastral_code)
                f.cadastral_code, f.area_ha,
                ST_AsGeoJSON(f.geom) AS geojson,
                f.latitude, f.longitude,
                ds.latest_ndvi, ds.wsi, ds.max_disease_risk, ds.max_disease_name,
                CAST(ds.frost_risk AS integer) AS frost_risk,
                sat.ndvi_mean AS sat_ndvi, sat.ndwi_mean AS sat_ndwi
            FROM fields f
            LEFT JOIN LATERAL (SELECT * FROM daily_summary WHERE field_id = f.id ORDER BY date DESC LIMIT 1) ds ON true
            LEFT JOIN LATERAL (SELECT * FROM satellite_indices WHERE field_id = f.id ORDER BY acquisition_date DESC LIMIT 1) sat ON true
            WHERE f.is_active = TRUE AND f.geom IS NOT NULL
            ORDER BY f.cadastral_code, f.id
        """, conn)
    finally:
        conn.close()


@st.cache_data(ttl=300)
def load_points():
    """წერტილოვანი ფენა — stations ან სხვა"""
    conn = get_connection()
    try:
        # ჯერ შევამოწმოთ რა წერტილოვანი ცხრილები არსებობს
        tables = pd.read_sql("""
            SELECT table_name FROM information_schema.columns
            WHERE column_name = 'geom' AND table_schema = 'public'
            AND table_name NOT IN ('fields', 'sakadastroebi_polygons')
        """, conn)

        points = []
        for _, row in tables.iterrows():
            tbl = row['table_name']
            try:
                pt = pd.read_sql(f"""
                    SELECT *, ST_X(ST_Centroid(geom::geometry)) AS lon,
                           ST_Y(ST_Centroid(geom::geometry)) AS lat
                    FROM {tbl}
                    WHERE geom IS NOT NULL
                    LIMIT 500
                """, conn)
                pt['_layer'] = tbl
                points.append(pt)
            except Exception:
                pass
        return pd.concat(points) if points else pd.DataFrame()
    except Exception:
        return pd.DataFrame()
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
            ORDER BY forecast_date, fetched_at DESC LIMIT 14
        """, conn)
    finally:
        conn.close()


@st.cache_data(ttl=300)
def load_history(days=30):
    conn = get_connection()
    try:
        return pd.read_sql("""
            SELECT DISTINCT ON (f.cadastral_code, ds.date)
                ds.field_id, f.cadastral_code, ds.date,
                ds.latest_ndvi, ds.wsi, ds.max_disease_risk,
                ds.temp_max, ds.temp_min, ds.precipitation
            FROM daily_summary ds
            JOIN fields f ON f.id = ds.field_id
            WHERE ds.date >= CURRENT_DATE - %s
            ORDER BY f.cadastral_code, ds.date, ds.id DESC
        """, conn, params=[days])
    finally:
        conn.close()


@st.cache_data(ttl=300)
def load_alerts():
    conn = get_connection()
    try:
        return pd.read_sql("""
            SELECT DISTINCT ON (a.field_id, a.alert_date, a.alert_type)
                a.*, f.cadastral_code, f.name AS field_name
            FROM alerts a JOIN fields f ON f.id = a.field_id
            WHERE a.alert_date >= CURRENT_DATE - 7
            ORDER BY a.field_id, a.alert_date, a.alert_type, a.id DESC
        """, conn)
    finally:
        conn.close()


@st.cache_data(ttl=300)
def load_full_table():
    conn = get_connection()
    try:
        return pd.read_sql("""
            SELECT DISTINCT ON (f.cadastral_code, ds.date)
                f.cadastral_code, f.area_ha, ds.date,
                ds.latest_ndvi, ds.latest_ndwi, ds.wsi, ds.vhi,
                ds.max_disease_risk, ds.max_disease_name,
                ds.temp_max, ds.temp_min, ds.precipitation,
                ds.gdd_cumulative, ds.phase_name,
                CAST(ds.frost_risk AS integer) AS frost_risk,
                CAST(ds.heat_stress AS integer) AS heat_stress
            FROM daily_summary ds
            JOIN fields f ON f.id = ds.field_id
            WHERE f.is_active = TRUE
            ORDER BY f.cadastral_code, ds.date DESC, ds.id DESC
        """, conn)
    finally:
        conn.close()


# ═══════════════════════════════════════════
# Problem Score
# ═══════════════════════════════════════════

def calc_problem_score(row):
    sc = 0
    ndvi = row.get('sat_ndvi') or row.get('latest_ndvi')
    wsi = row.get('wsi')
    disease = row.get('max_disease_risk')
    frost = row.get('frost_risk')
    heat = row.get('heat_stress')

    if ndvi is not None:
        if ndvi < 0.15: sc += 3
        elif ndvi < 0.2: sc += 1
    if wsi is not None:
        if wsi >= 0.7: sc += 3
        elif wsi >= 0.5: sc += 1
    if disease is not None:
        if disease >= 3: sc += 3
        elif disease >= 2: sc += 1
    if frost: sc += 2
    if heat: sc += 2
    return sc


def ndvi_color(v):
    if v is None or pd.isna(v): return '#808080'
    if v > 0.5: return '#2E7D32'
    if v > 0.4: return '#4CAF50'
    if v > 0.3: return '#8BC34A'
    if v > 0.2: return '#FFC107'
    return '#F44336'


def wsi_color(v):
    if v is None or pd.isna(v): return '#808080'
    if v >= 0.7: return '#F44336'
    if v >= 0.5: return '#FF9800'
    if v >= 0.3: return '#FFC107'
    return '#4CAF50'


# ═══════════════════════════════════════════
# Style
# ═══════════════════════════════════════════

st.markdown("""<style>
    .block-container { padding-top: 0.8rem; }
    [data-testid="stMetricValue"] { font-size: 24px !important; font-weight: 700; }
    [data-testid="stMetricDelta"] { font-size: 11px !important; }
    .problem-card {
        background: linear-gradient(135deg, #4a1a1a 0%, #2d1010 100%);
        padding: 14px; border-radius: 10px; margin: 6px 0;
        border-left: 4px solid #F44336;
    }
    .ok-card {
        background: linear-gradient(135deg, #1a3a1a 0%, #102d10 100%);
        padding: 14px; border-radius: 10px; margin: 6px 0;
        border-left: 4px solid #4CAF50;
    }
</style>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════
# Header
# ═══════════════════════════════════════════

c1, c2, c3 = st.columns([4, 1, 1])
with c1:
    st.markdown("# 🫒 OliveGuard")
    st.caption(f"დედოფლისწყარო, კახეთი  •  {date.today().strftime('%d.%m.%Y')}")
with c3:
    if st.button("🔄 განახლება", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

try:
    df = load_parcels()
    forecast = load_forecast()
    alerts = load_alerts()
except Exception as e:
    st.error(f"❌ ბაზასთან კავშირი: {e}")
    st.stop()

if df.empty:
    st.warning("მონაცემები ცარიელია")
    st.stop()

# Problem score
df['problem_score'] = df.apply(calc_problem_score, axis=1)
df['is_problematic'] = df['problem_score'] >= 2
n_problems = int(df['is_problematic'].sum())


# ═══════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚙️ ფილტრები")

    if df['phase_name'].notna().any():
        phase = df['phase_name'].iloc[0]
        st.markdown(f"""<div style="background:#1a472a; padding:12px; border-radius:8px; margin-bottom:12px;">
            <span style="font-size:12px; color:#aaa;">ფენოლოგიური ფაზა</span><br>
            <span style="font-size:17px; color:#4CAF50; font-weight:bold;">🌱 {phase}</span>
        </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        gdd = df['cumulative_gdd'].iloc[0] if df['cumulative_gdd'].notna().any() else 0
        st.metric("🌡 GDD", f"{gdd:.0f}")
    with c2:
        ch = df['chilling_hours'].iloc[0] if df['chilling_hours'].notna().any() else 0
        st.metric("❄️ CH", f"{ch:.0f}/700")
    st.progress(min(1.0, ch / 700), text=f"{min(100, ch/700*100):.0f}%")

    st.markdown("---")

    # პრობლემური ნაკვეთები
    if n_problems > 0:
        st.markdown(f"""<div style="background:#4a1a1a; padding:10px; border-radius:8px; text-align:center;">
            <span style="color:#F44336; font-size:20px; font-weight:bold;">⚠️ {n_problems}</span><br>
            <span style="color:#aaa; font-size:12px;">პრობლემური ნაკვეთი</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.success("✅ ყველა ნაკვეთი ნორმაშია")

    st.markdown("---")

    all_codes = ['ყველა'] + sorted(df['cadastral_code'].dropna().tolist())
    selected = st.selectbox("📍 ნაკვეთი", all_codes)
    if selected != 'ყველა':
        df = df[df['cadastral_code'] == selected]

    history_days = st.slider("📅 ისტორია", 7, 90, 30)
    st.markdown("---")
    st.caption(f"📊 {len(df)} ნაკვეთი")
    if df['summary_date'].notna().any():
        st.caption(f"📅 {df['summary_date'].iloc[0]}")
    if df['sat_date'].notna().any():
        st.caption(f"🛰 {df['sat_date'].iloc[0]}")


# ═══════════════════════════════════════════
# KPI
# ═══════════════════════════════════════════

ndvi_vals = df['sat_ndvi'].dropna()
wsi_vals = df['wsi'].dropna()

k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    st.metric("🛰 NDVI", f"{ndvi_vals.mean():.3f}" if not ndvi_vals.empty else "—",
              f"min {ndvi_vals.min():.3f}" if not ndvi_vals.empty else None)
with k2:
    st.metric("💧 WSI", f"{wsi_vals.mean():.2f}" if not wsi_vals.empty else "—",
              f"max {wsi_vals.max():.2f}" if not wsi_vals.empty else None, delta_color="inverse")
with k3:
    st.metric("🦠 დაავადება", f"{int((df['max_disease_risk']>=2).sum())}/{len(df)}")
with k4:
    if df['temp_max'].notna().any():
        st.metric("🌡 ტემპ.", f"{df['temp_min'].iloc[0]:.0f}°/{df['temp_max'].iloc[0]:.0f}°C")
    else: st.metric("🌡 ტემპ.", "—")
with k5:
    if df['precipitation'].notna().any():
        st.metric("🌧 ნალექი", f"{df['precipitation'].iloc[0]:.1f} მმ")
    else: st.metric("🌧 ნალექი", "—")
with k6:
    st.metric("⚠️ პრობლემა", f"{n_problems}/{len(df)}")


# ═══════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════

st.markdown("")
tab_map, tab_problems, tab_ndvi, tab_disease, tab_forecast, tab_history, tab_alerts, tab_table = st.tabs([
    "🗺️ რუკა", "⚠️ პრობლემური", "🛰 NDVI/WSI", "🦠 დაავადებები",
    "⛅ პროგნოზი", "📈 ისტორია", "🚨 ალერტები", "📋 მონაცემები"
])


# ═══════════════════════════════════════════
# რუკა — პოლიგონებით
# ═══════════════════════════════════════════

with tab_map:
    if HAS_FOLIUM:
        poly_df = load_polygons()
        points_df = load_points()

        map_mode = st.radio("შეფერადება", ["NDVI", "WSI", "დაავადება"], horizontal=True)

        # ცენტრი
        center_lat = df['latitude'].mean() if df['latitude'].notna().any() else 41.33
        center_lon = df['longitude'].mean() if df['longitude'].notna().any() else 46.06

        m = folium.Map(location=[center_lat, center_lon], zoom_start=13,
                       tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                       attr='Esri World Imagery')

        # პოლიგონები
        if not poly_df.empty:
            for _, row in poly_df.iterrows():
                if row['geojson'] is None:
                    continue
                try:
                    geoj = json.loads(row['geojson'])
                except Exception:
                    continue

                if map_mode == "NDVI":
                    color = ndvi_color(row.get('sat_ndvi'))
                    val = f"NDVI: {row.get('sat_ndvi', 'N/A')}"
                elif map_mode == "WSI":
                    color = wsi_color(row.get('wsi'))
                    val = f"WSI: {row.get('wsi', 'N/A')}"
                else:
                    dr = row.get('max_disease_risk', 0) or 0
                    color = '#F44336' if dr >= 2 else '#FFC107' if dr >= 1 else '#4CAF50'
                    val = f"Disease: {dr}"

                popup_html = f"""
                <b>{row['cadastral_code']}</b><br>
                {row.get('area_ha', '')} ჰა<br>
                NDVI: {row.get('sat_ndvi', 'N/A')}<br>
                WSI: {row.get('wsi', 'N/A')}<br>
                დაავადება: {row.get('max_disease_name', 'N/A')}
                """

                folium.GeoJson(
                    geoj,
                    style_function=lambda x, c=color: {
                        'fillColor': c, 'color': '#ffffff',
                        'weight': 1.5, 'fillOpacity': 0.6
                    },
                    popup=folium.Popup(popup_html, max_width=250),
                    tooltip=f"{row['cadastral_code']} | {val}"
                ).add_to(m)

        # წერტილოვანი ფენა
        if not points_df.empty and 'lat' in points_df.columns:
            for _, pt in points_df.iterrows():
                if pd.notna(pt.get('lat')) and pd.notna(pt.get('lon')):
                    folium.CircleMarker(
                        location=[pt['lat'], pt['lon']],
                        radius=6, color='#00BCD4', fill=True,
                        fill_color='#00BCD4', fill_opacity=0.8,
                        tooltip=f"{pt.get('_layer', 'point')}",
                        popup=str({k: v for k, v in pt.items() if k not in ['geom', '_layer', 'lat', 'lon']})
                    ).add_to(m)

        # ლეგენდა
        if map_mode == "NDVI":
            legend = """<div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:rgba(0,0,0,0.7);padding:10px;border-radius:8px;color:white;font-size:12px;">
                <b>NDVI</b><br>
                <span style="color:#2E7D32;">■</span> &gt;0.5 კარგი<br>
                <span style="color:#8BC34A;">■</span> 0.3-0.5 საშუალო<br>
                <span style="color:#FFC107;">■</span> 0.2-0.3 დაბალი<br>
                <span style="color:#F44336;">■</span> &lt;0.2 კრიტიკული
            </div>"""
            m.get_root().html.add_child(folium.Element(legend))

        st_folium(m, width=None, height=600, use_container_width=True)

    else:
        # Fallback — plotly scatter
        st.info("📦 `folium` არ არის დაინსტალირებული — წერტილოვანი რუკა")
        c1, c2 = st.columns(2)
        map_df = df[df['latitude'].notna() & df['sat_ndvi'].notna()].copy()
        with c1:
            st.markdown("#### NDVI")
            if not map_df.empty:
                map_df['size'] = 300
                fig = px.scatter_mapbox(map_df, lat='latitude', lon='longitude',
                    color='sat_ndvi', color_continuous_scale='RdYlGn', range_color=[0, 0.7],
                    size='size', hover_name='cadastral_code',
                    hover_data={'sat_ndvi':':.3f','wsi':':.2f','size':False},
                    zoom=11, height=500, mapbox_style='open-street-map')
                fig.update_layout(margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("#### WSI")
            map_df2 = df[df['latitude'].notna() & df['wsi'].notna()].copy()
            if not map_df2.empty:
                map_df2['size'] = 300
                fig2 = px.scatter_mapbox(map_df2, lat='latitude', lon='longitude',
                    color='wsi', color_continuous_scale='RdYlGn_r', range_color=[0,1],
                    size='size', hover_name='cadastral_code',
                    hover_data={'wsi':':.2f','size':False},
                    zoom=11, height=500, mapbox_style='open-street-map')
                fig2.update_layout(margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════
# პრობლემური ნაკვეთები
# ═══════════════════════════════════════════

with tab_problems:
    st.markdown("#### ⚠️ პრობლემური ნაკვეთები")

    # Reload full df for problem tab (not filtered)
    all_df = load_parcels()
    all_df['problem_score'] = all_df.apply(calc_problem_score, axis=1)
    problems = all_df[all_df['problem_score'] >= 2].sort_values('problem_score', ascending=False)

    if problems.empty:
        st.success("✅ პრობლემური ნაკვეთები არ არის!")
    else:
        st.error(f"**{len(problems)}** ნაკვეთს სჭირდება ყურადღება")

        for _, p in problems.iterrows():
            issues = []
            ndvi = p.get('sat_ndvi')
            wsi = p.get('wsi')
            disease = p.get('max_disease_risk')

            if ndvi is not None and ndvi < 0.2:
                issues.append(f"🔴 NDVI: {ndvi:.3f}")
            if wsi is not None and wsi >= 0.5:
                issues.append(f"💧 WSI: {wsi:.2f}")
            if disease is not None and disease >= 2:
                issues.append(f"🦠 {p.get('max_disease_name', '')}: {disease:.0f}/3")
            if p.get('frost_risk'):
                issues.append("❄️ ყინვის რისკი")
            if p.get('heat_stress'):
                issues.append("🔥 სიცხის სტრესი")

            score = p['problem_score']
            severity = "🔴 კრიტიკული" if score >= 5 else "🟠 მაღალი" if score >= 3 else "🟡 საშუალო"

            st.markdown(f"""<div class="problem-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:16px; font-weight:bold; color:#fff;">
                        📍 {p['cadastral_code']}
                    </span>
                    <span style="font-size:13px; color:#F44336;">{severity} (score: {score})</span>
                </div>
                <div style="color:#ccc; font-size:13px; margin-top:6px;">
                    {' &nbsp;|&nbsp; '.join(issues)}
                </div>
                <div style="color:#888; font-size:11px; margin-top:4px;">
                    {p.get('area_ha', ''):.1f} ჰა &nbsp;|&nbsp; {p.get('phase_name', '')}
                </div>
            </div>""", unsafe_allow_html=True)

        # პრობლემების განაწილება
        st.markdown("#### პრობლემების ტიპები")
        issue_counts = {
            '🔴 დაბალი NDVI': int((all_df['sat_ndvi'] < 0.2).sum()),
            '💧 წყლის სტრესი': int((all_df['wsi'] >= 0.5).sum()),
            '🦠 დაავადება': int((all_df['max_disease_risk'] >= 2).sum()),
            '❄️ ყინვა': int(all_df['frost_risk'].sum()),
            '🔥 სიცხე': int(all_df['heat_stress'].sum()),
        }
        issue_df = pd.DataFrame({'პრობლემა': issue_counts.keys(), 'რაოდენობა': issue_counts.values()})
        fig = px.bar(issue_df, x='რაოდენობა', y='პრობლემა', orientation='h',
                     color='რაოდენობა', color_continuous_scale=['#4CAF50', '#F44336'])
        fig.update_layout(height=250, showlegend=False, plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)


# ═══ NDVI / WSI ═══
with tab_ndvi:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### NDVI — ნაკვეთები")
        nd = df[df['sat_ndvi'].notna()].sort_values('sat_ndvi').copy()
        if not nd.empty:
            nd['color'] = nd['sat_ndvi'].apply(lambda v: '#4CAF50' if v>0.5 else '#FFC107' if v>0.3 else '#FF9800' if v>0.2 else '#F44336')
            nd['label'] = nd['cadastral_code'].str[-6:]
            fig = go.Figure(go.Bar(y=nd['label'], x=nd['sat_ndvi'], orientation='h',
                marker_color=nd['color'], text=nd['sat_ndvi'].round(3), textposition='outside', textfont_size=9))
            fig.add_vline(x=0.5, line_dash='dash', line_color='green', opacity=0.3, annotation_text="კარგი")
            fig.add_vline(x=0.3, line_dash='dash', line_color='orange', opacity=0.3)
            fig.update_layout(height=max(400, len(nd)*24), xaxis_title='NDVI', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("#### WSI — წყლის სტრესი")
        wd = df[df['wsi'].notna()].sort_values('wsi', ascending=False).copy()
        if not wd.empty:
            wd['color'] = wd['wsi'].apply(lambda v: '#F44336' if v>=0.7 else '#FF9800' if v>=0.5 else '#FFC107' if v>=0.3 else '#4CAF50')
            wd['label'] = wd['cadastral_code'].str[-6:]
            fig = go.Figure(go.Bar(y=wd['label'], x=wd['wsi'], orientation='h',
                marker_color=wd['color'], text=wd['wsi'].round(2), textposition='outside', textfont_size=9))
            fig.add_vline(x=0.5, line_dash='dash', line_color='red', opacity=0.3, annotation_text="მაღალი")
            fig.update_layout(height=max(400, len(wd)*24), xaxis_title='WSI', xaxis_range=[0,1], plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)


# ═══ დაავადებები ═══
with tab_disease:
    st.markdown("#### 🦠 დაავადებების რისკი")
    dcols = ['peacock_risk_score','verticillium_risk_score','anthracnose_risk_score','olive_fly_risk_score','olive_moth_risk_score']
    dnames = ['Peacock','Verticillium','Anthracnose','Olive Fly','Olive Moth']
    ac = [c for c in dcols if c in df.columns and df[c].notna().any()]
    if ac:
        ddf = df[['cadastral_code']+ac].set_index('cadastral_code')
        ddf.columns = [dnames[dcols.index(c)] for c in ac]
        fig = px.imshow(ddf.T.values, x=ddf.index.tolist(), y=ddf.columns.tolist(),
            color_continuous_scale=['#1a472a','#4CAF50','#FFC107','#FF9800','#F44336'],
            zmin=0, zmax=3, aspect='auto', text_auto=True)
        fig.update_layout(height=280, xaxis_tickangle=45, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)


# ═══ პროგნოზი ═══
with tab_forecast:
    st.markdown("#### ⛅ 14-დღიანი პროგნოზი")
    if not forecast.empty:
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=forecast['forecast_date'], y=forecast['temp_max'],
                name='Max', line=dict(color='#FF5252', width=2.5), mode='lines+markers'))
            fig.add_trace(go.Scatter(x=forecast['forecast_date'], y=forecast['temp_min'],
                name='Min', line=dict(color='#448AFF', width=2.5), mode='lines+markers'))
            fig.add_hline(y=0, line_dash='solid', line_color='cyan', opacity=0.4)
            fig.update_layout(title='ტემპერატურა (°C)', height=400, plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=forecast['forecast_date'], y=forecast['precipitation_sum'],
                name='ნალექი მმ', marker_color='#2196F3', opacity=0.7), secondary_y=False)
            if 'gdd' in forecast.columns and df['gdd_cumulative'].notna().any():
                base = df['gdd_cumulative'].iloc[0] or 0
                fig.add_trace(go.Scatter(x=forecast['forecast_date'],
                    y=forecast['gdd'].fillna(0).cumsum() + base,
                    name='GDD', line=dict(color='#66BB6A', width=2.5), mode='lines+markers'), secondary_y=True)
            fig.update_layout(title='ნალექი + GDD', height=400, plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)


# ═══ ისტორია ═══
with tab_history:
    st.markdown(f"#### 📈 ბოლო {history_days} დღე")
    hist = load_history(history_days)
    if not hist.empty:
        fig = px.line(hist, x='date', y='latest_ndvi', color='cadastral_code', title='NDVI ტრენდი')
        fig.add_hline(y=0.3, line_dash='dash', line_color='red', opacity=0.3)
        fig.update_layout(height=380, plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.line(hist, x='date', y='wsi', color='cadastral_code', title='WSI ტრენდი')
            fig.add_hline(y=0.5, line_dash='dash', line_color='red', opacity=0.3)
            fig.update_layout(height=300, showlegend=False, plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.line(hist, x='date', y='temp_max', color='cadastral_code', title='Max ტემპ.')
            fig.update_layout(height=300, showlegend=False, plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)


# ═══ ალერტები ═══
with tab_alerts:
    st.markdown("#### 🚨 ალერტები")
    if not alerts.empty:
        for _, a in alerts.sort_values('alert_date', ascending=False).head(20).iterrows():
            icon = "🔴" if a.get('priority') == 'CRITICAL' else "🟡"
            with st.expander(f"{icon} {a.get('cadastral_code','')} — {a['alert_title'][:60]}"):
                st.caption(f"📅 {a.get('alert_date','')} | {a.get('field_name','')}")
                if pd.notna(a.get('alert_description')): st.write(a['alert_description'])
                if pd.notna(a.get('recommendations')): st.success(f"💡 {a['recommendations']}")
    else:
        st.success("✅ ალერტები არ არის")


# ═══ მონაცემები ═══
with tab_table:
    st.markdown("#### 📋 მონაცემები — ისტორიით")
    full = load_full_table()
    if not full.empty:
        c1, c2 = st.columns([2, 1])
        with c1:
            dr = st.date_input("📅 პერიოდი",
                value=(full['date'].min(), full['date'].max()),
                min_value=full['date'].min(), max_value=full['date'].max())
        with c2:
            sort_by = st.selectbox("სორტირება", ['date', 'cadastral_code', 'latest_ndvi', 'wsi'],
                format_func=lambda x: {'date':'თარიღი','cadastral_code':'კადასტრი','latest_ndvi':'NDVI','wsi':'WSI'}[x])

        if isinstance(dr, tuple) and len(dr) == 2:
            mask = (full['date'] >= pd.Timestamp(dr[0])) & (full['date'] <= pd.Timestamp(dr[1]))
            filtered = full[mask].sort_values(sort_by, ascending=(sort_by != 'wsi'))
        else:
            filtered = full.sort_values(sort_by, ascending=(sort_by != 'wsi'))

        display = filtered.rename(columns={
            'cadastral_code':'კადასტრი', 'area_ha':'ჰა', 'date':'თარიღი',
            'latest_ndvi':'NDVI', 'latest_ndwi':'NDWI', 'wsi':'WSI', 'vhi':'VHI',
            'max_disease_risk':'დაავ.', 'max_disease_name':'დაავ.სახელი',
            'temp_max':'Max°C', 'temp_min':'Min°C', 'precipitation':'ნალექი',
            'gdd_cumulative':'GDD', 'phase_name':'ფაზა',
            'frost_risk':'ყინვა', 'heat_stress':'სიცხე'
        })
        st.dataframe(display, use_container_width=True, height=600)
        st.caption(f"{len(filtered)} ჩანაწერი")
        st.download_button("📥 CSV", filtered.to_csv(index=False), "oliveguard.csv", "text/csv")


# Footer
st.markdown("---")
st.caption("🫒 OliveGuard — ზეთისხილის მონიტორინგი | დედოფლისწყარო, კახეთი")
