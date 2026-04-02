

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title      = "EpiTrack | COVID-19 Epidemic Dashboard",
    page_icon       = "🦠",
    layout          = "wide",
    initial_sidebar_state = "expanded"
)

st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #f4f6f9; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1a2e;
        color: white;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #e0e0e0 !important;
        font-size: 15px;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: #e0e0e0 !important;
    }

    /* KPI metric cards */
    [data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e4ea;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.07);
    }
    [data-testid="metric-container"] label {
        font-size: 13px !important;
        color: #6b7280 !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 26px !important;
        font-weight: 700 !important;
        color: #1a1a2e !important;
    }

    /* Section headers */
    .section-header {
        font-size: 22px;
        font-weight: 700;
        color: #1a1a2e;
        border-left: 5px solid #e63946;
        padding-left: 12px;
        margin-bottom: 8px;
    }

    /* Sub-headers */
    .sub-header {
        font-size: 16px;
        font-weight: 600;
        color: #374151;
        margin-bottom: 4px;
    }

    /* Info banner */
    .info-banner {
        background-color: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 10px;
        padding: 12px 18px;
        font-size: 13px;
        color: #1e40af;
        margin-bottom: 16px;
    }

    /* Divider */
    .custom-divider {
        border: none;
        border-top: 1px solid #e0e4ea;
        margin: 20px 0;
    }

    /* Chart containers */
    .chart-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        margin-bottom: 16px;
    }

    /* Tier badge colors */
    .tier-critical { color: #9b2226; font-weight: 700; }
    .tier-high     { color: #e76f51; font-weight: 700; }
    .tier-moderate { color: #f4a261; font-weight: 700; }
    .tier-low      { color: #2dc653; font-weight: 700; }

    /* Hide Streamlit default footer */
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_all_data():
    """Load all pre-computed CSV exports from Steps 1–3."""

    data = {}

    data['global_ts'] = pd.read_csv(
        "covid_global_timeseries.csv", parse_dates=['Date']
    )

    data['long'] = pd.read_csv(
        "covid_long_format.csv", parse_dates=['Date']
    )

    data['global_forecast'] = pd.read_csv(
        "global_forecast.csv", parse_dates=['ds']
    )

    data['country_forecasts'] = pd.read_csv(
        "country_forecasts.csv", parse_dates=['ds']
    )

    data['risk_snap'] = pd.read_csv("covid_risk_map_data.csv")
    data['country_risk'] = pd.read_csv("covid_country_risk.csv")

    return data


@st.cache_data(show_spinner=False)
def compute_kpis(global_ts, long_df, risk_snap):
    """Compute all KPI values once and cache them."""

    latest_date       = global_ts['Date'].max()
    total_confirmed   = int(global_ts['Global_Confirmed'].max())
    peak_daily        = int(global_ts['Daily_New_Cases'].max())
    peak_daily_date   = global_ts.loc[
        global_ts['Daily_New_Cases'].idxmax(), 'Date'
    ].strftime('%b %d, %Y')
    countries_count   = long_df['Country/Region'].nunique()
    critical_regions  = int((risk_snap['Risk_Tier'] == 'Critical').sum())

    # Top country by total cases
    top_country_row = (
        long_df[long_df['Date'] == latest_date]
        .groupby('Country/Region')['Confirmed_Cases']
        .sum()
        .idxmax()
    )
    top_country_cases = int(
        long_df[long_df['Date'] == latest_date]
        .groupby('Country/Region')['Confirmed_Cases']
        .sum()
        .max()
    )

    # Most recent 7-day avg
    latest_7day = global_ts['Rolling_7day_Avg'].dropna().iloc[-1]
    prev_7day   = global_ts['Rolling_7day_Avg'].dropna().iloc[-8]
    delta_7day  = latest_7day - prev_7day

    return {
        'latest_date'     : latest_date.strftime('%B %d, %Y'),
        'total_confirmed' : total_confirmed,
        'peak_daily'      : peak_daily,
        'peak_daily_date' : peak_daily_date,
        'countries_count' : countries_count,
        'critical_regions': critical_regions,
        'top_country'     : top_country_row,
        'top_country_cases': top_country_cases,
        'latest_7day'     : latest_7day,
        'delta_7day'      : delta_7day,
    }



def build_risk_bubble_map(df_snap):
    """Map C — Plotly province-level scatter_geo bubble map."""

    TIER_COLORS = {
        'Low'      : '#2dc653',
        'Moderate' : '#f4a261',
        'High'     : '#e76f51',
        'Critical' : '#9b2226'
    }

    fig = px.scatter_geo(
        df_snap,
        lat='Lat',
        lon='Long',
        size='Risk_Score_Norm',
        color='Risk_Tier',
        hover_name='Province/State',
        hover_data={
            'Country/Region'  : True,
            'Confirmed_Cases' : ':,',
            'Risk_Tier'       : True,
            'Lat'             : False,
            'Long'            : False,
            'Risk_Score_Norm' : False
        },
        color_discrete_map=TIER_COLORS,
        size_max=28,
        category_orders={'Risk_Tier': ['Low', 'Moderate', 'High', 'Critical']},
        projection='natural earth'
    )

    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor='#d1d5db',
            showland=True,
            landcolor='#f0ede8',
            showocean=True,
            oceancolor='#dbeafe',
            showlakes=True,
            lakecolor='#dbeafe',
            bgcolor='#f4f6f9'
        ),
        legend=dict(
            title='<b>Risk Tier</b>',
            orientation='v',
            font=dict(size=12)
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        height=520,
        paper_bgcolor='#ffffff',
    )
    return fig


def build_choropleth(df_country_risk):
    """Plotly choropleth country-level fill map."""

    fig = px.choropleth(
        df_country_risk.dropna(subset=['ISO_Alpha']),
        locations='ISO_Alpha',
        color='Log_Cases',
        hover_name='Country/Region',
        hover_data={
            'Confirmed_Cases' : ':,',
            'Log_Cases'       : ':.2f',
            'ISO_Alpha'       : False
        },
        color_continuous_scale=[
            (0.00, '#ffffb2'), (0.25, '#fecc5c'),
            (0.50, '#fd8d3c'), (0.75, '#f03b20'),
            (1.00, '#bd0026')
        ],
        labels={
            'Log_Cases'       : 'Log₁₀(Cases)',
            'Confirmed_Cases' : 'Total Cases'
        }
    )
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor='#d1d5db',
            projection_type='natural earth',
            bgcolor='#f4f6f9'
        ),
        coloraxis_colorbar=dict(
            title='Log₁₀<br>(Cases)',
            tickvals=[2, 3, 4, 5, 6, 7, 8],
            ticktext=['100', '1K', '10K', '100K', '1M', '10M', '100M']
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        height=480,
        paper_bgcolor='#ffffff'
    )
    return fig


def build_global_trend_chart(global_ts):
    """Dual-panel: cumulative cases + daily new cases with 7-day avg."""

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            'Cumulative Confirmed Cases (Millions)',
            'Daily New Cases (Thousands) with 7-Day Rolling Average'
        ),
        vertical_spacing=0.12
    )

    # Panel 1 — Cumulative
    fig.add_trace(go.Scatter(
        x=global_ts['Date'],
        y=global_ts['Global_Confirmed'] / 1e6,
        mode='lines',
        fill='tozeroy',
        fillcolor='rgba(230,57,70,0.10)',
        line=dict(color='#e63946', width=2),
        name='Cumulative Cases'
    ), row=1, col=1)

    # Panel 2 — Daily bars
    fig.add_trace(go.Bar(
        x=global_ts['Date'],
        y=global_ts['Daily_New_Cases'] / 1e3,
        marker_color='#457b9d',
        opacity=0.45,
        name='Daily New Cases'
    ), row=2, col=1)

    # Panel 2 — Rolling average
    fig.add_trace(go.Scatter(
        x=global_ts['Date'],
        y=global_ts['Rolling_7day_Avg'] / 1e3,
        mode='lines',
        line=dict(color='#e63946', width=2),
        name='7-Day Rolling Avg'
    ), row=2, col=1)

    fig.update_layout(
        height=580,
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode='x unified'
    )
    fig.update_yaxes(gridcolor='#f0f0f0')
    fig.update_xaxes(gridcolor='#f0f0f0')
    return fig

#this is used for building_top15_bar
def build_top15_bar(long_df):
    """Horizontal bar chart — Top 15 countries by total cases."""

    latest_date = long_df['Date'].max()

    df_top15 = (
        long_df[long_df['Date'] == latest_date]
        .groupby('Country/Region')['Confirmed_Cases']
        .sum()
        .reset_index()
        .sort_values('Confirmed_Cases', ascending=True)
        .tail(15)
    )

    fig = px.bar(
        df_top15,
        x='Confirmed_Cases',
        y='Country/Region',
        orientation='h',
        color='Confirmed_Cases',
        color_continuous_scale=['#fdd0ce', '#e63946'],
        labels={
            'Confirmed_Cases' : 'Total Confirmed Cases',
            'Country/Region'  : ''
        },
        text='Confirmed_Cases'
    )
    fig.update_traces(
        texttemplate='%{text:,.0f}',
        textposition='outside',
        textfont_size=10
    )
    fig.update_layout(
        height=520,
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        coloraxis_showscale=False,
        margin=dict(l=10, r=80, t=20, b=20),
        xaxis=dict(showgrid=True, gridcolor='#f0f0f0')
    )
    return fig


def build_top5_line(long_df):
    """Multi-line chart for top 5 countries over time."""

    latest_date  = long_df['Date'].max()

    top5_names = (
        long_df[long_df['Date'] == latest_date]
        .groupby('Country/Region')['Confirmed_Cases']
        .sum()
        .nlargest(5)
        .index.tolist()
    )

    df_top5 = (
        long_df[long_df['Country/Region'].isin(top5_names)]
        .groupby(['Country/Region', 'Date'])['Confirmed_Cases']
        .sum()
        .reset_index()
    )

    fig = px.line(
        df_top5,
        x='Date',
        y='Confirmed_Cases',
        color='Country/Region',
        labels={
            'Confirmed_Cases' : 'Confirmed Cases',
            'Date'            : 'Date',
            'Country/Region'  : 'Country'
        },
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig.update_layout(
        height=400,
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode='x unified',
        yaxis=dict(gridcolor='#f0f0f0'),
        xaxis=dict(gridcolor='#f0f0f0')
    )
    return fig


def build_global_forecast_chart(global_ts, global_forecast):
    """Interactive Prophet global forecast with CI bands."""

    cutoff       = global_ts['Date'].max()
    hist         = global_forecast[global_forecast['ds'] <= cutoff].copy()
    future       = global_forecast[global_forecast['ds'] >  cutoff].copy()

    fig = go.Figure()

    # Observed data
    fig.add_trace(go.Scatter(
        x=global_ts['Date'],
        y=global_ts['Rolling_7day_Avg'] / 1e3,
        mode='lines',
        name='Observed (7-Day Avg)',
        line=dict(color='#457b9d', width=1.8)
    ))

    # Historical model fit
    fig.add_trace(go.Scatter(
        x=hist['ds'],
        y=hist['yhat'] / 1e3,
        mode='lines',
        name='Model Fit',
        line=dict(color='#e63946', width=1.5, dash='dot'),
        opacity=0.7
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=future['ds'],
        y=future['yhat'] / 1e3,
        mode='lines',
        name='Forecast',
        line=dict(color='#f4a261', width=2.5, dash='dash')
    ))

    # 95% CI upper bound (invisible line for fill)
    fig.add_trace(go.Scatter(
        x=future['ds'],
        y=future['yhat_upper'] / 1e3,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))

    # 95% CI fill
    fig.add_trace(go.Scatter(
        x=future['ds'],
        y=future['yhat_lower'] / 1e3,
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(244,162,97,0.18)',
        line=dict(width=0),
        name='95% Confidence Interval',
        hoverinfo='skip'
    ))

# Forecast start vertical line
    fig.add_vline(
        x=cutoff,
        line_dash='dot',
        line_color='#9ca3af'
    )

    fig.update_layout(
        height=460,
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(title='Date', gridcolor='#f0f0f0'),
        yaxis=dict(title='Daily New Cases (Thousands)', gridcolor='#f0f0f0'),
        hovermode='x unified'
    )
    return fig


def build_country_forecast_chart(country_forecasts_df, long_df, country):
    """Individual country Prophet forecast from pre-computed CSV."""

    # Get country history
    latest_date = long_df['Date'].max()

    df_hist = (
        long_df[long_df['Country/Region'] == country]
        .groupby('Date')['Confirmed_Cases']
        .sum()
        .reset_index()
    )
    df_hist = df_hist.sort_values('Date')
    df_hist['Daily_New'] = df_hist['Confirmed_Cases'].diff().fillna(0).clip(lower=0)
    df_hist['Smooth']    = df_hist['Daily_New'].rolling(7).mean().fillna(0)

    # Get country forecast slice
    df_fc = country_forecasts_df[country_forecasts_df['Country'] == country].copy()

    if df_fc.empty:
        return None

    cutoff  = df_hist['Date'].max()
    fc_hist = df_fc[df_fc['ds'] <= cutoff]
    fc_fut  = df_fc[df_fc['ds'] >  cutoff]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_hist['Date'],
        y=df_hist['Smooth'] / 1e3,
        mode='lines',
        name='Observed (7-Day Avg)',
        line=dict(color='#457b9d', width=1.8)
    ))

    fig.add_trace(go.Scatter(
        x=fc_hist['ds'],
        y=fc_hist['yhat'] / 1e3,
        mode='lines',
        name='Model Fit',
        line=dict(color='#e63946', width=1.5, dash='dot'),
        opacity=0.7
    ))

    fig.add_trace(go.Scatter(
        x=fc_fut['ds'],
        y=fc_fut['yhat'] / 1e3,
        mode='lines',
        name='Forecast',
        line=dict(color='#f4a261', width=2.5, dash='dash')
    ))

    # CI upper (invisible)
    fig.add_trace(go.Scatter(
        x=fc_fut['ds'], y=fc_fut['yhat_upper'] / 1e3,
        mode='lines', line=dict(width=0),
        showlegend=False, hoverinfo='skip'
    ))

    # CI fill
    fig.add_trace(go.Scatter(
        x=fc_fut['ds'], y=fc_fut['yhat_lower'] / 1e3,
        mode='lines', fill='tonexty',
        fillcolor='rgba(244,162,97,0.18)',
        line=dict(width=0),
        name='95% Confidence Interval',
        hoverinfo='skip'
    ))

# Forecast start vertical line
    fig.add_vline(
        x=cutoff,
        line_dash='dot',
        line_color='#9ca3af'
    )

    fig.update_layout(
        height=420,
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(title='Date', gridcolor='#f0f0f0'),
        yaxis=dict(title='Daily New Cases (Thousands)', gridcolor='#f0f0f0'),
        hovermode='x unified',
        title=dict(
            text=f'<b>{country}</b> — 3-Month Outbreak Forecast',
            font=dict(size=14),
            x=0
        )
    )
    return fig


with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px 0;'>
        <span style='font-size:42px;'>🦠</span>
        <h2 style='color:white; margin:6px 0 2px 0; font-size:20px;'>EpiTrack</h2>
        <p style='color:#94a3b8; font-size:12px; margin:0;'>
            COVID-19 Epidemic Intelligence Dashboard
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#2d2d4e; margin:0 0 16px 0;'>", unsafe_allow_html=True)

    st.markdown("<p style='color:#94a3b8; font-size:11px; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;'>Navigation</p>", unsafe_allow_html=True)

    page = st.radio(
        label="",
        options=[
            "🌍  Overview & Risk Map",
            "📈  Trend Analysis",
            "🤖  AI Forecast (Prophet)"
        ],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:#2d2d4e; margin:16px 0;'>", unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:11px; color:#64748b; line-height:1.7;'>
        <b style='color:#94a3b8;'>Data Source</b><br>
        Johns Hopkins University<br>
        CSSE COVID-19 Dataset<br><br>
        <b style='color:#94a3b8;'>Model</b><br>
        Meta Prophet (Bayesian<br>structural time-series)<br><br>
        <b style='color:#94a3b8;'>Forecast Horizon</b><br>
        Global: 180 days<br>
        Country: 90 days<br><br>
        <b style='color:#94a3b8;'>Built for</b><br>
        CodeCure Biohackathon<br>
        Track C — Epidemic Modeling
    </div>
    """, unsafe_allow_html=True)




with st.spinner("Loading pre-computed epidemic data..."):
    data    = load_all_data()
    kpis    = compute_kpis(data['global_ts'], data['long'], data['risk_snap'])

global_ts          = data['global_ts']
long_df            = data['long']
global_forecast    = data['global_forecast']
country_forecasts  = data['country_forecasts']
risk_snap          = data['risk_snap']
country_risk       = data['country_risk']

available_countries = sorted(country_forecasts['Country'].unique().tolist())


if page == "🌍  Overview & Risk Map":

    # --- Page Header ---
    st.markdown(
        "<div class='section-header'>🌍 Global Overview & Risk Map</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='info-banner'>📅 Data through <b>{kpis['latest_date']}</b> — "
        f"All metrics are computed from pre-processed JHU CSSE time-series data.</div>",
        unsafe_allow_html=True
    )

    # --- KPI Row ---
    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        st.metric(
            label="🧪 Total Confirmed Cases",
            value=f"{kpis['total_confirmed'] / 1e6:.1f}M"
        )
    with k2:
        st.metric(
            label="🏔️ Peak Daily New Cases",
            value=f"{kpis['peak_daily'] / 1e6:.2f}M",
            help=f"Occurred on {kpis['peak_daily_date']}"
        )
    with k3:
        st.metric(
            label="🌐 Countries / Regions",
            value=f"{kpis['countries_count']}"
        )
    with k4:
        st.metric(
            label="🚨 Critical Risk Zones",
            value=f"{kpis['critical_regions']}",
            help="Provinces/regions above the 75th percentile of confirmed cases"
        )
    with k5:
        st.metric(
            label=f"🥇 Most Impacted Country",
            value=kpis['top_country'],
            help=f"{kpis['top_country_cases']:,} total confirmed cases"
        )

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # --- Map Toggle ---
    st.markdown("<div class='sub-header'>Select Map View</div>", unsafe_allow_html=True)
    map_choice = st.radio(
        label="",
        options=["🔵 Province-Level Bubble Map", "🗺️ Country Choropleth Map"],
        horizontal=True,
        label_visibility="collapsed"
    )

    with st.container():
        if map_choice == "🔵 Province-Level Bubble Map":
            st.markdown(
                "<div class='sub-header'>Province-Level Risk Bubble Map</div>",
                unsafe_allow_html=True
            )
            st.caption(
                "Bubble size and color reflect the risk tier (percentile-based). "
                "Hover for details · Click a bubble for full stats."
            )
            st.plotly_chart(
                build_risk_bubble_map(risk_snap),
                use_container_width=True
            )
        else:
            st.markdown(
                "<div class='sub-header'>Country-Level Choropleth Map</div>",
                unsafe_allow_html=True
            )
            st.caption(
                "Color intensity reflects Log₁₀(confirmed cases). "
                "Hover over any country for exact figures."
            )
            st.plotly_chart(
                build_choropleth(country_risk),
                use_container_width=True
            )

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # --- Risk Tier Summary Table ---
    st.markdown("<div class='sub-header'>📋 Risk Tier Summary by Province</div>", unsafe_allow_html=True)

    tier_order = ['Critical', 'High', 'Moderate', 'Low']
    tier_summary = (
        risk_snap.groupby('Risk_Tier')
        .agg(
            Regions=('Province/State', 'count'),
            Total_Cases=('Confirmed_Cases', 'sum'),
            Avg_Cases=('Confirmed_Cases', 'mean')
        )
        .reindex(tier_order)
        .reset_index()
    )
    tier_summary.columns = ['Risk Tier', 'Regions', 'Total Cases', 'Avg Cases per Region']
    tier_summary['Total Cases']          = tier_summary['Total Cases'].apply(lambda x: f"{x:,.0f}")
    tier_summary['Avg Cases per Region'] = tier_summary['Avg Cases per Region'].apply(lambda x: f"{x:,.0f}")

    st.dataframe(
        tier_summary,
        use_container_width=True,
        hide_index=True
    )


elif page == "📈  Trend Analysis":

    st.markdown(
        "<div class='section-header'>📈 Epidemic Trend Analysis</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div class='info-banner'>These charts are generated from the pre-processed "
        "JHU CSSE time-series. The 7-day rolling average smooths weekend reporting "
        "artifacts to reveal the true epidemic trend.</div>",
        unsafe_allow_html=True
    )

    # --- Panel 1: Global Trend ---
    st.markdown("<div class='sub-header'>🌐 Global Cumulative & Daily New Cases</div>", unsafe_allow_html=True)
    st.caption("Top panel: cumulative confirmed cases (millions). Bottom panel: daily new cases with 7-day rolling average.")

    with st.container():
        st.plotly_chart(
            build_global_trend_chart(global_ts),
            use_container_width=True
        )

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # --- Panel 2: Two columns for bar + line ---
    col_left, col_right = st.columns([1, 1], gap="medium")

    with col_left:
        st.markdown("<div class='sub-header'>🏆 Top 15 Countries — Total Cases</div>", unsafe_allow_html=True)
        st.caption("Ranked by cumulative confirmed cases as of the latest available date.")
        st.plotly_chart(
            build_top15_bar(long_df),
            use_container_width=True
        )

    with col_right:
        st.markdown("<div class='sub-header'>📊 Top 5 Countries — Case Trajectory</div>", unsafe_allow_html=True)
        st.caption("Cumulative confirmed cases over the full observation period.")
        st.plotly_chart(
            build_top5_line(long_df),
            use_container_width=True
        )

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # --- Panel 3: Raw Data Explorer ---
    with st.expander("🔍 Explore Raw Global Time-Series Data", expanded=False):
        st.markdown("<div class='sub-header'>Global Time-Series Table</div>", unsafe_allow_html=True)

        date_range = st.date_input(
            "Filter by date range",
            value=(global_ts['Date'].min().date(), global_ts['Date'].max().date()),
            min_value=global_ts['Date'].min().date(),
            max_value=global_ts['Date'].max().date()
        )

        if len(date_range) == 2:
            mask = (
                (global_ts['Date'].dt.date >= date_range[0]) &
                (global_ts['Date'].dt.date <= date_range[1])
            )
            df_display = global_ts[mask][
                ['Date', 'Global_Confirmed', 'Daily_New_Cases', 'Rolling_7day_Avg']
            ].copy()
            df_display.columns = ['Date', 'Cumulative Cases', 'Daily New Cases', '7-Day Avg']
            df_display['Date']             = df_display['Date'].dt.strftime('%Y-%m-%d')
            df_display['Cumulative Cases'] = df_display['Cumulative Cases'].apply(lambda x: f"{x:,.0f}")
            df_display['Daily New Cases']  = df_display['Daily New Cases'].apply(lambda x: f"{x:,.0f}")
            df_display['7-Day Avg']        = df_display['7-Day Avg'].apply(
                lambda x: f"{x:,.0f}" if pd.notna(x) else "—"
            )
            st.dataframe(df_display, use_container_width=True, hide_index=True)



elif page == "🤖  AI Forecast (Prophet)":

    st.markdown(
        "<div class='section-header'>🤖 AI Outbreak Forecast — Meta Prophet</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div class='info-banner'>"
        "⚡ Forecasts are <b>pre-computed</b> — no model training happens here. "
        "The Prophet model was trained offline on the full historical time-series and "
        "the results are loaded instantly from CSV. "
        "Global horizon: <b>180 days</b> · Country horizon: <b>90 days</b> · "
        "Confidence interval: <b>95%</b>."
        "</div>",
        unsafe_allow_html=True
    )

    # --- Model methodology expander ---
    with st.expander("📖 Why Prophet? — Epidemiological Justification", expanded=False):
        st.markdown("""
        **Meta Prophet** was selected over ARIMA for this dataset for the following reasons:

        | Criterion | ARIMA | Prophet ✅ |
        |---|---|---|
        | Handles multi-wave non-linear trends | ❌ Needs manual differencing | ✅ Automatic changepoint detection |
        | Weekly reporting artifacts (weekend dip) | ❌ Requires manual regressors | ✅ Built-in weekly seasonality |
        | Annual patterns (winter surges) | ❌ Requires seasonal ARIMA (SARIMA) | ✅ Built-in yearly seasonality |
        | 1,100+ day training series | ❌ Risk of overfitting | ✅ Robust at scale |
        | Uncertainty quantification | ❌ Requires extra work | ✅ Native CI bands |

        **Key hyperparameters applied:**
        - `changepoint_prior_scale=0.3` — flexible enough to capture COVID wave inflections
        - `seasonality_mode='multiplicative'` — case counts scale proportionally with trend
        - `interval_width=0.95` — 95% confidence bands for decision support
        - Input signal: **7-day rolling average** of daily new cases (removes reporting noise)
        """)

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # --- Section 1: Global Forecast ---
    st.markdown("<div class='sub-header'>🌐 Global 6-Month Forecast</div>", unsafe_allow_html=True)
    st.caption(
        "Dashed orange line = forecast · Shaded band = 95% confidence interval · "
        "Dotted red line = model fit on historical data."
    )

    st.plotly_chart(
        build_global_forecast_chart(global_ts, global_forecast),
        use_container_width=True
    )

    # Forecast summary metrics
    cutoff      = global_ts['Date'].max()
    fc_future   = global_forecast[global_forecast['ds'] > cutoff]

    if not fc_future.empty:
        fc_peak     = fc_future['yhat'].max()
        fc_end_val  = fc_future['yhat'].iloc[-1]
        fc_end_date = fc_future['ds'].iloc[-1].strftime('%b %d, %Y')
        fc_peak_date= fc_future.loc[fc_future['yhat'].idxmax(), 'ds'].strftime('%b %d, %Y')

        fm1, fm2, fm3 = st.columns(3)
        with fm1:
            st.metric("📅 Forecast End Date", fc_end_date)
        with fm2:
            st.metric("🏔️ Forecast Peak Daily Cases", f"{fc_peak / 1e3:,.1f}K",
                      help=f"Expected around {fc_peak_date}")
        with fm3:
            st.metric("📉 Projected Daily Cases at End", f"{fc_end_val / 1e3:,.1f}K")

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # --- Section 2: Country-Level Forecast ---
    st.markdown("<div class='sub-header'>🏳️ Country-Level 3-Month Forecast</div>", unsafe_allow_html=True)

    col_select, col_info = st.columns([1, 2], gap="medium")

    with col_select:
        selected_country = st.selectbox(
            label="Select a country to view its forecast:",
            options=available_countries,
            index=0,
            help="These are the Top 5 highest-impact countries. "
                 "Each has an independently trained Prophet model."
        )

    with col_info:
        # Show country's latest total cases as context
        latest_date = long_df['Date'].max()
        country_total = int(
            long_df[(long_df['Country/Region'] == selected_country) &
                    (long_df['Date'] == latest_date)]['Confirmed_Cases'].sum()
        )
        country_risk_tier = risk_snap[
            risk_snap['Country/Region'] == selected_country
        ]['Risk_Tier'].mode()
        tier_label = country_risk_tier.iloc[0] if not country_risk_tier.empty else "N/A"
        tier_color_map = {
            'Critical': '#9b2226', 'High': '#e76f51',
            'Moderate': '#f4a261', 'Low' : '#2dc653'
        }
        t_color = tier_color_map.get(tier_label, '#374151')

        st.markdown(f"""
        <div style='background:#ffffff; border-radius:10px; padding:14px 18px;
                    border:1px solid #e0e4ea; margin-top:4px;'>
            <span style='font-size:13px; color:#6b7280;'>Selected Country</span><br>
            <b style='font-size:20px; color:#1a1a2e;'>{selected_country}</b><br>
            <span style='font-size:13px; color:#374151;'>
                Total confirmed: <b>{country_total:,}</b> &nbsp;|&nbsp;
                Dominant risk tier: <b style='color:{t_color};'>{tier_label}</b>
            </span>
        </div>
        """, unsafe_allow_html=True)

    # Build and display the country forecast chart
    fig_country = build_country_forecast_chart(country_forecasts, long_df, selected_country)

    if fig_country:
        st.plotly_chart(fig_country, use_container_width=True)
    else:
        st.warning(f"No pre-computed forecast available for {selected_country}. "
                   "Please check that Step 2 completed successfully.")

    # Country forecast summary metrics
    fc_country = country_forecasts[country_forecasts['Country'] == selected_country]
    country_cutoff = long_df[long_df['Country/Region'] == selected_country]['Date'].max()
    fc_c_fut = fc_country[fc_country['ds'] > country_cutoff]

    if not fc_c_fut.empty:
        c_peak      = fc_c_fut['yhat'].max()
        c_end_val   = fc_c_fut['yhat'].iloc[-1]
        c_end_date  = fc_c_fut['ds'].iloc[-1].strftime('%b %d, %Y')
        c_peak_date = fc_c_fut.loc[fc_c_fut['yhat'].idxmax(), 'ds'].strftime('%b %d, %Y')

        cm1, cm2, cm3 = st.columns(3)
        with cm1:
            st.metric("📅 Forecast End Date", c_end_date)
        with cm2:
            st.metric("🏔️ Forecast Peak Daily Cases",
                      f"{c_peak / 1e3:,.1f}K" if c_peak >= 1000 else f"{c_peak:,.0f}",
                      help=f"Expected around {c_peak_date}")
        with cm3:
            st.metric("📉 Projected Daily Cases at End",
                      f"{c_end_val / 1e3:,.1f}K" if c_end_val >= 1000 else f"{c_end_val:,.0f}")
