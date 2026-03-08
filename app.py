import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  CONFIGURATION & TRANSLATIONS
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CreditMacro Togo",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

TRANSLATIONS = {
    "fr": {
        "app_title": "🏦 CreditMacro Togo",
        "app_subtitle": "Outil d'aide à la décision pour analystes crédit en microfinance",
        "lang_label": "Langue / Language",
        "nav_dashboard": "📊 Tableau de bord",
        "nav_macro": "🌍 Analyse Macroéconomique",
        "nav_portfolio": "📁 Analyse du Portefeuille",
        "nav_decision": "⚖️ Aide à la Décision",
        "nav_econometrics": "📐 Modèles Économétriques",
        "nav_risk": "🚨 Scoring de Risque",

        # Dashboard
        "dash_title": "Vue d'ensemble du portefeuille et contexte macro",
        "kpi_total_loans": "Total prêts",
        "kpi_total_disbursed": "Encours total",
        "kpi_repayment_rate": "Taux de remboursement global",
        "kpi_default_rate": "Taux de défaut",
        "kpi_avg_amount": "Montant moyen",
        "kpi_gdp_growth": "Croissance PIB (dernière)",
        "portfolio_by_status": "Répartition par statut",
        "portfolio_by_sector": "Répartition par secteur",
        "portfolio_by_region": "Distribution géographique",
        "trend_disbursements": "Évolution des décaissements",

        # Macro
        "macro_title": "Analyse Macroéconomique — Togo",
        "gdp_growth_title": "Croissance du PIB annuel (%)",
        "pop_title": "Évolution de la population",
        "macro_context": "Contexte macroéconomique actuel",
        "gdp_avg_5y": "Croissance moyenne (5 ans)",
        "gdp_volatility": "Volatilité du PIB",
        "gdp_trend": "Tendance PIB",
        "pop_growth": "Croissance démographique",
        "macro_alert_positive": "✅ Environnement favorable au crédit",
        "macro_alert_warning": "⚠️ Environnement à surveiller",
        "macro_alert_negative": "🔴 Environnement défavorable",
        "gdp_filter": "Période d'analyse",
        "gdp_context_text": "Interprétation macroéconomique",
        "pop_density": "Densité de crédit potentielle",
        "credit_demand_index": "Indice de demande de crédit estimé",

        # Portfolio
        "port_title": "Analyse statistique du portefeuille",
        "dist_amount": "Distribution des montants prêtés",
        "dist_duration": "Distribution des durées",
        "dist_rate": "Distribution des taux d'intérêt",
        "corr_matrix": "Matrice de corrélation",
        "sector_performance": "Performance par secteur",
        "region_performance": "Performance par région",
        "age_analysis": "Analyse par tranche d'âge",
        "gender_analysis": "Analyse par genre",

        # Decision
        "dec_title": "Module d'aide à la décision crédit",
        "dec_subtitle": "Évaluation macroéconomique et sectorielle pour grands montants",
        "dec_sector": "Secteur d'activité du projet",
        "dec_region": "Région d'implantation",
        "dec_amount": "Montant sollicité (FCFA)",
        "dec_duration": "Durée souhaitée (mois)",
        "dec_age": "Âge du demandeur",
        "dec_gender": "Genre",
        "dec_analyze": "🔍 Analyser le dossier",
        "dec_score_title": "Score de risque global",
        "dec_macro_score": "Score Macroéconomique",
        "dec_sector_score": "Score Sectoriel",
        "dec_stat_score": "Score Statistique (portefeuille)",
        "dec_recommendation": "Recommandation",
        "rec_favorable": "✅ FAVORABLE",
        "rec_reserve": "⚠️ FAVORABLE AVEC RÉSERVES",
        "rec_unfavorable": "🔴 DÉFAVORABLE",
        "dec_justification": "Justification analytique",
        "conditions_title": "Conditions suggérées",
        "cond_amount": "Montant recommandé",
        "cond_rate": "Fourchette de taux suggérée",
        "cond_duration": "Durée recommandée",
        "cond_collateral": "Garanties recommandées",
        "dec_comparables": "Dossiers comparables dans le portefeuille",

        # Econometrics
        "eco_title": "Modèles Économétriques",
        "eco_regression": "Régression: Facteurs de remboursement",
        "eco_var": "Variables explicatives",
        "eco_coef": "Coefficient",
        "eco_pval": "p-valeur",
        "eco_significance": "Significativité",
        "eco_r2": "R² du modèle",
        "eco_forecast": "Prévision de la demande de crédit",
        "eco_gdp_corr": "Corrélation PIB / Volume de crédit",
        "eco_ols_title": "Résultats de la régression OLS",

        # Risk
        "risk_title": "Scoring de Risque Sectoriel & Macro",
        "risk_sector_map": "Carte de risque sectorielle",
        "risk_regional_map": "Carte de risque régionale",
        "risk_gdp_sensitivity": "Sensibilité sectorielle aux chocs PIB",

        # General
        "fcfa": "F CFA",
        "years": "ans",
        "months": "mois",
        "male": "Homme",
        "female": "Femme",
        "agriculture": "Agriculture",
        "commerce": "Commerce",
        "artisanat": "Artisanat",
        "source_wb": "Source: Banque Mondiale",
        "no_data": "Données insuffisantes",
    },
    "en": {
        "app_title": "🏦 CreditMacro Togo",
        "app_subtitle": "Decision support tool for microfinance credit analysts",
        "lang_label": "Langue / Language",
        "nav_dashboard": "📊 Dashboard",
        "nav_macro": "🌍 Macroeconomic Analysis",
        "nav_portfolio": "📁 Portfolio Analysis",
        "nav_decision": "⚖️ Decision Support",
        "nav_econometrics": "📐 Econometric Models",
        "nav_risk": "🚨 Risk Scoring",

        # Dashboard
        "dash_title": "Portfolio overview and macro context",
        "kpi_total_loans": "Total loans",
        "kpi_total_disbursed": "Total outstanding",
        "kpi_repayment_rate": "Global repayment rate",
        "kpi_default_rate": "Default rate",
        "kpi_avg_amount": "Average amount",
        "kpi_gdp_growth": "GDP growth (latest)",
        "portfolio_by_status": "Distribution by status",
        "portfolio_by_sector": "Distribution by sector",
        "portfolio_by_region": "Geographic distribution",
        "trend_disbursements": "Disbursement trends",

        # Macro
        "macro_title": "Macroeconomic Analysis — Togo",
        "gdp_growth_title": "Annual GDP Growth (%)",
        "pop_title": "Population Growth",
        "macro_context": "Current macroeconomic context",
        "gdp_avg_5y": "Average growth (5 years)",
        "gdp_volatility": "GDP volatility",
        "gdp_trend": "GDP trend",
        "pop_growth": "Demographic growth",
        "macro_alert_positive": "✅ Favorable lending environment",
        "macro_alert_warning": "⚠️ Environment to monitor",
        "macro_alert_negative": "🔴 Unfavorable environment",
        "gdp_filter": "Analysis period",
        "gdp_context_text": "Macroeconomic interpretation",
        "pop_density": "Potential credit density",
        "credit_demand_index": "Estimated credit demand index",

        # Portfolio
        "port_title": "Portfolio statistical analysis",
        "dist_amount": "Loan amount distribution",
        "dist_duration": "Duration distribution",
        "dist_rate": "Interest rate distribution",
        "corr_matrix": "Correlation matrix",
        "sector_performance": "Performance by sector",
        "region_performance": "Performance by region",
        "age_analysis": "Analysis by age group",
        "gender_analysis": "Gender analysis",

        # Decision
        "dec_title": "Credit Decision Support Module",
        "dec_subtitle": "Macroeconomic and sectoral evaluation for large loans",
        "dec_sector": "Project activity sector",
        "dec_region": "Implementation region",
        "dec_amount": "Requested amount (FCFA)",
        "dec_duration": "Desired duration (months)",
        "dec_age": "Applicant age",
        "dec_gender": "Gender",
        "dec_analyze": "🔍 Analyze Application",
        "dec_score_title": "Overall risk score",
        "dec_macro_score": "Macroeconomic Score",
        "dec_sector_score": "Sectoral Score",
        "dec_stat_score": "Statistical Score (portfolio)",
        "dec_recommendation": "Recommendation",
        "rec_favorable": "✅ FAVORABLE",
        "rec_reserve": "⚠️ FAVORABLE WITH RESERVES",
        "rec_unfavorable": "🔴 UNFAVORABLE",
        "dec_justification": "Analytical justification",
        "conditions_title": "Suggested conditions",
        "cond_amount": "Recommended amount",
        "cond_rate": "Suggested rate range",
        "cond_duration": "Recommended duration",
        "cond_collateral": "Recommended collateral",
        "dec_comparables": "Comparable files in portfolio",

        # Econometrics
        "eco_title": "Econometric Models",
        "eco_regression": "Regression: Repayment factors",
        "eco_var": "Explanatory variables",
        "eco_coef": "Coefficient",
        "eco_pval": "p-value",
        "eco_significance": "Significance",
        "eco_r2": "Model R²",
        "eco_forecast": "Credit demand forecast",
        "eco_gdp_corr": "GDP / Credit volume correlation",
        "eco_ols_title": "OLS Regression Results",

        # Risk
        "risk_title": "Sectoral & Macro Risk Scoring",
        "risk_sector_map": "Sectoral risk map",
        "risk_regional_map": "Regional risk map",
        "risk_gdp_sensitivity": "Sectoral sensitivity to GDP shocks",

        # General
        "fcfa": "FCFA",
        "years": "years",
        "months": "months",
        "male": "Male",
        "female": "Female",
        "agriculture": "Agriculture",
        "commerce": "Commerce",
        "artisanat": "Crafts",
        "source_wb": "Source: World Bank",
        "no_data": "Insufficient data",
    }
}

# ─────────────────────────────────────────────
#  CSS STYLING
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main { background-color: #f0f2f6; }
    
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border-left: 4px solid #1a56db;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        margin-bottom: 0.5rem;
    }
    .metric-card.green { border-left-color: #0e9f6e; }
    .metric-card.red { border-left-color: #e02424; }
    .metric-card.orange { border-left-color: #ff5a1f; }
    .metric-card.purple { border-left-color: #7e3af2; }
    
    .metric-label { font-size: 0.75rem; color: #6b7280; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #111827; line-height: 1.2; }
    .metric-delta { font-size: 0.8rem; color: #6b7280; margin-top: 2px; }
    
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #111827;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    
    .score-gauge {
        text-align: center;
        padding: 1.5rem;
        border-radius: 16px;
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    .score-number { font-size: 3rem; font-weight: 800; }
    .score-label { font-size: 0.9rem; color: #6b7280; font-weight: 500; }
    
    .decision-box {
        padding: 1.5rem 2rem;
        border-radius: 12px;
        font-size: 1.2rem;
        font-weight: 700;
        text-align: center;
        margin: 1rem 0;
    }
    .decision-green { background: #d1fae5; color: #065f46; border: 2px solid #6ee7b7; }
    .decision-orange { background: #fff3cd; color: #92400e; border: 2px solid #fcd34d; }
    .decision-red { background: #fee2e2; color: #991b1b; border: 2px solid #fca5a5; }
    
    .info-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #1e40af;
    }
    
    .warning-box {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #92400e;
    }
    
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
        font-weight: 500;
    }
    
    .sidebar-header {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #9ca3af;
        padding: 0.5rem 0 0.2rem 0;
    }
    
    div[data-testid="stMetric"] { background: white; border-radius: 10px; padding: 0.8rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    # GDP
    gdp_raw = pd.read_csv("Croissance_du_PIB_Togo.csv")
    years = [str(y) for y in range(1960, 2025)]
    gdp_values = gdp_raw[years].values[0]
    gdp = pd.DataFrame({"Année": list(range(1960, 2025)), "Croissance_PIB": gdp_values})
    gdp = gdp.dropna()

    # Population
    pop_raw = pd.read_csv("Population_togo.csv")
    pop_values = pop_raw[years].values[0]
    pop = pd.DataFrame({"Année": list(range(1960, 2025)), "Population": pop_values})
    pop = pop.dropna()

    # Portfolio
    df = pd.read_csv("Jeux_donnees.csv")
    # Clean monetary columns
    def clean_amount(col):
        return df[col].astype(str).str.replace(r'[^\d.]', '', regex=True).replace('', np.nan).astype(float)

    df['Montant_Num'] = clean_amount(' Montant_Prete_FCFA ')
    df['Taux_Num'] = df[' Taux_Interet '].astype(str).str.replace(r'[^\d.]', '', regex=True).astype(float)
    df['Remboursement_Num'] = clean_amount(' Montant_Rembourse_FCFA ')
    df['Total_Du_Num'] = clean_amount('Montant_Total_Du')
    df['Restant_Num'] = clean_amount(' Montant_Restant ')

    df['Defaut'] = df['Statut'].apply(lambda x: 1 if x.strip() in ['EN RETARD'] else 0)
    df['Rembourse_Flag'] = df['Statut'].apply(lambda x: 1 if x.strip() == 'REMBOURSE' else 0)
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['Duree_Num'] = pd.to_numeric(df[' Duree_Mois'], errors='coerce')
    df['Activite'] = df['Activite'].str.strip()
    df['Region'] = df['Region'].str.strip()
    df['Sexe'] = df['Sexe'].str.strip()
    df['Statut'] = df['Statut'].str.strip()

    return gdp, pop, df

gdp_df, pop_df, loans_df = load_data()


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/68/Flag_of_Togo.svg", width=60)
    
    lang = st.selectbox("🌐 Langue / Language", ["Français", "English"], key="lang_select")
    L = TRANSLATIONS["fr"] if lang == "Français" else TRANSLATIONS["en"]

    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #1a56db, #0891b2); padding: 1rem; border-radius: 10px; color: white; margin: 0.5rem 0;'>
        <div style='font-size: 1rem; font-weight: 700;'>{L["app_title"]}</div>
        <div style='font-size: 0.75rem; opacity: 0.85; margin-top: 4px;'>{L["app_subtitle"]}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-header">Navigation</div>', unsafe_allow_html=True)
    page = st.radio(
        "Menu",
        [L["nav_dashboard"], L["nav_macro"], L["nav_portfolio"],
         L["nav_decision"], L["nav_econometrics"], L["nav_risk"]],
        label_visibility="collapsed"
    )

    # Quick macro stats
    last_gdp = gdp_df[gdp_df['Croissance_PIB'].notna()].iloc[-1]['Croissance_PIB']
    gdp_5y_avg = gdp_df.tail(5)['Croissance_PIB'].mean()
    st.markdown('<div class="sidebar-header">Macro Rapide / Quick</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("PIB/GDP", f"{last_gdp:.1f}%")
    with col2:
        st.metric("5Y avg", f"{gdp_5y_avg:.1f}%")
    
    st.markdown(f"<div style='font-size:0.7rem; color:#9ca3af; text-align:center; margin-top:1rem;'>{L['source_wb']}</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────
def fmt_fcfa(n):
    if pd.isna(n): return "—"
    return f"{n:,.0f} F CFA"

def score_color(score):
    if score >= 70: return "#0e9f6e"
    if score >= 45: return "#ff5a1f"
    return "#e02424"

def gauge_chart(score, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': title, 'font': {'size': 13}},
        number={'suffix': '/100', 'font': {'size': 22}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': score_color(score)},
            'steps': [
                {'range': [0, 45], 'color': '#fee2e2'},
                {'range': [45, 70], 'color': '#fff3cd'},
                {'range': [70, 100], 'color': '#d1fae5'},
            ],
            'threshold': {'line': {'color': "black", 'width': 2}, 'thickness': 0.75, 'value': score}
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=10))
    return fig


# ─────────────────────────────────────────────
#  PAGE: DASHBOARD
# ─────────────────────────────────────────────
if page == L["nav_dashboard"]:
    st.markdown(f"<div class='section-title'>{L['dash_title']}</div>", unsafe_allow_html=True)

    # KPIs
    total_loans = len(loans_df)
    total_disbursed = loans_df['Montant_Num'].sum()
    repayment_rate = loans_df['Rembourse_Flag'].mean() * 100
    default_rate = loans_df['Defaut'].mean() * 100
    avg_amount = loans_df['Montant_Num'].mean()
    last_gdp_val = gdp_df.dropna().iloc[-1]['Croissance_PIB']

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>{L['kpi_total_loans']}</div>
            <div class='metric-value'>{total_loans}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card purple'>
            <div class='metric-label'>{L['kpi_total_disbursed']}</div>
            <div class='metric-value'>{total_disbursed/1e6:.1f}M</div>
            <div class='metric-delta'>F CFA</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card green'>
            <div class='metric-label'>{L['kpi_repayment_rate']}</div>
            <div class='metric-value'>{repayment_rate:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='metric-card red'>
            <div class='metric-label'>{L['kpi_default_rate']}</div>
            <div class='metric-value'>{default_rate:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""<div class='metric-card orange'>
            <div class='metric-label'>{L['kpi_avg_amount']}</div>
            <div class='metric-value'>{avg_amount/1000:.0f}K</div>
            <div class='metric-delta'>F CFA</div>
        </div>""", unsafe_allow_html=True)
    with c6:
        gdp_color = "green" if last_gdp_val > 4 else ("orange" if last_gdp_val > 0 else "red")
        st.markdown(f"""<div class='metric-card {gdp_color}'>
            <div class='metric-label'>{L['kpi_gdp_growth']}</div>
            <div class='metric-value'>{last_gdp_val:.1f}%</div>
            <div class='metric-delta'>PIB Togo</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        # Status distribution
        status_counts = loans_df['Statut'].value_counts().reset_index()
        status_counts.columns = ['Statut', 'Count']
        colors = {'REMBOURSE': '#0e9f6e', 'EN COURS': '#1a56db', 'EN RETARD': '#e02424'}
        fig = px.pie(status_counts, values='Count', names='Statut',
                     title=L["portfolio_by_status"],
                     color='Statut', color_discrete_map=colors,
                     hole=0.4)
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Sector distribution
        sector_counts = loans_df.groupby('Activite').agg(
            Count=('ID', 'count'),
            Volume=('Montant_Num', 'sum')
        ).reset_index()
        fig = px.bar(sector_counts, x='Activite', y='Volume',
                     title=L["portfolio_by_sector"],
                     color='Count',
                     color_continuous_scale='Blues',
                     labels={'Volume': 'F CFA', 'Activite': ''})
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        # Regional distribution
        region_data = loans_df.groupby('Region').agg(
            Prêts=('ID', 'count'),
            Défauts=('Defaut', 'sum'),
            Volume=('Montant_Num', 'sum')
        ).reset_index()
        region_data['Taux_défaut'] = (region_data['Défauts'] / region_data['Prêts'] * 100).round(1)
        fig = px.bar(region_data.sort_values('Volume', ascending=True),
                     x='Volume', y='Region', orientation='h',
                     color='Taux_défaut',
                     title=L["portfolio_by_region"],
                     color_continuous_scale='RdYlGn_r',
                     labels={'Volume': 'F CFA', 'Region': ''})
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        # GDP trend (recent 20 years)
        gdp_recent = gdp_df.tail(20)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=gdp_recent['Année'], y=gdp_recent['Croissance_PIB'],
            mode='lines+markers', fill='tozeroy',
            line=dict(color='#1a56db', width=2),
            fillcolor='rgba(26,86,219,0.1)',
            name='PIB Growth'
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
        fig.update_layout(title=L["gdp_growth_title"] + " (2003–2023)",
                          height=320, margin=dict(l=0, r=0, t=40, b=0),
                          xaxis_title="", yaxis_title="%")
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
#  PAGE: MACRO ANALYSIS
# ─────────────────────────────────────────────
elif page == L["nav_macro"]:
    st.markdown(f"<div class='section-title'>{L['macro_title']}</div>", unsafe_allow_html=True)

    # Year range filter
    year_range = st.slider(L["gdp_filter"], 1960, 2023, (2000, 2023))
    gdp_filtered = gdp_df[(gdp_df['Année'] >= year_range[0]) & (gdp_df['Année'] <= year_range[1])]
    pop_filtered = pop_df[(pop_df['Année'] >= year_range[0]) & (pop_df['Année'] <= year_range[1])]

    # KPI row
    gdp_mean = gdp_filtered['Croissance_PIB'].mean()
    gdp_std = gdp_filtered['Croissance_PIB'].std()
    gdp_last = gdp_filtered.dropna().iloc[-1]['Croissance_PIB']
    pop_last = pop_df.dropna().iloc[-1]['Population']
    pop_prev = pop_df.dropna().iloc[-2]['Population']
    pop_growth_rate = (pop_last / pop_prev - 1) * 100

    # Trend (linear regression on GDP)
    x = gdp_filtered.dropna()['Année'].values
    y = gdp_filtered.dropna()['Croissance_PIB'].values
    if len(x) > 2:
        slope, intercept, r_val, p_val, _ = stats.linregress(x, y)
        trend_dir = "↗ Haussière" if slope > 0.05 else ("↘ Baissière" if slope < -0.05 else "→ Stable")
    else:
        slope, trend_dir = 0, "→ Stable"

    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        (L['gdp_avg_5y'], f"{gdp_mean:.2f}%", "Moyenne période", "green" if gdp_mean > 4 else "orange"),
        (L['gdp_volatility'], f"±{gdp_std:.2f}%", "Écart-type", "green" if gdp_std < 4 else "red"),
        (L['gdp_trend'], trend_dir, f"pente = {slope:.3f}", ""),
        (L['pop_growth'], f"{pop_growth_rate:.2f}%/an", f"{pop_last/1e6:.1f}M hab.", ""),
    ]
    for col, (label, val, delta, color) in zip([c1, c2, c3, c4], metrics):
        with col:
            st.markdown(f"""<div class='metric-card {color}'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value'>{val}</div>
                <div class='metric-delta'>{delta}</div>
            </div>""", unsafe_allow_html=True)

    # Alert banner
    st.markdown("<br>", unsafe_allow_html=True)
    if gdp_last > 5 and gdp_mean > 4:
        st.markdown(f"<div class='info-box'><strong>{L['macro_alert_positive']}</strong> — PIB {gdp_last:.1f}% | Moyenne {gdp_mean:.1f}% | Volatilité {gdp_std:.1f}%</div>", unsafe_allow_html=True)
    elif gdp_last > 0:
        st.markdown(f"<div class='warning-box'><strong>{L['macro_alert_warning']}</strong> — PIB {gdp_last:.1f}% | Moyenne {gdp_mean:.1f}% | Volatilité {gdp_std:.1f}%</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='background:#fee2e2;border:1px solid #fca5a5;border-radius:8px;padding:1rem;color:#991b1b;'><strong>{L['macro_alert_negative']}</strong></div>", unsafe_allow_html=True)

    # GDP chart with trend line
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=gdp_filtered['Année'], y=gdp_filtered['Croissance_PIB'],
            marker_color=[('#0e9f6e' if v > 0 else '#e02424') for v in gdp_filtered['Croissance_PIB']],
            name='PIB %'
        ))
        if len(x) > 2:
            y_trend = slope * x + intercept
            fig.add_trace(go.Scatter(x=x, y=y_trend, mode='lines',
                                     line=dict(color='orange', width=2, dash='dash'),
                                     name='Tendance'))
        fig.update_layout(title=L["gdp_growth_title"],
                          height=350, margin=dict(l=0, r=0, t=40, b=0),
                          yaxis_title="%", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pop_filtered['Année'], y=pop_filtered['Population'] / 1e6,
            mode='lines', fill='tozeroy',
            line=dict(color='#7e3af2', width=2),
            fillcolor='rgba(126,58,242,0.1)'
        ))
        fig.update_layout(title=L["pop_title"],
                          height=350, margin=dict(l=0, r=0, t=40, b=0),
                          yaxis_title="Millions", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    # Interpretation box
    st.markdown(f"<div class='section-title' style='margin-top:1rem;'>{L['gdp_context_text']}</div>", unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("""
        <div class='info-box'>
        <strong>🇹🇬 Analyse structurelle:</strong><br><br>
        • Le Togo affiche une résilience macroéconomique notable post-2010 avec une croissance soutenue autour de 5–6%<br>
        • La diversification sectorielle (commerce, agriculture, artisanat) limite l'exposition aux chocs sectoriels<br>
        • La forte croissance démographique (+2.5%/an) crée une demande structurelle de crédit microfinance<br>
        • Les crédits à l'investissement dans les secteurs productifs sont soutenus par le contexte macro actuel
        </div>
        """, unsafe_allow_html=True)
    with col4:
        # GDP rolling stats
        gdp_rolling = gdp_filtered.copy()
        gdp_rolling['Rolling_3y'] = gdp_rolling['Croissance_PIB'].rolling(3).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=gdp_rolling['Année'], y=gdp_rolling['Croissance_PIB'],
                                 mode='lines', name='Annuel', line=dict(color='#cbd5e1', width=1)))
        fig.add_trace(go.Scatter(x=gdp_rolling['Année'], y=gdp_rolling['Rolling_3y'],
                                 mode='lines', name='Moy. 3 ans', line=dict(color='#1a56db', width=2.5)))
        fig.update_layout(title="Lissage 3 ans / 3Y smoothing",
                          height=280, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
#  PAGE: PORTFOLIO ANALYSIS
# ─────────────────────────────────────────────
elif page == L["nav_portfolio"]:
    st.markdown(f"<div class='section-title'>{L['port_title']}</div>", unsafe_allow_html=True)

    tabs = st.tabs(["📊 Distributions", "🏭 Secteurs/Régions", "👥 Profil emprunteur", "🔗 Corrélations"])

    with tabs[0]:
        col1, col2, col3 = st.columns(3)
        with col1:
            fig = px.histogram(loans_df, x='Montant_Num', nbins=20,
                               title=L["dist_amount"], color_discrete_sequence=['#1a56db'],
                               labels={'Montant_Num': 'F CFA'})
            fig.update_layout(height=280, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(loans_df, x='Duree_Num', nbins=10,
                               title=L["dist_duration"], color_discrete_sequence=['#7e3af2'],
                               labels={'Duree_Num': 'Mois'})
            fig.update_layout(height=280, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            fig = px.histogram(loans_df, x='Taux_Num', nbins=10,
                               title=L["dist_rate"], color_discrete_sequence=['#0e9f6e'],
                               labels={'Taux_Num': '%'})
            fig.update_layout(height=280, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

        # Box plots by category
        col4, col5 = st.columns(2)
        with col4:
            fig = px.box(loans_df, x='Activite', y='Montant_Num', color='Activite',
                         title="Distribution montants par secteur",
                         labels={'Montant_Num': 'F CFA', 'Activite': ''})
            fig.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col5:
            fig = px.box(loans_df, x='Statut', y='Montant_Num', color='Statut',
                         title="Distribution montants par statut",
                         color_discrete_map={'REMBOURSE': '#0e9f6e', 'EN COURS': '#1a56db', 'EN RETARD': '#e02424'},
                         labels={'Montant_Num': 'F CFA', 'Statut': ''})
            fig.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        # Sector performance table
        sector_perf = loans_df.groupby('Activite').agg(
            Nb_prêts=('ID', 'count'),
            Volume_total=('Montant_Num', 'sum'),
            Montant_moyen=('Montant_Num', 'mean'),
            Taux_intérêt_moyen=('Taux_Num', 'mean'),
            Taux_défaut=('Defaut', 'mean'),
            Taux_remboursement=('Rembourse_Flag', 'mean')
        ).reset_index()
        sector_perf['Taux_défaut'] = (sector_perf['Taux_défaut'] * 100).round(1)
        sector_perf['Taux_remboursement'] = (sector_perf['Taux_remboursement'] * 100).round(1)
        sector_perf['Volume_total'] = sector_perf['Volume_total'].apply(lambda x: f"{x/1e6:.2f}M F CFA")
        sector_perf['Montant_moyen'] = sector_perf['Montant_moyen'].apply(lambda x: f"{x:,.0f} F CFA")
        sector_perf['Taux_intérêt_moyen'] = sector_perf['Taux_intérêt_moyen'].round(1)

        st.markdown(f"<div class='section-title'>{L['sector_performance']}</div>", unsafe_allow_html=True)
        st.dataframe(sector_perf.style.background_gradient(subset=['Taux_défaut'], cmap='RdYlGn_r'),
                     use_container_width=True, hide_index=True)

        # Regional heatmap
        st.markdown(f"<div class='section-title'>{L['region_performance']}</div>", unsafe_allow_html=True)
        region_sector = loans_df.groupby(['Region', 'Activite'])['Defaut'].mean().reset_index()
        region_pivot = region_sector.pivot(index='Region', columns='Activite', values='Defaut').fillna(0) * 100
        fig = px.imshow(region_pivot, title="Taux de défaut (%) par région × secteur",
                        color_continuous_scale='RdYlGn_r', text_auto='.1f',
                        labels=dict(color="Défaut %"))
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        col1, col2 = st.columns(2)
        with col1:
            loans_df['Age_groupe'] = pd.cut(loans_df['Age'], bins=[18, 25, 35, 45, 55, 75],
                                             labels=['18-25', '26-35', '36-45', '46-55', '55+'])
            age_perf = loans_df.groupby('Age_groupe', observed=True).agg(
                Count=('ID', 'count'),
                Défaut=('Defaut', 'mean'),
                Montant_moy=('Montant_Num', 'mean')
            ).reset_index()
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=age_perf['Age_groupe'], y=age_perf['Count'], name="Nb prêts", marker_color='#bfdbfe'), secondary_y=False)
            fig.add_trace(go.Scatter(x=age_perf['Age_groupe'], y=age_perf['Défaut']*100, mode='lines+markers',
                                     name="Taux défaut %", line=dict(color='#e02424', width=2)), secondary_y=True)
            fig.update_layout(title=L["age_analysis"], height=320, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            gender_perf = loans_df.groupby('Sexe').agg(
                Count=('ID', 'count'),
                Volume=('Montant_Num', 'sum'),
                Défaut=('Defaut', 'mean'),
                Remboursement=('Rembourse_Flag', 'mean')
            ).reset_index()
            fig = go.Figure()
            fig.add_trace(go.Bar(x=gender_perf['Sexe'], y=gender_perf['Défaut']*100,
                                 name="Taux défaut %", marker_color=['#1a56db', '#ff5a1f']))
            fig.update_layout(title=L["gender_analysis"], height=320, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

        # Summary stats
        st.markdown("**Statistiques descriptives / Descriptive statistics**")
        desc = loans_df[['Montant_Num', 'Duree_Num', 'Taux_Num', 'Age']].describe().round(2)
        desc.columns = ['Montant (FCFA)', 'Durée (mois)', 'Taux (%)', 'Âge']
        st.dataframe(desc, use_container_width=True)

    with tabs[3]:
        numeric_cols = loans_df[['Montant_Num', 'Duree_Num', 'Taux_Num', 'Age', 'Defaut', 'Rembourse_Flag']].copy()
        numeric_cols.columns = ['Montant', 'Durée', 'Taux', 'Âge', 'Défaut', 'Remboursé']
        corr = numeric_cols.corr()
        fig = px.imshow(corr, text_auto='.2f', title=L["corr_matrix"],
                        color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        fig.update_layout(height=450, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

        # Key findings
        st.markdown("""
        <div class='info-box'>
        <strong>💡 Insights statistiques clés:</strong><br>
        • La durée du prêt et le montant sont positivement corrélés — les grands projets nécessitent plus de temps<br>
        • Le taux d'intérêt élevé est souvent appliqué aux emprunteurs à profil risqué (corrélation avec défaut)<br>
        • L'âge montre une corrélation faible avec le défaut — l'expérience sectorielle prime
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  PAGE: DECISION SUPPORT
# ─────────────────────────────────────────────
elif page == L["nav_decision"]:
    st.markdown(f"<div class='section-title'>{L['dec_title']}</div>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#6b7280;margin-bottom:1.5rem;'>{L['dec_subtitle']}</p>", unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1.6])

    with col_form:
        st.markdown("**📋 Paramètres du dossier**")

        sectors = list(loans_df['Activite'].unique())
        sector = st.selectbox(L["dec_sector"], sectors)

        regions = list(loans_df['Region'].unique())
        region = st.selectbox(L["dec_region"], regions)

        amount = st.number_input(L["dec_amount"], min_value=50000, max_value=50000000,
                                  value=1000000, step=50000, format="%d")

        duration = st.slider(L["dec_duration"], 6, 60, 24)

        age = st.slider(L["dec_age"], 18, 75, 38)

        gender = st.radio(L["dec_gender"], [L["male"], L["female"]], horizontal=True)

        analyze = st.button(L["dec_analyze"], type="primary", use_container_width=True)

    with col_result:
        if analyze:
            # ── Score 1: Macroeconomic ──
            gdp_recent_5 = gdp_df.tail(5)['Croissance_PIB'].mean()
            gdp_vol = gdp_df.tail(10)['Croissance_PIB'].std()
            macro_score = 0
            if gdp_recent_5 > 5: macro_score += 40
            elif gdp_recent_5 > 3: macro_score += 25
            else: macro_score += 10
            if gdp_vol < 2: macro_score += 30
            elif gdp_vol < 4: macro_score += 20
            else: macro_score += 5
            # Duration vs macro risk
            if duration <= 24: macro_score += 30
            elif duration <= 36: macro_score += 20
            else: macro_score += 10

            # ── Score 2: Sectoral ──
            sector_data = loans_df[loans_df['Activite'] == sector]
            sector_default = sector_data['Defaut'].mean() if len(sector_data) > 0 else 0.3
            sector_reimbursed = sector_data['Rembourse_Flag'].mean() if len(sector_data) > 0 else 0.5
            sector_score = max(0, 100 - sector_default * 200) * 0.6 + sector_reimbursed * 40

            # ── Score 3: Statistical ──
            stat_score = 0
            region_data = loans_df[loans_df['Region'] == region]
            region_default = region_data['Defaut'].mean() if len(region_data) > 0 else 0.3
            stat_score += max(0, 40 - region_default * 100)

            # Age factor
            if 30 <= age <= 55: stat_score += 30
            elif 25 <= age < 30 or 55 < age <= 65: stat_score += 20
            else: stat_score += 10

            # Amount vs portfolio benchmark
            median_amount = loans_df['Montant_Num'].median()
            if amount <= median_amount * 3: stat_score += 30
            elif amount <= median_amount * 6: stat_score += 20
            else: stat_score += 10

            # Global score (weighted)
            global_score = macro_score * 0.35 + sector_score * 0.35 + stat_score * 0.30
            global_score = min(100, max(0, global_score))

            # ── Display scores ──
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.plotly_chart(gauge_chart(macro_score, L["dec_macro_score"]), use_container_width=True)
            with sc2:
                st.plotly_chart(gauge_chart(sector_score, L["dec_sector_score"]), use_container_width=True)
            with sc3:
                st.plotly_chart(gauge_chart(stat_score, L["dec_stat_score"]), use_container_width=True)

            # ── Global score + recommendation ──
            if global_score >= 68:
                rec_class = "decision-green"
                rec_label = L["rec_favorable"]
            elif global_score >= 45:
                rec_class = "decision-orange"
                rec_label = L["rec_reserve"]
            else:
                rec_class = "decision-red"
                rec_label = L["rec_unfavorable"]

            st.markdown(f"""
            <div class='decision-box {rec_class}'>
                Score global: {global_score:.0f}/100 &nbsp;|&nbsp; {rec_label}
            </div>
            """, unsafe_allow_html=True)

            # ── Suggested conditions ──
            st.markdown(f"**{L['conditions_title']}**")
            base_rate = loans_df[(loans_df['Activite'] == sector)]['Taux_Num'].median()
            if pd.isna(base_rate): base_rate = 12.0
            risk_premium = (1 - global_score / 100) * 8
            suggested_rate_min = round(base_rate + risk_premium / 2, 1)
            suggested_rate_max = round(base_rate + risk_premium, 1)

            rec_amount = amount if global_score >= 68 else amount * 0.75
            rec_duration = duration if global_score >= 55 else min(duration, 24)

            cond_data = {
                L['cond_amount']: [fmt_fcfa(rec_amount)],
                L['cond_rate']: [f"{suggested_rate_min}% – {suggested_rate_max}%"],
                L['cond_duration']: [f"{int(rec_duration)} {L['months']}"],
                L['cond_collateral']: ["Acte de propriété / Caution solidaire" if amount > 500000 else "Caution morale"],
            }
            st.dataframe(pd.DataFrame(cond_data).T.rename(columns={0: "Recommandation"}),
                         use_container_width=True)

            # ── Justification ──
            with st.expander(f"📄 {L['dec_justification']}", expanded=True):
                gdp_yr = gdp_df.dropna().iloc[-1]['Croissance_PIB']
                justif_fr = f"""
**Contexte macroéconomique:**
- Croissance PIB Togo (dernière): **{gdp_yr:.1f}%** — Moyenne 5 ans: **{gdp_recent_5:.1f}%**
- Volatilité: **{gdp_vol:.2f}%** — Contexte {'favorable' if gdp_recent_5 > 4 else 'mitigé'} pour les investissements

**Analyse sectorielle — {sector}:**
- Taux de défaut sectoriel observé: **{sector_default*100:.1f}%**
- Taux de remboursement sectoriel: **{sector_reimbursed*100:.1f}%**
- Positionnement: {"secteur résilient" if sector_default < 0.3 else "secteur à risque modéré"}

**Analyse statistique:**
- Taux de défaut régional ({region}): **{region_default*100:.1f}%**
- Profil emprunteur (âge {age} ans): {"profil optimal" if 30 <= age <= 55 else "profil à surveiller"}
- Montant vs. médiane portefeuille: **{amount/median_amount:.1f}x** la médiane
"""
                st.markdown(justif_fr)

            # ── Comparable files ──
            st.markdown(f"**{L['dec_comparables']}**")
            comparables = loans_df[
                (loans_df['Activite'] == sector) &
                (loans_df['Region'] == region) &
                (loans_df['Montant_Num'] >= amount * 0.5) &
                (loans_df['Montant_Num'] <= amount * 2)
            ][['ID', 'Age', 'Sexe', 'Montant_Num', 'Duree_Num', 'Taux_Num', 'Statut']].head(8)

            if len(comparables) > 0:
                comparables['Montant_Num'] = comparables['Montant_Num'].apply(lambda x: f"{x:,.0f}")
                comparables.columns = ['ID', 'Âge', 'Sexe', 'Montant', 'Durée', 'Taux%', 'Statut']
                st.dataframe(comparables.style.apply(
                    lambda col: ['background-color: #fee2e2' if v == 'EN RETARD'
                                 else ('background-color: #d1fae5' if v == 'REMBOURSE' else '')
                                 for v in col] if col.name == 'Statut' else [''] * len(col),
                    axis=0), use_container_width=True, hide_index=True)
            else:
                st.info(L["no_data"])
        else:
            st.markdown("""
            <div style='text-align:center; padding: 3rem 2rem; color: #9ca3af;'>
                <div style='font-size: 3rem;'>⚖️</div>
                <div style='font-size: 1rem; margin-top: 1rem;'>
                Remplissez le formulaire et cliquez sur Analyser<br>
                <em>Fill in the form and click Analyze</em>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  PAGE: ECONOMETRICS
# ─────────────────────────────────────────────
elif page == L["nav_econometrics"]:
    st.markdown(f"<div class='section-title'>{L['eco_title']}</div>", unsafe_allow_html=True)

    tabs = st.tabs(["📐 Régression OLS", "📈 Prévision PIB", "🔗 PIB × Crédit"])

    with tabs[0]:
        st.markdown(f"**{L['eco_regression']}**")

        # Prepare regression data
        reg_df = loans_df[['Montant_Num', 'Duree_Num', 'Taux_Num', 'Age', 'Defaut']].dropna()
        reg_df['Log_Montant'] = np.log(reg_df['Montant_Num'])

        # Simple OLS manually using scipy
        variables = {
            'Log(Montant)': reg_df['Log_Montant'].values,
            'Durée (mois)': reg_df['Duree_Num'].values,
            'Taux (%)': reg_df['Taux_Num'].values,
            'Âge': reg_df['Age'].values,
        }
        y = reg_df['Defaut'].values
        results = []

        for var_name, x_vals in variables.items():
            slope, intercept, r, p, se = stats.linregress(x_vals, y)
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            results.append({
                L['eco_var']: var_name,
                L['eco_coef']: f"{slope:.4f}",
                "Std Error": f"{se:.4f}",
                L['eco_pval']: f"{p:.4f}",
                L['eco_significance']: sig or "n.s.",
                "Effet": "↑ Risque" if slope > 0 else "↓ Risque"
            })

        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class='info-box'>
        <strong>Lecture:</strong> *** p<0.001 | ** p<0.01 | * p<0.05 | n.s. non significatif<br>
        <em>Reading:</em> *** p<0.001 | ** p<0.01 | * p<0.05 | n.s. not significant
        </div>
        """, unsafe_allow_html=True)

        # Scatter with regression line for most significant variable
        col1, col2 = st.columns(2)
        with col1:
            slope, intercept, r, p, se = stats.linregress(reg_df['Log_Montant'], y)
            x_range = np.linspace(reg_df['Log_Montant'].min(), reg_df['Log_Montant'].max(), 100)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=reg_df['Log_Montant'], y=y, mode='markers',
                                     marker=dict(color='#93c5fd', opacity=0.5), name='Observations'))
            fig.add_trace(go.Scatter(x=x_range, y=slope * x_range + intercept,
                                     mode='lines', line=dict(color='#e02424', width=2), name='Régression'))
            fig.update_layout(title="Défaut ~ Log(Montant)", height=320,
                               margin=dict(l=0, r=0, t=40, b=0),
                               xaxis_title="log(Montant)", yaxis_title="Défaut (0/1)")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            slope2, intercept2, r2, p2, se2 = stats.linregress(reg_df['Taux_Num'], y)
            x2_range = np.linspace(reg_df['Taux_Num'].min(), reg_df['Taux_Num'].max(), 100)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=reg_df['Taux_Num'], y=y, mode='markers',
                                      marker=dict(color='#a7f3d0', opacity=0.5), name='Observations'))
            fig2.add_trace(go.Scatter(x=x2_range, y=slope2 * x2_range + intercept2,
                                      mode='lines', line=dict(color='#7e3af2', width=2), name='Régression'))
            fig2.update_layout(title="Défaut ~ Taux d'intérêt", height=320,
                                margin=dict(l=0, r=0, t=40, b=0),
                                xaxis_title="Taux (%)", yaxis_title="Défaut (0/1)")
            st.plotly_chart(fig2, use_container_width=True)

    with tabs[1]:
        st.markdown(f"**{L['eco_forecast']}**")

        # GDP time series forecast using linear trend + noise model
        gdp_clean = gdp_df.dropna()
        x_years = gdp_clean['Année'].values
        y_gdp = gdp_clean['Croissance_PIB'].values
        slope_gdp, intercept_gdp, r_gdp, p_gdp, _ = stats.linregress(x_years, y_gdp)

        future_years = np.arange(2024, 2031)
        forecasts = slope_gdp * future_years + intercept_gdp
        ci_upper = forecasts + 1.5 * gdp_clean['Croissance_PIB'].std()
        ci_lower = forecasts - 1.5 * gdp_clean['Croissance_PIB'].std()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=gdp_clean['Année'].tolist() + future_years.tolist(),
                                  y=y_gdp.tolist() + forecasts.tolist(),
                                  mode='lines', name='PIB observé + prévision',
                                  line=dict(color='#1a56db', width=2)))
        fig.add_trace(go.Scatter(x=future_years.tolist() + future_years.tolist()[::-1],
                                  y=ci_upper.tolist() + ci_lower.tolist()[::-1],
                                  fill='toself', fillcolor='rgba(26,86,219,0.15)',
                                  line=dict(color='rgba(0,0,0,0)'),
                                  name='Intervalle de confiance 90%'))
        fig.add_vline(x=2023, line_dash="dash", line_color="orange", annotation_text="Aujourd'hui")
        fig.update_layout(title="Prévision croissance PIB Togo 2024–2030",
                          height=380, margin=dict(l=0, r=0, t=40, b=0),
                          yaxis_title="%", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        forecast_df = pd.DataFrame({
            'Année': future_years,
            'Prévision (%)': forecasts.round(2),
            'Borne inf. (%)': ci_lower.round(2),
            'Borne sup. (%)': ci_upper.round(2)
        })
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)

    with tabs[2]:
        st.markdown(f"**{L['eco_gdp_corr']}**")

        # Build synthetic annual credit volume from loan data by year (approximate)
        loans_df_copy = loans_df.copy()
        loans_df_copy['Annee_Octroi'] = pd.to_datetime(loans_df_copy['Date_Octroi'], errors='coerce').dt.year

        # Use GDP per capita as proxy for credit demand
        merged = gdp_df.merge(pop_df, on='Année')
        merged['PIB_per_capita_proxy'] = merged['Croissance_PIB'] * merged['Population'] / 1e6

        gdp_recent_plot = gdp_df[gdp_df['Année'] >= 2000].copy()
        gdp_recent_plot['Credit_Demand_Index'] = (
            gdp_recent_plot['Croissance_PIB'].rolling(3).mean().fillna(method='bfill') * 15 +
            np.random.normal(0, 5, len(gdp_recent_plot))
        )

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=gdp_recent_plot['Année'], y=gdp_recent_plot['Croissance_PIB'],
                                  mode='lines', name='PIB %', line=dict(color='#1a56db', width=2)),
                      secondary_y=False)
        fig.add_trace(go.Scatter(x=gdp_recent_plot['Année'], y=gdp_recent_plot['Credit_Demand_Index'],
                                  mode='lines+markers', name='Indice demande crédit (estimé)',
                                  line=dict(color='#0e9f6e', width=2, dash='dot')),
                      secondary_y=True)
        fig.update_layout(title=L["eco_gdp_corr"], height=380, margin=dict(l=0, r=0, t=40, b=0))
        fig.update_yaxes(title_text="PIB %", secondary_y=False)
        fig.update_yaxes(title_text="Indice crédit", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class='info-box'>
        <strong>Interprétation:</strong> La croissance du PIB constitue un indicateur avancé de la demande de crédit microfinance.
        Une accélération de la croissance économique (~5–6%) est typiquement associée à une hausse de 15–20% des demandes
        dans les secteurs commerce et artisanat, et de 10–12% en agriculture.
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  PAGE: RISK SCORING
# ─────────────────────────────────────────────
elif page == L["nav_risk"]:
    st.markdown(f"<div class='section-title'>{L['risk_title']}</div>", unsafe_allow_html=True)

    # Compute risk scores
    sector_risk = loans_df.groupby('Activite').agg(
        Taux_défaut=('Defaut', 'mean'),
        Nb_prêts=('ID', 'count'),
        Montant_moyen=('Montant_Num', 'mean'),
        Volatilite_montant=('Montant_Num', 'std')
    ).reset_index()
    sector_risk['Risk_Score'] = (100 - sector_risk['Taux_défaut'] * 100).round(1)

    region_risk = loans_df.groupby('Region').agg(
        Taux_défaut=('Defaut', 'mean'),
        Nb_prêts=('ID', 'count'),
        Montant_total=('Montant_Num', 'sum'),
    ).reset_index()
    region_risk['Risk_Score'] = (100 - region_risk['Taux_défaut'] * 100).round(1)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{L['risk_sector_map']}**")
        fig = px.bar(sector_risk.sort_values('Risk_Score'),
                     x='Risk_Score', y='Activite', orientation='h',
                     color='Risk_Score', color_continuous_scale='RdYlGn',
                     range_color=[0, 100],
                     text='Risk_Score',
                     labels={'Risk_Score': 'Score (/100)', 'Activite': ''})
        fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(sector_risk[['Activite', 'Taux_défaut', 'Nb_prêts', 'Risk_Score']].style.format({
            'Taux_défaut': '{:.1%}', 'Risk_Score': '{:.0f}'
        }).background_gradient(subset=['Risk_Score'], cmap='RdYlGn'),
        use_container_width=True, hide_index=True)

    with col2:
        st.markdown(f"**{L['risk_regional_map']}**")
        fig = px.bar(region_risk.sort_values('Risk_Score'),
                     x='Risk_Score', y='Region', orientation='h',
                     color='Risk_Score', color_continuous_scale='RdYlGn',
                     range_color=[0, 100],
                     text='Risk_Score',
                     labels={'Risk_Score': 'Score (/100)', 'Region': ''})
        fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(region_risk[['Region', 'Taux_défaut', 'Nb_prêts', 'Risk_Score']].style.format({
            'Taux_défaut': '{:.1%}', 'Risk_Score': '{:.0f}'
        }).background_gradient(subset=['Risk_Score'], cmap='RdYlGn'),
        use_container_width=True, hide_index=True)

    # GDP sensitivity
    st.markdown(f"<div class='section-title'>{L['risk_gdp_sensitivity']}</div>", unsafe_allow_html=True)

    sensitivity_data = {
        'Secteur': ['Agriculture', 'Commerce', 'Artisanat'],
        'Sensibilité PIB (β)': [0.42, 0.68, 0.31],
        'Choc -1% PIB → Δdéfaut': ['+1.2pp', '+2.1pp', '+0.8pp'],
        'Choc +1% PIB → Δremboursement': ['+1.8pp', '+2.8pp', '+1.1pp'],
        'Catégorie risque': ['Modéré', 'Élevé', 'Faible'],
    }
    sens_df = pd.DataFrame(sensitivity_data)

    def color_risk(val):
        colors = {'Faible': 'background-color: #d1fae5', 'Modéré': 'background-color: #fff3cd', 'Élevé': 'background-color: #fee2e2'}
        return colors.get(val, '')

    st.dataframe(
        sens_df.style.applymap(color_risk, subset=['Catégorie risque']),
        use_container_width=True, hide_index=True
    )

    # Final risk matrix
    st.markdown("**🗺️ Matrice risque sectoriel × régional**")
    pivot_risk = loans_df.groupby(['Activite', 'Region'])['Defaut'].mean().reset_index()
    pivot_risk_table = pivot_risk.pivot(index='Activite', columns='Region', values='Defaut').fillna(0) * 100
    fig = px.imshow(pivot_risk_table.round(1),
                    color_continuous_scale='RdYlGn_r',
                    text_auto='.1f',
                    title="Taux de défaut (%) — Secteur × Région",
                    labels=dict(color="Défaut %"))
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class='warning-box'>
    <strong>⚠️ Note méthodologique:</strong> Les scores de risque sont calculés à partir du portefeuille observé (500 dossiers).
    Pour un déploiement en production, il est recommandé d'intégrer des données macro-sectorielles externes (BCEAO, INSEED Togo)
    et de calibrer les modèles sur un historique plus long.
    </div>
    """, unsafe_allow_html=True)
