"""CreditMacro — Risk Engine for Microfinance"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="CreditMacro · Risk Engine", page_icon="🏦", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"] { background: #0f172a !important; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
.kpi-card { background:white; border-radius:14px; padding:1.1rem 1.4rem; border-left:5px solid #3b82f6; box-shadow:0 1px 4px rgba(0,0,0,.07); margin-bottom:.4rem; }
.kpi-card.green  { border-left-color:#10b981; }
.kpi-card.red    { border-left-color:#ef4444; }
.kpi-card.orange { border-left-color:#f59e0b; }
.kpi-card.purple { border-left-color:#8b5cf6; }
.kpi-card.teal   { border-left-color:#06b6d4; }
.kpi-label { font-size:.7rem; font-weight:600; text-transform:uppercase; letter-spacing:.07em; color:#94a3b8; }
.kpi-val   { font-size:1.75rem; font-weight:800; color:#0f172a; line-height:1.1; }
.kpi-sub   { font-size:.75rem; color:#64748b; margin-top:2px; }
.sec-title { font-size:1.2rem; font-weight:700; color:#0f172a; border-bottom:2px solid #e2e8f0; padding-bottom:.4rem; margin-bottom:1rem; margin-top:.5rem; }
.upload-hero { background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 100%); border-radius:16px; padding:2.5rem; color:white; text-align:center; margin-bottom:1.5rem; }
.upload-hero h2 { font-size:1.6rem; font-weight:800; margin:0 0 .5rem; }
.upload-hero p { font-size:.9rem; color:#94a3b8; margin:0; }
.info-box    { background:#eff6ff; border:1px solid #bfdbfe; border-radius:8px; padding:.9rem 1.1rem; font-size:.85rem; color:#1e40af; margin:.5rem 0; }
.warn-box    { background:#fffbeb; border:1px solid #fde68a; border-radius:8px; padding:.9rem 1.1rem; font-size:.85rem; color:#92400e; margin:.5rem 0; }
.success-box { background:#f0fdf4; border:1px solid #bbf7d0; border-radius:8px; padding:.9rem 1.1rem; font-size:.85rem; color:#166534; margin:.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Translations ──
T = {
    "fr": dict(
        title="🏦 CreditMacro · Risk Engine", subtitle="Analyse de risque crédit microfinance",
        nav_upload="Chargement", nav_pd="Probabilité de Défaut", nav_ols="Régression OLS",
        nav_corr="Corrélations", nav_logit="Régression Logistique", nav_export="Export",
        upload_title="Chargez votre portefeuille de prêts", upload_desc="CSV ou Excel · Toute institution de microfinance",
        col_map="Mapping des colonnes", col_map_desc="Associez vos colonnes aux champs analytiques",
        col_default="Colonne statut / défaut", col_amount="Montant du prêt", col_duration="Durée (mois)",
        col_rate="Taux d'intérêt (%)", col_age="Âge emprunteur", col_sector="Secteur d'activité",
        col_region="Région", col_gender="Genre", val_default="Valeurs = défaut (ex: EN RETARD, 1)",
        apply_map="✅ Valider le mapping", no_data="⬅️ Chargez d'abord vos données dans Chargement",
        not_enough="Données insuffisantes (min. 30 lignes valides)",
        pd_title="Probabilité de Défaut — Analyse empirique", pd_global="PD globale",
        pd_by_sector="PD par secteur", pd_by_region="PD par région", pd_by_amount="PD par tranche de montant",
        pd_by_age="PD par tranche d'âge", pd_by_gender="PD par genre", pd_by_duration="PD par durée",
        ols_title="Régression OLS — Facteurs explicatifs du défaut",
        ols_dep="Variable dépendante", ols_indep="Variables indépendantes",
        ols_r2="R² ajusté", ols_fstat="F-statistique", ols_nobs="Observations",
        corr_title="Matrice de corrélations", corr_desc="Corrélations de Pearson entre variables numériques",
        corr_top="Corrélations les plus fortes avec le défaut", corr_scatter="Nuage de points",
        logit_title="Régression Logistique — Modèle de scoring PD",
        logit_roc="Courbe ROC", logit_auc="AUC", logit_cm="Matrice de confusion",
        logit_calib="Courbe de calibration", logit_score="Distribution des scores PD",
        logit_thresh="Seuil de décision", logit_report="Rapport de classification", logit_gini="Gini",
        export_title="Export & Rapport de synthèse",
        export_dl_csv="⬇️ Télécharger portefeuille scoré (CSV)", export_dl_sum="⬇️ Rapport analytique (TXT)",
    ),
    "en": dict(
        title="🏦 CreditMacro · Risk Engine", subtitle="Credit risk analysis for microfinance",
        nav_upload="Data Upload", nav_pd="Probability of Default", nav_ols="OLS Regression",
        nav_corr="Correlations", nav_logit="Logistic Regression", nav_export="Export",
        upload_title="Upload your loan portfolio", upload_desc="CSV or Excel · Any microfinance institution",
        col_map="Column Mapping", col_map_desc="Map your columns to the required analytical fields",
        col_default="Default / status column", col_amount="Loan amount", col_duration="Duration (months)",
        col_rate="Interest rate (%)", col_age="Borrower age", col_sector="Activity sector",
        col_region="Region", col_gender="Gender", val_default="Default values (e.g. LATE, 1)",
        apply_map="✅ Confirm mapping", no_data="⬅️ Please load your data in the Upload tab first",
        not_enough="Not enough data (min. 30 valid rows)",
        pd_title="Probability of Default — Empirical Analysis", pd_global="Portfolio PD",
        pd_by_sector="PD by sector", pd_by_region="PD by region", pd_by_amount="PD by loan size",
        pd_by_age="PD by age group", pd_by_gender="PD by gender", pd_by_duration="PD by duration",
        ols_title="OLS Regression — Default determinants",
        ols_dep="Dependent variable", ols_indep="Independent variables",
        ols_r2="Adjusted R²", ols_fstat="F-statistic", ols_nobs="Observations",
        corr_title="Correlation Matrix", corr_desc="Pearson correlations between numeric variables",
        corr_top="Strongest correlations with default", corr_scatter="Scatter plot",
        logit_title="Logistic Regression — PD Scoring Model",
        logit_roc="ROC Curve", logit_auc="AUC", logit_cm="Confusion Matrix",
        logit_calib="Calibration Curve", logit_score="Score distribution (PD)",
        logit_thresh="Decision threshold", logit_report="Classification report", logit_gini="Gini",
        export_title="Export & Summary Report",
        export_dl_csv="⬇️ Download scored portfolio (CSV)", export_dl_sum="⬇️ Analytical report (TXT)",
    )
}

# ── Session state ──
for _k in ["df","df_raw","mapping","default_col","default_values",
           "logit_model","scaler","feature_cols","X_test","y_test","y_prob"]:
    if _k not in st.session_state:
        st.session_state[_k] = None

# ── Helpers ──
def smart_num(series):
    return (series.astype(str)
            .str.replace(r'[^\d.\-]', '', regex=True)
            .replace('', np.nan).astype(float))

def guess(cols, kws):
    for kw in kws:
        for c in cols:
            if kw.lower() in c.lower():
                return c
    return None

def sig_stars(p):
    if p < .001: return "***"
    if p < .01:  return "**"
    if p < .05:  return "*"
    if p < .10:  return "."
    return "n.s."

def kpi(col, cls, lbl, val, sub=""):
    with col:
        st.markdown(
            f"<div class='kpi-card {cls}'>"
            f"<div class='kpi-label'>{lbl}</div>"
            f"<div class='kpi-val'>{val}</div>"
            f"<div class='kpi-sub'>{sub}</div></div>",
            unsafe_allow_html=True)

def build_df(df_raw, mapping, dcol, dvals):
    rename = {v: k for k, v in mapping.items() if v and v != "—" and v in df_raw.columns}
    df = df_raw.rename(columns=rename).copy()
    for nc in ["amount", "duration", "rate", "age"]:
        if nc in df.columns:
            df[nc] = smart_num(df[nc])
    if dcol and dcol in df_raw.columns:
        raw = df_raw[dcol].astype(str).str.strip()
        if dvals:
            dvlist = [v.strip() for v in dvals.split(",")]
            df["default"] = raw.isin(dvlist).astype(int)
        else:
            df["default"] = pd.to_numeric(raw, errors="coerce").fillna(0).astype(int)
    else:
        df["default"] = np.nan
    return df

# ── Sidebar ──
with st.sidebar:
    st.markdown(
        "<div style='padding:.8rem .5rem .4rem;font-size:1.05rem;"
        "font-weight:800;color:white;'>🏦 CreditMacro</div>",
        unsafe_allow_html=True)
    lang = st.selectbox("🌐", ["Français", "English"], label_visibility="collapsed")
    L = T["fr"] if lang == "Français" else T["en"]
    st.markdown("<hr style='border-color:#1e293b;margin:.4rem 0;'>", unsafe_allow_html=True)
    page = st.radio(
        "Nav",
        [L["nav_upload"], L["nav_pd"], L["nav_ols"],
         L["nav_corr"], L["nav_logit"], L["nav_export"]],
        label_visibility="collapsed")
    st.markdown("<hr style='border-color:#1e293b;margin:.4rem 0;'>", unsafe_allow_html=True)
    if st.session_state.df is not None:
        _d = st.session_state.df
        _nd = int(_d["default"].sum()) if "default" in _d.columns else "?"
        st.markdown(
            f"<div style='background:#0f2744;border-radius:10px;padding:.7rem 1rem;'>"
            f"<div style='font-size:.65rem;color:#64748b;text-transform:uppercase;'>Portefeuille</div>"
            f"<div style='font-size:1.2rem;font-weight:800;color:white;'>{len(_d):,} prêts</div>"
            f"<div style='font-size:.75rem;color:#94a3b8;'>{_nd} défauts</div></div>",
            unsafe_allow_html=True)
    else:
        st.markdown("<div style='font-size:.7rem;color:#475569;'>Aucun fichier chargé</div>",
                    unsafe_allow_html=True)

# ════════════════════════════════════════════
# GUARD FUNCTION
# ════════════════════════════════════════════
def require():
    if st.session_state.df is None:
        st.markdown(f"<div class='warn-box'>{L['no_data']}</div>", unsafe_allow_html=True)
        st.stop()
    _df = st.session_state.df
    if len(_df) < 10:
        st.markdown(f"<div class='warn-box'>{L['not_enough']}</div>", unsafe_allow_html=True)
        st.stop()
    return _df

# ════════════════════════════════════════════
# PAGE 1 — UPLOAD
# ════════════════════════════════════════════
def page_upload():
    st.markdown(
        f"<div class='upload-hero'><h2>{L['upload_title']}</h2>"
        f"<p>{L['upload_desc']}</p></div>",
        unsafe_allow_html=True)
    uploaded = st.file_uploader("Fichier CSV ou Excel", type=["csv", "xlsx", "xls"])
    if uploaded:
        try:
            if uploaded.name.endswith((".xlsx", ".xls")):
                df_raw = pd.read_excel(uploaded)
            else:
                raw = uploaded.read(); uploaded.seek(0)
                sample = raw[:2000].decode("utf-8", errors="replace")
                sep = ";" if sample.count(";") > sample.count(",") else ","
                df_raw = pd.read_csv(uploaded, sep=sep, encoding="utf-8", on_bad_lines="skip")
        except Exception as e:
            st.error(f"Erreur lecture : {e}"); return
        st.session_state.df_raw = df_raw
        cols = list(df_raw.columns)
        st.markdown(
            f"<div class='success-box'>✅ {len(df_raw):,} lignes · "
            f"{len(cols)} colonnes · {uploaded.name}</div>",
            unsafe_allow_html=True)
        with st.expander("👁️ Aperçu des données brutes"):
            st.dataframe(df_raw.head(10), use_container_width=True)
        st.markdown(f"<div class='sec-title'>{L['col_map']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='info-box'>{L['col_map_desc']}</div>", unsafe_allow_html=True)
        NONE = "—"; opts = [NONE] + cols
        gd   = guess(cols, ["statut","status","defaut","default","retard","late","impaye"])
        ga   = guess(cols, ["montant","amount","prete","loan","credit"])
        gdu  = guess(cols, ["duree","duration","mois","month","terme"])
        gr   = guess(cols, ["taux","rate","interet","interest"])
        gage = guess(cols, ["age"])
        gs   = guess(cols, ["activite","sector","secteur","activity"])
        greg = guess(cols, ["region","zone","ville","city","localite"])
        gg   = guess(cols, ["sexe","genre","gender","sex"])
        c1, c2 = st.columns(2)
        with c1:
            md  = st.selectbox(L["col_default"],  opts, index=opts.index(gd)   if gd   in opts else 0)
            ma  = st.selectbox(L["col_amount"],   opts, index=opts.index(ga)   if ga   in opts else 0)
            mdu = st.selectbox(L["col_duration"], opts, index=opts.index(gdu)  if gdu  in opts else 0)
            mra = st.selectbox(L["col_rate"],     opts, index=opts.index(gr)   if gr   in opts else 0)
        with c2:
            mage = st.selectbox(L["col_age"],    opts, index=opts.index(gage) if gage in opts else 0)
            mse  = st.selectbox(L["col_sector"], opts, index=opts.index(gs)   if gs   in opts else 0)
            mreg = st.selectbox(L["col_region"], opts, index=opts.index(greg) if greg in opts else 0)
            mge  = st.selectbox(L["col_gender"], opts, index=opts.index(gg)   if gg   in opts else 0)
        if md != NONE:
            sv = df_raw[md].astype(str).str.strip().value_counts().head(8).index.tolist()
            st.markdown(f"**{L['val_default']}**")
            st.caption(f"Valeurs observées dans la colonne : `{sv}`")
            hint = ", ".join([v for v in sv if any(k in v.upper()
                for k in ["RETARD","LATE","DEFAULT","IMPAY","BAD"]) or v == "1"])
            dvi = st.text_input("Valeurs de défaut (séparées par virgule)", value=hint)
        else:
            dvi = ""
        if st.button(L["apply_map"], type="primary"):
            mapping = {
                "amount": ma if ma != NONE else None,
                "duration": mdu if mdu != NONE else None,
                "rate": mra if mra != NONE else None,
                "age": mage if mage != NONE else None,
                "sector": mse if mse != NONE else None,
                "region": mreg if mreg != NONE else None,
                "gender": mge if mge != NONE else None,
            }
            dfc = build_df(df_raw, mapping, md, dvi)
            st.session_state.df = dfc
            st.session_state.mapping = mapping
            st.session_state.default_col = md
            st.session_state.default_values = dvi
            st.session_state.logit_model = None
            nd2 = int(dfc["default"].sum()); pdg = dfc["default"].mean() * 100
            st.markdown(
                f"<div class='success-box'>✅ {len(dfc):,} prêts · "
                f"{nd2} défauts · PD globale = {pdg:.1f}%</div>",
                unsafe_allow_html=True)
            st.dataframe(dfc.head(8), use_container_width=True)
    else:
        st.markdown(
            "<div class='warn-box'>💡 Aucun fichier chargé — "
            "utilisez le bouton ci-dessous pour la démo.</div>",
            unsafe_allow_html=True)
        if st.button("Charger données de démonstration (Togo)"):
            try:
                dfd = pd.read_csv("Jeux_donnees.csv")
                mapping = {
                    "amount": " Montant_Prete_FCFA ", "duration": "Duree_Mois",
                    "rate": " Taux_Interet ", "age": "Age",
                    "sector": "Activite", "region": "Region", "gender": "Sexe"}
                dfc = build_df(dfd, mapping, "Statut", "EN RETARD")
                st.session_state.df = dfc; st.session_state.df_raw = dfd
                st.session_state.mapping = mapping
                st.session_state.default_col = "Statut"
                st.session_state.default_values = "EN RETARD"
                st.session_state.logit_model = None
                st.success(f"✅ {len(dfc)} prêts chargés depuis le fichier démo")
                st.rerun()
            except FileNotFoundError:
                st.error("Fichier Jeux_donnees.csv introuvable dans le répertoire courant.")

# ════════════════════════════════════════════
# PAGE 2 — PD
# ════════════════════════════════════════════
def page_pd():
    df = require()
    st.markdown(f"<div class='sec-title'>{L['pd_title']}</div>", unsafe_allow_html=True)
    n = len(df); nd = int(df["default"].sum()); pdp = df["default"].mean() * 100
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "",        "Total prêts / Total loans", f"{n:,}")
    kpi(c2, "red",     "Défauts / Defaults",         f"{nd:,}",       f"PD = {pdp:.2f}%")
    kpi(c3, "green",   "Sains / Performing",          f"{n - nd:,}",   f"{100 - pdp:.1f}%")
    kpi(c4, "orange",  L["pd_global"],                f"{pdp:.2f}%",   "Taux observé")
    st.markdown("<br>", unsafe_allow_html=True)

    def pd_bar(cat, title):
        if cat not in df.columns: return
        g = df.groupby(cat)["default"].agg(["mean", "sum", "count"]).reset_index()
        g.columns = [cat, "PD", "Def", "N"]
        g["PD_pct"] = (g["PD"] * 100).round(2)
        g = g.sort_values("PD_pct", ascending=True)
        fig = px.bar(g, x="PD_pct", y=cat, orientation="h",
                     color="PD_pct", color_continuous_scale="RdYlGn_r",
                     range_color=[0, max(g["PD_pct"].max(), 1)],
                     text="PD_pct", title=title,
                     labels={"PD_pct": "PD (%)", cat: ""},
                     custom_data=["Def", "N"])
        fig.update_traces(
            texttemplate="%{text:.1f}%", textposition="outside",
            hovertemplate="<b>%{y}</b><br>PD: %{x:.1f}%<br>Défauts: %{customdata[0]}<br>N: %{customdata[1]}<extra></extra>")
        fig.update_layout(height=max(230, len(g) * 45),
                          margin=dict(l=0, r=40, t=40, b=0), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1: pd_bar("sector", L["pd_by_sector"])
    with col2: pd_bar("region", L["pd_by_region"])

    if "amount" in df.columns and df["amount"].notna().sum() > 10:
        df["_amt_q"] = pd.qcut(df["amount"].dropna(), q=5, duplicates="drop",
                                labels=["Q1 Très petit", "Q2 Petit", "Q3 Moyen", "Q4 Grand", "Q5 Très grand"])
        col3, col4 = st.columns(2)
        with col3: pd_bar("_amt_q", L["pd_by_amount"])
        with col4:
            if "duration" in df.columns and df["duration"].notna().sum() > 10:
                df["_dur_q"] = pd.cut(df["duration"], bins=[0, 6, 12, 24, 36, 120],
                                       labels=["≤6m", "7-12m", "13-24m", "25-36m", "36m+"])
                pd_bar("_dur_q", L["pd_by_duration"])

    col5, col6 = st.columns(2)
    with col5:
        if "age" in df.columns and df["age"].notna().sum() > 10:
            df["_age_g"] = pd.cut(df["age"], bins=[0, 25, 35, 45, 55, 100],
                                   labels=["≤25", "26-35", "36-45", "46-55", "55+"])
            pd_bar("_age_g", L["pd_by_age"])
    with col6:
        pd_bar("gender", L["pd_by_gender"])

    st.markdown("<div class='sec-title'>Tableau récapitulatif</div>", unsafe_allow_html=True)
    cat_avail = [c for c in ["sector", "region", "gender"] if c in df.columns]
    if cat_avail:
        dim = st.selectbox("Dimension", cat_avail)
        tbl = df.groupby(dim)["default"].agg(
            N="count", Défauts="sum",
            PD_pct=lambda x: round(x.mean() * 100, 2)).reset_index()
        tbl["PD_pct"] = tbl["PD_pct"].apply(lambda x: f"{x:.2f}%")
        st.dataframe(tbl, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════
# PAGE 3 — OLS
# ════════════════════════════════════════════
def page_ols():
    df = require()
    st.markdown(f"<div class='sec-title'>{L['ols_title']}</div>", unsafe_allow_html=True)
    num_cols = [c for c in ["amount", "duration", "rate", "age"]
                if c in df.columns and df[c].notna().sum() > 10]
    if not num_cols:
        st.warning("Aucune variable numérique disponible."); return
    cc1, cc2 = st.columns([1, 2])
    with cc1:
        dep = st.selectbox(L["ols_dep"], ["default"] + num_cols)
    with cc2:
        indep = st.multiselect(L["ols_indep"],
                               [c for c in num_cols if c != dep],
                               default=[c for c in num_cols if c != dep][:4])
    if not indep:
        st.info("Sélectionnez au moins une variable indépendante."); return
    rdf = df[[dep] + indep].dropna()
    if len(rdf) < 20:
        st.warning(L["not_enough"]); return
    Y = rdf[dep].values

    # Univariate table
    rows = []
    for v in indep:
        X = rdf[v].values
        sl, ic, r, p, se = stats.linregress(X, Y)
        t = sl / se if se > 0 else 0
        ss_r = np.sum((Y - (sl * X + ic)) ** 2)
        ss_t = np.sum((Y - Y.mean()) ** 2)
        r2 = 1 - ss_r / ss_t if ss_t > 0 else 0
        rows.append({"Variable": v, "β": round(sl, 6), "Intercept": round(ic, 4),
                     "Std.Err": round(se, 6), "t-stat": round(t, 3),
                     "p-value": round(p, 4), "R²": round(r2, 4), "Sig.": sig_stars(p)})
    res_df = pd.DataFrame(rows)
    st.markdown(f"**OLS Bivariée — Y = `{dep}`**")
    st.dataframe(
        res_df.style
        .map(lambda v: "color:#166534;font-weight:700" if v in ["***", "**", "*"]
                else ("color:#9ca3af" if v == "n.s." else ""), subset=["Sig."])
        .background_gradient(subset=["R²"], cmap="Blues"),
        use_container_width=True, hide_index=True)
    st.caption("*** p<.001 | ** p<.01 | * p<.05 | . p<.10 | n.s. non significatif")

    # Multivariate OLS
    st.markdown("<div class='sec-title'>OLS Multivarié</div>", unsafe_allow_html=True)
    Xm = np.column_stack([np.ones(len(rdf))] + [rdf[v].values for v in indep])
    try:
        beta = np.linalg.lstsq(Xm, Y, rcond=None)[0]
        Yh = Xm @ beta; res = Y - Yh
        n, k = len(Y), len(beta)
        ss_r = np.sum(res ** 2); ss_t = np.sum((Y - Y.mean()) ** 2)
        r2m = 1 - ss_r / ss_t if ss_t > 0 else 0
        r2a = 1 - (1 - r2m) * (n - 1) / (n - k - 1) if n > k + 1 else r2m
        mse = ss_r / (n - k)
        cov = mse * np.linalg.pinv(Xm.T @ Xm)
        se_m = np.sqrt(np.maximum(np.diag(cov), 0))
        t_m = beta / np.where(se_m > 0, se_m, 1e-10)
        p_m = 2 * (1 - stats.t.cdf(np.abs(t_m), df=n - k))
        fstat = ((ss_t - ss_r) / (k - 1)) / mse if mse > 0 and k > 1 else 0
        kc1, kc2, kc3 = st.columns(3)
        kpi(kc1, "", L["ols_r2"], f"{r2a:.4f}")
        kpi(kc2, "purple", L["ols_fstat"], f"{fstat:.2f}")
        kpi(kc3, "teal", L["ols_nobs"], f"{n:,}")
        mrows = []
        for i, vn in enumerate(["Intercept"] + indep):
            mrows.append({"Variable": vn, "β": round(beta[i], 6),
                          "Std.Error": round(se_m[i], 6), "t-stat": round(t_m[i], 3),
                          "p-value": round(p_m[i], 4), "Sig.": sig_stars(p_m[i])})
        mdf = pd.DataFrame(mrows)
        st.dataframe(
            mdf.style
            .map(lambda v: "color:#166534;font-weight:700" if v in ["***", "**", "*"]
                      else ("color:#9ca3af" if v == "n.s." else ""), subset=["Sig."])
            .background_gradient(subset=["β"], cmap="coolwarm"),
            use_container_width=True, hide_index=True)

        # Residuals & QQ plot
        rc1, rc2 = st.columns(2)
        with rc1:
            fig = px.scatter(x=Yh, y=res, opacity=.5, color_discrete_sequence=["#3b82f6"],
                             labels={"x": "Valeurs ajustées", "y": "Résidus"},
                             title="Résidus vs. Valeurs ajustées")
            fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=.6)
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
        with rc2:
            (osm, osr), (sl2, ic2, _) = stats.probplot(res)
            fqq = go.Figure([
                go.Scatter(x=osm, y=osr, mode="markers",
                           marker=dict(color="#8b5cf6", size=4, opacity=.6), name="Résidus"),
                go.Scatter(x=[min(osm), max(osm)],
                           y=[sl2 * min(osm) + ic2, sl2 * max(osm) + ic2],
                           mode="lines", line=dict(color="red", dash="dash"), name="Normale théorique")
            ])
            fqq.update_layout(title="Q-Q Plot des résidus", height=300,
                               margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fqq, use_container_width=True)

        # Scatter per variable
        st.markdown("<div class='sec-title'>Relations bivariées</div>", unsafe_allow_html=True)
        vcols = st.columns(min(len(indep), 3))
        for i, v in enumerate(indep):
            with vcols[i % 3]:
                s2, ic3, _, p2, _ = stats.linregress(rdf[v], rdf[dep])
                xr = np.linspace(rdf[v].min(), rdf[v].max(), 100)
                fs = go.Figure([
                    go.Scatter(x=rdf[v], y=rdf[dep], mode="markers",
                               marker=dict(size=4, opacity=.35, color="#94a3b8")),
                    go.Scatter(x=xr, y=s2 * xr + ic3, mode="lines",
                               line=dict(color="#ef4444", width=2),
                               name=f"β={s2:.4f} {sig_stars(p2)}")
                ])
                fs.update_layout(title=f"{dep} ~ {v}", height=260,
                                  margin=dict(l=0, r=0, t=40, b=0),
                                  showlegend=True, legend=dict(font_size=9))
                st.plotly_chart(fs, use_container_width=True)
    except np.linalg.LinAlgError:
        st.error("Multicolinéarité détectée — réduisez le nombre de variables.")

# ════════════════════════════════════════════
# PAGE 4 — CORRELATIONS
# ════════════════════════════════════════════
def page_corr():
    df = require()
    st.markdown(f"<div class='sec-title'>{L['corr_title']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='info-box'>{L['corr_desc']}</div>", unsafe_allow_html=True)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    avail = [c for c in num_cols if df[c].notna().sum() > 10]
    if len(avail) < 2:
        st.warning("Pas assez de variables numériques."); return
    sel = st.multiselect("Variables à inclure", avail, default=avail)
    if len(sel) < 2:
        st.info("Sélectionnez au moins 2 variables."); return
    cdf = df[sel].dropna()
    cm = cdf.corr()
    fh = px.imshow(cm, text_auto=".2f", aspect="auto",
                   color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title=L["corr_title"])
    fh.update_layout(height=max(350, len(sel) * 60), margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fh, use_container_width=True)

    if "default" in sel:
        st.markdown(f"<div class='sec-title'>{L['corr_top']}</div>", unsafe_allow_html=True)
        cwd = cm["default"].drop("default").sort_values(key=abs, ascending=False).reset_index()
        cwd.columns = ["Variable", "r avec défaut"]
        cwd["Interprétation"] = cwd["r avec défaut"].apply(
            lambda x: "Fort" if abs(x) > .3 else ("Modéré" if abs(x) > .15 else "Faible"))
        fb = px.bar(cwd, x="r avec défaut", y="Variable", orientation="h",
                    color="r avec défaut", color_continuous_scale="RdBu_r", range_color=[-1, 1],
                    text="r avec défaut", title=L["corr_top"])
        fb.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fb.add_vline(x=0, line_color="black", line_width=1)
        fb.update_layout(height=max(230, len(cwd) * 50),
                         margin=dict(l=0, r=60, t=40, b=0), coloraxis_showscale=False)
        st.plotly_chart(fb, use_container_width=True)
        st.dataframe(cwd, use_container_width=True, hide_index=True)

    st.markdown(f"<div class='sec-title'>{L['corr_scatter']}</div>", unsafe_allow_html=True)
    sc1, sc2 = st.columns(2)
    with sc1: xv = st.selectbox("Variable X", sel, index=0)
    with sc2: yv = st.selectbox("Variable Y", sel, index=min(1, len(sel) - 1))
    sdf = df[[xv, yv]].dropna()
    cv = "default" if "default" in df.columns else None
    if cv:
        sdf = sdf.copy()
        sdf["default"] = df.loc[sdf.index, "default"].astype(str)
    fsc = px.scatter(sdf, x=xv, y=yv, color="default" if cv else None,
                     color_discrete_map={"0": "#3b82f6", "1": "#ef4444"},
                     opacity=.55, trendline="ols", title=f"{yv} ~ {xv}")
    fsc.update_layout(height=360, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fsc, use_container_width=True)

    # Significance table
    st.markdown("<div class='sec-title'>Signification des corrélations (Pearson)</div>",
                unsafe_allow_html=True)
    sigr = []
    for i in range(len(sel)):
        for j in range(i + 1, len(sel)):
            v1, v2 = sel[i], sel[j]
            pair = df[[v1, v2]].dropna()
            if len(pair) < 5: continue
            rv, pv = stats.pearsonr(pair[v1], pair[v2])
            sigr.append({"Var 1": v1, "Var 2": v2, "r": round(rv, 4),
                         "p-value": round(pv, 4), "Sig.": sig_stars(pv), "n": len(pair)})
    if sigr:
        st_df = pd.DataFrame(sigr).sort_values("p-value")
        st.dataframe(
            st_df.style
            .applymap(lambda v: "color:#166534;font-weight:700" if v in ["***", "**", "*"]
                      else ("color:#9ca3af" if v == "n.s." else ""), subset=["Sig."])
            .background_gradient(subset=["r"], cmap="RdBu_r", vmin=-1, vmax=1),
            use_container_width=True, hide_index=True)

# ════════════════════════════════════════════
# PAGE 5 — LOGISTIC REGRESSION
# ════════════════════════════════════════════
def page_logit():
    df = require()
    st.markdown(f"<div class='sec-title'>{L['logit_title']}</div>", unsafe_allow_html=True)
    if "default" not in df.columns or df["default"].nunique() < 2:
        st.warning("Variable défaut manquante ou constante."); return
    num_cols = [c for c in ["amount", "duration", "rate", "age"]
                if c in df.columns and df[c].notna().sum() > 20]
    if not num_cols:
        st.warning("Aucune variable numérique disponible."); return
    cfg1, cfg2 = st.columns([2, 1])
    with cfg1:
        feats = st.multiselect("Variables explicatives (features)", num_cols, default=num_cols)
    with cfg2:
        thr = st.slider(L["logit_thresh"], .1, .9, .5, .05)
        ts  = st.slider("Taille test set", .1, .4, .2, .05)
    if not feats:
        st.info("Sélectionnez au moins une variable."); return
    rdf = df[feats + ["default"]].dropna()
    if len(rdf) < 30:
        st.warning(L["not_enough"]); return
    Xa = rdf[feats].values; ya = rdf["default"].values
    try:
        Xtr, Xte, ytr, yte = train_test_split(Xa, ya, test_size=ts, random_state=42, stratify=ya)
    except ValueError:
        Xtr, Xte, ytr, yte = train_test_split(Xa, ya, test_size=ts, random_state=42)
    sc = StandardScaler()
    Xtrs = sc.fit_transform(Xtr); Xtes = sc.transform(Xte)
    mdl = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    mdl.fit(Xtrs, ytr)
    yprob = mdl.predict_proba(Xtes)[:, 1]; ypred = (yprob >= thr).astype(int)
    st.session_state.logit_model = mdl; st.session_state.scaler = sc
    st.session_state.feature_cols = feats; st.session_state.X_test = Xte
    st.session_state.y_test = yte; st.session_state.y_prob = yprob
    try: auc = roc_auc_score(yte, yprob)
    except: auc = .5
    gini = 2 * auc - 1
    cm_arr = confusion_matrix(yte, ypred)
    tn, fp, fn, tp = cm_arr.ravel() if cm_arr.shape == (2, 2) else (0, 0, 0, 0)
    prec = tp / (tp + fp) if tp + fp > 0 else 0
    rec  = tp / (tp + fn) if tp + fn > 0 else 0
    acc  = (tp + tn) / len(yte) if len(yte) > 0 else 0
    # KPIs
    kk1, kk2, kk3, kk4, kk5 = st.columns(5)
    kpi(kk1, "purple", L["logit_auc"],   f"{auc:.4f}",  "Discriminance")
    kpi(kk2, "orange", L["logit_gini"],  f"{gini:.4f}", "2·AUC−1")
    kpi(kk3, "green",  "Accuracy",       f"{acc:.1%}",  f"seuil {thr}")
    kpi(kk4, "",       "Précision",      f"{prec:.1%}", "VP/(VP+FP)")
    kpi(kk5, "red",    "Rappel (Recall)",f"{rec:.1%}",  "VP/(VP+FN)")
    st.markdown("<br>", unsafe_allow_html=True)

    t1, t2, t3, t4 = st.tabs(["ROC & Calibration", "Coefficients & Odds-Ratios",
                                "Matrice de confusion", "Distribution des scores"])
    with t1:
        rc1, rc2 = st.columns(2)
        with rc1:
            fpr_v, tpr_v, _ = roc_curve(yte, yprob)
            fr = go.Figure([
                go.Scatter(x=fpr_v, y=tpr_v, mode="lines",
                           line=dict(color="#3b82f6", width=2.5),
                           fill="tozeroy", fillcolor="rgba(59,130,246,.1)",
                           name=f"ROC (AUC={auc:.4f})"),
                go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                           line=dict(dash="dash", color="#94a3b8", width=1), name="Aléatoire")
            ])
            fr.add_annotation(x=.65, y=.2,
                              text=f"AUC = {auc:.4f}<br>Gini = {gini:.4f}",
                              showarrow=False, bgcolor="white",
                              bordercolor="#3b82f6", font=dict(size=11))
            fr.update_layout(title=L["logit_roc"], height=360,
                             margin=dict(l=0, r=0, t=40, b=0),
                             xaxis_title="Taux FP (1-Spécificité)",
                             yaxis_title="Taux VP (Sensibilité)")
            st.plotly_chart(fr, use_container_width=True)
        with rc2:
            if len(yte) >= 20:
                nb = min(10, max(5, len(yte) // 20))
                fp_cal, mp = calibration_curve(yte, yprob, n_bins=nb)
                fc = go.Figure([
                    go.Scatter(x=mp, y=fp_cal, mode="lines+markers", name="Modèle",
                               line=dict(color="#8b5cf6", width=2), marker=dict(size=8)),
                    go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                               line=dict(dash="dash", color="#94a3b8"), name="Parfaite")
                ])
                fc.update_layout(title=L["logit_calib"], height=360,
                                 margin=dict(l=0, r=0, t=40, b=0),
                                 xaxis_title="PD prédite (moy. bin)",
                                 yaxis_title="Fréquence réelle de défaut")
                st.plotly_chart(fc, use_container_width=True)
    with t2:
        coefs = mdl.coef_[0]; icpt = mdl.intercept_[0]; OR = np.exp(coefs)
        ptrp = mdl.predict_proba(Xtrs)[:, 1]; W = np.diag(ptrp * (1 - ptrp))
        try:
            XtWX = Xtrs.T @ W @ Xtrs
            cov_l = np.linalg.pinv(XtWX)
            se_l = np.sqrt(np.maximum(np.diag(cov_l), 0))
            z_l = coefs / np.where(se_l > 0, se_l, 1e-10)
            pv_l = 2 * (1 - stats.norm.cdf(np.abs(z_l)))
        except Exception:
            se_l = np.zeros(len(coefs)); z_l = np.zeros(len(coefs)); pv_l = np.ones(len(coefs))
        cdf2 = pd.DataFrame({
            "Feature": feats, "β": coefs.round(4), "Std.Err": se_l.round(4),
            "z-stat": z_l.round(3), "p-value": pv_l.round(4),
            "Sig.": [sig_stars(p) for p in pv_l], "Odds Ratio": OR.round(4),
            "Effet": ["↑ Risque+" if c > 0 else "↓ Risque−" for c in coefs]
        }).sort_values("β", key=abs, ascending=False)
        st.dataframe(
            cdf2.style
            .applymap(lambda v: "color:#166534;font-weight:700" if v in ["***", "**", "*"]
                      else ("color:#9ca3af" if v == "n.s." else ""), subset=["Sig."])
            .background_gradient(subset=["β"], cmap="RdBu_r"),
            use_container_width=True, hide_index=True)
        fco = px.bar(cdf2.sort_values("β"), x="β", y="Feature", orientation="h",
                     color="β", color_continuous_scale="RdBu_r",
                     title="Coefficients logistiques standardisés", text="Odds Ratio")
        fco.update_traces(texttemplate="OR=%{text:.3f}", textposition="outside")
        fco.add_vline(x=0, line_dash="dash", line_color="black")
        fco.update_layout(height=max(280, len(feats) * 55),
                          margin=dict(l=0, r=80, t=40, b=0), coloraxis_showscale=False)
        st.plotly_chart(fco, use_container_width=True)
        st.markdown(
            f"<div class='info-box'><strong>Intercept:</strong> {icpt:.4f}<br>"
            "OR > 1 → augmente le risque de défaut | OR &lt; 1 → diminue le risque</div>",
            unsafe_allow_html=True)
    with t3:
        tc1, tc2 = st.columns([1, 1])
        with tc1:
            flb = ["Sain (0)", "Défaut (1)"]
            fcm = px.imshow(cm_arr, text_auto=True, aspect="auto",
                            x=flb, y=flb, color_continuous_scale="Blues",
                            title=L["logit_cm"], labels=dict(x="Prédit", y="Réel", color="N"))
            fcm.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fcm, use_container_width=True)
        with tc2:
            rpt = classification_report(yte, ypred, target_names=flb, output_dict=True)
            st.markdown(f"**{L['logit_report']}**")
            st.dataframe(pd.DataFrame(rpt).T.round(3), use_container_width=True)
    with t4:
        sdf2 = pd.DataFrame({"PD estimée": yprob, "Réel": yte.astype(str)})
        fd = px.histogram(sdf2, x="PD estimée", color="Réel", nbins=30, barmode="overlay",
                          color_discrete_map={"0": "#3b82f6", "1": "#ef4444"}, opacity=.7,
                          title=L["logit_score"])
        fd.add_vline(x=thr, line_dash="dash", line_color="black",
                     annotation_text=f"Seuil={thr}", annotation_position="top right")
        fd.update_layout(height=340, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fd, use_container_width=True)
        if len(yprob) >= 10:
            sca = pd.DataFrame({"s": yprob, "d": yte})
            n_dec = min(10, max(3, len(sca) // 3))
            sca["decile"] = pd.qcut(sca["s"], q=n_dec,
                                    labels=[f"D{i}" for i in range(1, n_dec + 1)],
                                    duplicates="drop")
            dtbl = sca.groupby("decile", observed=True).agg(
                N=("d", "count"), Déf=("d", "sum"),
                PD_réelle=("d", "mean"), PD_moy=("s", "mean")).reset_index()
            dtbl["PD_réelle"] = (dtbl["PD_réelle"] * 100).round(2)
            dtbl["PD_moy"]    = (dtbl["PD_moy"]    * 100).round(2)
            fdc = px.bar(dtbl, x="decile", y="PD_réelle", title="PD réelle par décile de score",
                         color="PD_réelle", color_continuous_scale="RdYlGn_r", text="PD_réelle")
            fdc.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fdc.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0), coloraxis_showscale=False)
            st.plotly_chart(fdc, use_container_width=True)
            st.dataframe(dtbl, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════
# PAGE 6 — EXPORT
# ════════════════════════════════════════════
def page_export():
    df = require()
    st.markdown(f"<div class='sec-title'>{L['export_title']}</div>", unsafe_allow_html=True)
    n = len(df); pdg = df["default"].mean() * 100 if "default" in df.columns else 0
    ec1, ec2, ec3 = st.columns(3)
    kpi(ec1, "", "Total prêts", f"{n:,}")
    kpi(ec2, "red", "PD globale", f"{pdg:.2f}%")
    ms = "✅ Modèle entraîné" if st.session_state.logit_model else "⚠️ Non entraîné — voir Régression Logistique"
    kpi(ec3, "green" if st.session_state.logit_model else "orange", "Modèle logistique", ms)
    st.markdown("<br>", unsafe_allow_html=True)
    edf = df.copy()
    if st.session_state.logit_model and st.session_state.feature_cols:
        try:
            fs = st.session_state.feature_cols
            Xall = edf[fs].fillna(edf[fs].median())
            Xsc = st.session_state.scaler.transform(Xall.values)
            edf["PD_score"] = st.session_state.logit_model.predict_proba(Xsc)[:, 1].round(4)
            edf["Classe_risque"] = pd.cut(edf["PD_score"], bins=[0, .15, .30, .50, 1.0],
                                           labels=["Faible", "Modéré", "Élevé", "Très élevé"])
            st.markdown(
                "<div class='success-box'>✅ Scores PD calculés pour l'ensemble du portefeuille "
                "(colonnes <strong>PD_score</strong> et <strong>Classe_risque</strong>)</div>",
                unsafe_allow_html=True)
            fe = px.histogram(edf, x="PD_score", color="Classe_risque", nbins=30,
                              title="Distribution PD — portefeuille complet",
                              color_discrete_map={"Faible": "#10b981", "Modéré": "#f59e0b",
                                                   "Élevé": "#f97316", "Très élevé": "#ef4444"})
            fe.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fe, use_container_width=True)
        except Exception as e:
            st.warning(f"Scoring impossible : {e}")
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(label=L["export_dl_csv"],
                           data=edf.to_csv(index=False).encode("utf-8-sig"),
                           file_name="portefeuille_scored.csv", mime="text/csv",
                           use_container_width=True)
    with dl2:
        lines = ["RAPPORT ANALYTIQUE — CreditMacro Risk Engine", "=" * 50,
                 f"Nombre de prêts : {n}", f"PD globale observée : {pdg:.2f}%", ""]
        if "sector" in df.columns:
            lines.append("PD par secteur:")
            for kv, vv in df.groupby("sector")["default"].mean().items():
                lines.append(f"  {kv}: {vv * 100:.1f}%")
        if "region" in df.columns:
            lines.append("\nPD par région:")
            for kv, vv in df.groupby("region")["default"].mean().items():
                lines.append(f"  {kv}: {vv * 100:.1f}%")
        if st.session_state.logit_model and st.session_state.y_prob is not None:
            try:
                auc2 = roc_auc_score(st.session_state.y_test, st.session_state.y_prob)
                lines += ["", f"Modèle logistique — AUC: {auc2:.4f}  Gini: {2 * auc2 - 1:.4f}",
                          f"Features : {', '.join(st.session_state.feature_cols)}"]
            except Exception:
                pass
        st.download_button(label=L["export_dl_sum"],
                           data="\n".join(lines).encode("utf-8"),
                           file_name="rapport_analytique.txt", mime="text/plain",
                           use_container_width=True)
    st.markdown("<div class='sec-title'>Aperçu du portefeuille scoré</div>", unsafe_allow_html=True)
    st.dataframe(edf.head(25), use_container_width=True)

# ════════════════════════════════════════════
# ROUTER
# ════════════════════════════════════════════
if   page == L["nav_upload"]: page_upload()
elif page == L["nav_pd"]:     page_pd()
elif page == L["nav_ols"]:    page_ols()
elif page == L["nav_corr"]:   page_corr()
elif page == L["nav_logit"]:  page_logit()
elif page == L["nav_export"]: page_export()
