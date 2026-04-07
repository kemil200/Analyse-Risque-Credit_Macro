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
.explain-box { background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:.85rem 1.1rem; font-size:.85rem; color:#334155; margin:.5rem 0 1rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Translations ──
T = {
    "fr": dict(
        title="🏦 CreditMacro · Risk Engine", subtitle="Analyse de risque crédit microfinance",
        nav_upload="Chargement", nav_pd="Taux de Défaut", nav_ols="Facteurs de Risque",
        nav_corr="Liens entre Variables", nav_logit="Modèle de Scoring", nav_export="Export",
        upload_title="Chargez votre portefeuille de prêts", upload_desc="CSV ou Excel · Toute institution de microfinance",
        col_map="Mapping des colonnes", col_map_desc="Associez vos colonnes aux champs analytiques",
        col_default="Colonne statut / défaut", col_amount="Montant du prêt", col_duration="Durée (mois)",
        col_rate="Taux d'intérêt (%)", col_age="Âge emprunteur", col_sector="Secteur d'activité",
        col_region="Région", col_gender="Genre", val_default="Valeurs = défaut (ex: EN RETARD, 1)",
        apply_map="✅ Valider le mapping", no_data="⬅️ Chargez d'abord vos données dans Chargement",
        not_enough="Données insuffisantes (min. 30 lignes valides)",
        pd_title="Taux de Défaut — Qui ne rembourse pas ?",
        pd_global="Taux de défaut global",
        pd_by_sector="Taux de défaut par secteur d'activité",
        pd_by_region="Taux de défaut par région",
        pd_by_gender="Taux de défaut par genre",
        ols_title="Facteurs de Risque — Qu'est-ce qui influence le défaut ?",
        ols_explain="Ce graphique montre quels facteurs augmentent ou diminuent le risque de non-remboursement. Les barres rouges signifient que la variable est liée à plus de défauts ; les barres vertes à moins de défauts.",
        corr_title="Liens entre Variables",
        corr_explain="Ce graphique montre le lien entre chaque variable et le fait de ne pas rembourser. Plus la barre est longue, plus le lien est fort. Rouge = lié à plus de défauts, Bleu = lié à moins de défauts.",
        logit_title="Modèle de Scoring — Capacité à prédire le défaut",
        logit_explain_roc="La courbe ROC montre la capacité du modèle à distinguer les bons et mauvais payeurs. Plus la courbe est haute et à gauche, meilleur est le modèle. Un score de 0,5 = modèle aléatoire ; 1,0 = modèle parfait.",
        logit_explain_dist="Ce graphique montre comment le modèle distribue les scores de risque. Les barres rouges sont les clients qui ont réellement fait défaut : elles doivent idéalement se concentrer à droite (score élevé).",
        logit_auc="Score de fiabilité du modèle",
        logit_thresh="Seuil de décision",
        logit_cm="Résultats de la prédiction",
        export_title="Export & Rapport de synthèse",
        export_dl_csv="⬇️ Télécharger portefeuille scoré (CSV)",
        export_dl_sum="⬇️ Rapport analytique (TXT)",
    ),
    "en": dict(
        title="🏦 CreditMacro · Risk Engine", subtitle="Credit risk analysis for microfinance",
        nav_upload="Data Upload", nav_pd="Default Rate", nav_ols="Risk Factors",
        nav_corr="Variable Links", nav_logit="Scoring Model", nav_export="Export",
        upload_title="Upload your loan portfolio", upload_desc="CSV or Excel · Any microfinance institution",
        col_map="Column Mapping", col_map_desc="Map your columns to the required analytical fields",
        col_default="Default / status column", col_amount="Loan amount", col_duration="Duration (months)",
        col_rate="Interest rate (%)", col_age="Borrower age", col_sector="Activity sector",
        col_region="Region", col_gender="Gender", val_default="Default values (e.g. LATE, 1)",
        apply_map="✅ Confirm mapping", no_data="⬅️ Please load your data in the Upload tab first",
        not_enough="Not enough data (min. 30 valid rows)",
        pd_title="Default Rate — Who is not repaying?",
        pd_global="Overall default rate",
        pd_by_sector="Default rate by sector",
        pd_by_region="Default rate by region",
        pd_by_gender="Default rate by gender",
        ols_title="Risk Factors — What drives default?",
        ols_explain="This chart shows which factors increase or decrease the risk of non-repayment. Red bars mean the variable is linked to more defaults; green bars to fewer defaults.",
        corr_title="Links Between Variables",
        corr_explain="This chart shows how strongly each variable is linked to non-repayment. The longer the bar, the stronger the link. Red = linked to more defaults, Blue = linked to fewer defaults.",
        logit_title="Scoring Model — Ability to predict default",
        logit_explain_roc="The ROC curve shows how well the model distinguishes good from bad payers. The higher and further left the curve, the better the model. A score of 0.5 = random guess; 1.0 = perfect model.",
        logit_explain_dist="This chart shows how the model distributes risk scores. Red bars are clients who actually defaulted — ideally they should cluster on the right (high score).",
        logit_auc="Model reliability score",
        logit_thresh="Decision threshold",
        logit_cm="Prediction results",
        export_title="Export & Summary Report",
        export_dl_csv="⬇️ Download scored portfolio (CSV)",
        export_dl_sum="⬇️ Analytical report (TXT)",
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

def kpi(col, cls, lbl, val, sub=""):
    with col:
        st.markdown(
            f"<div class='kpi-card {cls}'>"
            f"<div class='kpi-label'>{lbl}</div>"
            f"<div class='kpi-val'>{val}</div>"
            f"<div class='kpi-sub'>{sub}</div></div>",
            unsafe_allow_html=True)

def explain(txt):
    st.markdown(f"<div class='explain-box'>💡 {txt}</div>", unsafe_allow_html=True)

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
                f"{nd2} défauts · Taux de défaut global = {pdg:.1f}%</div>",
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
# PAGE 2 — TAUX DE DÉFAUT (simplifié)
# ════════════════════════════════════════════
def page_pd():
    df = require()
    st.markdown(f"<div class='sec-title'>{L['pd_title']}</div>", unsafe_allow_html=True)
    n = len(df); nd = int(df["default"].sum()); pdp = df["default"].mean() * 100

    # KPIs simples
    c1, c2, c3 = st.columns(3)
    kpi(c1, "",       "Nombre total de prêts",  f"{n:,}")
    kpi(c2, "red",    "Clients en défaut",       f"{nd:,}",     f"{pdp:.1f}% du portefeuille")
    kpi(c3, "green",  "Clients qui remboursent", f"{n - nd:,}", f"{100 - pdp:.1f}% du portefeuille")

    explain("Le taux de défaut indique la proportion de clients qui n'ont pas remboursé leur prêt. Plus ce taux est élevé, plus le risque pour l'institution est important.")

    st.markdown("<br>", unsafe_allow_html=True)

    def pd_bar(cat, title, note=""):
        if cat not in df.columns: return
        g = df.groupby(cat)["default"].agg(["mean", "sum", "count"]).reset_index()
        g.columns = [cat, "PD", "Def", "N"]
        g["PD_pct"] = (g["PD"] * 100).round(1)
        g = g.sort_values("PD_pct", ascending=True)
        fig = px.bar(g, x="PD_pct", y=cat, orientation="h",
                     color="PD_pct", color_continuous_scale="RdYlGn_r",
                     range_color=[0, max(g["PD_pct"].max(), 1)],
                     text="PD_pct", title=title,
                     labels={"PD_pct": "% clients en défaut", cat: ""},
                     custom_data=["Def", "N"])
        fig.update_traces(
            texttemplate="%{text:.1f}%", textposition="outside",
            hovertemplate="<b>%{y}</b><br>%{x:.1f}% en défaut<br>%{customdata[0]} défauts sur %{customdata[1]} prêts<extra></extra>")
        fig.update_layout(height=max(230, len(g) * 50),
                          margin=dict(l=0, r=40, t=40, b=0), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        if note:
            explain(note)

    col1, col2 = st.columns(2)
    with col1:
        pd_bar("sector", L["pd_by_sector"],
               "Les secteurs en rouge sont les plus risqués : ils concentrent le plus de clients en défaut.")
    with col2:
        pd_bar("region", L["pd_by_region"],
               "Certaines régions présentent un taux de défaut plus élevé, ce qui peut orienter les décisions de déploiement.")

    if "gender" in df.columns:
        col3, _ = st.columns(2)
        with col3:
            pd_bar("gender", L["pd_by_gender"],
                   "Comparaison du taux de défaut entre genres.")

    # Tableau récapitulatif simplifié
    st.markdown("<div class='sec-title'>Tableau récapitulatif</div>", unsafe_allow_html=True)
    cat_avail = [c for c in ["sector", "region", "gender"] if c in df.columns]
    if cat_avail:
        dim = st.selectbox("Voir les résultats par", cat_avail)
        tbl = df.groupby(dim)["default"].agg(
            **{"Nombre de prêts": "count",
               "Clients en défaut": "sum"}).reset_index()
        tbl["Taux de défaut"] = (df.groupby(dim)["default"].mean().values * 100).round(1)
        tbl["Taux de défaut"] = tbl["Taux de défaut"].apply(lambda x: f"{x:.1f}%")
        st.dataframe(tbl, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════
# PAGE 3 — FACTEURS DE RISQUE (simplifié OLS)
# ════════════════════════════════════════════
def page_ols():
    df = require()
    st.markdown(f"<div class='sec-title'>{L['ols_title']}</div>", unsafe_allow_html=True)
    explain(L["ols_explain"])

    num_cols = [c for c in ["amount", "duration", "rate", "age"]
                if c in df.columns and df[c].notna().sum() > 10]
    if not num_cols:
        st.warning("Aucune variable numérique disponible."); return

    label_map = {
        "amount": "Montant du prêt",
        "duration": "Durée du prêt",
        "rate": "Taux d'intérêt",
        "age": "Âge de l'emprunteur"
    }

    rows = []
    for v in num_cols:
        pair = df[[v, "default"]].dropna()
        if len(pair) < 10: continue
        sl, _, _, p, _ = stats.linregress(pair[v], pair["default"])
        rows.append({
            "Facteur": label_map.get(v, v),
            "Direction": "↑ Plus de défauts" if sl > 0 else "↓ Moins de défauts",
            "Intensité": abs(sl),
            "beta_raw": sl,
        })

    if not rows:
        st.warning("Pas assez de données pour calculer les facteurs."); return

    rdf = pd.DataFrame(rows).sort_values("Intensité", ascending=True)
    rdf["Couleur"] = rdf["beta_raw"].apply(lambda x: "#ef4444" if x > 0 else "#10b981")

    fig = go.Figure()
    for _, row in rdf.iterrows():
        fig.add_trace(go.Bar(
            x=[row["beta_raw"]],
            y=[row["Facteur"]],
            orientation="h",
            marker_color=row["Couleur"],
            name=row["Facteur"],
            text=row["Direction"],
            textposition="outside",
            hovertemplate=f"<b>{row['Facteur']}</b><br>{row['Direction']}<extra></extra>"
        ))
    fig.add_vline(x=0, line_color="#334155", line_width=1.5)
    fig.update_layout(
        showlegend=False,
        height=max(250, len(rows) * 70),
        margin=dict(l=0, r=120, t=20, b=0),
        xaxis=dict(visible=False),
        yaxis_title="",
        plot_bgcolor="white"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Scatter plots simplifiés
    st.markdown("<div class='sec-title'>Relation entre chaque facteur et le défaut</div>",
                unsafe_allow_html=True)
    vcols = st.columns(min(len(num_cols), 2))
    for i, v in enumerate(num_cols):
        pair = df[[v, "default"]].dropna()
        if len(pair) < 10: continue
        with vcols[i % 2]:
            avg = pair.groupby("default")[v].mean().reset_index()
            avg["Statut"] = avg["default"].map({0: "✅ Rembourse", 1: "❌ En défaut"})
            avg[label_map.get(v, v)] = avg[v].round(1)
            fb = px.bar(avg, x="Statut", y=v,
                        color="Statut",
                        color_discrete_map={"✅ Rembourse": "#10b981", "❌ En défaut": "#ef4444"},
                        title=f"Moyenne : {label_map.get(v, v)}",
                        labels={v: label_map.get(v, v), "Statut": ""},
                        text=v)
            fb.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fb.update_layout(height=280, margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
            st.plotly_chart(fb, use_container_width=True)

# ════════════════════════════════════════════
# PAGE 4 — LIENS ENTRE VARIABLES (simplifié)
# ════════════════════════════════════════════
def page_corr():
    df = require()
    st.markdown(f"<div class='sec-title'>{L['corr_title']}</div>", unsafe_allow_html=True)
    explain(L["corr_explain"])

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    avail = [c for c in num_cols if df[c].notna().sum() > 10]

    label_map = {
        "amount": "Montant du prêt",
        "duration": "Durée du prêt",
        "rate": "Taux d'intérêt",
        "age": "Âge de l'emprunteur",
        "default": "Défaut"
    }

    if "default" not in avail or len(avail) < 2:
        st.warning("Pas assez de variables numériques."); return

    cm = df[avail].dropna().corr()
    cwd = cm["default"].drop("default").reset_index()
    cwd.columns = ["variable", "r"]
    cwd["Facteur"] = cwd["variable"].map(lambda x: label_map.get(x, x))
    cwd["Lien"] = cwd["r"].apply(
        lambda x: "Fort" if abs(x) > .3 else ("Modéré" if abs(x) > .15 else "Faible"))
    cwd["Couleur"] = cwd["r"].apply(lambda x: "#ef4444" if x > 0 else "#3b82f6")
    cwd = cwd.sort_values("r", key=abs, ascending=True)

    fig = go.Figure()
    for _, row in cwd.iterrows():
        direction = "lié à plus de défauts" if row["r"] > 0 else "lié à moins de défauts"
        fig.add_trace(go.Bar(
            x=[row["r"]],
            y=[row["Facteur"]],
            orientation="h",
            marker_color=row["Couleur"],
            name=row["Facteur"],
            text=f"{row['Lien']} ({direction})",
            textposition="outside",
            hovertemplate=f"<b>{row['Facteur']}</b><br>Lien: {row['Lien']}<br>{direction}<extra></extra>"
        ))
    fig.add_vline(x=0, line_color="#334155", line_width=1.5)
    fig.update_layout(
        showlegend=False,
        height=max(250, len(cwd) * 70),
        margin=dict(l=0, r=200, t=20, b=0),
        xaxis=dict(visible=False),
        yaxis_title="",
        plot_bgcolor="white"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tableau lisible
    st.markdown("<div class='sec-title'>Résumé des liens</div>", unsafe_allow_html=True)
    tbl = cwd[["Facteur", "Lien"]].copy()
    tbl["Direction"] = cwd["r"].apply(
        lambda x: "↑ Lié à plus de défauts" if x > 0 else "↓ Lié à moins de défauts")
    tbl = tbl.sort_values("Lien", ascending=False)
    st.dataframe(tbl, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════
# PAGE 5 — MODÈLE DE SCORING (simplifié)
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

    label_map = {
        "amount": "Montant du prêt",
        "duration": "Durée du prêt",
        "rate": "Taux d'intérêt",
        "age": "Âge de l'emprunteur"
    }

    with st.expander("⚙️ Paramètres du modèle", expanded=False):
        feats = st.multiselect("Variables utilisées par le modèle",
                               num_cols, default=num_cols,
                               format_func=lambda x: label_map.get(x, x))
        thr = st.slider(L["logit_thresh"], .1, .9, .5, .05,
                        help="Au-dessus de ce seuil, le client est classé 'à risque'")
        ts  = st.slider("Proportion de données de test", .1, .4, .2, .05)

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
    yprob = mdl.predict_proba(Xtes)[:, 1]
    ypred = (yprob >= thr).astype(int)

    st.session_state.logit_model = mdl; st.session_state.scaler = sc
    st.session_state.feature_cols = feats; st.session_state.X_test = Xte
    st.session_state.y_test = yte; st.session_state.y_prob = yprob

    try: auc = roc_auc_score(yte, yprob)
    except: auc = .5

    cm_arr = confusion_matrix(yte, ypred)
    tn, fp, fn, tp = cm_arr.ravel() if cm_arr.shape == (2, 2) else (0, 0, 0, 0)
    acc  = (tp + tn) / len(yte) if len(yte) > 0 else 0
    # Défauts détectés = recall
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # KPIs simples
    k1, k2, k3 = st.columns(3)
    fiab = "Excellent" if auc >= .8 else ("Bon" if auc >= .7 else ("Acceptable" if auc >= .6 else "Faible"))
    kpi(k1, "purple", L["logit_auc"],           f"{auc:.2f} / 1.00", f"Niveau : {fiab}")
    kpi(k2, "green",  "Prédictions correctes",  f"{acc:.0%}",        f"seuil {thr}")
    kpi(k3, "orange", "Défauts détectés",        f"{recall:.0%}",     "parmi les vrais défauts")

    explain(f"Le modèle a un score de fiabilité de **{auc:.2f}** sur 1,00 (niveau : {fiab}). "
            f"Il détecte correctement **{recall:.0%}** des clients qui vont réellement faire défaut.")

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📈 Performance du modèle", "🎯 Résultats de la prédiction"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            explain(L["logit_explain_roc"])
            fpr_v, tpr_v, _ = roc_curve(yte, yprob)
            fr = go.Figure([
                go.Scatter(x=fpr_v, y=tpr_v, mode="lines",
                           line=dict(color="#3b82f6", width=3),
                           fill="tozeroy", fillcolor="rgba(59,130,246,.1)",
                           name=f"Notre modèle (score {auc:.2f})"),
                go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                           line=dict(dash="dash", color="#94a3b8", width=1),
                           name="Décision aléatoire")
            ])
            fr.add_annotation(x=.6, y=.2,
                              text=f"Score de fiabilité<br><b>{auc:.2f} / 1.00</b><br>Niveau : {fiab}",
                              showarrow=False, bgcolor="white",
                              bordercolor="#3b82f6", font=dict(size=12))
            fr.update_layout(
                title="Capacité du modèle à distinguer bon/mauvais payeur",
                height=360, margin=dict(l=0, r=0, t=40, b=0),
                xaxis_title="Fausses alertes (clients sains classés à risque)",
                yaxis_title="Vrais défauts détectés")
            st.plotly_chart(fr, use_container_width=True)

        with col2:
            explain(L["logit_explain_dist"])
            sdf2 = pd.DataFrame({
                "Score de risque estimé": yprob,
                "Statut réel": yte.astype(str).map({"0": "✅ Rembourse", "1": "❌ En défaut"})
            })
            fd = px.histogram(sdf2, x="Score de risque estimé", color="Statut réel",
                              nbins=25, barmode="overlay", opacity=.75,
                              color_discrete_map={"✅ Rembourse": "#3b82f6", "❌ En défaut": "#ef4444"},
                              title="Distribution des scores de risque")
            fd.add_vline(x=thr, line_dash="dash", line_color="#0f172a",
                         annotation_text=f"Seuil de décision ({thr})",
                         annotation_position="top right")
            fd.update_layout(height=360, margin=dict(l=0, r=0, t=40, b=0),
                             xaxis_title="Score de risque (0 = aucun risque, 1 = risque maximal)",
                             yaxis_title="Nombre de clients",
                             legend_title="Statut réel")
            st.plotly_chart(fd, use_container_width=True)

    with tab2:
        explain(f"Sur {len(yte)} clients de test, le modèle a correctement classé **{tp + tn}** clients. "
                f"Il a manqué **{fn}** vrais défauts et généré **{fp}** fausses alertes.")

        col3, col4 = st.columns(2)
        with col3:
            labels = ["✅ Rembourse", "❌ En défaut"]
            fcm = px.imshow(cm_arr, text_auto=True, aspect="auto",
                            x=labels, y=labels,
                            color_continuous_scale="Blues",
                            title="Matrice de résultats",
                            labels=dict(x="Prédit par le modèle", y="Réalité", color="Clients"))
            fcm.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fcm, use_container_width=True)

        with col4:
            st.markdown("**Lecture de la matrice :**")
            st.markdown(f"""
- ✅ **{tn}** clients sains correctement identifiés
- ❌ **{tp}** vrais défauts correctement détectés
- ⚠️ **{fp}** clients sains classés à tort comme risqués *(fausses alertes)*
- 🔴 **{fn}** vrais défauts non détectés *(manqués)*
""")
            st.markdown(f"""
**En résumé :** Pour 100 clients analysés, le modèle prédit correctement le statut de **{acc:.0%}** d'entre eux.
""")

# ════════════════════════════════════════════
# PAGE 6 — EXPORT
# ════════════════════════════════════════════
def page_export():
    df = require()
    st.markdown(f"<div class='sec-title'>{L['export_title']}</div>", unsafe_allow_html=True)
    n = len(df); pdg = df["default"].mean() * 100 if "default" in df.columns else 0
    ec1, ec2, ec3 = st.columns(3)
    kpi(ec1, "", "Total prêts", f"{n:,}")
    kpi(ec2, "red", "Taux de défaut global", f"{pdg:.1f}%")
    ms = "✅ Modèle entraîné" if st.session_state.logit_model else "⚠️ Non entraîné — voir Modèle de Scoring"
    kpi(ec3, "green" if st.session_state.logit_model else "orange", "Modèle de scoring", ms)
    st.markdown("<br>", unsafe_allow_html=True)

    edf = df.copy()
    if st.session_state.logit_model and st.session_state.feature_cols:
        try:
            fs = st.session_state.feature_cols
            Xall = edf[fs].fillna(edf[fs].median())
            Xsc = st.session_state.scaler.transform(Xall.values)
            edf["Score_risque"] = st.session_state.logit_model.predict_proba(Xsc)[:, 1].round(4)
            edf["Classe_risque"] = pd.cut(edf["Score_risque"], bins=[0, .15, .30, .50, 1.0],
                                           labels=["🟢 Faible", "🟡 Modéré", "🟠 Élevé", "🔴 Très élevé"])
            st.markdown(
                "<div class='success-box'>✅ Scores de risque calculés pour l'ensemble du portefeuille "
                "(colonnes <strong>Score_risque</strong> et <strong>Classe_risque</strong>)</div>",
                unsafe_allow_html=True)
            fe = px.histogram(edf, x="Score_risque", color="Classe_risque", nbins=30,
                              title="Répartition des clients par niveau de risque",
                              color_discrete_map={
                                  "🟢 Faible": "#10b981", "🟡 Modéré": "#f59e0b",
                                  "🟠 Élevé": "#f97316", "🔴 Très élevé": "#ef4444"},
                              labels={"Score_risque": "Score de risque", "Classe_risque": "Niveau de risque"})
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
                 f"Nombre de prêts : {n}", f"Taux de défaut global : {pdg:.1f}%", ""]
        if "sector" in df.columns:
            lines.append("Taux de défaut par secteur d'activité :")
            for kv, vv in df.groupby("sector")["default"].mean().items():
                lines.append(f"  {kv}: {vv * 100:.1f}%")
        if "region" in df.columns:
            lines.append("\nTaux de défaut par région :")
            for kv, vv in df.groupby("region")["default"].mean().items():
                lines.append(f"  {kv}: {vv * 100:.1f}%")
        if st.session_state.logit_model and st.session_state.y_prob is not None:
            try:
                auc2 = roc_auc_score(st.session_state.y_test, st.session_state.y_prob)
                fiab2 = "Excellent" if auc2 >= .8 else ("Bon" if auc2 >= .7 else ("Acceptable" if auc2 >= .6 else "Faible"))
                lines += ["", f"Modèle de scoring — Fiabilité: {auc2:.2f}/1.00 ({fiab2})",
                          f"Variables utilisées : {', '.join(st.session_state.feature_cols)}"]
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
