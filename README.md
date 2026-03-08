# 🏦 CreditMacro Togo — Outil d'aide à la décision crédit en microfinance

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Application Streamlit bilingue (FR/EN) pour analystes crédit en microfinance au Togo.  
> Va au-delà du dossier comptable client en intégrant des analyses **macroéconomiques**, **économétriques** et **statistiques** pour les décisions de crédit à l'investissement.

---

## 🎯 Objectif

Fournir aux analystes crédit un outil de décision rigoureux intégrant :
- La conjoncture macroéconomique (PIB, population)
- La performance historique du portefeuille par secteur et région
- Des modèles économétriques pour évaluer le risque
- Un système de scoring multicritère pour les grands montants destinés à l'investissement

---

## 📦 Structure du projet

```
microfinance_app/
├── app.py                        # Application principale Streamlit
├── requirements.txt              # Dépendances Python
├── Croissance_du_PIB_Togo.csv   # Données PIB Togo (Banque Mondiale, 1960–2023)
├── Population_togo.csv           # Données population Togo (Banque Mondiale)
├── Jeux_donnees.csv              # Portefeuille microfinance (500 prêts réels)
└── README.md
```

---

## 🚀 Installation & lancement

```bash
# Cloner le dépôt
git clone https://github.com/VOTRE_USERNAME/creditMacro-togo.git
cd creditMacro-togo

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run app.py
```

---

## 📊 Fonctionnalités

### 1. 📊 Tableau de bord
- KPIs clés du portefeuille (encours, taux de défaut, remboursement)
- Répartition par statut, secteur, région
- Évolution du PIB intégrée en contexte

### 2. 🌍 Analyse Macroéconomique
- Croissance du PIB Togo (1960–2023) avec tendance linéaire
- Évolution de la population
- Indicateurs de contexte : moyenne 5 ans, volatilité, tendance
- Alerte macro : favorable / à surveiller / défavorable

### 3. 📁 Analyse du Portefeuille
- Distributions statistiques : montants, durées, taux
- Performance par secteur et région (matrice de défaut)
- Analyse démographique (âge, genre)
- Matrice de corrélations entre variables

### 4. ⚖️ Aide à la Décision (module central)
- Formulaire de saisie du dossier d'investissement
- **Score macroéconomique** : conjoncture, volatilité, durée
- **Score sectoriel** : taux de défaut et remboursement du secteur
- **Score statistique** : profil région, âge, montant vs. médiane
- Score global pondéré + recommandation (Favorable / Avec réserves / Défavorable)
- Conditions suggérées : montant, taux, durée, garanties
- Justification analytique détaillée
- Dossiers comparables dans le portefeuille

### 5. 📐 Modèles Économétriques
- Régression OLS : facteurs explicatifs du défaut
- Prévision de la croissance PIB 2024–2030 (avec intervalles de confiance)
- Corrélation PIB × demande de crédit

### 6. 🚨 Scoring de Risque
- Score de risque sectoriel (0–100)
- Score de risque régional (0–100)
- Sensibilité sectorielle aux chocs PIB (β)
- Matrice risque Secteur × Région

---

## 🌐 Bilinguisme

L'interface est entièrement disponible en **Français** et en **English**, sélectionnable depuis la barre latérale.

---

## 📈 Données utilisées

| Fichier | Source | Période |
|---------|--------|---------|
| `Croissance_du_PIB_Togo.csv` | Banque Mondiale (NY.GDP.MKTP.KD.ZG) | 1960–2023 |
| `Population_togo.csv` | Banque Mondiale (SP.POP.TOTL) | 1960–2023 |
| `Jeux_donnees.csv` | Portefeuille microfinance réel | 2024–2025 |

---

## 🔧 Technologies

- **[Streamlit](https://streamlit.io/)** — Interface web interactive
- **[Plotly](https://plotly.com/)** — Visualisations interactives
- **[Pandas / NumPy](https://pandas.pydata.org/)** — Traitement des données
- **[SciPy](https://scipy.org/)** — Régressions et statistiques

---

## 📜 Licence

MIT License — Libre utilisation pour fins académiques et professionnelles.

---

## 👤 Auteur

Développé par SOWAH Kemil Alberto, étudiant en fin de parcours finance comptabilité, Université de Lomé (IUT-G)
Tout apport et amélioration est le bienvenu
