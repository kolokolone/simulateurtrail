# Simulateur Trail — Allure ↔ Pente (Fork)

> Analyse de traces **GPX** avec **allure ajustée à la pente**.  
> Deux modes : **théorique** (Minetti/Strava) et **empirique** (à partir de *tes vitesses observées*).

## ✨ TL;DR

- **Objectif** : prédire ou expliquer un **temps total** sur un parcours trail en tenant compte de la pente.  
- **Nouveauté du fork** : un **mode empirique** qui construit **ta courbe perso Allure ↔ Pente** depuis ton GPX, puis **rejoue la trace** pour estimer un temps réaliste basé sur tes vitesses mesurées.
- **Interface** : Streamlit + graphiques Plotly, carte Folium, récap **VAP** + **allure équivalente au plat**.

## 🧭 Pourquoi ce fork ?

Le dépôt amont proposait surtout le calcul “temps objectif” via modèles **Minetti** et **Strava** (GAP-like).  
Ce fork **garde cet esprit** mais ajoute un **pipeline de données réelles** pour ancrer les estimations dans **ta physiologie de course**.

> Projet amont : [theotimroger/simulateurtrail](https://github.com/theotimroger/simulateurtrail)

## 🚀 Ce que fait l’application

- **Charge un GPX**, reconstruit distance, D+/D-, profil, carte.
- **Mode Théorique** : tu saisis un **temps total visé**, l’app calibre la **VAP** et simule le parcours selon **Minetti** et **Strava** → courbes Allure ↔ Pente, temps de passage, allures lissées.
- **Mode Empirique (nouveau)** : l’app extrait tes vitesses, fait un **binning par pente**, calcule les **médianes**, **rejoue la trace** pour obtenir un **temps total reconstruit** et une **courbe Allure ↔ Pente** (médiane + lissage).

## 🆚 Changements majeurs vs l’original

1. **Mode “Données réelles”**  
   - `build_empirical_curve_from_gpx` : courbe **vitesse(m/s) ↔ pente** (binning, médiane, IQR) depuis les timestamps du GPX.  
   - `compute_cumulative_time_empirical` : **rejeu de la trace** (mêmes distances/altitudes) en appliquant la courbe perso ; gestion des cas hors-domaine.  
   - `compute_paces_empirical` : table **allure par km** à partir du **temps cumulé** (interpolation propre).

2. **UX & affichages**  
   - **Récap enrichi** : **VAP** + **allure équivalente au plat** affichées clairement.  
   - **Sélecteur de mode** et explications (rappels Minetti/Strava) ; **sélection de segment** avec D+/D-, allure, temps estimé.  
   - **Lisibilité** : la **table d’allures par km** est masquée par **défaut** pour se concentrer sur les **courbes** et le **profil**.

3. **Outillage GAP Strava**  
   - `approximation_courbe_modele_strava.py` pour (re)fiter le polynôme Strava **hors production**.

## 📦 Installation

```bash
# Python 3.11 conseillé
python -m venv .venv
# Windows :
.venv\Scripts\activate
# macOS / Linux :
# source .venv/bin/activate
pip install -r requirements.txt
```

Dépendances clés : `streamlit`, `gpxpy`, `plotly`, `numpy`, `pandas`, `folium`, `streamlit_folium`.

## ▶️ Lancer l’application

```bash
streamlit run app.py
```

Ouvre ensuite l’UI, charge un GPX, puis choisis **Temps objectif (théorique)** ou **Temps du GPX (empirique)**.

## 🗂️ Arborescence & points d’entrée

- `app.py` — UI Streamlit (chargement GPX, graphes, carte, contrôles).  
- `utils.py` — cœur métier :  
  - GPX → `process_gpx`, `calculate_deniv`.  
  - Modèles → `minetti_cost_running`, `strava_cost`, `adjusted_speed_*`, `compute_cumulative_time[_strava]`, `compute_paces[_strava]`, `trouver_vitesse_plate[_strava]`.  
  - Empirique → `build_empirical_curve_from_gpx`, `compute_cumulative_time_empirical`, `compute_paces_empirical`.  
- `approximation_courbe_modele_strava.py` — recalibrage du polynôme Strava (outil).

## ⚠️ Limites & hypothèses

- **VAP constante** en mode théorique.  
- **Bruitage GPS** et **pauses** : filtrage simple (dt, dd, vitesse min).  
- **Descente rapide** plafonnée côté Minetti (réalisme mécanique).

## 🛣️ Roadmap (idées)

- Pondération par **temps passé** (vs distance) dans l’empirique.  
- Option de **lissage robuste** (médiane glissante).  
- **Comparaison de traces** (empirique vs empirique) sur un même graphique.  
- Export **CSV** des tables d’allures / temps de passage.

## 🤝 Remerciements

- Merci au projet amont **theotimroger/simulateurtrail** (référence Minetti/Strava).

## 📜 Licence

Ce fork respecte la **licence** du dépôt amont. Reporte-toi au fichier LICENSE de l’original.
