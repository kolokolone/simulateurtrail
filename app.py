# app.py — choix entre calcul théorique (Minetti/Strava) et données réelles (empirique)
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import gpxpy
import folium
from streamlit_folium import st_folium

from utils import (
    # Lecture et prépa trace
    process_gpx,                    # -> distances(km), elevations(m), distances_pace(km), coords[(lat, lon)]

    # Modèles théoriques
    trouver_vitesse_plate,
    trouver_vitesse_plate_strava,
    adjusted_speed_minetti,
    adjusted_speed_strava,
    compute_cumulative_time,
    compute_cumulative_time_strava,
    compute_paces,                  # (si tu l'utilises ailleurs)
    compute_paces_strava,           # (si tu l'utilises ailleurs)

    # Outils d'affichage / conversions
    vitesse_to_allure,
    allure_to_v_asc,
    allure_to_seconds,
    format_time,

    # Données réelles (empirique)
    build_empirical_curve_from_gpx,
    compute_cumulative_time_empirical,
    compute_paces_empirical,
)

# -----------------------
# Mise en page & entête
# -----------------------
st.set_page_config(page_title="Simulateur Trail — Allure ↔ Pente", layout="centered")
st.title("Analyse de trace GPX — Allure ajustée à la pente")
st.info(
    """
    🏔️ **Bienvenue !**  
    Fork de **theotimroger/simulateurtrail** — même esprit, deux chemins de calcul :

    **Ce qui reste (hérité de l’original)**  
    - Saisie d’un **temps total objectif** et calibration de la **VAP**.  
    - Simulation avec les modèles **Minetti** et **Strava** pour obtenir profils et temps de passage.

    **Ce que ce fork ajoute / change**  
    - Nouveau mode **« Données réelles »** : on **extrait tes vitesses observées** du GPX, on construit **ta courbe perso Allure ↔ Pente**, puis on **rejoue la trace** pour reconstruire le temps total.  
    - **Sélecteur de mode** : *Temps objectif (théorique)* **ou** *Temps du GPX (empirique)*.  
    - Récap enrichi : affichage direct **VAP** *et* **allure équivalente au plat**.  
    - UI allégée : la **table d’allures par km** est masquée par défaut pour se concentrer sur les **courbes Allure ↔ Pente** et le **profil**.

    👉 En bref : **prédire** (théorique) ou **expliquer** (empirique) à partir de ta trace — au choix.
    """
)


# -----------------------
# Utilitaire: durée GPX
# -----------------------
def gpx_duration_seconds(gpx_text: str):
    """Durée brute issue des timestamps du fichier GPX (en secondes)."""
    try:
        gpx = gpxpy.parse(gpx_text)
    except Exception:
        return None
    t0, t1 = None, None
    for trk in gpx.tracks:
        for seg in trk.segments:
            for pt in seg.points:
                if getattr(pt, "time", None):
                    if t0 is None:
                        t0 = pt.time
                    t1 = pt.time
    if t0 and t1:
        return int((t1 - t0).total_seconds())
    return None

def _seconds_to_mmss(x):
    m = int(x // 60)
    s = int(x % 60)
    return f"{m}:{s:02d}"

def _moving_average(arr, window=5):
    """Moyenne mobile simple pour lisser une série."""
    window = max(1, int(window))
    if window == 1 or len(arr) <= 2:
        return arr
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(arr, kernel, mode="same")

# -----------------------
# Upload GPX
# -----------------------
uploaded = st.file_uploader("Charge un fichier GPX", type=["gpx"])

if uploaded is not None:
    gpx_text = uploaded.read().decode("utf-8")

    # 1) Lecture GPX -> distances (km), elevations (m), coords
    distances_km, elevations_m, distances_pace_km, coords = process_gpx(gpx_text)  # utils.py OK
    distances_m = np.array(distances_km, dtype=float) * 1000.0

    # 2) Carte Folium
    with st.expander("🗺️ Carte du parcours", expanded=True):
        if coords:
            lat_moy = sum(lat for lat, _ in coords) / len(coords)
            lon_moy = sum(lon for _, lon in coords) / len(coords)
            m = folium.Map(location=[lat_moy, lon_moy], zoom_start=13, tiles="OpenStreetMap")
            m.fit_bounds(coords)
            folium.PolyLine(coords, color="blue", weight=3).add_to(m)
            st_folium(m, width=None, height=460)
        else:
            st.warning("Aucune coordonnée trouvée pour afficher la carte.")

    # 3) Sélecteur réutilisé : temps objectif = MODE THÉORIQUE, temps GPX = MODE EMPIRIQUE
    mode = st.radio(
        "Choisis le mode de calcul du temps :",
        ["Saisir un temps objectif (hh:mm:ss)", "Utiliser le temps du GPX"],
        index=0,
        horizontal=True,
    )

    # =========================
    # MODE THÉORIQUE (Minetti / Strava)
    # =========================
    if mode == "Saisir un temps objectif (hh:mm:ss)":
        temps_str = st.text_input("Temps objectif (hh:mm:ss)", value="03:30:00")
        modele = st.radio("Modèle théorique :", ("Minetti", "Strava"), index=0, horizontal=True)

        if st.button("Calculer (modèle théorique)"):
            # Parse temps
            try:
                h, m, s = map(int, temps_str.split(":"))
                temps_espere_sec = h * 3600 + m * 60 + s
                assert temps_espere_sec > 0
            except Exception:
                st.error("Format invalide. Utilise hh:mm:ss, ex. 03:30:00.")
                st.stop()

            # Calibrage VAP (vitesse équivalente plat) selon le modèle choisi
            if modele == "Minetti":
                flat_speed = trouver_vitesse_plate(distances_km, elevations_m, temps_espere_sec)  # m/s
                t_cum = compute_cumulative_time(flat_speed, distances_km, elevations_m)          # s
            else:
                flat_speed = trouver_vitesse_plate_strava(distances_km, elevations_m, temps_espere_sec)  # m/s
                t_cum = compute_cumulative_time_strava(flat_speed, distances_km, elevations_m)          # s

            # ➜ Ajout de l'allure (min/km) dérivée de la VAP
            allure_flat = vitesse_to_allure(flat_speed)
            st.success(
                f"Temps total estimé : {format_time(t_cum[-1])} — VAP: {flat_speed*3.6:.2f} km/h — Allure: {allure_flat}/km"
            )

            # ⬇️ Tableau d’allures par km retiré en mode théorique (demande utilisateur)
            # paces_km_df = compute_paces_empirical(distances_m, np.array(t_cum, dtype=float))
            # st.dataframe(paces_km_df, use_container_width=True)

            # Courbe Allure vs Pente (on trace les deux modèles pour comparer)
            pentes = list(range(-30, 35, 1))  # -30% → +34%
            allures_minetti, allures_strava, vasc_minetti, vasc_strava = [], [], [], []

            flat_speed_minetti = trouver_vitesse_plate(distances_km, elevations_m, temps_espere_sec)
            flat_speed_strava  = trouver_vitesse_plate_strava(distances_km, elevations_m, temps_espere_sec)

            for pente in pentes:
                v_m = adjusted_speed_minetti(flat_speed_minetti, pente)
                v_s = adjusted_speed_strava(flat_speed_strava, pente)
                a_m = vitesse_to_allure(v_m)
                a_s = vitesse_to_allure(v_s)
                allures_minetti.append(a_m); allures_strava.append(a_s)
                vasc_minetti.append(round(allure_to_v_asc(a_m, pente)))
                vasc_strava.append(round(allure_to_v_asc(a_s, pente)))

            y_minetti = [allure_to_seconds(a) for a in allures_minetti]
            y_strava  = [allure_to_seconds(a) for a in allures_strava]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pentes, y=y_minetti, mode="lines", name="Minetti",
                customdata=list(zip(allures_minetti, vasc_minetti)),
                hovertemplate="Pente: %{x}%<br>Allure: %{customdata[0]}/km<br>V. verticale: %{customdata[1]} m/h<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=pentes, y=y_strava, mode="lines", name="Strava",
                customdata=list(zip(allures_strava, vasc_strava)),
                hovertemplate="Pente: %{x}%<br>Allure: %{customdata[0]}/km<br>V. verticale: %{customdata[1]} m/h<extra></extra>",
                line=dict(dash="dash"),
            ))

            ymin = int(min(min(y_minetti), min(y_strava)) // 60) * 60
            ymax = int(max(max(y_minetti), max(y_strava)) // 60 + 2) * 60
            fig.update_layout(
                title="Allure (théorique) en fonction de la pente",
                xaxis_title="Pente (%)",
                yaxis_title="Allure (min/km)",
                height=520,
                legend=dict(x=0.02, y=0.98),
                hovermode="x",
                yaxis=dict(
                    autorange="reversed",
                    tickmode="array",
                    tickvals=list(range(ymin, ymax + 1, 120)),
                    ticktext=[_seconds_to_mmss(v) for v in range(ymin, ymax + 1, 120)],
                ),
            )
            with st.expander("⌳ Afficher l'allure (théorique) en fonction de la pente", expanded=True):
                st.plotly_chart(fig, use_container_width=True)

    # =========================
    # MODE DONNÉES RÉELLES (empirique sur GPX)
    # =========================
    else:
        st.caption("Mode « Données réelles » : on exploite tes vitesses observées vs pente pour construire ta courbe perso.")
        if st.button("Analyser mes données réelles"):
            # Construit la courbe empirique (vitesse vs pente) à partir des points horodatés du GPX
            curve_df, t_obs = build_empirical_curve_from_gpx(gpx_text)
            if curve_df.empty:
                st.error("Impossible de construire la courbe empirique (timestamps ou élévations insuffisants). Saisis un temps objectif pour utiliser le mode théorique.")
                st.stop()

            # Rejoue la trace (mêmes distances/altitudes) avec ta courbe empirique
            t_cum_emp = compute_cumulative_time_empirical(distances_m, np.array(elevations_m, dtype=float), curve_df)
            if t_cum_emp.size == 0:
                st.error("Échec du recalcul empirique sur cette trace (données insuffisantes).")
                st.stop()

            # Affichage des temps
            gpx_dur = gpx_duration_seconds(gpx_text)
            col1, col2 = st.columns(2)
            with col1:
                if gpx_dur:
                    st.success(f"Temps total observé (timestamps GPX) : {format_time(gpx_dur)}")
                else:
                    st.info("Aucun timestamp exploitable dans le GPX.")
            with col2:
                st.success(f"Temps total reconstruit (courbe empirique) : {format_time(float(t_cum_emp[-1]))}")

            # ⬇️ Note courte sous les deux temps (explications simplifiées)
            st.caption(
                "ℹ️ **Rappel rapide** — "
                "**Théorique** : `trouver_vitesse_plate[_strava]` → `compute_cumulative_time[_strava]` → allures/km (interpolation). "
                "**Empirique** : `build_empirical_curve_from_gpx` → `compute_cumulative_time_empirical` sur la **même trace** pour fiabiliser l’interpolation aux km."
            )

            # (Demandé) NE PAS afficher le tableau d’allures par km en mode empirique
            # paces_emp = compute_paces_empirical(distances_m, t_cum_emp)
            # st.dataframe(paces_emp, use_container_width=True)

            # Courbe Allure ↔ Pente (empirique) + courbe lissée
            x_pct = curve_df["slope_bin"].values * 100.0
            y_min_per_km = 1000.0 / (curve_df["speed_mps_median"].values * 60.0)

            # Lissage (moyenne mobile)
            n = max(5, int(round(len(y_min_per_km) * 0.05)))
            if n % 2 == 0:
                n += 1
            y_smooth = _moving_average(y_min_per_km, window=n)

            fig2 = go.Figure()
            fig2.add_scatter(
                x=x_pct,
                y=y_min_per_km,
                mode="lines+markers",
                name="Allure empirique (médiane)",
                hovertemplate="Pente: %{x:.1f}%<br>Allure médiane: %{y:.2f} min/km<extra></extra>"
            )
            fig2.add_scatter(
                x=x_pct,
                y=y_smooth,
                mode="lines",
                name=f"Lissé (moyenne mobile, fenêtre={n})",
                hovertemplate="Pente: %{x:.1f}%<br>Allure lissée: %{y:.2f} min/km<extra></extra>"
            )
            fig2.update_layout(
                title="Relation empirique Allure ↔ Pente (médiane & lissée)",
                xaxis_title="Pente (%)",
                yaxis_title="Allure (min/km)",
                height=520,
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig2, use_container_width=True)
