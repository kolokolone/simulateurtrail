import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from streamlit_plotly_events import plotly_events
import folium
from streamlit_folium import st_folium
from utils import (
    process_gpx,
    trouver_vitesse_plate,
    compute_cumulative_time,
    format_time,
    compute_paces,
    calculate_deniv,
    trouver_vitesse_plate_strava,
    compute_cumulative_time_strava,
    compute_paces_strava,
    vitesse_to_allure,
    adjusted_speed_minetti,
    adjusted_speed_strava,
    allure_to_v_asc,
    allure_to_seconds,
    smooth
)

st.title("Analyse de trace GPX - Allure ajustée à la pente")
st.info(
    """
    🏔️ **Bienvenue sur le simulateur d'allure ajustée à la pente !**

    Cette application vous permet d'analyser une trace GPX en tenant compte de votre **temps total espéré**,
 

    ➡️ L'algorithme calcule alors :
    - votre **allure ajustée à la pente**, c'est l'allure pour laquelle vous fournissez le même effort sur du plat.
    - votre **profil d'allure ajustée** tout au long du parcours,
    - et vos **temps de passage estimés** à chaque point.

    2 Modèles sont proposés ici: celui de Minetti et celui de Strava (détails en bas de page).

    **Chargez simplement votre fichier GPX et entrez votre objectif de temps.**
    """
)



uploaded_file = st.file_uploader("Chargez votre fichier GPX", type=["gpx"])

## -- AFFICHAGE DES INFOS RELATIVE AU PARCOURS --

if uploaded_file is not None:

    gpx_content = uploaded_file.read().decode("utf-8")

    # Lecture brute : distances et altitudes
    distances, elevations, distances_pace, coords = process_gpx(gpx_content)
    d_plus, d_moins = calculate_deniv(elevations)

    st.markdown("""
    <div style='background-color: rgba(255,0,0,0.25); padding: 0px; border-radius: 10px; margin-bottom: 0px;'>
        <div style='text-align: center; margin-bottom: 0px;'>
            <h4 style='margin: 0 0 0 0; color: rgba(255,0,0,1);'>🗺️ Résumé du parcours</h4>
        </div>
    """, unsafe_allow_html=True)
    # Créer la carte centrée sur le point moyen
    if coords:
        lat_moy = sum(lat for lat, lon in coords) / len(coords)
        lon_moy = sum(lon for lat, lon in coords) / len(coords)

        m = folium.Map(location=[lat_moy, lon_moy], zoom_start=13, tiles='OpenStreetMap')
        m.fit_bounds(coords)

        folium.PolyLine(coords, color="blue", weight=3).add_to(m)
        st_folium(m, width=700, height=500)
    else:
        st.warning("Impossible de récupérer les coordonnées GPS.")

    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<h4 style='text-align: center;'>Distance</h4>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: center; font-size: 16px; margin-bottom: 10;'>{distances[-1]:.1f} km</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<h4 style='text-align: center;'>D+</h4>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: center; font-size: 16px; margin-bottom: 10;'>{d_plus} m</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<h4 style='text-align: center;'>D-</h4>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: center; font-size: 16px; margin-bottom: 10;'>{d_moins} m</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # fermeture du cadre

## -----------------------------------------


## INPUT DU TEMPS ESPERE SUR LA COURSE
    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.markdown(
            """
            <div style="display: flex; align-items: center; justify-content: flex-end; height: 40px;">
                <p style="font-weight: bold; margin: 0;">Temps espéré (hh:mm:ss)</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

    with col_b:
        temps_espere = st.text_input("", value="06:15:30", label_visibility="collapsed")

## -----------------------------------------

## -- CALCUL DE L'ALLURE AU COURS DU TEMPS
if uploaded_file is not None and temps_espere:

    # Conversion temps espéré en secondes
    try:
        h, m, s = map(int, temps_espere.split(":"))
        temps_espere_sec = h * 3600 + m * 60 + s
    except ValueError:
        st.error("Format invalide. Utilisez hh:mm:ss")
        st.stop()


    ## ALLURE MOYENNE
    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.markdown(
            """
            <div style="display: flex; align-items: center; justify-content: flex-end; height: 40px;">
                <p style="font-weight: bold; margin: 0;">Allure moyenne</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

    allure_moy = vitesse_to_allure(distances[-1] * 1000 / temps_espere_sec)

    with col_b:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; justify-content: flex-start; height: 40px;">
                <p style="margin: 0;">{allure_moy}/km</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    
    # Trouver la vitesse sur plat correcte
    flat_speed = trouver_vitesse_plate(distances, elevations, temps_espere_sec)
    flat_speed_strava = trouver_vitesse_plate_strava(distances, elevations, temps_espere_sec)


    # Calcul de l'allure ajustée correspondante

    allure_plat_str = vitesse_to_allure(flat_speed)

    allure_plat_str_strava = vitesse_to_allure(flat_speed_strava)

    st.markdown("""
    <div style='background-color: rgba(0, 180, 0,0.25); padding: 0px; border-radius: 10px; margin-bottom: 0px;'>
        <div style='text-align: center; margin-bottom: 0px;'>
            <h4 style='margin: 0 0 0 0; color: rgba(0, 180, 0,1);'>Allure ajustée à la pente</h4>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h5 style='text-align: center;'>Modèle Minetti</h5>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: center; font-size: 16px;'>{allure_plat_str} min/km</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<h5 style='text-align: center;'>Modèle Strava</h5>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: center; font-size: 16px;'>{allure_plat_str_strava} min/km</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    ## -----------------------------------------




        
    ## GRAPHE DES ALLURES EN FONCTION DES PENTES

    pentes = list(range(-30, 35, 1))  # de -30% à +35% tous les 5%

        # Construire les listes d'allures
    allures_minetti = []
    allures_strava = []
    vitesses_asc_minetti = []
    vitesses_asc_strava = []

    for pente in pentes:
        v_minetti = adjusted_speed_minetti(flat_speed, pente)
        v_strava = adjusted_speed_strava(flat_speed, pente)
        allure_minetti = vitesse_to_allure(v_minetti)
        allure_strava = vitesse_to_allure(v_strava)
        v_asc_minetti = allure_to_v_asc(allure_minetti,pente)
        v_asc_strava = allure_to_v_asc(allure_strava,pente)
        vitesses_asc_minetti.append(round(v_asc_minetti))
        vitesses_asc_strava.append(round(v_asc_strava))
        allures_minetti.append(allure_minetti)
        allures_strava.append(allure_strava)

    y_minetti = [allure_to_seconds(a) for a in allures_minetti]
    y_strava = [allure_to_seconds(a) for a in allures_strava]

    # Fonction pour reformatter en mm:ss
    def seconds_to_mmss(x):
        m = int(x // 60)
        s = int(x % 60)
        return f"{m}:{s:02d}"

    # 📈 Tracer avec Plotly
    fig = go.Figure()

    # Courbe Minetti
    fig.add_trace(go.Scatter(
        x=pentes,
        y=y_minetti,
        mode='lines',
        name='Minetti',
        customdata=list(zip(allures_minetti, vitesses_asc_minetti)),
        hovertemplate=(
            '<span style="color:#1f77b4;">Allure: %{customdata[0]}/km, V verticale: %{customdata[1]} m/h</span><br><extra></extra>'
        ),
        line=dict(color='#1f77b4')
    ))

    # Courbe Strava
    fig.add_trace(go.Scatter(
        x=pentes,
        y=y_strava,
        mode='lines',
        name='Strava',
        customdata=list(zip(allures_strava, vitesses_asc_strava)),
        hovertemplate=(
            '<span style="color:orange;">Allure: %{customdata[0]}/km, V verticale: %{customdata[1]} m/h</span><br><extra></extra>'
        ),
        line=dict(color='orange', dash='dash'),
    ))

    # Définir les limites et ticks en allure (y)
    min_val = int(min(min(y_minetti), min(y_strava)) // 60) * 60
    max_val = int(max(max(y_minetti), max(y_strava)) // 60 + 2) * 60

    fig.update_layout(
        title="Allure ajustée en fonction de la pente",
        xaxis_title="Pente (%)",
        yaxis_title="Allure (min/km)",
        height=500,
        legend=dict(x=0.05, y=0.95),
        yaxis=dict(
            autorange='reversed',
            tickmode='array',
            tickvals=list(range(min_val, max_val + 1, 120)),
            ticktext=[seconds_to_mmss(v) for v in range(min_val, max_val + 1, 120)]
        ),
        hovermode="x",
        hoverdistance=100,
        spikedistance=1000,
        xaxis=dict(
            showspikes=True,
            spikecolor="grey",
            spikethickness=1,
            spikedash="dot",
            spikemode="across",
        ),
        dragmode=False,
    )
        

    # Streamlit affichage
    with st.expander("⌳ Afficher l'allure en fonction de la pente"):
        st.plotly_chart(fig, use_container_width=True)
    
    ## --------------------------------------------------------------


    ## GRAPHE ALTIMETRIE ET TEMPS DE PASSAGE

    # Recalcul du temps cumulé avec la bonne vitesse
    cumulative_time = compute_cumulative_time(flat_speed, distances, elevations)
    cumulative_time_strava = compute_cumulative_time_strava(flat_speed_strava, distances, elevations)
    
    ## PROFIL ALTIMETRIQUE ET DUREE DES SEGMENTS

    with st.expander("🏔️ Afficher le profil altimétrique et les temps de passage estimés"):

        # Affichage du graphe comme avant
        fig = go.Figure()

        # Profil Altitude
        fig.add_trace(go.Scatter(
            x=distances_pace,
            y=elevations[1:],  # pour correspondre aux distances_pace
            mode='lines',
            name='Altitude',
            hovertemplate=(
                'Distance: %{x:.2f} km<br>'
                'Altitude: %{y:.0f} m<br>'
                '<span style="color:#1f77b4;">Minetti: %{customdata[0]}</span><br>'
                '<span style="color:orange;">Strava: %{customdata[1]}</span>'
            ),
            customdata=[[format_time(t), format_time(ts)] for t, ts in zip(cumulative_time, cumulative_time_strava)]
        ))

        # Configuration
        fig.update_layout(
            title="Profil Altimétrique et Temps de passage estimé",
            xaxis=dict(
                title='Distance (km)',
                showspikes=True,
                spikecolor="grey",
                spikethickness=1,
                spikedash="dot",
                spikemode="across"
            ),
            yaxis=dict(title='Altitude (m)', side='left'),
            legend=dict(x=0, y=1),
            height=300,
            hoverdistance=100,
            spikedistance=1000,
            hovermode = 'x',
            dragmode=False  # activation de la sélection
        )
        st.plotly_chart(fig, use_container_width=True)

        # Sélection avec slider
        st.markdown("### 📍 Sélectionner un segment")
        start_km, end_km = st.slider(
            "Choisissez deux distances (en km)",
            min_value=0.0,
            max_value=float(distances[-1]),
            value=(0.0, float(distances[-1])),
            step=0.1
        )

        if start_km >= end_km:
            st.warning("⚠️ La distance d'arrivée doit être supérieure à la distance de départ.")
        else:
            indices = [i for i, d in enumerate(distances_pace) if start_km <= d <= end_km]

            if len(indices) >= 2:
                # Distance, temps, vitesse, allure
                d_total = distances_pace[indices[-1]] - distances_pace[indices[0]]
                t_total = cumulative_time[indices[-1]] - cumulative_time[indices[0]]
                v_moy = (d_total * 1000) / t_total  # m/s
                allure_moyenne = vitesse_to_allure(v_moy)

                t_total_strava = cumulative_time_strava[indices[-1]] - cumulative_time_strava[indices[0]]
                v_moy_strava = (d_total * 1000) / t_total_strava
                allure_moyenne_strava = vitesse_to_allure(v_moy_strava)

                # D+ / D- sur le segment
                sub_elevations = elevations[indices[0]+1 : indices[-1]+2]  # +1 car distances_pace démarre à point 1
                d_plus_seg, d_moins_seg = calculate_deniv(sub_elevations)

                st.success(f"📏 Distance : {d_total:.2f} km  •  🧗 D+ : {d_plus_seg} m  •  ⬇️ D- : {abs(d_moins_seg)} m")
                st.info(f"Minetti - ⏱ Durée estimée : {format_time(t_total)} • 🏃 Allure : {allure_moyenne}/km")
                st.warning(f"Strava  - ⏱ Durée estimée : {format_time(t_total_strava)} • 🏃 Allure : {allure_moyenne_strava}/km")
            else:
                st.warning("Pas assez de points pour faire le calcul.")




    ## ALLURE INSTANTANEE
    # L'allure instantanée donne un résultats  illisisible puisque changeant avec la moindre variation de pente

    paces = compute_paces(distances, elevations, flat_speed)
    paces_strava = compute_paces_strava(distances, elevations, flat_speed_strava)
    paces = smooth(paces,4)
    paces_strava = smooth(paces_strava,4)
    paces_str = []
    paces_str_strava = []
    for i in range(len(paces)):
        paces_str.append(vitesse_to_allure(1000/(60*paces[i])))
        paces_str_strava.append(vitesse_to_allure(1000/(60*paces_strava[i])))


    fig2 = go.Figure()

        # 1. Courbe d’élévation en gris clair, remplie depuis la base, légèrement transparente (axe secondaire)
    fig2.add_trace(go.Scatter(
        x=distances_pace,
        y=elevations,
        mode='lines',
        name='Élévation',
        line=dict(color='lightgray'),
        fill='tozeroy',
        yaxis='y2',
        hoverinfo='skip',
        opacity=0.4,
        hovertemplate='Distance: %{x:.2f} km<br>Élévation: %{y:.0f} m'
    ))

    # 2. Courbe Minetti (bleu)
    fig2.add_trace(go.Scatter(
        x=distances_pace,
        y=paces,
        mode='lines',
        name='Allure Minetti',
        line=dict(color='#1f77b4'),
        customdata=paces_str,
        hovertemplate='Distance: %{x:.2f} km<br>Allure Minetti: %{customdata}/km'
    ))

    # 3. Courbe Strava (orange)
    fig2.add_trace(go.Scatter(
        x=distances_pace,
        y=paces_strava,
        mode='lines',
        name='Allure Strava',
        line=dict(color='orange', dash='dash'),
        customdata=paces_str_strava,
        hovertemplate='Distance: %{x:.2f} km<br>Allure Strava: %{customdata}/km'
    ))



    # Configuration générale
    fig2.update_layout(
        title="Comparaison des Allures Instantanées",
        xaxis=dict(
            title='Distance (km)',
            showspikes=True,
            spikecolor="grey",
            spikethickness=1,
            spikedash="dot",
            spikemode="across",
        ),
        yaxis=dict(
            title='Allure (mm:ss/km)',
            side='left',
            autorange='reversed'
        ),
        yaxis2=dict(
            title='Élévation (m)',
            side='right',
            overlaying='y',
            showgrid=False
        ),

        legend=dict(x=0, y=1),
        height=400,
        dragmode=False,
        hovermode='x',
        hoverdistance=100,
        spikedistance=1000,
    )




    with st.expander("🏃‍♂️ Afficher l'allure instantanée"):
        st.plotly_chart(fig2, use_container_width=True)





with st.expander("⚙️ Voir explication du calcul"):
    st.markdown(
        """
        ### Comment fonctionne ce simulateur ?

        L'algorithme repose sur **le modèle biomécanique de Minetti**<sup>1</sup>, qui estime le **coût énergétique** de la course à pied en fonction de la pente, et **le modèle fréquence cardiaque identique de Strava**<sup>2</sup>, qui estime l'allure ajustée en fonction de la fréquence cardiaque.
        Pour les calculs, on suppose que la VAP est constante.

        ➡️ **Sur terrain plat**, le coût énergétique est minimal.\n
        ➡️ **En montée**, le coût énergétique augmente (on dépense plus pour s'élever), la fréquence cardiaque aussi.\n
        ➡️ **En descente**, le coût diminue... mais on ne peut pas courir infiniment vite sans limite physique.

        ### Calculs effectués :

        - L'algorithme cherche une **vitesse ajustée à la pente (VAP)**<sup>3</sup> qui vous permettrait d'atteindre ce temps,
          tout en tenant compte :
            - du profil de pente de votre trace GPX,
            - du temps total espéré

        - Pour chaque segment du parcours :
            - Le coût (énergétique ou en fréquence cardiaque) est calculé selon la pente locale,
            - La vitesse instantanée est adaptée en fonction de ce coût,
            - En descente, la vitesse est plafonnée à 1.3 x VAP pour le modèle de Minetti<sup>4</sup>.

        - Différence entre les 2 modèles :
            - Le modèle Minetti se base sur le coût énergétique et les tests ont été réalisés en laboratoire. Il est très généreux sur la vitesse en descente.
            - Le modèle de Strava se base sur sa base de données d'activités de traileurs du monde entier. L'estimation est donc issue d'activités en conditions "réelles". Ce modèle reflète sûrement mieux l'aspect technique des descentes.

        - Remarques et Conseils:
            - L'algorithme n'effectue que des calculs et ne prend pas en compte votre état de fatigue au cours de la course.
            - Utilisez les valeurs en sortie comme indicateurs mais pas comme l'unique possibilité.
            - Ajustez les allures proposées en fonction de vos forces et faiblesse. A ce sujet, et vues les allures proposées, le modèle de Minetti semble plus adapté à des bons descendeurs.
        <hr>
        <p style="font-size: 0.8em;">
        <sup>1</sup> Minetti AE, Moia C, Roi GS, Susta D, Ferretti G. (2002), *Energy cost of walking and running at extreme uphill and downhill slopes*, Journal of Applied Physiology.<br>
        <sup>2</sup> <a href="https://medium.com/strava-engineering/an-improved-gap-model-8b07ae8886c3" target="_blank">Source Strava Engineering</a><br>
        <sup>3</sup> Pour les utilisateurs de Strava, cette allure est également disponible dans les rapports d'activité.<br>
        <sup>4</sup> Le modèle de Minetti est basé uniquement sur la dépense énergétique. Selon ce modèle, courir à un peu plus de 20 km/h sur une pente de -15% correspond à environ 11km/h sur du plat. Cette limitation n'a été validée par aucune expérience et peut être discutée. J'ai essayée ici de prendre en compte la technicité des chemins et les limites biomécaniques des traileurs.
        </p>
        <hr>
        """,
        unsafe_allow_html=True
    )

