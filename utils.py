import gpxpy
import numpy as np



# --- Fonctions ---
def format_time(seconds):
    """Formate un temps en secondes vers hh:mm:ss."""
    if seconds is None:
        return "-"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def vitesse_to_allure(vitesse):
    if vitesse == 0:
        return "∞"
    sec_per_km = 1000 / vitesse
    sec_per_km = round(sec_per_km)  # ARRONDIR AVANT de séparer en minutes + secondes
    minutes = int(sec_per_km // 60)
    seconds = int(sec_per_km % 60)
    return f"{minutes:02d}:{seconds:02d}"

def allure_to_seconds(allure_str):
    minutes, seconds = map(int, allure_str.split(":"))
    return minutes * 60 + seconds

def parse_allure(allure_str):
    minutes, seconds = map(int, allure_str.split(":"))
    total_seconds = minutes * 60 + seconds
    return 1000 / total_seconds  # vitesse en m/s

def minetti_cost_running(i):
    a, b, c, d, e, f = 155.4, -30.4, -43.3, 46.3, 19.5, 3.6
    return a * i**5 + b * i**4 + c*i**3 + d*i**2 + e*i + f

def strava_cost(i):
    a, b, c, d = -3.32959069, 14.61846764, 3.07428877, 1.03357331
    return a * i**3 + b * i**2 + c*i + d

def allure_to_v_asc(allure,pente):
    """calcul de la vitesse verticale en fonction de l'allure et de la pente
        INPUT:
        - allure (str) au format mm:ss
        - pente (float) en m/100m (%)
        
        OUTPUT:
        - v_asc en m/h
    """
    m, s = map(int, allure.split(":"))
    allure_sec = m * 60 + s
    km_par_heure = 3600/allure_sec
    v_asc = km_par_heure*pente*10
    return v_asc

def adjusted_speed_minetti(flat_speed, slope):
    i = slope / 100
    C0 = minetti_cost_running(0)
    Ci = minetti_cost_running(i)
    v = min(1.3*flat_speed,flat_speed * (C0 / Ci)) #vitesse ajustée avec max car difficle d'aller bcp plus vite en descente. Sans cette limite, un coureur avec une VAP de 10,6km/h descdndrait à plus de 20km/h pour des pentes entre -13 et -21%
    return v

def adjusted_speed_strava(flat_speed, slope):
    i = slope/100
    C0 = strava_cost(0)
    Ci = strava_cost(i)
    v = flat_speed*(C0/Ci)
    return v


def smooth(y, box_pts=5):
    y = np.array(y)
    n = len(y)
    y_smooth = np.copy(y)
    half_window = box_pts // 2
    for i in range(n):
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        y_smooth[i] = np.mean(y[start:end])
    return y_smooth


def simulate_temps_total(flat_speed, distances, elevations):
    """
    Calcule le temps total estimé sur un parcours pour une vitesse sur plat donnée,
    en utilisant les vraies distances entre points successifs.

    Args:
        flat_speed (float): Vitesse sur plat en m/s.
        distances (list of float): Distances cumulées en km (vrai chemin parcouru).
        elevations (list of float): Altitudes correspondantes en mètres.

    Returns:
        float: Temps total estimé en secondes.
    """
    total_time = 0
    for i in range(1, len(distances)):
        d = (distances[i] - distances[i-1]) * 1000  # mètres
        dz = elevations[i] - elevations[i-1]
        slope = (dz / d) * 100 if d != 0 else 0
        v_adj = adjusted_speed_minetti(flat_speed, slope)
        if v_adj != 0:
            total_time += d / v_adj
    return total_time



def trouver_vitesse_plate(distances, elevations, temps_espere_sec, precision=1):
    """
    Trouve la vitesse sur plat équivalente pour correspondre au temps espéré
    en utilisant une recherche par dichotomie.

    Args:
        distances_pace (list of float): Distances intermédiaires en km.
        elevations (list of float): Altitudes correspondantes en mètres.
        temps_espere_sec (float): Temps total espéré en secondes.
        precision (float): Précision souhaitée en secondes pour l'approximation.

    Returns:
        float: Vitesse sur plat en m/s.
    """
    v_min = 1.0  # m/s
    v_max = 6.0  # m/s
    iteration = 0

    while v_max - v_min > 0.0001:
        iteration += 1
        v_mid = (v_min + v_max) / 2
        temps_mid = simulate_temps_total(v_mid, distances, elevations)

        if abs(temps_mid - temps_espere_sec) < precision:
            return v_mid  # Vitesse trouvée !

        if temps_mid > temps_espere_sec:
            v_min = v_mid  # Temps trop long -> aller plus vite
        else:
            v_max = v_mid  # Temps trop court -> aller moins vite

        if iteration > 100:
            break

    return (v_min + v_max) / 2

def compute_cumulative_time(flat_speed, distances, elevations):
    """
    Calcule cumulative_time proprement à partir de distances et elevations.

    Args:
        flat_speed (float): vitesse sur plat m/s
        distances (list of float): distances cumulées (km)
        elevations (list of float): altitudes (m)

    Returns:
        list of float: temps cumulé en secondes pour chaque point
    """
    cumulative_time = [0.0]

    for i in range(1, len(distances)):
        d = (distances[i] - distances[i-1]) * 1000  # mètres
        dz = elevations[i] - elevations[i-1]
        slope = (dz / d) * 100 if d != 0 else 0
        v_adj = adjusted_speed_minetti(flat_speed, slope)

        if v_adj > 0:
            next_time = cumulative_time[-1] + d / v_adj
        else:
            next_time = cumulative_time[-1]

        cumulative_time.append(next_time)

    return cumulative_time

def compute_paces(distances, elevations, flat_speed):
    """
    Calcule l'allure ajustée (min/km) pour chaque segment du parcours.

    Args:
        distances (list of float): Distances cumulées en km.
        elevations (list of float): Altitudes correspondantes en mètres.
        flat_speed (float): Vitesse sur plat en m/s.

    Returns:
        list of float: Allures ajustées (min/km) pour chaque segment.
    """
    paces = []
    for i in range(1, len(distances)):
        d = (distances[i] - distances[i-1]) * 1000  # mètres
        dz = elevations[i] - elevations[i-1]
        slope = (dz / d) * 100 if d != 0 else 0

        v_adj = adjusted_speed_minetti(flat_speed, slope)

        if v_adj > 0:
            pace = (1000 / v_adj) / 60  # min/km
        else:
            pace = None  # ou pace = valeur maximale comme 70 min/km si tu préfères

        paces.append(pace)
    return paces





## MEMES FONCTIONS MAIS AVEC LE MODELE STRAVA



def simulate_temps_total_strava(flat_speed, distances, elevations):
    """
    Calcule le temps total estimé sur un parcours pour une vitesse sur plat donnée,
    en utilisant les vraies distances entre points successifs.

    Args:
        flat_speed (float): Vitesse sur plat en m/s.
        distances (list of float): Distances cumulées en km (vrai chemin parcouru).
        elevations (list of float): Altitudes correspondantes en mètres.

    Returns:
        float: Temps total estimé en secondes.
    """
    total_time = 0
    for i in range(1, len(distances)):
        d = (distances[i] - distances[i-1]) * 1000  # mètres
        dz = elevations[i] - elevations[i-1]
        slope = (dz / d) * 100 if d != 0 else 0
        v_adj = adjusted_speed_strava(flat_speed, slope)
        if v_adj != 0:
            total_time += d / v_adj
    return total_time



def trouver_vitesse_plate_strava(distances, elevations, temps_espere_sec, precision=1):
    """
    Trouve la vitesse sur plat équivalente pour correspondre au temps espéré
    en utilisant une recherche par dichotomie.

    Args:
        distances_pace (list of float): Distances intermédiaires en km.
        elevations (list of float): Altitudes correspondantes en mètres.
        temps_espere_sec (float): Temps total espéré en secondes.
        precision (float): Précision souhaitée en secondes pour l'approximation.

    Returns:
        float: Vitesse sur plat en m/s.
    """
    v_min = 1.0  # m/s
    v_max = 6.0  # m/s
    iteration = 0

    while v_max - v_min > 0.0001:
        iteration += 1
        v_mid = (v_min + v_max) / 2
        temps_mid = simulate_temps_total_strava(v_mid, distances, elevations)

        if abs(temps_mid - temps_espere_sec) < precision:
            return v_mid  # Vitesse trouvée !

        if temps_mid > temps_espere_sec:
            v_min = v_mid  # Temps trop long -> aller plus vite
        else:
            v_max = v_mid  # Temps trop court -> aller moins vite

        if iteration > 100:
            break

    return (v_min + v_max) / 2

def compute_cumulative_time_strava(flat_speed, distances, elevations):
    """
    Calcule cumulative_time proprement à partir de distances et elevations.

    Args:
        flat_speed (float): vitesse sur plat m/s
        distances (list of float): distances cumulées (km)
        elevations (list of float): altitudes (m)

    Returns:
        list of float: temps cumulé en secondes pour chaque point
    """
    cumulative_time = [0.0]

    for i in range(1, len(distances)):
        d = (distances[i] - distances[i-1]) * 1000  # mètres
        dz = elevations[i] - elevations[i-1]
        slope = (dz / d) * 100 if d != 0 else 0
        v_adj = adjusted_speed_strava(flat_speed, slope)

        if v_adj > 0:
            next_time = cumulative_time[-1] + d / v_adj
        else:
            next_time = cumulative_time[-1]

        cumulative_time.append(next_time)

    return cumulative_time

def compute_paces_strava(distances, elevations, flat_speed):
    """
    Calcule l'allure ajustée (min/km) pour chaque segment du parcours.

    Args:
        distances (list of float): Distances cumulées en km.
        elevations (list of float): Altitudes correspondantes en mètres.
        flat_speed (float): Vitesse sur plat en m/s.

    Returns:
        list of float: Allures ajustées (min/km) pour chaque segment.
    """
    paces = []
    for i in range(1, len(distances)):
        d = (distances[i] - distances[i-1]) * 1000  # mètres
        dz = elevations[i] - elevations[i-1]
        slope = (dz / d) * 100 if d != 0 else 0

        v_adj = adjusted_speed_strava(flat_speed, slope)

        if v_adj > 0:
            pace = (1000 / v_adj) / 60  # min/km
        else:
            pace = None  # ou pace = valeur maximale comme 70 min/km si tu préfères

        paces.append(pace)
    return paces


##-------------------------------------------

def process_gpx(gpx_content):
    """Lis le fichier GPX et retourne distances, elevations, etc."""
    gpx = gpxpy.parse(gpx_content)

    last_point = None
    total_distance = 0
    distances = []
    elevations = []
    coords = []

    DISTANCE_MIN = 30  # mètres entre 2 points retenus, pour 
    distance_since_last_save = 0

    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                coords.append((point.latitude, point.longitude))
                if last_point is not None:
                    d = point.distance_3d(last_point) or 0
                    total_distance += d
                    distance_since_last_save += d

                    if distance_since_last_save >= DISTANCE_MIN:
                        distances.append(total_distance / 1000)  # en km
                        elevations.append(point.elevation)
                        distance_since_last_save = 0  # reset après avoir sauvé
                else:
                    # Sauvegarder tout premier point
                    distances.append(total_distance / 1000)
                    elevations.append(point.elevation)

                last_point = point

    distances_pace = [(distances[i] + distances[i-1]) / 2 for i in range(1, len(distances))]

    return distances, elevations, distances_pace, coords

def calculate_deniv(elevations):
    d_plus = [0]
    d_moins = [0]
    for i in range(1,len(elevations)):
        deniv_segment = elevations[i]-elevations[i-1]
        if deniv_segment > 0:
            d_plus.append(d_plus[-1]+deniv_segment)
            d_moins.append(d_moins[-1])
        else:
            d_plus.append(d_plus[-1])
            d_moins.append(d_moins[-1]+deniv_segment)
    return round(d_plus[-1]), round(d_moins[-1])
