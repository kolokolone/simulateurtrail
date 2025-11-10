import gpxpy
import numpy as np
import datetime as _dt
import pandas as _pd

# -----------------------------
# Format & conversions d'allure
# -----------------------------
def format_time(seconds):
    """Formate un temps (s) vers hh:mm:ss."""
    if seconds is None:
        return "-"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def vitesse_to_allure(vitesse):
    """
    Convertit une vitesse (m/s) en allure 'm:ss/km'.
    Pour v = 0 → '∞'.
    """
    if vitesse is None or vitesse <= 0:
        return "∞"
    sec_per_km = 1000.0 / vitesse  # s / km
    m = int(sec_per_km // 60)
    s = int(round(sec_per_km % 60))
    if s == 60:
        m += 1
        s = 0
    return f"{m}:{s:02d}"


def allure_to_seconds(allure):
    """
    Convertit une allure 'm:ss' (ou 'mm:ss') en secondes par km (int).
    Tolère 'm:s', 'mm:ss' et espace.
    """
    if allure is None:
        return None
    txt = str(allure).strip()
    if ":" not in txt:
        # on tolère mss → "5'00" sans deux-points, mais ici on reste strict
        raise ValueError("Format d'allure attendu: mm:ss")
    parts = txt.split(":")
    if len(parts) != 2:
        raise ValueError("Format d'allure attendu: mm:ss")
    m = int(parts[0])
    s = int(parts[1])
    if m < 0 or s < 0 or s >= 60:
        raise ValueError("Allure invalide")
    return m * 60 + s


def parse_allure(allure):
    """
    Compatibilité: renvoie les secondes par km (comme allure_to_seconds).
    Gardée pour compat avec d'anciens appels éventuels.
    """
    return allure_to_seconds(allure)


def allure_to_v_asc(allure, pente):
    """
    Convertit une allure 'm:ss' et une pente (%) en vitesse verticale (m/h).
    Hypothèse: v_kmh = 3600 / allure_sec → v_vert (m/h) = v_kmh * pente(%) * 10
    """
    if allure is None:
        return 0.0
    m, s = map(int, str(allure).split(":"))
    allure_sec = m * 60 + s
    if allure_sec <= 0:
        return 0.0
    km_par_heure = 3600.0 / allure_sec
    v_asc = km_par_heure * float(pente) * 10.0
    return v_asc


# --------------------------
# Coûts énergétiques (modèles)
# --------------------------
def minetti_cost_running(i):
    """
    Coût de transport (running) selon Minetti (polynôme 5e degré).
    i = pente en ratio (ex: +10% -> 0.10 ; -6% -> -0.06).
    Retourne un coût relatif (unité arbitraire).
    """
    a, b, c, d, e, f = 155.4, -30.4, -43.3, 46.3, 19.5, 3.6
    return a * i**5 + b * i**4 + c * i**3 + d * i**2 + e * i + f


def strava_cost(i):
    """
    Coût 'Strava-like' approximé (polynôme 3e degré) issu d'un fit.
    i = pente en ratio. Modèle valide sur ~[-0.32 ; +0.32].
    """
    a, b, c, d = -3.32959069, 14.61846764, 3.07428877, 1.03357331
    return a * i**3 + b * i**2 + c * i + d


# -----------------------------------
# Ajustements de vitesse vs. la pente
# -----------------------------------
def adjusted_speed_minetti(flat_speed, slope, _C0=minetti_cost_running(0.0)):
    """
    Vitesse ajustée (m/s) selon Minetti, plafonnée en descente (1.3x).
    slope en %.
    """
    if flat_speed is None or flat_speed <= 0:
        return 1e-6
    i = float(slope) / 100.0
    Ci = max(minetti_cost_running(i), 1e-6)
    v_raw = flat_speed * (_C0 / Ci)
    v_capped = min(1.3 * flat_speed, v_raw)  # plafonne la descente
    return max(v_capped, 1e-6)


def adjusted_speed_strava(flat_speed, slope, _C0=strava_cost(0.0)):
    """
    Vitesse ajustée (m/s) Strava-like.
    Clamp du domaine du polynôme à [-0.32 ; +0.32] (±32%).
    """
    if flat_speed is None or flat_speed <= 0:
        return 1e-6
    i = float(slope) / 100.0
    if i < -0.32:
        i = -0.32
    elif i > 0.32:
        i = 0.32
    Ci = max(strava_cost(i), 1e-6)
    v = flat_speed * (_C0 / Ci)
    return max(v, 1e-6)


# ------------------------------------------------
# Simulation de temps total et recherche de la VAP
# ------------------------------------------------
def simulate_temps_total(flat_speed, distances, elevations):
    """Temps total (s) via Minetti pour une vitesse équivalente sur plat donnée."""
    total_time = 0.0
    for i in range(1, len(distances)):
        d = (distances[i] - distances[i - 1]) * 1000.0  # m
        if d <= 0:
            continue
        dz = elevations[i] - elevations[i - 1]
        slope = (dz / d) * 100.0
        v_adj = adjusted_speed_minetti(flat_speed, slope)
        total_time += d / v_adj
    return total_time


def simulate_temps_total_strava(flat_speed, distances, elevations):
    """Temps total (s) via Strava-like pour une vitesse équivalente sur plat donnée."""
    total_time = 0.0
    for i in range(1, len(distances)):
        d = (distances[i] - distances[i - 1]) * 1000.0  # m
        if d <= 0:
            continue
        dz = elevations[i] - elevations[i - 1]
        slope = (dz / d) * 100.0
        v_adj = adjusted_speed_strava(flat_speed, slope)
        total_time += d / v_adj
    return total_time


def _bracket_speed(distances, temps_espere_sec):
    """
    Encadrement initial 'intelligent' pour la dichotomie autour de la vitesse naïve.
    Retombe sur [1.0 ; 6.0] m/s si estimation non exploitable.
    """
    try:
        total_km = float(distances[-1]) if distances else 0.0
        est = (total_km * 1000.0) / max(temps_espere_sec, 1e-6)  # m/s
        if est > 0:
            v_min = max(0.5, est * 0.5)   # >= 1.8 km/h
            v_max = min(10.0, est * 2.5)  # <= 36 km/h
            if v_max - v_min >= 0.1:
                return v_min, v_max
    except Exception:
        pass
    return 1.0, 6.0


def trouver_vitesse_plate(distances, elevations, temps_espere_sec, precision=1):
    """
    Recherche par dichotomie de la vitesse équivalente sur plat (m/s) — modèle Minetti.
    distances (km), elevations (m), temps_espere_sec (s).
    """
    v_min, v_max = _bracket_speed(distances, temps_espere_sec)
    iteration = 0
    while v_max - v_min > 1e-4:
        iteration += 1
        v_mid = (v_min + v_max) / 2.0
        temps_mid = simulate_temps_total(v_mid, distances, elevations)
        if abs(temps_mid - temps_espere_sec) < precision:
            return v_mid
        if temps_mid > temps_espere_sec:
            v_min = v_mid  # trop lent → augmenter la vitesse
        else:
            v_max = v_mid  # trop rapide → diminuer
        if iteration > 100:
            break
    return (v_min + v_max) / 2.0


def trouver_vitesse_plate_strava(distances, elevations, temps_espere_sec, precision=1):
    """
    Recherche par dichotomie de la vitesse équivalente sur plat (m/s) — modèle Strava-like.
    """
    v_min, v_max = _bracket_speed(distances, temps_espere_sec)
    iteration = 0
    while v_max - v_min > 1e-4:
        iteration += 1
        v_mid = (v_min + v_max) / 2.0
        temps_mid = simulate_temps_total_strava(v_mid, distances, elevations)
        if abs(temps_mid - temps_espere_sec) < precision:
            return v_mid
        if temps_mid > temps_espere_sec:
            v_min = v_mid
        else:
            v_max = v_mid
        if iteration > 100:
            break
    return (v_min + v_max) / 2.0


# ------------------------------
# Temps cumulés & allures par km
# ------------------------------
def compute_cumulative_time(flat_speed, distances, elevations):
    """Temps cumulé (s) pour chaque point — Minetti."""
    cumulative_time = [0.0]
    for i in range(1, len(distances)):
        d = (distances[i] - distances[i - 1]) * 1000.0
        if d <= 0:
            cumulative_time.append(cumulative_time[-1])
            continue
        dz = elevations[i] - elevations[i - 1]
        slope = (dz / d) * 100.0
        v_adj = adjusted_speed_minetti(flat_speed, slope)
        cumulative_time.append(cumulative_time[-1] + d / v_adj)
    return cumulative_time


def compute_cumulative_time_strava(flat_speed, distances, elevations):
    """Temps cumulé (s) pour chaque point — Strava-like."""
    cumulative_time = [0.0]
    for i in range(1, len(distances)):
        d = (distances[i] - distances[i - 1]) * 1000.0
        if d <= 0:
            cumulative_time.append(cumulative_time[-1])
            continue
        dz = elevations[i] - elevations[i - 1]
        slope = (dz / d) * 100.0
        v_adj = adjusted_speed_strava(flat_speed, slope)
        cumulative_time.append(cumulative_time[-1] + d / v_adj)
    return cumulative_time


def compute_paces(distances, elevations, flat_speed):
    """
    Allure (min/km) par segment — modèle Minetti.
    Renvoie une liste de floats (jamais None), bornée à [0.1 ; 70] min/km.
    """
    paces = []
    for i in range(1, len(distances)):
        d = (distances[i] - distances[i - 1]) * 1000.0
        if d <= 0:
            paces.append(70.0)
            continue
        dz = elevations[i] - elevations[i - 1]
        slope = (dz / d) * 100.0
        v_adj = adjusted_speed_minetti(flat_speed, slope)
        pace = (1000.0 / v_adj) / 60.0  # min/km
        paces.append(min(max(pace, 0.1), 70.0))
    return paces


def compute_paces_strava(distances, elevations, flat_speed):
    """
    Allure (min/km) par segment — modèle Strava-like.
    Renvoie une liste de floats (jamais None), bornée à [0.1 ; 70] min/km.
    """
    paces = []
    for i in range(1, len(distances)):
        d = (distances[i] - distances[i - 1]) * 1000.0
        if d <= 0:
            paces.append(70.0)
            continue
        dz = elevations[i] - elevations[i - 1]
        slope = (dz / d) * 100.0
        v_adj = adjusted_speed_strava(flat_speed, slope)
        pace = (1000.0 / v_adj) / 60.0  # min/km
        paces.append(min(max(pace, 0.1), 70.0))
    return paces


# --------------
# Lissage simple
# --------------
def smooth(y, box_pts=5):
    """
    Moyenne glissante simple O(n) via convolution.
    - y : séquence de floats
    - box_pts : taille de fenêtre (>=1)
    Retourne un np.ndarray.
    """
    arr = np.asarray(y, dtype=float)
    n = arr.size
    box_pts = max(1, int(box_pts))
    if n == 0 or box_pts == 1:
        return arr
    half = box_pts // 2
    arr_pad = np.pad(arr, (half, half), mode="edge")
    kernel = np.ones(box_pts, dtype=float) / box_pts
    out = np.convolve(arr_pad, kernel, mode="valid")
    return out


# ---------------------------
# Lecture & préparation du GPX
# ---------------------------
def process_gpx(gpx_content):
    """
    Parse GPX -> distances(km), elevations(m), distances_pace(km), coords[(lat, lon)].
    - Sous-échantillonnage ~30 m.
    - Altitude manquante: propage la dernière valeur connue (0 si aucune).
    """
    gpx = gpxpy.parse(gpx_content)

    last_point = None
    total_distance = 0.0
    distances, elevations, coords = [], [], []

    DISTANCE_MIN = 30.0  # ~30 m
    distance_since_last_save = 0.0
    last_elev = None
    started = False

    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                coords.append((point.latitude, point.longitude))
                elev = point.elevation
                if elev is None:
                    elev = last_elev if last_elev is not None else 0.0

                if last_point is not None:
                    d = point.distance_3d(last_point) or 0.0
                    total_distance += d
                    distance_since_last_save += d
                    if distance_since_last_save >= DISTANCE_MIN:
                        distances.append(total_distance / 1000.0)  # km
                        elevations.append(float(elev))
                        distance_since_last_save = 0.0
                        started = True
                else:
                    # Premier point
                    distances.append(0.0)
                    elevations.append(float(elev))
                    started = True

                last_point = point
                last_elev = elev

    # distances_pace = milieux des segments pour coller aux paces segmentaires
    distances_pace = [(distances[i] + distances[i - 1]) / 2.0 for i in range(1, len(distances))]
    return distances, elevations, distances_pace, coords


# -------------------------
# Dénivelé + / -
# -------------------------
def calculate_deniv(elevations):
    """
    Retourne (D+, D-) en mètres, arrondis.
    Implémentation simple sans état cumulatif coûteux.
    """
    d_plus = 0.0
    d_moins = 0.0
    for i in range(1, len(elevations)):
        delta = elevations[i] - elevations[i - 1]
        if delta > 0:
            d_plus += delta
        else:
            d_moins += delta
    return round(d_plus), round(d_moins)

# === Utils pour "Données réelles" (empirique) ===

def _haversine_m(dx_lat, dx_lon, lat1, lon1):
    """Distance en mètres entre deux points lat/lon (radians internes)."""
    R = 6371000.0
    return 2 * R * np.arcsin(
        np.sqrt(
            np.sin(dx_lat / 2) ** 2
            + np.cos(lat1) * np.cos(lat1 + dx_lat) * np.sin(dx_lon / 2) ** 2
        )
    )

def _parse_gpx_points_with_time(gpx_content):
    """
    Retourne des listes (lat, lon en rad; ele en m; t en secondes; cumdist en m).
    """
    gpx = gpxpy.parse(gpx_content)
    lats, lons, eles, ts = [], [], [], []
    for trk in gpx.tracks:
        for seg in trk.segments:
            for p in seg.points:
                if p.time is None or p.elevation is None:
                    continue
                lats.append(np.deg2rad(p.latitude))
                lons.append(np.deg2rad(p.longitude))
                eles.append(float(p.elevation))
                ts.append(p.time)

    if len(lats) < 2:
        return [], [], [], [], []

    lats = np.array(lats); lons = np.array(lons)
    eles = np.array(eles)
    ts = np.array(ts, dtype="datetime64[ns]")
    t0 = ts[0].astype("datetime64[s]").astype(int)
    t_sec = ts.astype("datetime64[s]").astype(int) - t0

    # Distances cumulées
    dists = [0.0]
    for i in range(1, len(lats)):
        dx_lat = lats[i] - lats[i - 1]
        dx_lon = lons[i] - lons[i - 1]
        d = _haversine_m(dx_lat, dx_lon, lats[i - 1], lons[i - 1])
        dists.append(dists[-1] + float(d))
    return lats, lons, eles, t_sec.astype(float), np.array(dists, dtype=float)

def build_empirical_curve_from_gpx(gpx_content, bin_size=0.02, min_speed_mps=0.5):
    """
    Construit une courbe empirique vitesse(m/s) vs pente à partir d'un GPX réel.
    - bin_size: pas de pente (ex: 0.02 = 2%)
    Retourne un DataFrame: [slope_bin (float), speed_mps_median, speed_p25, speed_p75]
    et le cumul de temps observé (array en s) pour réutilisation directe.
    """

    if isinstance(gpx_content, str):
        gpx_content = gpx_content

    lats, lons, eles, t_sec, dists = _parse_gpx_points_with_time(gpx_content)
    if len(dists) < 2:
        return _pd.DataFrame(columns=["slope_bin", "speed_mps_median", "speed_p25", "speed_p75"]), np.array([])

    # Segments
    dd = np.diff(dists)  # m
    dh = np.diff(eles)   # m
    dt = np.diff(t_sec)  # s

    # Filtres robustes
    mask = (dt > 0.5) & (dd > 1.0)  # éviter pauses et bruit
    dd = dd[mask]; dh = dh[mask]; dt = dt[mask]
    slope = dh / dd                   # pente en décimal (0.1 = +10%)
    speed = dd / dt                   # m/s

    # Garde les points en mouvement
    mask2 = (speed >= min_speed_mps) & (np.isfinite(speed)) & (np.isfinite(slope)) & (np.abs(slope) <= 0.6)
    slope = slope[mask2]; speed = speed[mask2]

    if slope.size == 0:
        return _pd.DataFrame(columns=["slope_bin", "speed_mps_median", "speed_p25", "speed_p75"]), np.array([])

    # Binning par pente
    bins = np.arange(-0.60, 0.60 + bin_size, bin_size)
    idx = np.digitize(slope, bins) - 1
    df = _pd.DataFrame({"slope": slope, "speed": speed, "bin": idx})
    grouped = df.groupby("bin")

    rows = []
    for b, g in grouped:
        if b < 0 or b >= len(bins) - 1 or len(g) < 5:
            continue
        sbin = float((bins[b] + bins[b + 1]) / 2)
        rows.append({
            "slope_bin": sbin,
            "speed_mps_median": float(np.median(g["speed"])),
            "speed_p25": float(np.percentile(g["speed"], 25)),
            "speed_p75": float(np.percentile(g["speed"], 75)),
        })
    curve = _pd.DataFrame(rows).sort_values("slope_bin").reset_index(drop=True)
    return curve, np.array(t_sec, dtype=float)

def _speed_from_empirical(slope, curve_df):
    """Vitesse m/s pour une pente donnée via plus proche bin; extrapole aux bords."""
    if curve_df is None or curve_df.empty:
        return np.nan
    s = float(slope)
    # Nearest neighbor
    k = int(np.argmin(np.abs(curve_df["slope_bin"].values - s)))
    return float(curve_df["speed_mps_median"].iloc[k])

def compute_cumulative_time_empirical(distances_m, elevations_m, curve_df, fallback_speed_mps=1.5):
    """
    Recalcule un temps cumulé sur une trace (distances/elevations) à partir de la courbe empirique.
    Si la courbe ne couvre pas certaines pentes, on tombe sur fallback_speed_mps.
    """
    distances_m = np.asarray(distances_m, dtype=float)
    elevations_m = np.asarray(elevations_m, dtype=float)
    if len(distances_m) != len(elevations_m) or len(distances_m) < 2:
        return np.array([])

    cum_t = [0.0]
    for i in range(1, len(distances_m)):
        dd = distances_m[i] - distances_m[i - 1]
        if dd <= 0:
            cum_t.append(cum_t[-1]); continue
        dh = elevations_m[i] - elevations_m[i - 1]
        slope = dh / dd
        v = _speed_from_empirical(slope, curve_df)
        if not np.isfinite(v) or v <= 0:
            v = fallback_speed_mps
        cum_t.append(cum_t[-1] + dd / v)
    return np.array(cum_t, dtype=float)

def compute_paces_empirical(distances_m, cumulative_time_s):
    """
    Renvoie un DataFrame paces/km à partir du temps cumulé.
    """
    if len(distances_m) == 0 or len(cumulative_time_s) == 0:
        return _pd.DataFrame(columns=["km", "pace_min_per_km"])

    km_marks = np.arange(1, int(np.ceil(distances_m[-1] / 1000.0)) + 1)
    rows = []
    for k in km_marks:
        target_m = k * 1000.0
        # interpolation linéaire du temps au passage du km k
        t_k = np.interp(target_m, distances_m, cumulative_time_s, left=np.nan, right=np.nan)
        t_km_1 = np.interp(target_m - 1000.0, distances_m, cumulative_time_s, left=np.nan, right=np.nan)
        if np.isfinite(t_k) and np.isfinite(t_km_1):
            pace_min = (t_k - t_km_1) / 60.0
            rows.append({"km": int(k), "pace_min_per_km": float(pace_min)})
    return _pd.DataFrame(rows)