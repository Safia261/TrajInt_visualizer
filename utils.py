import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.path import Path
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.spatial.distance import cdist, directed_hausdorff
from config import *


def compute_interaction_intervals(df, ped_id, cyc_id, fps, distance_threshold=5.0):
    """
    Retourne les intervalles de temps où un piéton et un cycliste sont en interaction.

    Returns:
        intervals : liste de tuples [(t_start, t_end), ...]
    """

    # récupérer toutes les interactions
    from analysis_interactions import compute_ped_cyc_interactions_with_time
    interactions = compute_ped_cyc_interactions_with_time(df, distance_threshold)

    # clé triée (important)
    key = tuple(sorted((ped_id, cyc_id)))

    frames = interactions.get(key, [])

    if len(frames) == 0:
        return []

    frames = sorted(frames)

    # conversion en intervalles
    intervals_fps = []
    start = frames[0]
    prev = frames[0]

    for t in frames[1:]:
        # continuité temporelle
        if t == prev:
            continue

        # rupture -> nouvel intervalle
        if t > prev + 1:
            intervals_fps.append((int(start), int(prev)))
            start = t

        prev = t

    # dernier intervalle
    intervals_fps.append((int(start), int(prev)))

    intervals_sec = [(float(s/fps), float(e/fps)) for s, e in intervals_fps]

    total_duration_sec = sum((e - s) for s, e in intervals_sec)

    return intervals_fps, intervals_sec, total_duration_sec

def frames_to_intervals(frames):
    """
    Convertit une liste de frames en intervalles continus (start, end)
    """
    if len(frames) == 0:
        return []

    frames = sorted(frames)

    intervals = []
    start = frames[0]
    prev = frames[0]

    for t in frames[1:]:
        if t == prev + 1:
            # continuité
            prev = t
        else:
            intervals.append((start, prev))
            start = t
            prev = t

    intervals.append((start, prev))
    return intervals


def add_time_markers(ax, intervals):
    for i, (start, end) in enumerate(intervals):
        ax.axvline(start, color="green", linestyle="--",
                   label="Début interaction" if i == 0 else "")
        ax.axvline(end, color="red", linestyle="--",
                   label="Fin interaction" if i == 0 else "")

        ax.axvspan(start, end, color="orange", alpha=0.2)


def add_spatial_markers(ax, df, ped_id, intervals):
    ped = df[df[COL_ID] == ped_id]

    for i, (start, end) in enumerate(intervals):

        p_start = ped[ped[COL_TIME] == start]
        p_end = ped[ped[COL_TIME] == end]

        if not p_start.empty:
            p_start = p_start.iloc[0]
            ax.scatter(
                p_start["x_m"], p_start["y_m"],
                color="green", s=80, marker="o",
                label="Début interaction" if i == 0 else ""
            )

        if not p_end.empty:
            p_end = p_end.iloc[0]
            ax.scatter(
                p_end["x_m"], p_end["y_m"],
                color="red", s=80, marker="x",
                label="Fin interaction" if i == 0 else ""
            )


def filter_series_by_intervals(times, values, intervals):
    """
    Garde uniquement les valeurs dans les intervalles d'interaction
    """
    mask = np.zeros_like(times, dtype=bool)

    for start, end in intervals:
        mask |= (times >= start) & (times <= end)

    return times[mask], values[mask]


###############################################
# Critères spatio-temporels d'une interaction paire (individu-individu)
###############################################

def compute_speed(g, fps):
    """
    Calcule la vitesse (m/s et km/h) d'un agent au cours du temps.
    """
    g = g.sort_values(COL_TIME)

    xs = g["x_m"].values
    ys = g["y_m"].values
    times = g[COL_TIME].values

    if len(xs) < 2:
        return None, None

    dx = np.diff(xs)
    dy = np.diff(ys)

    # à voir si je mets pas simplement que la ligne suivante
    # dt = np.ones_like(dx) / fps # (c'est exactement équivalent à dt = 1/fps en s)

    if len(np.unique(times)) > 1: # si y a au moins 2 timestamps différents
        dt = np.diff(times) / fps
    else:
        dt = np.ones_like(dx) / fps # (c'est exactement équivalent à dt = 1/fps en s)

    # dt[dt == 0] = 1e-6 # pour éviter division par zéro
    dt[dt <= 0] = 1 / fps # pour éviter division par zéro

    speeds = np.hypot(dx, dy) / dt # exactement équivalent à dist euclidienne

    # conversion km/h
    speeds_kmh = speeds * 3.6

    return times[1:], speeds, speeds_kmh


def compute_distance_ped_cyc(df, ped_id, cyc_id, fps, distance_threshold=5.0, plot=False, return_class=False):
    """
    Calcule et trace la distance entre un piéton et un cycliste.

    Returns:
        times
        distances
        (optionnel) intervals
    """

    # extraction
    ped = df[df[COL_ID] == ped_id].sort_values(COL_TIME)
    cyc = df[df[COL_ID] == cyc_id].sort_values(COL_TIME)

    # synchronisation
    common_times = np.intersect1d(
        ped[COL_TIME].values,
        cyc[COL_TIME].values
    )

    if len(common_times) == 0:
        return None, None

    distances = []
    valid_times = []

    for t in common_times:
        p = ped[ped[COL_TIME] == t].iloc[0]
        c = cyc[cyc[COL_TIME] == t].iloc[0]

        dx = p["x_m"] - c["x_m"]
        dy = p["y_m"] - c["y_m"]

        dist = np.hypot(dx, dy)

        distances.append(dist)
        valid_times.append(t)

    distances = np.array(distances)
    valid_times = np.array(valid_times)

    dist_min = distances.min()
    idx_min = np.nan
    t_min = np.nan
    t_min_plot = np.nan
    if len(distances) != 0:
        idx_min = np.argmin(distances) 
        t_min = valid_times[idx_min]
        t_min_plot = t_min / fps
    

    # intervalles interaction
    from analysis_interactions import compute_ped_cyc_interactions_with_time
    interactions = compute_ped_cyc_interactions_with_time(df, distance_threshold)
    key = tuple(sorted((ped_id, cyc_id)))
    frames = interactions.get(key, [])

    intervals = frames_to_intervals(frames) if len(frames) > 0 else []

    intervals_plot = [(s / fps, e / fps) for s, e in intervals]

    # plot
    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(valid_times / fps, distances, label="Distance (m)")
        plt.axhline(distance_threshold, color="red", linestyle="--", label="Seuil interaction (5m)")
        add_time_markers(plt.gca(), intervals_plot)
        if t_min_plot != np.nan:
            plt.scatter(t_min_plot, dist_min, color="red", zorder=5, label=f"Distance minimale = {dist_min:.2f} m")
        plt.xlabel("Temps (s)")
        plt.ylabel("Distance (m)")
        plt.title(f"Distance piéton {ped_id} - cycliste {cyc_id}")
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()
    
    if return_class:
        from analysis_interactions import classify_distance_interaction
        return valid_times, distances, dist_min, classify_distance_interaction(valid_times, distances, intervals)

    return valid_times, distances, dist_min


def find_closest_ped_cyc(df):
    """
    Renvoie le piéton et cycliste les plus proches entre eux.
    """
    min_dist = float("inf")
    best_pair = (None, None)

    for t in df[COL_TIME].unique():
        frame = df[df[COL_TIME] == t]

        peds = frame[frame[COL_CLASS] == 1]
        cycs = frame[frame[COL_CLASS] == 2]

        if len(peds) == 0 or len(cycs) == 0:
            continue

        for _, ped in peds.iterrows():
            for _, cyc in cycs.iterrows():
                dx = ped["x_m"] - cyc["x_m"]
                dy = ped["y_m"] - cyc["y_m"]
                dist = np.hypot(dx, dy)

                if dist < min_dist:
                    min_dist = dist
                    best_pair = (ped[COL_ID], cyc[COL_ID])

    return best_pair



def compute_velocity_vectors(g, fps, plot=False):
    """
    Calcule les vecteurs vitesse (vx, vy) pour un agent donné.
    Returns:
        times, vx, vy
    """
    g = g.sort_values(COL_TIME)

    xs = g["x_m"].values
    ys = g["y_m"].values
    times = g[COL_TIME].values

    if len(xs) < 2:
        return None, None, None

    dx = np.diff(xs)
    dy = np.diff(ys)

    if len(np.unique(times)) > 1:
        dt = np.diff(times) / fps
    else:
        dt = np.ones_like(dx) / fps

    dt[dt <= 0] = 1 / fps

    vx = dx / dt
    vy = dy / dt

    if plot:
        plt.figure(figsize=(6, 6))

        plt.plot(xs, ys, 'k--', alpha=0.5, label="Trajectoire")

        plt.quiver(
            xs[:-1], ys[:-1],
            vx, vy,
            angles='xy', scale_units='xy', scale=1,
            color="blue"
        )
        
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title(f"Vecteurs vitesse - agent {g[COL_ID].iloc[0]}")
        plt.gca().invert_yaxis()
        plt.grid()
        plt.legend()
        plt.axis("equal")

        plt.show()

    return times[1:], vx, vy


def compute_direction_angle_velocity_based(df, ped_id, cyc_id, fps, angle_unit="deg", plot=False, return_class=False):
    """
    Calcule l'angle de direction entre les vecteurs vitesse du piéton et du cycliste. Indique la direction des usagers lors de l'interaction.

    Returns:
        times, angles
    """

    ped = df[df[COL_ID] == ped_id].sort_values(COL_TIME)
    cyc = df[df[COL_ID] == cyc_id].sort_values(COL_TIME)

    # vitesses
    t_p, vx_p, vy_p = compute_velocity_vectors(ped, fps)
    t_c, vx_c, vy_c = compute_velocity_vectors(cyc, fps)

    if t_p is None or t_c is None:
        return None, None

    # synchronisation des timestamps
    common_times = np.intersect1d(t_p, t_c)

    angles = []
    valid_times = []

    for t in common_times:
        i_p = np.where(t_p == t)[0][0]
        i_c = np.where(t_c == t)[0][0]

        v_p = np.array([vx_p[i_p], vy_p[i_p]])
        v_c = np.array([vx_c[i_c], vy_c[i_c]])

        norm_p = np.linalg.norm(v_p)
        norm_c = np.linalg.norm(v_c)

        if norm_p == 0 or norm_c == 0:
            continue

        cos_theta = np.dot(v_p, v_c) / (norm_p * norm_c)
        cos_theta = np.clip(cos_theta, -1.0, 1.0) # en radians

        angle = np.arccos(cos_theta)

        if angle_unit == "deg":
            angle = np.degrees(angle)

        angles.append(angle)
        valid_times.append(t)

    angles = np.array(angles)
    valid_times = np.array(valid_times)

    # plot
    if plot and len(angles) > 0:
        from analysis_interactions import compute_ped_cyc_interactions_with_time
        interactions = compute_ped_cyc_interactions_with_time(df)

        frames = interactions.get((ped_id, cyc_id), [])
        intervals = frames_to_intervals(frames)

        plt.figure(figsize=(10, 4))
        plt.plot(valid_times/fps, angles)
        plt.xlabel("Temps (s)")
        plt.ylabel(f"Angle ({angle_unit})")
        plt.title(f"Angle de direction (entre vecteurs vitesses) (cycliste {cyc_id} - piéton {ped_id})")
        if angle_unit == "deg":
            plt.axhline(0, linestyle="--", color="black", alpha=0.5, label="Même direction (0°)")
            plt.axhline(90, linestyle="--", color="orange", label="Perpendiculaire (90°)") # croisement latéral ou perpendiculaire mais pas forcément de collision
            plt.axhline(180, linestyle="--", color="red", label="Opposition (180°)")
        plt.grid()
        intervals_sec = [(s / fps, e / fps) for (s, e) in intervals]
        add_time_markers(plt.gca(), intervals_sec)
        plt.legend()
        plt.show()

    if return_class:
        from analysis_interactions import classify_direction_angle_interaction
        return valid_times, angles, classify_direction_angle_interaction(df, ped_id, cyc_id, valid_times, angles)
    
    return valid_times, angles


def compute_approach_angle(
    df,
    ped_id,
    cyc_id,
    fps,
    angle_unit="deg",
    plot=False,
    return_class=False):
    """
    Calcule l'angle d'approche entre un cycliste et un piéton (entre la lignée de visée et la vitesse relative).

    Basé sur :
        - vecteur position relative
        - vecteur vitesse relative

    Returns:
        times, angles
    """

    ped = df[df[COL_ID] == ped_id].sort_values(COL_TIME)
    cyc = df[df[COL_ID] == cyc_id].sort_values(COL_TIME)

    # vecteurs vitesses
    t_p, vx_p, vy_p = compute_velocity_vectors(ped, fps)
    t_c, vx_c, vy_c = compute_velocity_vectors(cyc, fps)

    if t_p is None or t_c is None:
        return None, None

    # positions
    t_pos = np.intersect1d(ped[COL_TIME].values, cyc[COL_TIME].values)
    t_vel = np.intersect1d(t_p, t_c)

    common_times = np.intersect1d(t_pos, t_vel)

    angles = []
    valid_times = []

    for t in common_times:
        # indices positions
        i_p_pos = np.where(ped[COL_TIME].values == t)[0][0]
        i_c_pos = np.where(cyc[COL_TIME].values == t)[0][0]

        x_p, y_p = ped.iloc[i_p_pos][["x_m", "y_m"]]
        x_c, y_c = cyc.iloc[i_c_pos][["x_m", "y_m"]]

        # indices vitesses
        i_p_vel = np.where(t_p == t)[0][0]
        i_c_vel = np.where(t_c == t)[0][0]

        v_p = np.array([vx_p[i_p_vel], vy_p[i_p_vel]])
        v_c = np.array([vx_c[i_c_vel], vy_c[i_c_vel]])

        # vceteurs importants
        r = np.array([x_p - x_c, y_p - y_c])   # position relative cycliste par rapp au piéton
        v_rel = v_c - v_p                      # vitesse relative

        norm_r = np.linalg.norm(r)
        norm_v = np.linalg.norm(v_rel)

        if norm_r == 0 or norm_v == 0:
            continue

        cos_theta = np.dot(v_rel, r) / (norm_r * norm_v)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        angle = np.arccos(cos_theta)

        if angle_unit == "deg":
            angle = np.degrees(angle)

        angles.append(angle)
        valid_times.append(t)

    angles = np.array(angles)
    valid_times = np.array(valid_times)

    # plot
    if plot and len(angles) > 0:
        from analysis_interactions import compute_ped_cyc_interactions_with_time

        interactions = compute_ped_cyc_interactions_with_time(df)
        pair = tuple(sorted((ped_id, cyc_id)))
        frames = interactions.get(pair, [])

        # intervals = [(min(frames), max(frames))] if frames else []    
        intervals = frames_to_intervals(frames)

        plt.figure(figsize=(10, 4))
        plt.plot(valid_times/fps, angles, label="Angle d'approche")

        # for (t_start, t_end) in intervals:
        #     plt.axvspan(t_start, t_end, color="orange", alpha=0.3)

        plt.xlabel("Temps (s)")
        plt.ylabel(f"Angle ({angle_unit})")
        plt.title(f"Angle d'approche (cycliste {cyc_id} - piéton {ped_id})")
        if angle_unit == "deg":
            plt.axhline(0, linestyle="--", color="black", alpha=0.5, label="Frontal (0°)")
            plt.axhline(90, linestyle="--", color="orange", label="Croisement (90°)") # croisement latéral ou perpendiculaire mais pas forcément de collision
            plt.axhline(180, linestyle="--", color="red", label="Opposition (180°)")
        plt.grid()
        intervals_sec = [(s / fps, e / fps) for (s, e) in intervals]
        add_time_markers(plt.gca(), intervals_sec)
        plt.legend()
        plt.show()
    
    if return_class:
        from analysis_interactions import classify_approach_angle_interaction
        return valid_times, angles, classify_approach_angle_interaction(df, ped_id, cyc_id, valid_times, angles)

    return valid_times, angles


def compute_relative_speed(df, ped_id, cyc_id, fps, angle_unit="deg", return_distance=True, distance_threshold=5.0, plot=False, return_class=False):
    """
    Calcule le mouvement relatif entre un piéton et un cycliste :
    - vitesse relative (m/s)
    - vitesse relative (km/h)
    - angle d'approche (rad)
    - distance (optionnel)

    Returns:
        times, rel_speeds, rel_speeds_kmh, angles, distances (optionnel)
    """
    ped = df[df[COL_ID] == ped_id].sort_values(COL_TIME)
    cyc = df[df[COL_ID] == cyc_id].sort_values(COL_TIME)

    if len(ped) < 2 or len(cyc) < 2:
        return None

    # vecteurs vitesses
    t_p, vx_p, vy_p = compute_velocity_vectors(ped, fps)
    t_c, vx_c, vy_c = compute_velocity_vectors(cyc, fps)

    # synchronisation timestamps
    common_times = np.intersect1d(t_p, t_c)

    if len(common_times) == 0:
        print("Pas de temps communs")
        return None

    # index correspondants
    idx_p = np.isin(t_p, common_times)
    idx_c = np.isin(t_c, common_times)

    vx_p = vx_p[idx_p]
    vy_p = vy_p[idx_p]

    vx_c = vx_c[idx_c]
    vy_c = vy_c[idx_c]

    times = common_times

    # vitesse relative
    speed_p = np.hypot(vx_p, vy_p)
    speed_c = np.hypot(vx_c, vy_c)

    dot = vx_p * vx_c + vy_p * vy_c
    norm_p = np.maximum(speed_p, 1e-8)
    norm_c = np.maximum(speed_c, 1e-8)

    cos_theta = dot / (norm_p * norm_c)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    theta = np.arccos(cos_theta)

    rel_speeds = np.sqrt(speed_p**2 + speed_c**2 - 2 * speed_p * speed_c * cos_theta)
    rel_speeds_kmh = rel_speeds * 3.6

    # angle d'approche
    angles = theta
    if angle_unit == "deg":
        angles = np.degrees(angles)

    if plot:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        ax1, ax2, ax3 = axes

        # vitesse m/s
        ax1.plot(common_times/fps, rel_speeds, label="Vitesse relative (m/s)")
        ax1.set_ylabel("m/s")
        ax1.set_title("Vitesse relative")
        ax1.grid()

        # vitesse km/h
        ax2.plot(common_times/fps, rel_speeds_kmh, label="Vitesse relative (km/h)", color="orange")
        ax2.set_ylabel("km/h")
        ax2.grid()

        # angle
        ax3.plot(common_times/fps, angles, label="Angle (deg)", color="green")
        ax3.set_ylabel("Angle (°)")
        ax3.set_xlabel("Temps (s)")
        ax3.grid()

        from analysis_interactions import compute_ped_cyc_interactions_with_time
        interactions = compute_ped_cyc_interactions_with_time(df, distance_threshold)

        pair = tuple(sorted((ped_id, cyc_id)))
        interaction_times = interactions.get(pair, [])

        # Marqueurs d'interaction
        if len(interaction_times) > 0:
            t_start = min(interaction_times)
            t_end = max(interaction_times)

            for ax in axes:
                ax.axvline(t_start/fps, color="red", linestyle="--", label="début interaction")
                ax.axvline(t_end/fps, color="purple", linestyle="--", label="fin interaction")

        # pour éviter doublons de légende
        for ax in axes:
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())

        plt.tight_layout()
        plt.show()

    # distance (optionnel)
    distances = None

    if return_distance:
        ped_pos = ped[ped[COL_TIME].isin(times)]
        cyc_pos = cyc[cyc[COL_TIME].isin(times)]

        dx = cyc_pos["x_m"].values - ped_pos["x_m"].values
        dy = cyc_pos["y_m"].values - ped_pos["y_m"].values

        distances = np.hypot(dx, dy)

    # Résultats
    if return_distance:
        if return_class:
            from analysis_interactions import classify_relative_speed_interaction
            return times, rel_speeds, rel_speeds_kmh, angles, distances, classify_relative_speed_interaction(df, ped_id, cyc_id, times, rel_speeds_kmh)
        else:
            return times, rel_speeds, rel_speeds_kmh, angles, distances
    else:
        if return_class:
            from analysis_interactions import classify_relative_speed_interaction
            return times, rel_speeds, rel_speeds_kmh, angles, classify_relative_speed_interaction(df, ped_id, cyc_id, times, rel_speeds_kmh)
        else:
            return times, rel_speeds, rel_speeds_kmh, angles


def compute_pet(df, ped_id, cyc_id, fps, distance_threshold=1.0, plot=False, return_class=False):
    """
    Calcule le PET (Post-Encroachment Time) entre un piéton et un cycliste.

    Parameters:
        df : DataFrame
        ped_id : ID du piéton
        cyc_id : ID du cycliste
        distance_threshold : rayon pour considérer "même point" (en mètres)

    Returns:
        pet (float) ou None
    """

    ped = df[df[COL_ID] == ped_id].sort_values(COL_TIME)
    cyc = df[df[COL_ID] == cyc_id].sort_values(COL_TIME)

    times_p = ped[COL_TIME].values
    times_c = cyc[COL_TIME].values

    # 1. Trouver point de conflit (distance minimale)
    min_dist = float("inf")
    conflict_point = None

    for _, p in ped.iterrows():
        for _, c in cyc.iterrows():
            dx = p["x_m"] - c["x_m"]
            dy = p["y_m"] - c["y_m"]
            dist = np.hypot(dx, dy)

            if dist < min_dist:
                min_dist = dist
                conflict_point = ((p["x_m"] + c["x_m"]) / 2,
                                  (p["y_m"] + c["y_m"]) / 2)

    if conflict_point is None:
        return None

    cx, cy = conflict_point

    # 2. Trouver temps de passage près du point
    def get_crossing_time(agent_df):
        for _, row in agent_df.iterrows():
            dx = row["x_m"] - cx
            dy = row["y_m"] - cy
            dist = np.hypot(dx, dy)

            if dist < distance_threshold:
                return row[COL_TIME], row["x_m"], row["y_m"]

        return None, None, None

    t_ped, x_ped, y_ped = get_crossing_time(ped)
    t_cyc, x_cyc, y_cyc = get_crossing_time(cyc)

    if t_ped is None or t_cyc is None:
        return None

    # 3. PET
    pet = abs(t_ped - t_cyc) / fps

    if t_ped <= t_cyc:
        first_out = "Ped"
        last_in = "Cyc"
    else:
        first_out = "Cyc"
        last_in = "Ped"

    if plot:
        plt.figure(figsize=(8, 8))

        # trajectoires
        plt.plot(ped["x_m"], ped["y_m"],
                 color=CLASS_COLORS.get(1, "blue"),
                 label=f"Ped {ped_id}")

        plt.plot(cyc["x_m"], cyc["y_m"],
                 color=CLASS_COLORS.get(2, "green"),
                 label=f"Cyc {cyc_id}")

        # point de conflit
        plt.scatter(cx, cy,
                    color="red",
                    s=120,
                    marker="X",
                    label="Point conflit")

        # essayer d'ajouter les flèches de direction des agents plus tard si temps

        plt.text(cx + 2.0, cy + 2.0, f"First out: {first_out}\nLast in: {last_in}", bbox=dict(facecolor="white", alpha=0.8))

        # cercle zone conflit
        circle = plt.Circle(
            (cx, cy),
            distance_threshold,
            color="red",
            fill=False,
            linestyle="--"
        )
        plt.gca().add_patch(circle)

        # annotation PET
        plt.text(
            cx + 1.0, cy - 1.0,
            f"PET = {pet:.2f}s",
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.8)
        )

        plt.title(f"Interaction PET (Ped {ped_id} / Cyc {cyc_id})")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.gca().invert_yaxis()
        plt.grid()
        plt.legend()
        plt.axis("equal")

        plt.show()

    if return_class:
        from analysis_interactions import classify_pet
        return pet, classify_pet(pet)
    
    return pet


def compute_ttc(df, ped_id, cyc_id, fps, distance_threshold=5.0, plot=False, return_class=False):
    """
    Calcule le TTC (Time-To-Collision) entre un piéton et un cycliste.

    Returns:
        times, ttc_values
    """

    ped = df[df[COL_ID] == ped_id].sort_values(COL_TIME)
    cyc = df[df[COL_ID] == cyc_id].sort_values(COL_TIME)

    # vitesses
    t_p, vx_p, vy_p = compute_velocity_vectors(ped, fps)
    t_c, vx_c, vy_c = compute_velocity_vectors(cyc, fps)

    if t_p is None or t_c is None:
        return None, None

    # positions (alignées sur t_p[1:])
    ped_pos = ped[["x_m", "y_m"]].values[1:]
    cyc_pos = cyc[["x_m", "y_m"]].values[1:]

    # synchronisation temporelle
    common_times = np.intersect1d(t_p, t_c)

    ttc_values = []
    valid_times = []

    for t in common_times:
        i_p = np.where(t_p == t)[0][0]
        i_c = np.where(t_c == t)[0][0]

        # vecteurs 
        # (peut-être que ça ne fonctionne pas là ? car TTC généralement quand usagers dans même direction et l'un devant l'autre d'après Josué)
        r = cyc_pos[i_c] - ped_pos[i_p]
        v = np.array([vx_c[i_c] - vx_p[i_p],
                      vy_c[i_c] - vy_p[i_p]])

        v_norm_sq = np.dot(v, v)

        if v_norm_sq == 0:
            continue

        dot = np.dot(r, v)

        # condition approche 
        if dot >= 0:
            continue

        ttc = - dot / v_norm_sq

        if ttc < 0:
            continue

        ttc_values.append(ttc)
        valid_times.append(t)

    ttc_values = np.array(ttc_values)
    valid_times = np.array(valid_times)

    # TTC min
    idx_min = np.argmin(ttc_values)
    ttc_min = ttc_values[idx_min]
    ttc_min_time = valid_times[idx_min]

    if plot and len(ttc_values) > 0:

        from analysis_interactions import compute_ped_cyc_interactions_with_time

        interactions = compute_ped_cyc_interactions_with_time(df)
        frames = interactions.get((ped_id, cyc_id), [])
        intervals = frames_to_intervals(frames)

        times_d, rel_speeds, rel_speeds_kmh, a = compute_relative_speed(
            df, ped_id, cyc_id, fps, return_distance=False
        )

        t, angles = compute_approach_angle(df, ped_id, cyc_id, fps)

        common_times = times_d
        distances = []

        for t in common_times:
            p = ped[ped[COL_TIME] == t]
            c = cyc[cyc[COL_TIME] == t]

            if p.empty or c.empty:
                distances.append(np.nan)
                continue

            dx = p.iloc[0]["x_m"] - c.iloc[0]["x_m"]
            dy = p.iloc[0]["y_m"] - c.iloc[0]["y_m"]
            distances.append(np.hypot(dx, dy))

        distances = np.array(distances)

        mask = np.zeros_like(valid_times, dtype=bool)
        for start, end in intervals:
            mask |= (valid_times >= start) & (valid_times <= end)
        ttc_inter = ttc_values[mask]
        if len(ttc_inter) == 0:
            return None
        ttc_min_inter = np.min(ttc_inter)

        # plt.figure(figsize=(10, 4))
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        ax_dist, ax_ttc, ax_angle = axes

        ax_ttc.plot(valid_times/fps, ttc_values, color="purple", label="TTC")
        ax_ttc.axhline(2, linestyle="--", color="red", label="Seuil critique (2s)")
        ax_ttc.scatter(
            ttc_min_time/fps,
            ttc_min,
            color="red",
            s=80,
            label=f"TTC min = {ttc_min:.2f}s"
        )
        ax_ttc.set_ylabel("TTC (s)")
        ax_ttc.set_title("Time-To-Collision")
        ax_ttc.grid()
        ax_ttc.legend()
        ax_ttc.text(
            0.02, 0.95,
            f"TTC min = {ttc_min:.2f}s\n(t = {ttc_min_time:.2f})",
            transform=plt.gca().transAxes,
            bbox=dict(facecolor="white", alpha=0.8)
        )

        ax_dist.plot(common_times/fps, distances, color="blue", label="Distance")
        ax_dist.axhline(distance_threshold, linestyle="--", color="red", label="Seuil interaction (5m)")
        ax_dist.set_ylabel("Distance (m)")
        ax_dist.set_title("Distance piéton-cycliste")
        ax_dist.grid()
        ax_dist.legend()

        ax_angle.plot(common_times/fps, angles, color="green", label="Angle")
        ax_angle.axhline(0, linestyle="--", color="black", alpha=0.5, label="Frontal (0°)")
        ax_angle.axhline(90, linestyle="--", color="orange", label="Croisement (90°)") # croisement latéral ou perpendiculaire mais pas forcément de collision
        ax_angle.axhline(180, linestyle="--", color="red", label="Opposition (180°)")
        ax_angle.set_ylabel("Angle (deg)")
        ax_angle.set_xlabel("Temps (s)")
        ax_angle.set_title("Angle d'approche")
        ax_angle.grid()
        ax_angle.legend()

        intervals_sec = [(s / fps, e / fps) for (s, e) in intervals]
        for ax in axes:
            add_time_markers(ax, intervals_sec)

        plt.tight_layout()
        # plt.legend()
        plt.show()
    
    if return_class:
        from analysis_interactions import classify_ttc
        return valid_times, ttc_values, ttc_min, classify_ttc(ttc_min_inter)

    return valid_times, ttc_values, ttc_min


def compute_ttac(df, id_A, id_B, fps, plot=False, return_class=False): # A REVOIR CAR FAIBLE MEME QUAND PAS DE RISQUE
    """
    Calcule le TTAC (Time-To-Avoided-Collision point) entre deux agents.

    TTAC = différence de temps d'arrivée au point de conflit (intersection des trajectoires)

    Returns:
        times, ttac_series, ttac_min, classification (optionnel)
    """

    # récupérer données alignées
    gA = df[df[COL_ID] == id_A].sort_values(COL_TIME)
    gB = df[df[COL_ID] == id_B].sort_values(COL_TIME)

    merged = gA.merge(gB, on=COL_TIME, suffixes=("_A", "_B"))

    if len(merged) < 2:
        return None, None, None, None

    times = merged[COL_TIME].values

    ttac_values = []

    for i in range(len(merged) - 1):

        # positions
        pA = np.array([merged.iloc[i]["x_m_A"], merged.iloc[i]["y_m_A"]])
        pB = np.array([merged.iloc[i]["x_m_B"], merged.iloc[i]["y_m_B"]])

        # vitesses (approx dérivée)
        pA_next = np.array([merged.iloc[i+1]["x_m_A"], merged.iloc[i+1]["y_m_A"]])
        pB_next = np.array([merged.iloc[i+1]["x_m_B"], merged.iloc[i+1]["y_m_B"]])

        vA = (pA_next - pA) * fps
        vB = (pB_next - pB) * fps

        # éviter cas dégénérés
        if np.linalg.norm(vA) < 1e-3 or np.linalg.norm(vB) < 1e-3:
            ttac_values.append(None)
            continue

        # calcul intersection des trajectoires
        # résoudre : pA + tA*vA = pB + tB*vB

        A_mat = np.column_stack((vA, -vB))
        b_vec = pB - pA

        if np.linalg.matrix_rank(A_mat) < 2:
            ttac_values.append(None)
            continue

        try:
            t_vals = np.linalg.solve(A_mat, b_vec)
            tA, tB = t_vals

            # on ne garde que les futurs points
            if tA < 0 or tB < 0:
                ttac_values.append(None)
                continue

            ttac = abs(tA - tB)
            ttac_values.append(ttac)

        except:
            ttac_values.append(None)

    ttac_values = np.array(ttac_values, dtype=float)

    # nettoyage
    valid = ttac_values[np.isfinite(ttac_values)]

    if len(valid) == 0:
        return times[:-1], ttac_values, None, None

    ttac_min = np.min(valid)

    # classification
    def classify_ttac(x):
        if x < 1:
            return "CRITICAL"
        elif x < 2:
            return "HIGH"
        elif x < 4:
            return "MEDIUM"
        else:
            return "LOW"

    c_ttac = classify_ttac(ttac_min)

    # plot
    if plot:
        from analysis_interactions import compute_ped_cyc_interactions_with_time
        interactions = compute_ped_cyc_interactions_with_time(df)

        frames = interactions.get((id_A, id_B), [])
        intervals = frames_to_intervals(frames)
        intervals_sec = [(float(s/fps), float(e/fps)) for s, e in intervals]

        plt.figure(figsize=(8, 4))
        plt.plot(times[:-1]/fps, ttac_values)
        plt.axhline(ttac_min, linestyle="--", label=f"min={ttac_min:.2f}")
        plt.title(f"TTAC en fonction du temps (ped {id_A} - cyc {id_B})")
        plt.xlabel("Temps (s)")
        plt.ylabel("TTAC (s)")
        add_time_markers(plt.gca(), intervals_sec)
        plt.legend()
        plt.grid()
        plt.show()

    if return_class:
        return times[:-1], ttac_values, ttac_min, c_ttac
    else:
        return times[:-1], ttac_values, ttac_min


###############################################
# Critères spatio-temporels des interactions groupe-individu et groupe-goupe
###############################################

def compute_vector_direction_series(df_agent):
    """
    Retourne une série de vecteurs direction normalisés
    entre chaque frame consécutive.
    """
    df_agent = df_agent.sort_values(COL_TIME)

    directions = []
    times = df_agent[COL_TIME].values

    for i in range(1, len(df_agent)):
        dx = df_agent["x_m"].iloc[i] - df_agent["x_m"].iloc[i - 1]
        dy = df_agent["y_m"].iloc[i] - df_agent["y_m"].iloc[i - 1]

        norm = np.hypot(dx, dy)

        if norm == 0:
            directions.append(np.array([0.0, 0.0]))
        else:
            directions.append(np.array([dx / norm, dy / norm]))

    return np.array(directions), times[1:]


def compute_directions_for_ids(df, ids, t):
    # interprétation: [1, 0]-> droite, [0, 1] -> haut, [-1, 0] -> gauche
    dirs = []
    valid_ids = []

    for aid in ids:
        traj = df[df[COL_ID] == aid].sort_values(COL_TIME)

        curr = traj[traj[COL_TIME] == t]
        prev = traj[traj[COL_TIME] == t - 1]

        if curr.empty or prev.empty:
            continue

        dx = curr.iloc[0]["x_m"] - prev.iloc[0]["x_m"]
        dy = curr.iloc[0]["y_m"] - prev.iloc[0]["y_m"]

        norm = np.hypot(dx, dy)
        if norm == 0:
            continue

        dirs.append([dx / norm, dy / norm])
        valid_ids.append(aid)

    return np.array(dirs), np.array(valid_ids)


def compute_direction_variation(directions):
    """
    Calcule les variations angulaires (en radians) entre directions successives.
    Pour déterminer ensuite quel agent est plus récatif que l'autre dans une interaction.
    """

    variations = []

    for i in range(1, len(directions)):
        v1 = directions[i - 1]
        v2 = directions[i]

        dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle = np.arccos(dot)  # radians

        variations.append(angle)

    return np.array(variations)


def compute_distance_series(df, id_A, id_B, fps=None, return_seconds=False):
    """
    Distance entre deux agents au cours du temps.

    Returns:
        times, distances
    """

    A = df[df[COL_ID] == id_A].sort_values(COL_TIME)
    B = df[df[COL_ID] == id_B].sort_values(COL_TIME)

    common_times = np.intersect1d(
        A[COL_TIME].values,
        B[COL_TIME].values
    )

    if len(common_times) == 0:
        return None, None

    A_sync = A[A[COL_TIME].isin(common_times)]
    B_sync = B[B[COL_TIME].isin(common_times)]

    dx = A_sync["x_m"].values - B_sync["x_m"].values
    dy = A_sync["y_m"].values - B_sync["y_m"].values

    distances = np.hypot(dx, dy)

    if return_seconds and fps is not None:
        return common_times / fps, distances

    return common_times, distances


def compute_relative_position_series(df, id_A, id_B, fps=None, return_seconds=False):
    """
    Position relative A par rapport à B (A - B)

    Returns:
        times, dx, dy, distances
    """

    A = df[df[COL_ID] == id_A].sort_values(COL_TIME)
    B = df[df[COL_ID] == id_B].sort_values(COL_TIME)

    common_times = np.intersect1d(
        A[COL_TIME].values,
        B[COL_TIME].values
    )

    # if len(common_times) == 0:
    #     return None

    if len(common_times) == 0:
        # print("ERREUR ICI !!!")
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([])
        )

    A_sync = A[A[COL_TIME].isin(common_times)]
    B_sync = B[B[COL_TIME].isin(common_times)]

    dx = A_sync["x_m"].values - B_sync["x_m"].values
    dy = A_sync["y_m"].values - B_sync["y_m"].values

    distances = np.hypot(dx, dy)

    if return_seconds and fps is not None:
        return common_times / fps, dx, dy, distances

    return common_times, dx, dy, distances



def compute_clusters_and_hulls_over_time(df, min_samples=2,
                                         eps_dir=0.3,
                                         plot=False, fps=None,
                                         save_gif=False, output_path="clusters.gif"):
    """
    Calcule DBSCAN + convex hull frame par frame.

    2 DBSCAN appliqués:
        1. DBSCAN directionnel (clustering en fonciton de la direction des agents)
        2. DBSCAN spatial (clustering en fonction de la distance entre agents allant dans la même direction)

    Returns:
        history[t] = {
            "ped": {...},
            "cyc": {...}
        }
    """

    history = {}

    for t in sorted(df[COL_TIME].unique()):
        frame = df[df[COL_TIME] == t]
        history[t] = {}

        for cls, name in [(1, "ped"), (2, "cyc")]:
            if name == "ped":
                eps = 1.5
            else:
                eps = 2.5

            sub = frame[frame[COL_CLASS] == cls]

            if len(sub) == 0:
                history[t][name] = None
                continue

            points = sub[["x_m", "y_m"]].values
            ids = sub[COL_ID].values

            clusters = []
            clusters_ids = []
            hulls = []

            noise_points = []
            noise_ids = []

            # =========================================================
            # 1) DBSCAN DIRECTIONNEL (premier filtrage)
            # =========================================================
            dirs, valid_ids_all = compute_directions_for_ids(df, ids, t)

            if len(dirs) < 2:
                # pas assez de direction -> fallback spatial direct
                clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
                labels = clustering.labels_
                valid_groups = [(labels, ids, points)]
            else:
                clustering_dir = DBSCAN(eps=eps_dir, min_samples=2, metric="cosine").fit(dirs)
                dir_labels = clustering_dir.labels_

                valid_groups = []

                for dlab in set(dir_labels):
                    mask_dir = dir_labels == dlab
                    sub_ids = valid_ids_all[mask_dir]

                    if dlab == -1:
                        # bruit directionnel -> bruit final direct
                        noise_pts = df[
                            (df[COL_TIME] == t) &
                            (df[COL_ID].isin(sub_ids))
                        ][["x_m", "y_m"]].values

                        noise_points.append(noise_pts)
                        noise_ids.extend(sub_ids)
                        continue

                    sub_pts = df[
                        (df[COL_TIME] == t) &
                        (df[COL_ID].isin(sub_ids))
                    ][["x_m", "y_m"]].values

                    if len(sub_pts) > 0:
                        valid_groups.append((None, sub_ids, sub_pts))

            # =========================================================
            # 2) DBSCAN SPATIAL (sur chaque groupe directionnel)
            # =========================================================
            for _, group_ids, group_pts in valid_groups:

                if len(group_pts) < min_samples:
                    noise_points.append(group_pts)
                    noise_ids.extend(group_ids)
                    continue

                spatial_labels = DBSCAN(
                    eps=eps,
                    min_samples=min_samples
                ).fit_predict(group_pts)

                for lab in set(spatial_labels):

                    mask = spatial_labels == lab
                    pts = group_pts[mask]
                    ids_cluster = np.array(list(group_ids))[mask]

                    if lab == -1:
                        noise_points.append(pts)
                        noise_ids.extend(ids_cluster.tolist())
                        continue

                    clusters.append(pts)
                    clusters_ids.append(set(ids_cluster))

                    if len(pts) >= 3:
                        hulls.append((pts, ConvexHull(pts)))

            # concat bruit
            if len(noise_points) > 0:
                noise_points = np.vstack(noise_points)
            else:
                noise_points = np.empty((0, 2))

            history[t][name] = {
                "points": points,
                "ids": ids,
                "clusters": clusters,
                "clusters_ids": clusters_ids,
                "hulls": hulls,
                "noise": noise_points,
                "noise_ids": np.array(noise_ids),
                "n_clusters": len(clusters),
                "n_noise": len(noise_ids)
            }

    # =========================================================
    # VISUALISATION
    # =========================================================
    if plot:
        times = sorted(history.keys())
        fig, ax = plt.subplots(figsize=(6, 6))

        def update(i):
            ax.clear()

            t = times[i]
            data = history[t]

            legend = {}

            for name, color, noise_color, marker in [
                ("ped", "blue", "cyan", "o"),
                ("cyc", "green", "lime", "s")
            ]:

                if data[name] is None:
                    continue

                # clusters
                for idx, pts in enumerate(data[name]["clusters"]):
                    sc = ax.scatter(
                        pts[:, 0], pts[:, 1],
                        c=color,
                        marker=marker,
                        label=f"{name} cluster {idx}"
                    )
                    legend[f"{name} cluster {idx}"] = sc

                # noise
                if len(data[name]["noise"]) > 0:
                    sc = ax.scatter(
                        data[name]["noise"][:, 0],
                        data[name]["noise"][:, 1],
                        c=noise_color,
                        marker="x",
                        label=f"{name} noise"
                    )
                    legend[f"{name} noise"] = sc

                # hulls
                for pts, hull in data[name]["hulls"]:
                    for simplex in hull.simplices:
                        ln, = ax.plot(
                            pts[simplex, 0],
                            pts[simplex, 1],
                            color=color,
                            linewidth=2
                        )

                # ids
                for (x, y), aid in zip(data[name]["points"], data[name]["ids"]):
                    ax.text(x + 0.1, y + 0.1, str(aid), fontsize=8, color=color)

            ax.set_title(f"Frame {t}")
            ax.set_xlim(df["x_m"].min(), df["x_m"].max())
            ax.set_ylim(df["y_m"].max(), df["y_m"].min())
            ax.grid()

        ani = FuncAnimation(fig, update, frames=len(times), interval=1000 / fps)

        if save_gif:
            ani.save(output_path, writer=PillowWriter(fps=fps))
            print(f"Saved: {output_path}")

        plt.show()

    return history


def match_clusters(prev_clusters, curr_clusters):
    """
    Associe clusters entre t-1 et t via overlap des points.

    prev_clusters / curr_clusters = liste de sets d'IDs

    Returns:
        matches: dict prev_id -> list of curr_ids
    """

    matches = {}

    for i, prev in enumerate(prev_clusters):
        matches[i] = []

        for j, curr in enumerate(curr_clusters):
            overlap = len(prev.intersection(curr))

            if overlap > 0:
                matches[i].append(j)

    return matches


def detect_cluster_splits(prev_clusters, curr_clusters):
    """
    Détecte si un cluster se sépare.

    Returns:
        splits = [(prev_id, [new_ids])]
    """

    matches = match_clusters(prev_clusters, curr_clusters)

    splits = []

    for prev_id, curr_ids in matches.items():
        if len(curr_ids) >= 2:
            splits.append((prev_id, curr_ids))

    return splits


def is_point_in_hull(point, hull_pts):
    """
    Test si un point est dans un convex hull.
    """
    path = Path(hull_pts)
    return path.contains_point(point)


def is_cyclist_near_cluster(cluster_pts, cyclists_pts, threshold=2.0):
    """
    Vérifie si un cycliste est proche d'un cluster.
    Permet de vérifier si la séparation d'un cluster en 2 est due au passage d'un cycliste.
    """
    for c in cyclists_pts:
        dists = np.linalg.norm(cluster_pts - c, axis=1)
        if np.min(dists) < threshold:
            return True
    return False


def detect_split_events_with_cyclists(history, distance_threshold=2.0):
    """
    Détecte les splits de clusters piétons et vérifie si un cycliste est impliqué.
    """

    events = []
    times = sorted(history.keys())

    prev_clusters = None

    for t in times:
        data = history[t]

        if data["ped"] is None:
            prev_clusters = None
            continue

        curr_clusters = data["ped"]["clusters_ids"]
        ped_points = data["ped"]["points"]
        ped_ids = data["ped"]["ids"]

        # cyclistes
        cyc_pts = []
        cyc_ids = []

        if data["cyc"] is not None:
            cyc_pts = data["cyc"]["points"]
            cyc_ids = data["cyc"]["ids"]

        if prev_clusters is not None:

            splits = detect_cluster_splits(prev_clusters, curr_clusters)

            for prev_id, new_ids in splits:

                prev_ids = list(prev_clusters[prev_id])

                # récupérer positions du cluster AVANT split
                mask = np.isin(ped_ids, prev_ids)
                prev_pts = ped_points[mask]

                if len(prev_pts) == 0:
                    continue

                # vérifier cycliste
                involved = False
                involved_cyclists = []

                if len(cyc_pts) > 0:
                    involved = is_cyclist_near_cluster(
                        prev_pts,
                        cyc_pts,
                        threshold=distance_threshold
                    )
                
                for c_pt, c_id in zip(cyc_pts, cyc_ids):
                    dists = np.linalg.norm(prev_pts - c_pt, axis=1)

                    if np.min(dists) < distance_threshold:
                        involved_cyclists.append(c_id)

                events.append({
                    "time": t,
                    "type": "SPLIT",
                    "parent_cluster": prev_id,
                    "child_clusters": new_ids,
                    # "cyclist_involved": involved,
                    "cyclist_involved": len(involved_cyclists) > 0,
                    "cyclist_ids": [int(c) for c in involved_cyclists]
                })

        prev_clusters = curr_clusters

    return events


def detect_cyclists_in_hulls(history):
    results = []

    for t, data in history.items():

        if data["ped"] is None or data["cyc"] is None:
            continue

        cyc_pts = data["cyc"]["points"]
        cyc_ids = data["cyc"]["ids"]

        for i, (pts, hull) in enumerate(data["ped"]["hulls"]):
            hull_pts = pts[hull.vertices]
            cyclists_inside = []

            for c_pt, c_id in zip(cyc_pts, cyc_ids):
                if is_point_in_hull(c_pt, hull_pts):
                    cyclists_inside.append(c_id)

            if len(cyclists_inside) > 0:
                results.append({
                    "time_frame": t,
                    # "time_s": t / fps,
                    "event": "CYCLIST_IN_HULL",
                    "hull_id": i,
                    "cyclist_ids": [int(c) for c in cyclists_inside]
                })

    return results


def min_distance_inter_agent(A, B):
    """
    A, B : arrays (N,2) et (M,2). Distance minimale inter-agent (cycliste VS piéton, ou cycliste VS groupe).
    """
    if len(A) == 0 or len(B) == 0:
        return np.nan

    dists = cdist(A, B)
    return np.min(dists)


def min_distance_point_cluster(point, cluster_pts):
    d = np.linalg.norm(cluster_pts - point, axis=1)
    return np.min(d)


def min_distance_clusters(a_pts, b_pts):
    # Calcule la distance minimale entre 2 clusters, soit entre les 2 points de chaque cluster les plus proches
    d = np.linalg.norm(a_pts[:, None, :] - b_pts[None, :, :], axis=2)
    return np.min(d)


def hausdorff_distance(A, B):
    if len(A) == 0 or len(B) == 0:
        return np.nan

    d_ab = directed_hausdorff(A, B)[0]
    d_ba = directed_hausdorff(B, A)[0]

    return max(d_ab, d_ba)


def modified_hausdorff(A, B):
    if len(A) == 0 or len(B) == 0:
        return np.nan

    dists = cdist(A, B)

    mean_ab = np.mean(np.min(dists, axis=1))
    mean_ba = np.mean(np.min(dists, axis=0))

    return max(mean_ab, mean_ba)


def compute_cluster_distances(history, fps=None, plot=False):
    results = {}

    for t, data in history.items():

        if data["ped"] is None or data["cyc"] is None:
            continue

        cyc = data["cyc"]["points"]
        ped_clusters = data["ped"]["clusters"]

        results[t] = []

        for i, cluster in enumerate(ped_clusters):

            res = {
                "cluster_id": i,
                "min_dist": min_distance_inter_agent(cluster, cyc),
                "hausdorff": hausdorff_distance(cluster, cyc),
                "mod_hausdorff": modified_hausdorff(cluster, cyc)
            }

            results[t].append(res)
    
    if plot:
        # récupérer tous les cluster_ids existants
        cluster_ids = set()
        for t in results:
            for r in results[t]:
                cluster_ids.add(r["cluster_id"])

        cluster_ids = sorted(cluster_ids)

        times = sorted(results.keys())
        times_arr = np.array(times) / fps if fps else np.array(times)

        plt.figure(figsize=(10,5))

        for cid in cluster_ids:

            min_vals = []
            haus_vals = []
            mod_vals = []

            for t in times:
                r_list = results.get(t, [])

                r = next((r for r in r_list if r["cluster_id"] == cid), None)

                if r is None:
                    min_vals.append(np.nan)
                    haus_vals.append(np.nan)
                    mod_vals.append(np.nan)
                else:
                    min_vals.append(r["min_dist"])
                    haus_vals.append(r["hausdorff"])
                    mod_vals.append(r["mod_hausdorff"])

            # plot
            plt.plot(times_arr, min_vals, label=f"Cluster {cid} - min")
            plt.plot(times_arr, haus_vals, linestyle="--", label=f"Cluster {cid} - haus")
            plt.plot(times_arr, mod_vals, linestyle=":", label=f"Cluster {cid} - mod")

        plt.xlabel("Temps (s)" if fps else "Frame")
        plt.ylabel("Distance (m)")
        plt.title("Distances cluster piétons - cyclistes")
        plt.legend(fontsize=8)
        plt.grid()

        plt.show()


    return results



def detect_ped_cluster_interactions(history, df, threshold=5.0):
    frame_events = {}

    for t, data in history.items():

        if data["ped"] is None or data["cyc"] is None:
            continue

        ped_clusters = data["ped"]["clusters_ids"]
        cyc_points = df[(df[COL_TIME] == t) & (df[COL_CLASS] == 2)]

        if len(cyc_points) == 0:
            continue

        events = []

        for _, cyc in cyc_points.iterrows():
            cyc_id = cyc[COL_ID]
            cyc_pos = np.array([cyc["x_m"], cyc["y_m"]])

            for ped_cluster in ped_clusters:

                ped_pts = df[
                    (df[COL_TIME] == t) &
                    (df[COL_ID].isin(ped_cluster))
                ][["x_m", "y_m"]].values

                if len(ped_pts) == 0:
                    continue

                d = min_distance_point_cluster(cyc_pos, ped_pts)

                if d <= threshold:
                    events.append({
                        "cyclist": cyc_id,
                        "ped_cluster": ped_cluster,
                        "distance": d,
                        "type": "CYCLIST_INDIVIDUAL - PEDESTRIAN_CLUSTER"
                    })

        frame_events[t] = events

    return frame_events


def detect_cluster_interactions(history, df, threshold=5.0):
    frame_events = {}

    for t, data in history.items():

        if data["ped"] is None or data["cyc"] is None:
            continue

        ped_clusters = data["ped"]["clusters_ids"]
        cyc_clusters = data["cyc"]["clusters_ids"]

        events = []

        for cyc_cluster in cyc_clusters:
            cyc_pts = df[
                (df[COL_TIME] == t) &
                (df[COL_ID].isin(cyc_cluster))
            ][["x_m", "y_m"]].values

            if len(cyc_pts) == 0:
                continue

            for ped_cluster in ped_clusters:
                ped_pts = df[
                    (df[COL_TIME] == t) &
                    (df[COL_ID].isin(ped_cluster))
                ][["x_m", "y_m"]].values

                if len(ped_pts) == 0:
                    continue

                d = min_distance_clusters(cyc_pts, ped_pts)

                if d <= threshold:
                    events.append({
                        "cyclist_cluster": cyc_cluster,
                        "ped_cluster": ped_cluster,
                        "distance": d,
                        "type": "CYCLIST_CLUSTER - PEDESTRIAN_CLUSTER"
                    })

        frame_events[t] = events

    return frame_events


def extract_entities(frame_data):
    """
    Retourne les entités avec leurs IDs :
    - clusters (avec ids)
    - noise (avec ids)
    """

    entities = []

    for cls_name in ["ped", "cyc"]:
        data = frame_data[cls_name]
        if data is None:
            continue

        # ===== CLUSTERS =====
        for pts, ids in zip(data["clusters"], data["clusters_ids"]):
            entities.append({
                "type": cls_name + "_cluster",
                "points": pts,
                "ids": set(ids)
            })

        # ===== NOISE =====
        if len(data["noise"]) > 0:
            # chaque point bruit = entité individuelle
            for pt, aid in zip(data["noise"], data["noise_ids"]):
                entities.append({
                    "type": cls_name + "_noise",
                    "points": np.array([pt]),
                    "ids": {aid}
                })

    return entities


def min_distance(A, B):
    return np.min(cdist(A, B))


def detect_interactions_at_frame(frame_data, threshold=5.0):
    """
    Interactions à UNE frame avec IDs
    """

    entities = extract_entities(frame_data)
    interactions = []

    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):

            A = entities[i]
            B = entities[j]

            d = min_distance(A["points"], B["points"])

            if d <= threshold:
                if A["type"].startswith("ped") and B["type"].startswith("cyc"):
                    interactions.append({
                        "type_ped": A["type"],
                        "type_cyc": B["type"],
                        "ids_ped": A["ids"],
                        "ids_cyc": B["ids"]
                    })

                elif A["type"].startswith("cyc") and B["type"].startswith("ped"):
                    interactions.append({
                        "type_ped": B["type"],
                        "type_cyc": A["type"],
                        "ids_ped": B["ids"],
                        "ids_cyc": A["ids"]
                    })

    return interactions


def same_interaction(inter, active_inter):
    """
    Vérifie si une interaction correspond à une interaction active
    Critère : overlap des IDs
    """

    if inter["type_ped"] != active_inter["type_ped"]:
        return False
    if inter["type_cyc"] != active_inter["type_cyc"]:
        return False

    overlap_A = len(inter["ids_ped"].intersection(active_inter["ids_ped"])) > 0
    overlap_B = len(inter["ids_cyc"].intersection(active_inter["ids_cyc"])) > 0

    return overlap_A and overlap_B


def build_interaction_events(history, threshold=5.0, fps=None):
    """
    Construit des interactions (avec cluster et bruit) avec :
    - start
    - end
    - ids impliqués (union sur le temps)
    Fonciton finale qui doit être utilisée pour capter toutes les interactions d'un ensemble de trajectoires filtré.
    """

    active_events = []
    finished_events = []

    for t in sorted(history.keys()):

        frame_data = history[t]
        interactions = detect_interactions_at_frame(frame_data, threshold)

        updated = [False] * len(active_events)

        for inter in interactions:

            matched = False

            for i, act in enumerate(active_events):

                if same_interaction(inter, act):

                    # update event
                    act["end"] = t
                    act["ids_ped"].update(inter["ids_ped"])
                    act["ids_cyc"].update(inter["ids_cyc"])

                    updated[i] = True
                    matched = True
                    break

            if not matched:
                # nouvelle interaction
                active_events.append({
                    "type_ped": inter["type_ped"],
                    "type_cyc": inter["type_cyc"],
                    "ids_ped": set(inter["ids_ped"]),
                    "ids_cyc": set(inter["ids_cyc"]),
                    "start": t,
                    "end": t
                })
                updated.append(True)

        # fermer celles non vues
        new_active = []
        for i, act in enumerate(active_events):
            if i < len(updated) and updated[i]:
                new_active.append(act)
            else:
                finished_events.append(act)

        active_events = new_active

    # fermer les restantes
    finished_events.extend(active_events)

    allowed_pairs = {
        frozenset(["cyc_noise", "ped_cluster"]),
        frozenset(["ped_cluster", "cyc_noise"]),
        frozenset(["ped_cluster", "cyc_cluster"]),
        frozenset(["cyc_cluster", "ped_cluster"]),
        frozenset(["cyc_cluster", "ped_noise"]),
        frozenset(["ped_noise", "cyc_cluster"]),
        frozenset(["cyc_noise", "ped_noise"]),
        frozenset(["ped_noise", "cyc_noise"])
    }

    filtered = []

    for e in finished_events:
        pair = frozenset([e["type_ped"], e["type_cyc"]])

        # if pair in allowed_pairs:
        #     filtered.append(e)

        if pair not in allowed_pairs:
            continue
            
        # filtre durée (une interaction doit au moins durer 1 sec)
        if fps is not None:
            duration_frames = e["end"] - e["start"]
            duration_s = duration_frames / fps

            if duration_s < 0.1:
                continue

        filtered.append(e)


    return filtered


def get_closest_points_with_ids(A_pts, B_pts, A_ids, B_ids):
    min_dist = np.inf
    best = None

    for i, a in enumerate(A_pts):
        for j, b in enumerate(B_pts):
            d = np.linalg.norm(a - b)
            if d < min_dist:
                min_dist = d
                best = (a, b, A_ids[i], B_ids[j])

    return best  # (pA, pB, idA, idB)


def compute_agent_velocity(df, agent_id, t, fps):
    traj = df[df[COL_ID] == agent_id].sort_values(COL_TIME)

    if t not in traj[COL_TIME].values:
        return None

    idx = traj[traj[COL_TIME] == t].index[0]

    if idx == traj.index.min():
        return None

    prev = traj.loc[idx - 1]

    curr = traj.loc[idx]

    dt_frames = curr[COL_TIME] - prev[COL_TIME]
    dt = dt_frames / fps

    if dt == 0:
        return None

    vx = (curr["x_m"] - prev["x_m"]) / dt
    vy = (curr["y_m"] - prev["y_m"]) / dt

    return np.array([vx, vy])


def compute_velocity(df_agent):
    df_agent = df_agent.sort_values(COL_TIME)

    dx = np.diff(df_agent["x_m"])
    dy = np.diff(df_agent["y_m"])
    dt = np.diff(df_agent[COL_TIME])

    speeds = np.hypot(dx, dy) / np.maximum(dt, 1e-6)
    return np.mean(speeds) if len(speeds) > 0 else 0


def compute_group_velocity_at_t(df, ids, t, fps):
    speeds = []

    for aid in ids:
        v = compute_agent_velocity(df, aid, t, fps)
        if v is not None:
            speeds.append(np.linalg.norm(v))  # norme

    if len(speeds) == 0:
        return None
    
    return np.mean(speeds) if speeds else None


def compute_group_relative_speed(df, ped_ids, cyc_ids, t, fps):
    v_ped = compute_group_velocity_at_t(df, ped_ids, t, fps)
    v_cyc = compute_group_velocity_at_t(df, cyc_ids, t, fps)

    if v_ped is None or v_cyc is None:
        return None

    return abs(v_cyc - v_ped)


def compute_group_diameter(points):
    if len(points) < 2:
        return 0
    return np.max(cdist(points, points))


def compute_group_density(points):
    if len(points) < 3:
        return 0

    hull = ConvexHull(points)
    area = hull.volume  # en 2D = aire

    return len(points) / area if area > 0 else 0


def compute_direction(df_agent):
    df_agent = df_agent.sort_values(COL_TIME)

    if len(df_agent) < 2:
        return np.array([0, 0])

    dx = df_agent["x_m"].iloc[-1] - df_agent["x_m"].iloc[0]
    dy = df_agent["y_m"].iloc[-1] - df_agent["y_m"].iloc[0]

    vec = np.array([dx, dy])
    norm = np.linalg.norm(vec)

    return vec / norm if norm > 0 else vec



def compute_group_direction_angle(df, ped_ids, cyc_ids, t):
    """
    Angle entre direction moyenne des deux groupes (en degrés)
    basé sur le mouvement temporel (t-1 -> t)
    """

    def group_velocity(ids):
        df_now = df[(df[COL_TIME] == t) & (df[COL_ID].isin(ids))]
        df_prev = df[(df[COL_TIME] == t - 1) & (df[COL_ID].isin(ids))]

        if len(df_now) == 0 or len(df_prev) == 0:
            return None

        # centroid à t
        c_now = df_now[["x_m", "y_m"]].mean().values
        c_prev = df_prev[["x_m", "y_m"]].mean().values

        v = c_now - c_prev
        norm = np.linalg.norm(v)

        if norm == 0:
            return None

        return v / norm

    v_ped = group_velocity(ped_ids)
    v_cyc = group_velocity(cyc_ids)

    if v_ped is None or v_cyc is None:
        return None

    dot = np.clip(np.dot(v_ped, v_cyc), -1.0, 1.0)
    angle = np.degrees(np.arccos(dot))

    return angle


def angle_between(v1, v2):
    dot = np.dot(v1, v2)
    n = np.linalg.norm(v1) * np.linalg.norm(v2)

    if n == 0:
        return 0

    return np.degrees(np.arccos(np.clip(dot / n, -1, 1)))


def get_points_from_ids(frame_data, ids, cls_name):
    data = frame_data[cls_name]
    if data is None:
        return None

    points = []
    for (pt, aid) in zip(data["points"], data["ids"]):
        if aid in ids:
            points.append(pt)

    return np.array(points) if points else None


def compute_interaction_features(event, history, df):
    start = event["start"]
    end = event["end"]

    distances = []
    hausdorffs = []
    hausdorff_mods = []
    # rel_speeds = []
    rel_speeds_series = []
    times = []
    angles = []

    diam_A = []
    diam_B = []
    dens_A = []
    dens_B = []

    for t in range(start, end + 1):

        frame = history.get(t)
        if frame is None:
            continue

        A_type = "cyc" if "cyc" in event["type_ped"] else "ped"
        B_type = "cyc" if "cyc" in event["type_cyc"] else "ped"

        A = get_points_from_ids(frame, event["ids_cyc"], A_type)
        B = get_points_from_ids(frame, event["ids_ped"], B_type)

        if A is None or B is None:
            continue

        # ===== DISTANCES =====
        distances.append(min_distance(A, B))
        hausdorffs.append(hausdorff_distance(A, B))
        hausdorff_mods.append(modified_hausdorff(A, B))

        # ===== VITESSE =====
        vA = compute_group_velocity_at_t(df, event["ids_A"], t)
        vB = compute_group_velocity_at_t(df, event["ids_B"], t)
        # rel_speeds.append(abs(vA - vB))

        if vA is not None and vB is not None:
            rel_speeds_series.append(abs(vA - vB))
            times.append(t)

        # ===== DIAM / DENS =====
        diam_A.append(compute_group_diameter(A))
        diam_B.append(compute_group_diameter(B))

        dens_A.append(compute_group_density(A))
        dens_B.append(compute_group_density(B))

        # ===== ANGLE =====
        # dir_A = compute_direction(df[df[COL_ID].isin(event["ids_A"])])
        dir_A = compute_group_direction_angle(df, event["ids_A"], event["ids_B"], t)
        # vec_AB = relative_position(A, B)

        # angles.append(angle_between(dir_A, vec_AB))
        angles.append(dir_A)

    return {
        "distance": distances,
        "hausdorff": hausdorffs,
        "modified_hausdorff": hausdorff_mods,
        "relative_speed_series": rel_speeds_series,
        "diameter_A_series": diam_A,
        "diameter_B_series": diam_B,
        "density_A_series": dens_A,
        "density_B_series": dens_B,
        "direction_angle_series": angles
    }


def is_noise_only_interaction(event):
    return (
        "noise" in event["type_ped"]
        and "noise" in event["type_cyc"]
    )


def get_entity(frame, ids, cls_name):
    data = frame[cls_name]
    if data is None:
        return None, None

    pts = []
    valid_ids = []

    for p, aid in zip(data["points"], data["ids"]):
        if aid in ids:
            pts.append(p)
            valid_ids.append(aid)

    return np.array(pts), set(valid_ids)


def get_closest_points(A, B):
    D = cdist(A, B)
    i, j = np.unravel_index(np.argmin(D), D.shape)
    return A[i], B[j], D[i, j]


def get_cyclists_in_hull_for_interaction(interaction, hull_events):
    t_start = interaction["start"]
    t_end = interaction["end"]

    cyc_ids = set(interaction["ids_cyc"])

    relevant = []

    for ev in hull_events:
        t = ev["time_frame"]

        if t_start <= t <= t_end:
            # intersection des cyclistes
            if any(c in cyc_ids for c in ev["cyclist_ids"]):
                relevant.append(ev)

    return relevant


def get_split_events_for_interaction(interaction, split_events):
    t_start = interaction["start"]
    t_end = interaction["end"]

    cyc_ids = set(interaction["ids_cyc"])

    relevant = []

    for ev in split_events:
        t = ev["time"]

        if t_start <= t <= t_end:
            if ev["cyclist_involved"]:
                # vérifier que c’est un cycliste de l’interaction
                if any(c in cyc_ids for c in ev["cyclist_ids"]):
                    relevant.append(ev)

    return relevant