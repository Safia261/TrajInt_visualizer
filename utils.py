import numpy as np
import matplotlib.pyplot as plt
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


def compute_speed(g, fps):
    g = g.sort_values(COL_TIME)

    xs = g["x_m"].values
    ys = g["y_m"].values
    times = g[COL_TIME].values

    if len(xs) < 2:
        return None, None

    dx = np.diff(xs)
    dy = np.diff(ys)

    # à voir si je mets pas simplement que la logne suivante
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


def find_closest_ped_cyc(df):
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

def compute_agent_direction(
    df,
    agent_id,
    fps,
    mode="vector",   # "vector" or "angle"
    angle_unit = "red", # "rad" or "deg"
    normalize=False,
    plot=False
):
    """
    Calcule la direction d'un agent à partir de ses déplacements successifs.

    Parameters:
        df : DataFrame
        agent_id : int
        mode : "vector" ou "angle"
        angle_unit: "rad" ou "deg"
        normalize : si True, retourne des vecteurs unitaires
        plot : affiche la direction sur la trajectoire

    Returns:
        - mode="vector" -> array (dx, dy)
        - mode="angle"  -> array theta (radians)
    """

    g = df[df[COL_ID] == agent_id].sort_values(COL_TIME)

    x = g["x_m"].values
    y = g["y_m"].values
    t = g[COL_TIME].values

    if len(x) < 2:
        return None # impossible de calculer la direction

    dx = np.diff(x)
    dy = np.diff(y)

    if normalize:
        norm = np.hypot(dx, dy)
        norm[norm == 0] = 1e-9
        dx = dx / norm
        dy = dy / norm

    if mode == "angle":
        angles = np.arctan2(dy, dx) # en radians
        if angle_unit == "deg":
            angles = np.degrees(angles)
        direction = angles
    else: # sinon s/s forme de vecteur
        direction = np.stack([dx, dy], axis=1)

    
    if plot:

        if mode == "vector":
            plt.figure(figsize=(8, 6))
            plt.plot(x, y, "k--", alpha=0.4, label="trajectory")
            plt.quiver(
                x[:-1], y[:-1],
                dx, dy,
                angles="xy",
                scale_units="xy",
                scale=1,
                color="red"
            )
            plt.title(f"Direction vectors - ID {agent_id}")
            plt.gca().invert_yaxis()
            plt.grid()
            plt.legend()
            plt.show()

        else:
            # plt.plot(t[1:], direction)
            # plt.title(f"Direction angle ({angle_unit}) - ID {agent_id}")
            # plt.ylabel(f"angle ({angle_unit})")
            # plt.xlabel("time")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

            ax1.plot(t[1:]/fps, angles, label="angle")
            ax1.set_xlabel("Temps (s)")
            ax1.set_ylabel(f"Angle ({angle_unit})")  # ou degrés
            ax1.set_title(f"Direction angle ({angle_unit}) au cours du temps - ID {agent_id}")
            ax1.grid()

            ax2.plot(x[:-1], y[:-1], linestyle="--", color="gray", label="trajectory")
            ax2.set_xlabel("x (m)")
            ax2.set_ylabel("y (m)")
            ax2.set_title("Trajectoire spatiale")
            ax2.invert_yaxis()
            ax2.grid()
            sc = ax2.scatter(x[:-1], y[:-1], c=t[1:], cmap="viridis", s=10)
            plt.colorbar(sc, ax=ax2, label="Temps")

            plt.legend()
            plt.show()


    return direction


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

    # ===== synchronisation des timestamps =====
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

    # ===== PLOT =====
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

    # ===== vitesses =====
    t_p, vx_p, vy_p = compute_velocity_vectors(ped, fps)
    t_c, vx_c, vy_c = compute_velocity_vectors(cyc, fps)

    if t_p is None or t_c is None:
        return None, None

    # ===== positions =====
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

        # ===== vecteurs clés =====
        r = np.array([x_p - x_c, y_p - y_c])   # position relative
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

    # ===== PLOT =====
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

    # ===============================
    # 1. Extraction des données
    # ===============================
    ped = df[df[COL_ID] == ped_id].sort_values(COL_TIME)
    cyc = df[df[COL_ID] == cyc_id].sort_values(COL_TIME)

    if len(ped) < 2 or len(cyc) < 2:
        return None

    # ===============================
    # 2. Vecteurs vitesse
    # ===============================
    t_p, vx_p, vy_p = compute_velocity_vectors(ped, fps)
    t_c, vx_c, vy_c = compute_velocity_vectors(cyc, fps)

    # ===============================
    # 3. Synchronisation temporelle
    # ===============================
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

    # ===============================
    # 4. Vitesse relative
    # ===============================
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

    # ===============================
    # 5. Angle d’approche
    # ===============================
    angles = theta
    if angle_unit == "deg":
        angles = np.degrees(angles)

    if plot:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        ax1, ax2, ax3 = axes

        # --- vitesse m/s ---
        ax1.plot(common_times/fps, rel_speeds, label="Vitesse relative (m/s)")
        ax1.set_ylabel("m/s")
        ax1.set_title("Vitesse relative")
        ax1.grid()

        # --- vitesse km/h ---
        ax2.plot(common_times/fps, rel_speeds_kmh, label="Vitesse relative (km/h)", color="orange")
        ax2.set_ylabel("km/h")
        ax2.grid()

        # --- angle ---
        ax3.plot(common_times/fps, angles, label="Angle (deg)", color="green")
        ax3.set_ylabel("Angle (°)")
        ax3.set_xlabel("Temps (s)")
        ax3.grid()

        from analysis_interactions import compute_ped_cyc_interactions_with_time
        interactions = compute_ped_cyc_interactions_with_time(df, distance_threshold)

        pair = tuple(sorted((ped_id, cyc_id)))
        interaction_times = interactions.get(pair, [])

        # ===============================
        # MARQUEURS INTERACTION
        # ===============================
        if len(interaction_times) > 0:
            t_start = min(interaction_times)
            t_end = max(interaction_times)

            for ax in axes:
                ax.axvline(t_start/fps, color="red", linestyle="--", label="début interaction")
                ax.axvline(t_end/fps, color="purple", linestyle="--", label="fin interaction")

        # éviter doublons de légende
        for ax in axes:
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())

        plt.tight_layout()
        plt.show()

    # ===============================
    # 6. Distance (optionnel)
    # ===============================
    distances = None

    if return_distance:
        ped_pos = ped[ped[COL_TIME].isin(times)]
        cyc_pos = cyc[cyc[COL_TIME].isin(times)]

        dx = cyc_pos["x_m"].values - ped_pos["x_m"].values
        dy = cyc_pos["y_m"].values - ped_pos["y_m"].values

        distances = np.hypot(dx, dy)

    # ===============================
    # 7. Résultat
    # ===============================
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

    # === 1. Trouver point de conflit (distance minimale) ===
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

    # === 2. Trouver temps de passage près du point ===
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

    # === 3. PET ===
    pet = abs(t_ped - t_cyc) / fps

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

        # points de passage
        plt.scatter(x_ped, y_ped,
                    color="blue",
                    s=80,
                    edgecolor="black",
                    label="Passage piéton")

        plt.scatter(x_cyc, y_cyc,
                    color="green",
                    s=80,
                    edgecolor="black",
                    label="Passage cycliste")

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

    # synchronisation
    common_times = np.intersect1d(t_p, t_c)

    ttc_values = []
    valid_times = []

    for t in common_times:
        i_p = np.where(t_p == t)[0][0]
        i_c = np.where(t_c == t)[0][0]

        # vecteurs
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

    idx_min = np.argmin(ttc_values)
    ttc_min = ttc_values[idx_min]
    ttc_min_time = valid_times[idx_min]

    # =====================================================
    # PLOT
    # =====================================================
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

        # ax_ttc.plot(valid_times, ttc_values, label="TTC")
        # # zones d'interaction
        # # add_time_markers(ax_ttc.gca(), intervals)
        # ax_ttc.xlabel("Temps")
        # ax_ttc.ylabel("TTC (s)")
        # ax_ttc.title(f"TTC (Cycliste {cyc_id} / Piéton {ped_id})")
        # ax_ttc.grid()
        # # ligne seuil critique
        # ax_ttc.axhline(2, linestyle="--", color="red", label="Seuil critique (2s)")

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

# def compute_ttc(p_rel, v_rel):
#     """
#     p_rel : vecteur position relative (p_c - p_p)
#     v_rel : vecteur vitesse relative (v_c - v_p)
#     """

#     dot = np.dot(p_rel, v_rel)

#     # ===== condition de rapprochement =====
#     if dot >= 0:
#         return None  # ou np.inf

#     norm_v2 = np.dot(v_rel, v_rel)

#     if norm_v2 < 1e-6:
#         return None  # vitesse trop faible

#     ttc = - dot / norm_v2

#     if ttc < 0:
#         return None

#     return ttc