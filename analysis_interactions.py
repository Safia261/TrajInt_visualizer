import numpy as np
import matplotlib.pyplot as plt
from config import *
from utils import *
from statsmodels.nonparametric.smoothers_lowess import lowess # pour lisser les courbes et visualiser les tendances globales
from scipy.stats import linregress


def analyze_initial_nb_traj_interactions(df, verbose=True):

    # Comptage par classe
    initial_counts = df.groupby(COL_CLASS)[COL_ID].nunique().to_dict()

    initial_ped = initial_counts.get(1, 0)
    initial_cyc = initial_counts.get(2, 0)
    # initial_car = initial_counts.get(3, 0)
    initial_vehicle = sum(initial_counts.get(c, 0) for c in VEHICLE_CLASSES)

    initial_ids = set(df[COL_ID].unique())
    initial_nb_traj = len(initial_ids)

    initial_interactions = compute_ped_cyc_interactions(df)
    initial_nb_interactions = len(initial_interactions)

    if verbose:
        print("\nAnalyse avant filtrage")
        print(f"Nb piétons au total : {initial_ped}")
        print(f"Nb cyclistes au total : {initial_cyc}")
        print(f"Nb autres usagers au total : {initial_vehicle}")
        print(f"Trajectoires totales : {initial_nb_traj}")
        print(f"Interactions piéton-cycliste (rayon 5m): {initial_nb_interactions}")

    return initial_nb_traj, initial_nb_interactions



def analyze_cycl_ped_distances(df):
    """
    Fonction pour analyser les distances entre piétons et cyclistes.
    """
    distances = []

    for t in df[COL_TIME].unique():
        frame = df[df[COL_TIME] == t]

        peds = frame[frame[COL_CLASS] == 1]
        cycls = frame[frame[COL_CLASS] == 2]

        if len(peds) == 0 or len(cycls) == 0:
            continue

        for _, ped in peds.iterrows():
            for _, cycl in cycls.iterrows():
                dx = ped["x_m"] - cycl["x_m"]
                dy = ped["y_m"] - cycl["y_m"]
                dist = np.hypot(dx, dy)

                distances.append(dist)

    if len(distances) == 0:
        print("Aucune distance calculée (pas de co-présence piéton/cycliste).")
        return

    distances = np.array(distances)
    mean_val = distances.mean()

    # Percentile
    percentiles = [5, 10, 25, 50, 75, 90, 95]

    print("\nANALYSE DISTANCES PIETON-CYCLISTE")
    print(f"Nombre de distances : {len(distances)}")
    print(f"Distance min        : {distances.min():.2f} m")
    print(f"Distance max        : {distances.max():.2f} m")
    print(f"Distance moyenne    : {mean_val:.2f} m")

    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=50)
    plt.axvline(mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.2f} m")

    for p in percentiles:
        val = np.percentile(distances, p)
        print(f"Percentile {p:>2}%     : {val:.2f} m")
        plt.axvline(val, color="black", linestyle=":", linewidth=1.8, label=f"P{p}: {val:.2f} m")

    plt.xlabel("Distance voiture - VRU (m)")
    plt.ylabel("Fréquence")
    plt.title("Distribution des distances voiture - VRU")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return distances



def analyze_car_vru_distances(df):
    """
    Analyzes distances between users to determine the distance threshold
    defining potential vehicle influence on VRU trajectories.
    """
    distances = []

    for t in df[COL_TIME].unique():
        frame = df[df[COL_TIME] == t]

        cars = frame[frame[COL_CLASS] == 3]
        vrus = frame[frame[COL_CLASS].isin([1, 2])]

        if len(cars) == 0 or len(vrus) == 0:
            continue

        for _, vru in vrus.iterrows():
            for _, car in cars.iterrows():
                dx = vru["x_m"] - car["x_m"]
                dy = vru["y_m"] - car["y_m"]
                dist = np.hypot(dx, dy)

                distances.append(dist)

    if len(distances) == 0:
        print("Aucune distance calculée (pas de co-présence voiture/VRU).")
        return

    distances = np.array(distances)
    mean_val = distances.mean()

    # Percentiles (peuvent être intéressants pour voir le pourcentage des distances des interactions les plus proches et pour choisir seuil)
    percentiles = [5, 10, 25, 50, 75, 90, 95]

    print("\nANALYSE DISTANCES VOITURE - VRU")
    print(f"Nombre de distances : {len(distances)}")
    print(f"Distance min        : {distances.min():.2f} m")
    print(f"Distance max        : {distances.max():.2f} m")
    print(f"Distance moyenne    : {mean_val:.2f} m")

    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=50)
    plt.axvline(mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.2f} m")

    for p in percentiles:
        val = np.percentile(distances, p)
        print(f"Percentile {p:>2}%     : {val:.2f} m")
        plt.axvline(val, color="black", linestyle=":", linewidth=1.8, label=f"P{p}: {val:.2f} m")

    plt.xlabel("Distance voiture - VRU (m)")
    plt.ylabel("Fréquence")
    plt.title("Distribution des distances voiture - VRU")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return distances


def analyze_speeds(df, cfg, fps, agent_ids=None, classes=None, start=None, end=None):
    """
    Fonciton pour analyser la vitesse d'un ou plusieurs au cours de leur trajectoire.
    """
    fps = cfg.get("fps", 1.0)
    selected_agents = []

    if agent_ids is not None:
        # --- sélection manuelle des agents ---
        for aid in agent_ids:
            sub = df[df[COL_ID] == aid]
            if len(sub) == 0:
                print(f"Agent {aid} non trouvé")
                continue

            cls = int(sub[COL_CLASS].iloc[0])

            if classes is None or cls in classes:
                selected_agents.append((aid, cls))

    else:
        # sélection automatique (interaction piéton-cycliste avec paire d'agents ayant été les plus proches spatialement)
        ped_id, cyc_id = find_closest_ped_cyc(df)

        if ped_id is None or cyc_id is None:
            print("Aucune interaction trouvée")
            return

        selected_agents.append((ped_id, 1))
        selected_agents.append((cyc_id, 2))

    if len(selected_agents) == 0:
        print("Aucun agent sélectionné")
        return

    print("\nAgents sélectionnés :")
    for aid, cls in selected_agents:
        print(f"- ID {aid} ({CLASS_NAMES.get(cls)})")

    # calcul vitesses
    data = []

    for aid, cls in selected_agents:
        g = df[df[COL_ID] == aid]

        times, speeds_ms, speeds_kmh = compute_speed(g, fps)

        if times is None:
            continue

        data.append({
            "id": aid,
            "class": cls,
            "times": times,
            "speeds_ms": speeds_ms,
            "speeds_kmh": speeds_kmh
        })

    if len(data) == 0:
        print("Pas de données exploitables")
        return
    
    # Intervalles d'interaction (uniquement si un piéton et un cycliste sont sélectionnés)
    intervals_sec = None

    if len(selected_agents) == 2:
        ped_ids = [aid for aid, cls in selected_agents if cls == 1]
        cyc_ids = [aid for aid, cls in selected_agents if cls == 2]

        if len(ped_ids) == 1 and len(cyc_ids) == 1:
            if start is None and end is None:
                from analysis_interactions import compute_ped_cyc_interactions_with_time
                interactions = compute_ped_cyc_interactions_with_time(df)
                pair = tuple(sorted((ped_ids[0], cyc_ids[0])))
                frames = interactions.get(pair, [])
                intervals = frames_to_intervals(frames)
                intervals_sec = [(s /fps, e / fps) for s, e in intervals]
            else:
                intervals_sec = [(start/fps, end/fps)]

    # Plot
    fig, ax_ms = plt.subplots(figsize=(18, 6))
    
    # courbes individuelles (m/s)
    for d in data:
        aid = d["id"]
        cls = d["class"]
        times = d["times"]
        speeds_ms = d["speeds_ms"]

        color = CLASS_COLORS.get(cls, "black")
        label = f"{CLASS_NAMES.get(cls)} {aid} speed"

        ax_ms.plot(times/fps, speeds_ms, label=label, color=color)

        smooth = lowess(speeds_ms, times/fps,frac=0.05)
        ax_ms.plot(smooth[:,0], smooth[:,1], linewidth=3, color ="red", label="Smoothed speed", alpha=0.5)

    ax_ms.set_title("Speed evolution (m/s)", fontsize=15, fontweight="bold")
    ax_ms.set_xlabel("Time (s)", fontsize=15)
    ax_ms.set_ylabel("Speed (m/s)", fontsize=15)
    ax_ms.tick_params(axis="both", labelsize=13)
    if intervals_sec is not None:
        add_time_markers(ax_ms, intervals_sec)
    ax_ms.grid()

    # axe secondaire (km/h)
    ax_kmh = ax_ms.twinx()
    ax_kmh.tick_params(axis="both", labelsize=13)
    for d in data:
        aid = d["id"]
        cls = d["class"]
        times = d["times"]
        speeds_kmh = d["speeds_kmh"]

        color = CLASS_COLORS.get(cls, "black")
        label = f"{CLASS_NAMES.get(cls)} {aid} speed"

        ax_kmh.plot(times/fps, speeds_kmh, label=label, color=color)

    ax_kmh.set_ylabel("Speed (km/h)", fontsize=15)

    # Légende fusionnée
    handles1, labels1 = ax_ms.get_legend_handles_labels()
    handles2, labels2 = ax_kmh.get_legend_handles_labels()
    by_label = dict(zip(labels1 + labels2, handles1 + handles2))
    ax_ms.legend(by_label.values(), by_label.keys(), fontsize=15)

    plt.tight_layout()
    plt.show()



def compute_ped_cyc_interactions(df, distance_threshold=5.0):
    """
    For now, an interaction is defined when a pedestrian and a cyclist are within a 5 m radius of each other.
    """
    interactions = set()

    for t in df[COL_TIME].unique():
        frame = df[df[COL_TIME] == t]

        pedestrians = frame[frame[COL_CLASS] == 1]
        cyclists = frame[frame[COL_CLASS] == 2]

        if len(pedestrians) == 0 or len(cyclists) == 0:
            continue

        for _, ped in pedestrians.iterrows():
            for _, cyc in cyclists.iterrows():
                dx = ped["x_m"] - cyc["x_m"]
                dy = ped["y_m"] - cyc["y_m"]
                dist = np.hypot(dx, dy)

                if dist < distance_threshold:
                    # tuple trié pour éviter doublons
                    pair = tuple(sorted((ped[COL_ID], cyc[COL_ID])))
                    interactions.add(pair)
    
    # print(interactions)

    return interactions


def compute_ped_cyc_interactions_with_time(df, distance_threshold=5.0):
    interactions = {}  # (ped, cyc) -> liste de frames

    for t in df[COL_TIME].unique():
        frame = df[df[COL_TIME] == t]

        pedestrians = frame[frame[COL_CLASS] == 1]
        cyclists = frame[frame[COL_CLASS] == 2]

        if len(pedestrians) == 0 or len(cyclists) == 0:
            continue

        for _, ped in pedestrians.iterrows():
            for _, cyc in cyclists.iterrows():
                dx = ped["x_m"] - cyc["x_m"]
                dy = ped["y_m"] - cyc["y_m"]
                dist = np.hypot(dx, dy)

                if dist < distance_threshold:
                    pair = tuple(sorted((ped[COL_ID], cyc[COL_ID])))

                    if pair not in interactions:
                        interactions[pair] = []

                    interactions[pair].append(t)

    return interactions


def classify_interactions_with_car_in_time(df, distance_threshold=5.0):
    """
    Classifies pedestrian–cyclist interactions over time based on the appearance of a car within a 5 m radius of either VRU
    """
    interactions = compute_ped_cyc_interactions_with_time(df, distance_threshold)

    car_frames = set(df[df[COL_CLASS] == 3][COL_TIME].unique())

    results = {}

    for pair, frames in interactions.items():
        frames = sorted(frames)

        has_during = any(f in car_frames for f in frames)

        if has_during:
            label = "PENDANT"
        else:
            first_frame = frames[0]

            cars_before = any(f < first_frame for f in car_frames)
            cars_after = any(f > frames[-1] for f in car_frames)

            if cars_before and not cars_after:
                label = "APRÈS voiture"
            elif cars_after and not cars_before:
                label = "AVANT voiture"
            elif cars_before and cars_after:
                label = "ENTOURÉE par voitures"
            else:
                label = "SANS voiture"

        results[pair] = label

    return results


def classify_interactions_with_car_in_time_and_space(df, distance_threshold=5.0, ind=False):
    interactions = compute_ped_cyc_interactions_with_time(df, distance_threshold)
    print(f"\nInteractions initiales: {compute_ped_cyc_interactions(df)}")

    results = {}

    # voitures actives (inD)
    active_cars = None
    if ind:
        car_speeds = {}
        for car_id, g in df[df[COL_CLASS] == 3].groupby(COL_ID):
            if "xVelocity" in g.columns and "yVelocity" in g.columns:
                speeds = np.hypot(g["xVelocity"], g["yVelocity"])
                mean_speed = speeds.mean()
            else:
                mean_speed = 0
            car_speeds[car_id] = mean_speed

        active_cars = {cid for cid, s in car_speeds.items() if s > 0.1}

    # analyse
    for (ped_id, cyc_id), frames in interactions.items():
        frames = sorted(frames)

        influencing_cars = set()
        has_during_close = False

        for t in frames:
            frame = df[df[COL_TIME] == t]

            ped = frame[frame[COL_ID] == ped_id]
            cyc = frame[frame[COL_ID] == cyc_id]

            if ped.empty or cyc.empty:
                continue

            ped = ped.iloc[0]
            cyc = cyc.iloc[0]

            if ind:
                cars = frame[(frame[COL_CLASS] == 3) & (frame[COL_ID].isin(active_cars))]
            else:
                # cars = frame[frame[COL_CLASS] == 3]
                cars = frame[(frame[COL_CLASS].isin(VEHICLE_CLASSES))]

            for _, car in cars.iterrows():
                dist_p = np.hypot(ped["x_m"] - car["x_m"], ped["y_m"] - car["y_m"])
                dist_c = np.hypot(cyc["x_m"] - car["x_m"], cyc["y_m"] - car["y_m"])

                if dist_p < distance_threshold or dist_c < distance_threshold:
                    has_during_close = True
                    influencing_cars.add(car[COL_ID])
                    # break

            if has_during_close:
                break

        # classification
        if has_during_close:
            label = "PENDANT_PROCHE"
        else:
            # car_frames = set(df[df[COL_CLASS] == 3][COL_TIME].unique())
            car_frames = set(df[df[COL_CLASS].isin(VEHICLE_CLASSES)][COL_TIME].unique())

            cars_before = any(f < frames[0] for f in car_frames)
            cars_after = any(f > frames[-1] for f in car_frames)

            if cars_before and not cars_after:
                label = "APRÈS voiture"
            elif cars_after and not cars_before:
                label = "AVANT voiture"
            elif cars_before and cars_after:
                label = "ENTOURÉE"
            else:
                label = "SANS voiture"

        results[(ped_id, cyc_id)] = {"label": label, "cars": influencing_cars}

    return results



###############################################
# Computation of the spatio-temporal criteria
###############################################

def compute_pair_interaction_features(event, df, fps=None, plot=False):
    """
    Computes the spatio-temporal criteria of an individual-individual interaction.
    """
    ids_A = list(event["ids_ped"])
    ids_B = list(event["ids_cyc"])

    start = event["start"]
    end = event["end"]

    if len(ids_A) != 1 or len(ids_B) != 1:
        return None  # juste par sécurité

    id_A = ids_A[0]
    id_B = ids_B[0]

    # DISTANCES
    times, distances = compute_distance_series(df, id_A, id_B, fps=fps, start=start, end=end, plot=plot)

    # ANGLE DIRECTION
    t_dir, angles_dir = compute_direction_angle_velocity_based(df, id_A, id_B, fps, start=start, end=end, plot=plot)
    angles_dir_i = None
    if t_dir is not None:
        mask_dir = (t_dir >= start) & (t_dir <= end)
        angles_dir_i = angles_dir[mask_dir]

    # ANGLE APPROCHE
    t_app, angles_app = compute_approach_angle(df, id_A, id_B, fps, start=start, end=end, plot=plot)
    angles_app_i = None
    if t_app is not None:
        mask_app = (t_app >= start) & (t_app <= end)
        angles_app_i = angles_app[mask_app]

    if plot:
        min_dist_idx = np.argmin(distances)
        min_dist = distances[min_dist_idx]
        min_dist_time = times[min_dist_idx] / fps

        plot_interaction_series(
        series=[
            {
                "times": times/fps,
                "values": distances,
                "label": "Distance",
                "color": "royalblue",
                "axis": "left"
            },
            {
                "times": t_app/fps,
                "values": angles_app,
                "label": "Approach angle",
                "color": "darkorange",
                "axis": "right"
            }
        ],
        interaction_intervals=[(start/fps, end/fps)],
        title="Pedestrian-Cyclist interaction",
        ylabel_left="Distance (m)",
        ylabel_right="Angle (°)",
        vertical_lines=[{
            "x": min_dist_time,
            "label": f"Min dist = {min_dist:.2f} m at t={min_dist_time:.2f}s",
            "color": "red"
        }]
        )

    # VITESSE RELATIVE
    t_rel, _, rel_speeds_kmh, _, _ = compute_relative_speed(df, id_A, id_B, fps, start=start, end=end, plot=plot)
    rel_speeds_kmh_i = None
    if t_rel is not None:
        mask_rel = (t_rel >= start) & (t_rel <= end)
        rel_speeds_kmh_i = rel_speeds_kmh[mask_rel]

    # POSITION RELATIVE
    _, dx, dy, _ = compute_relative_position_series(df, id_A, id_B, fps=fps, start=start, end=end, plot=plot)

    mask = np.zeros_like(times, dtype=bool)
    mask |= (times >= start) & (times <= end)

    # REACTIVITE AGENT (variation vitesse et direction)
    ped_df = df[df[COL_ID] == id_A]
    cyc_df = df[df[COL_ID] == id_B]
    speed_var_ped = compute_speed_variation_with_ref(ped_df, start, end, fps)
    speed_var_cyc = compute_speed_variation_with_ref(cyc_df, start, end, fps)
    spatial_var_ped = compute_spatial_deviation(ped_df, start, end)
    spatial_var_cyc = compute_spatial_deviation(cyc_df, start, end)
    # reactive_agent, stable_agent = detect_reactive_agent(df, ped_df, cyc_df)

    return {
        "distances": distances[mask],
        "direction_angle_series": angles_dir_i,
        "approach_angle_series": angles_app_i,
        "relative_speed_series": rel_speeds_kmh_i,
        "relative_position_series": np.column_stack((dx[mask], dy[mask])),
        "PET": compute_pet(df, id_A, id_B, fps, plot=plot),
        "TTAC": compute_ttac(df, id_A, id_B, fps, start=start, end=end, plot=plot),
        "speed_var_ped": speed_var_ped,
        "speed_var_cyc": speed_var_cyc,
        "spatial_var_ped": spatial_var_ped,
        "spatial_var_cyc": spatial_var_cyc
    }


def compute_group_interaction_features(event, history, df, split_events, hull_events, fps=None, plot=False):
    """
    Computes the spatio-temporal criteria of an interaction with 1 ou 2 groups.
    """

    times = []

    start = event["start"]
    end = event["end"]

    distances = []
    hausdorffs = []
    hausdorff_mods = []

    rel_positions = []
    rel_speeds = []
    direction_angles = []

    densities_A = []
    densities_B = []

    for t in range(start, end + 1):
        times.append(t)

        frame = history.get(t)
        if frame is None:
            continue

        A_type = "ped" if "ped" in event["type_ped"] else "cyc"
        B_type = "cyc" if "cyc" in event["type_cyc"] else "ped"

        A, ids_A = get_entity(frame, event["ids_ped"], A_type)
        B, ids_B = get_entity(frame, event["ids_cyc"], B_type)

        if A is None or B is None or len(A) == 0 or len(B) == 0:
            continue

        # DISTANCES
        pA, pB, dmin = get_closest_points(A, B)
        distances.append(dmin)

        hausdorffs.append(hausdorff_distance(A, B))
        hausdorff_mods.append(modified_hausdorff(A, B))

        # POSITION RELATIVE
        rel_positions.append(pB - pA)

        # VITESSE RELATIVE (moyenne)
        rel_speeds.append(compute_group_relative_speed(df, ids_A, ids_B, t, fps))

        # ANGLE DIRECTION
        dir_A = compute_group_direction_angle(df, ids_A, ids_B, t)

        direction_angles.append(dir_A)

        # DENSITE
        densities_A.append(compute_group_density(A))
        densities_B.append(compute_group_density(B))

        # CONVEX HULL
        hull_events_inter = get_cyclists_in_hull_for_interaction(event, hull_events)
        split_events_inter = get_split_events_for_interaction(event, split_events)
    
    # conversion en km/h
    rel_speeds_clean = np.array([
        v * 3.6 if v is not None else np.nan
        for v in rel_speeds
    ])

    features= {
        "time": times,
        "distances": distances,
        "hausdorff": hausdorffs,
        "modified_hausdorff": hausdorff_mods,

        "relative_position_series": rel_positions,
        "relative_speed_series": rel_speeds_clean, # en km/h
        "direction_angle_series": direction_angles,

        "density_ped_series": densities_A, # pedestrian gpe
        "density_cyc_series": densities_B, # cyclist gpe

        "cyclist_in_hull": len(hull_events_inter) > 0,
        "cyclist_in_hull_events": hull_events_inter,

        "cluster_split": len(split_events_inter) > 0,
        "cluster_split_events": split_events_inter
    }

    if plot:
        plot_group_interaction_features(features, fps, start, end)

    return features



def plot_group_interaction_features(features, fps, start, end):
    t = np.array(features["time"]) / fps

    from scipy.stats import linregress
    def plot_series(y, title, ylabel, start, end, show_density_stats=False, show_ms_axis=False, distance=False, angle_type=None):
        if y is None or len(y) == 0:
            return

        y = np.asarray(y)
        tt = t[:len(y)]

        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(tt, y, label=ylabel)

        if show_density_stats and len(y) > 1:

            # Régression linéaire
            slope, intercept, r, p, _ = linregress(tt, y)
            y_fit = intercept + slope * tt

            ax1.plot(
                tt,
                y_fit,
                '--',
                color='red',
                linewidth=2,
                label='Linear trend'
            )

            # Min / Max
            idx_min = np.argmin(y)
            idx_max = np.argmax(y)

            ax1.scatter(tt[idx_min], y[idx_min], color='blue', zorder=5)
            ax1.scatter(tt[idx_max], y[idx_max], color='green', zorder=5)

            ax1.annotate(
                f"Min = {y[idx_min]:.2f}",
                (tt[idx_min], y[idx_min]),
                xytext=(5, -15),
                textcoords="offset points"
            )

            ax1.annotate(
                f"Max = {y[idx_max]:.2f}",
                (tt[idx_max], y[idx_max]),
                xytext=(5, 10),
                textcoords="offset points"
            )

            ax1.text(
                0.02,
                0.98,
                f"Slope = {slope:.3f}",
                transform=plt.gca().transAxes,
                ha="left",
                va="top",
                bbox=dict(facecolor="white", alpha=0.8)
            )

        if show_ms_axis:
            ax2 = ax1.twinx()
            # même échelle mais convertie en m/s
            ymin, ymax = ax1.get_ylim()
            ax2.set_ylim(ymin / 3.6, ymax / 3.6)
            ax2.set_ylabel("Speed (m/s)", fontsize=15)

        if distance == True:
            idx_min = np.nanargmin(y)
            min_value = y[idx_min]
            min_time = tt[idx_min]

            ax1.scatter(
                min_time,
                min_value,
                color="red",
                s=55,
                linewidth=1.2,
                zorder=5
            )

            ax1.annotate(
                f"Min = {min_value:.2f} m",
                (min_time, min_value),
                xytext=(8, 10),
                textcoords="offset points",
                fontsize=9,
                color="red",
                weight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor="red",
                    alpha=0.9
                )
                # ,
                # arrowprops=dict(
                #     arrowstyle="->",
                #     color="royalblue",
                #     linewidth=1
                # )
            )

        if angle_type is not None:
            if angle_type == "direction":
                reference_angles = [
                    (0, "Same direction", "dimgray"),
                    (90, "Perpendicular", "orange"),
                    (180, "Opposition", "purple")
                ]
            elif angle_type == "approach":
                reference_angles = [
                    (0, "Frontal", "dimgray"),
                    (90, "Crossing", "orange"),
                    (180, "Opposition", "purple")
                ]
            else:
                reference_angles = []
            for angle, label, color in reference_angles:
                ax1.axhline(
                    angle,
                    color=color,
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.75,
                    zorder=0,
                    label=label
                )
            # Permet de garder les trois références visibles
            ax1.set_ylim(
                min(-5, np.nanmin(y) - 5),
                max(185, np.nanmax(y) + 5)
            )

        ax1.set_title(title, fontsize=15, fontweight="bold")
        ax1.set_xlabel("Time (s)", fontsize=15)
        ax1.set_ylabel(ylabel, fontsize=15)
        ax1.tick_params(axis="both", labelsize=13)
        add_time_markers(ax1, [(start/fps, end/fps)])
        ax1.grid(True)
        ax1.legend(fontsize=12)
        fig.tight_layout()
        plt.show()
    

    plot_series(features.get("distances"), "Minimum distance", "Distance (m)", start, end, distance=True)

    plot_series(features.get("hausdorff"), "Hausdorff Distance", "Distance (m)", start, end)

    plot_series(features.get("modified_hausdorff"), "Modified Hausdorff Distance", "Distance (m)", start, end)


    plot_series(features.get("relative_speed_series"), "Relative speed of cyclist(s) with respect to pedestrians", "Speed (km/h)", start, end, show_ms_axis=True)

    plot_series(features.get("direction_angle_series"), "Direction Angle", "Angle (°)", start, end, angle_type="direction")


    plot_series(features.get("density_ped_series"), "Evolution of Pedestrians Group Density", "Density", start, end, show_density_stats=True)

    plot_series(features.get("density_cyc_series"), "Evolution of Cyclists Group Density", "Density", start, end, show_density_stats=True)

    if features.get("cluster_split_events"):
        plt.figure(figsize=(8, 2))
        for e in features["cluster_split_events"]:
            t_event = e["time"] / fps if fps else e["time"]
            plt.axvline(t_event, color="red", linestyle="--")
        plt.title("Cluster Split Events")
        plt.xlabel("Time (s)")
        plt.yticks([])
        plt.grid(True)
        plt.show()

    if features.get("cyclist_in_hull_events"):
        plt.figure(figsize=(8, 2))
        dt = 1 / fps if fps else 1
        for e in features["cyclist_in_hull_events"]:
            t_event = e["time_frame"] / fps if fps else e["time_frame"]
            plt.axvspan(
                t_event - dt/2,
                t_event + dt/2,
                color="green",
                alpha=0.3
            )
            # plt.axvline(t_event, color="green", linestyle=":")
        plt.title("Cyclist in Hull Events")
        plt.xlabel("Time (s)")
        plt.yticks([])
        plt.grid(True)
        plt.show()


# Calcul des features (critères spatio-temporelles) d'une ou des interactions
def compute_one_interaction_features(df, history, interaction, fps=None, plot=False):
    split_events = detect_split_events_with_cyclists(history)
    hull_events = detect_cyclists_in_hulls(history)

    if is_noise_only_interaction(interaction):
        return compute_pair_interaction_features(interaction, df, fps=fps, plot=plot)
    else:
        return compute_group_interaction_features(interaction, history, df, split_events, hull_events, fps=fps, plot=plot)


def compute_all_interactions_features(df, history, interactions, fps=None, plot=False):

    split_events = detect_split_events_with_cyclists(history)
    hull_events = detect_cyclists_in_hulls(history)

    all_features = []

    for inter in interactions:
        is_noise_noise = is_noise_only_interaction(inter)

        if is_noise_noise:
            f = compute_pair_interaction_features(inter, df, fps=fps, plot=plot)
        else:
            f = compute_group_interaction_features(inter, history, df, split_events, hull_events, fps=fps, plot=plot)
        
        all_features.append(f)

    return all_features



###############################################
# Classification of the spatio-temporal criteria
###############################################

def classify_pet(pet):
    """
    PET classification (in seconds). Used to assess the risk level during the interaction.
    """

    if pet is None:
        return "None"

    if pet < 1:
        return "CRITICAL" # quasi-collision
    elif pet < 2:
        return "HIGH"
    elif pet < 4:
        return "MEDIUM"
    else:
        return "LOW"


def classify_ttac(ttac):
    """
    Classifies minimum TTAC classification (in seconds).
    """
    if ttac is None:
        return "None"

    if ttac < 1:
        return "CRITICAL"
    elif ttac < 2:
        return "HIGH"
    elif ttac < 4:
        return "MEDIUM"
    else:
        return "LOW"


def classify_speed_variation(speed_metrics, accel_thresh=0.05, slope_thresh=0.02):
    """
    Classifies speed behavior during the interaction:
    - ACCELERATING
    - DECELERATING
    - STABLE
    """

    mean_dv = speed_metrics.get("mean_dv", np.nan)
    slope = speed_metrics.get("trend_slope", np.nan)
    mean_speed_change = speed_metrics.get("mean_speed_change", np.nan)

    if np.isnan(mean_dv) or np.isnan(slope) or np.isnan(mean_speed_change):
        return "UNKNOWN"

    if abs(mean_dv) < accel_thresh and abs(slope) < slope_thresh:
        return "STABLE"

    if (mean_dv >= accel_thresh and mean_speed_change > 0.0) or mean_dv >= accel_thresh:
        return "ACCELERATING"

    if (mean_dv <= -accel_thresh and mean_speed_change < 0.0) or mean_dv <= -accel_thresh or mean_speed_change < 0.0:
        return "DECELERATING"
    
    return "STABLE"



def classify_spatial_deviation(spatial_metrics, low_thresh=0.2, high_thresh=0.3):
    """
    Classifies spatial deviation :
    - LINEAR (traj stable, no deviation)
    - SLIGHT_DEVIATION
    - MODERATE_DEVIATION
    - HIGH_DEVIATION
    """

    mean_dev = spatial_metrics.get("mean_deviation", np.nan)
    stability = spatial_metrics.get("trajectory_stability", np.nan)

    if np.isnan(mean_dev):
        return "UNKNOWN"

    # trajectoire quasi droite
    if mean_dev < low_thresh:
        return "LINEAR"

    # légère déviation
    if mean_dev < high_thresh:
        return "SLIGHT_DEVIATION"

    # déviation moyenne
    if mean_dev < 0.4:
        return "MODERATE_DEVIATION"

    return "HIGH_DEVIATION"


def clean_series(values):
    cleaned = []

    for v in values:
        if v is None:
            continue

        try:
            v = float(v)
            if np.isfinite(v):
                cleaned.append(v)
        except:
            continue

    return np.array(cleaned)


def compress_sequence(seq):
    if len(seq) == 0:
        return []

    compressed = [seq[0]]
    for x in seq[1:]:
        if x != compressed[-1]:
            compressed.append(x)
    return compressed


def classify_distance_series(distances):

    d = clean_series(distances)

    if len(d) == 0:
        return {"label_main": "UNKNOWN"}

    def label_dist(x):
        if x < 1.5:
            return "VERY_CLOSE"
        elif x < 3:
            return "CLOSE"
        elif x < 5:
            return "PRETTY_CLOSE"
        else:
            return "FAR"

    labels = [label_dist(x) for x in d]
    seq = compress_sequence(labels)

    # trouver pattern global
    if "VERY_CLOSE" in seq:
        label = "CRITICAL_PROXIMITY"
    elif "CLOSE" in seq:
        label = "CLOSE"
    elif seq == ["FAR"]:
        label = "FAR"
    else:
        label = "MODERATE_DISTANCE"

    return {
        "label_main": label,
        "sequence": seq,
        "min": float(np.min(d)),
        "mean": float(np.mean(d))
    }


def classify_direction_angle_series(angles):

    a = clean_series(angles)

    if len(a) == 0:
        return {"label_main": "UNKNOWN"}

    def label_angle(x):
        if x < 30:
            return "SAME_DIRECTION"
        elif x < 60:
            return "SLIGHT_DIVERGENCE"
        elif x < 120:
            return "CROSSING" # interaction perpendiculaire
        elif x < 150:
            return "STRONG_DIVERGENCE"
        else:
            return "OPPOSITE_DIRECTION"

    labels = [label_angle(x) for x in a]
    seq = compress_sequence(labels)

    # print("All labels : ", labels)
    # print("Compressed sequence : ", seq)

    # pattern temporel
    if "SAME_DIRECTION" in seq:
        label = "SAME_DIRECTION"
    elif "OPPOSITE_DIRECTION" in seq:
        label = "OPPOSITE_DIRECTION"
    elif "CROSSING" in seq:
        label = "CROSSING"
    else:
        label = "DIVERGING"

    return {
        "label_main": label,
        "sequence": seq,
        "median": float(np.median(a))
    }


def classify_approach_angle_series(angles):

    a = clean_series(angles)

    if len(a) == 0:
        return {"label_main": "UNKNOWN"}

    def label(x):
        if x < 30:
            return "FRONTAL_APPROACH" # rapprochement frontal, en direction du piéton
        elif x < 75:
            return "OBLIQUE_APPROACH" # rapprochement en diagonal (vers le piéton)
        elif x < 105:
            return "LATERAL_APPROACH" # interaction perpendiculaire ou position latérale par rapp au piéotn
        elif x < 150:
            return "OBLIQUE_DEPART" # éloignement en diagonal
        else:
            return "MOVING_AWAY" # éloignement (pointe à l'opposée du piéton)

    labels = [label(x) for x in a]
    seq = compress_sequence(labels)

    # pattern clé
    if seq == ["FRONTAL_APPROACH", "LATERAL_APPROACH", "MOVING_AWAY"] or seq == ["FRONTAL_APPROACH", "OBLIQUE_APPROACH", "LATERAL_APPROACH", "OBLIQUE_DEPART", "MOVING_AWAY"] or seq == ["OBLIQUE_APPROACH", "LATERAL_APPROACH", "OBLIQUE_DEPART"]:
        label_main = "APPROACH_AND_ESCAPE" # si dans des directions opposées, mais OVERTAKING (dépassement) si même direction

    elif "FRONTAL_APPROACH" in seq and "MOVING_AWAY" in seq:
        label_main = "APPROACH_AND_ESCAPE"

    elif "LATERAL_APPROACH" in seq:
        label_main = "LATERAL_APPROACH"
    
    elif "FRONTAL_APPROACH" in seq or "OBLIQUE_APPROACH" in seq:
        label_main = "FRONTAL_APPROACH"

    elif "MOVING_AWAY" in seq or "OBLIQUE_DEPART" in seq:
        label_main = "MOVING_AWAY" # éloignement / évitement

    else:
        "UNKNOWN"

    return {
        "label_main": label_main,
        "sequence": seq,
        "min": float(np.min(a)),
        "median": float(np.median(a))
    }



def classify_relative_speed_series(speeds):

    v = clean_series(speeds)

    if len(v) == 0:
        return {"label_main": "UNKNOWN"}

    def label(x):
        if x < 3:
            return "LOW"
        elif x < 15:
            return "MODERATE"
        elif x < 20:
            return "HIGH"
        else:
            return "VERY_HIGH"

    labels = [label(x) for x in v]
    seq = compress_sequence(labels)

    if "VERY_HIGH" in seq:
        main = "VERY_FAST"
    elif "HIGH" in seq:
        main = "FAST"
    elif "MODERATE" in seq:
        main = "MODERATE"
    else:
        main = "VERY_SLOW"

    return {
        "label_main": main,
        "sequence": seq,
        "v_max": float(np.max(v)),
        "v_mean": float(np.mean(v))
    }


def classify_relative_position_series(rel_positions):

    if len(rel_positions) == 0:
        return {"label_main": "UNKNOWN"}

    angles = []

    for vec in rel_positions:
        angle = np.degrees(np.arctan2(vec[1], vec[0]))
        angles.append(angle)

    angles = np.array(angles)

    # Classification angulaire
    front = np.sum((angles >= -45) & (angles <= 45))

    left = np.sum((angles > 45) & (angles <= 135))

    right = np.sum((angles < -45) & (angles >= -135))

    behind = np.sum(
        (angles > 135) | (angles < -135)
    )

    n = len(rel_positions)
    ratios = {
        "FRONT": front / n,
        "BEHIND": behind / n,
        "LEFT": left / n,
        "RIGHT": right / n
    }

    label = max(ratios, key=ratios.get)

    return {
        "label_main": label,
        "angles": angles.tolist(),
        "ratios": ratios
    }


def classify_density_series(density_series, slope_threshold=0.02, fps=None):

    if len(density_series) == 0:
        return {
            "label": "UNKNOWN",
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "trend": np.nan,
            "r2": np.nan
        }

    density = np.asarray(density_series)

    if len(density) == 1:
        return {
            "label": "STABLE",
            "mean": density[0],
            "std": 0.0,
            "min": density[0],
            "max": density[0],
            "trend": 0.0
        }

    t = np.arange(len(density))
    if fps is not None:
        t = t / fps

    slope, _, _, _, _ = linregress(t, density)

    if slope > slope_threshold:
        label = "DENSER"

    elif slope < -slope_threshold:
        label = "DISPERSED"

    else:
        label = "STABLE"

    return {
        "label": label,
        "mean": float(np.mean(density)),
        "std": float(np.std(density)),
        "min": float(np.min(density)),
        "max": float(np.max(density)),
        "trend": float(slope)
    }


###############################################
# Classification of the interactions
###############################################

def classify_pair_interaction(features):

    dist_res = classify_distance_series(features["distances"])

    dir_res = classify_direction_angle_series(features["direction_angle_series"])

    approach_res = classify_approach_angle_series(features["approach_angle_series"])

    speed_res = classify_relative_speed_series(features["relative_speed_series"])

    pos_res = classify_relative_position_series(features["relative_position_series"])

    pet_res = classify_pet(features["PET"])

    _, _, ttac_min = features["TTAC"]
    ttac_res = classify_ttac(ttac_min)

    speed_var_ped = features["speed_var_ped"]
    speed_var_res_ped = classify_speed_variation(speed_var_ped)
    speed_var_cyc = features["speed_var_cyc"]
    speed_var_res_cyc = classify_speed_variation(speed_var_cyc)
    spatial_var_ped = features["spatial_var_ped"]
    spatial_var_res_ped = classify_spatial_deviation(spatial_var_ped)
    spatial_var_cyc = features["spatial_var_cyc"]
    spatial_var_res_cyc = classify_spatial_deviation(spatial_var_cyc)

    # réactivité des agents
    stability_ped = spatial_var_ped.get("trajectory_stability", np.nan)
    stability_cyc = spatial_var_cyc.get("trajectory_stability", np.nan)
    reactive_agent = ""
    if not np.isnan(stability_ped) and not np.isnan(stability_cyc):
        if stability_ped > stability_cyc:
            reactive_agent = "cyc"
        elif stability_ped < stability_cyc:
            reactive_agent = "ped"
        else: # égalité
            reactive_agent = "both"

    # détection du type d'interaction
    interaction = "UNDEFINED"

    if approach_res["label_main"] == "APPROACH_AND_ESCAPE" or approach_res["label_main"] == "LATERAL_APPROACH":
        if dir_res["label_main"] == "SAME_DIRECTION":
            interaction = "OVERTAKING"
        elif dir_res["label_main"] in ["OPPOSITE_DIRECTION", "DIVERGING"]:
            interaction = "AVOIDANCE" # ici évitement = contournement
        elif dir_res["label_main"] == "CROSSING":
            interaction = "CROSSING" # sorte de contournement mais trajectoires perpendiculaires
            if pet_res == "CRITICAL" or (pet_res == "CRITICAL" and speed_res["label_main"] == "VERY_FAST") or (pet_res == "CRITICAL" and dist_res["label_main"] == "CRITICAL_PROXIMITY"):
                interaction = "CLOSE_COLLISION"
    
    # elif approach_res["label_main"] == "LATERAL_APPROACH":
#       interaction == "CROSSING"
#       if pet_res == "CRITICAL" or (pet_res == "CRITICAL" and speed_res["label_main"] == "VERY_FAST") or (pet_res == "CRITICAL" and dist_res["label_main"] == "CRITICAL_PROXIMITY"):
#           interaction = "CLOSE_COLLISION"

    elif approach_res["label_main"] == "FRONTAL_APPROACH":
        if dir_res["label_main"] == "SAME_DIRECTION":
            interaction = "FOLLOWING" # pas observé dans CTV, mais ajouté quand même pour la logique et les autres datasets
        elif dir_res["label_main"] in ["OPPOSITE_DIRECTION", "DIVERGING"]:
            interaction = "AVOIDANCE"
        elif dir_res["label_main"] == "CROSSING":
            interaction = "CROSSING"
            if pet_res == "CRITICAL" or (pet_res == "CRITICAL" and speed_res["label_main"] == "VERY_DYNAMIC") or (pet_res == "CRITICAL" and dist_res["label_main"] == "CRITICAL_PROXIMITY"):
                interaction = "CLOSE_COLLISION"
            elif pet_res == "MEDIUM" or pet_res == "LOW":
                if speed_res["label_main"] == "MODERATE" or speed_res["label_main"] == "VERY_SLOW":
                    interaction = "GIVE_WAY"
                
    
    elif approach_res["label_main"] == "MOVING_AWAY":
        interaction = "MOVING_AWAY" # éloignement / évitement
    
    # elif approach_res["label_main"] == "OBLIQUE_APPROACH":
    #     interaction = "OBLIQUE_APPROACH"

    # calcul niveau du risque
    score = 0

    if pet_res == "CRITICAL":
        score += 3
    elif pet_res == "HIGH":
        score += 2
    elif pet_res == "MEDIUM":
        score += 1
    
    if ttac_res == "CRITICAL":
        score += 3
    elif ttac_res == "HIGH":
        score += 2
    elif ttac_res == "MEDIUM":
        score += 1
    

    if dist_res["label_main"] == "CRITICAL_PROXIMITY":
        if speed_res["label_main"] == "VERY_FAST":
            score += 3
        elif speed_res["label_main"] == "FAST":
            score += 2
        elif speed_res["label_main"] == "MODERATE":
            score += 1
    elif dist_res["label_main"] == "CLOSE":
        if speed_res["label_main"] == "VERY_FAST":
            score += 3
        elif speed_res["label_main"] == "FAST":
            score += 2
    elif dist_res["label_main"] == "MODERATE_DISTANCE":
        if speed_res["label_main"] == "VERY_FAST":
            score += 2
        elif speed_res["label_main"] == "FAST":
            score += 1
    elif dist_res["label_main"] == "FAR":
        if speed_res["label_main"] == "VERY_FAST":
            score += 1
    

    if score >= 7:
        risk = "CRITICAL"
    elif score >= 5:
        risk = "HIGH"
    elif score >= 3:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    if interaction == "AVOIDANCE" or interaction == "OVERTAKING":
        if pos_res["ratios"]["LEFT"] > pos_res["ratios"]["RIGHT"]:
            pos_res["label_main"] = "LEFT"
        else:
            pos_res["label_main"] = "RIGHT"

    return {
        "interaction_type": interaction,
        "direction": dir_res,
        "approach": approach_res,
        "position": pos_res,
        "distance": dist_res,
        "speed": speed_res,
        "pet": pet_res,
        "ttac": ttac_res,
        "risk": risk,
        "risk_score": score,
        "speed_var_ped": speed_var_res_ped,
        "speed_var_cyc": speed_var_res_cyc,
        "spatial_var_ped": spatial_var_res_ped,
        "spatial_var_cyc": spatial_var_res_cyc,
        "most_reactive_agent": reactive_agent
    }



def classify_group_interaction(features, fps=None):
    dist_res = classify_distance_series(features["distances"])
    speed_res = classify_relative_speed_series(features["relative_speed_series"])

    direction_res = classify_direction_angle_series(features["direction_angle_series"])
    pos_res = classify_relative_position_series(features["relative_position_series"])

    in_hull = features["cyclist_in_hull"]
    split = features["cluster_split"]

    density_ped_res = classify_density_series(features["density_ped_series"], fps=fps)
    density_cyc_res = classify_density_series(features["density_cyc_series"], fps=fps)


    # pénétration dans groupe
    if in_hull and split:
        interaction = "WEAVING_AND_SPLIT" # faufilement du cycliste puis split du groupe

    elif in_hull and not split:
        interaction = "WEAVING" # faufilement du cycliste
    
    # cassure du groupe
    elif not in_hull and split:
        interaction = "GROUP_SPLIT"

    # dépassement (dans la même direction)
    elif direction_res["label_main"] == "SAME_DIRECTION":
        interaction = "OVERTAKING_GROUP"

    # contournement (dans des directions opposées ou autre, mais pas dans la même direction)
    elif direction_res["label_main"] == "OPPOSITE_DIRECTION":
        interaction = "BYPASSING_GROUP"

    # croisement
    elif direction_res["label_main"] == "CROSSING":
        interaction = "CROSSING_GROUP"

    else:
        interaction = "WEAK_INTERACTION"

    score = 0

    if dist_res["label_main"] == "CRITICAL_PROXIMITY":
        if speed_res["label_main"] == "VERY_FAST":
            score += 3
        elif speed_res["label_main"] == "FAST":
            score += 2
        elif speed_res["label_main"] == "MODERATE":
            score += 1
    elif dist_res["label_main"] == "CLOSE":
        if speed_res["label_main"] == "VERY_FAST":
            score += 3
        elif speed_res["label_main"] == "FAST":
            score += 2
    elif dist_res["label_main"] == "MODERATE_DISTANCE":
        if speed_res["label_main"] == "VERY_FAST":
            score += 2
        elif speed_res["label_main"] == "FAST":
            score += 1
    elif dist_res["label_main"] == "FAR":
        if speed_res["label_main"] == "VERY_FAST":
            score += 1
    

    if score >= 3:
        risk = "CRITICAL"
    elif score >= 2:
        risk = "HIGH"
    elif score >= 1:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        "interaction_type": interaction,
        "direction": direction_res,
        "position": pos_res,
        "distance": dist_res,
        "speed": speed_res,
        "in_hull": in_hull,
        "split": split,
        "density_ped_res": density_ped_res,
        "density_cyc_res": density_cyc_res,
        "risk": risk,
        "risk_score": score
    }


def classify_one_interaction(df, history, inter, fps):
    """
    Classify 1 interaction à partir des features.
    """
    features = compute_one_interaction_features(df, history, inter, fps=fps)

    label = {}

    type_A = inter["type_ped"]
    type_B = inter["type_cyc"]

    if "noise" in type_A and "noise" in type_B:
        label = classify_pair_interaction(features)
    
    else: 
        label = classify_group_interaction(features, fps=fps)
    
    return {"interaction": inter,
            "features": features,
            "label": label}


def classify_all_interactions(df, history, interactions, fps):
    """
    Classify all the interactions of a video.
    """

    results = []

    for inter in interactions:

        res = classify_one_interaction(df, history, inter, fps)

        results.append(res)
    
    return results



# def classify_ttc(ttc):
#     """
#     Classification du TTC (en secondes).
#     """
#     if ttc is None:
#         return "None"

#     if ttc < 1:
#         return "CRITICAL"
#     elif ttc < 2:
#         return "HIGH"
#     elif ttc < 4:
#         return "MEDIUM"
#     else:
#         return "LOW"