import numpy as np
import matplotlib.pyplot as plt
from config import *
from utils import *


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
    Fonction pour analyser les distances entre usagers, et déterminer le seuil de distance qui définit 
    une potentielle influence de la part d'une voiture sur les trajectoires des VRUs.
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


def analyze_speeds(
    df,
    cfg,
    agent_ids=None,
    classes=None
):
    fps = cfg.get("fps", 1.0)

    # ===============================
    # 1. Sélection des agents
    # ===============================

    selected_agents = []

    if agent_ids is not None:
        # --- sélection manuelle ---
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

    # ===============================
    # 2. Calcul des vitesses
    # ===============================
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

    # ===============================
    # 3. FIGURE
    # ===============================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax_hist = axes[0]
    ax_curve = axes[1]
    ax_curve_kmh = axes[2]

    # ===============================
    # 4. HISTOGRAMMES PAR CLASSE (km/h)
    # ===============================
    speeds_by_class = {}

    for d in data:
        cls = d["class"]
        speeds_by_class.setdefault(cls, []).extend(d["speeds_kmh"])

    for cls, speeds in speeds_by_class.items():
        speeds = np.array(speeds)

        label = CLASS_NAMES.get(cls, f"class {cls}")
        color = CLASS_COLORS.get(cls, None)

        ax_hist.hist(
            speeds,
            bins=30,
            alpha=0.5,
            label=label,
            color=color
        )

        print(f"\n=== {label} ===")
        print(f"Nb mesures : {len(speeds_kmh)}")
        print(f"Min : {speeds.min():.2f} km/h")
        print(f"Max : {speeds.max():.2f} km/h")
        print(f"Moy : {speeds.mean():.2f} km/h")

    ax_hist.set_title("Distribution des vitesses")
    ax_hist.set_xlabel("Vitesse (km/h)")
    ax_hist.set_ylabel("Fréquence")
    ax_hist.legend()
    ax_hist.grid()

    # ===============================
    # 5. COURBES INDIVIDUELLES (m/s)
    # ===============================
    for d in data:
        aid = d["id"]
        cls = d["class"]
        times = d["times"]
        speeds = d["speeds_ms"]

        color = CLASS_COLORS.get(cls, "black")
        # linestyle = "-" if cls == 1 else "--"
        # attention pas de différenciation entre les piétons entre eux, et entre les cyclistes entre eux...

        label = f"{CLASS_NAMES.get(cls)} ID={aid}"

        ax_curve.plot(
            times,
            speeds,
            label=label,
            color=color,
            # linestyle=linestyle,
            linewidth=2
        )

    ax_curve.set_title("Évolution des vitesses")
    ax_curve.set_xlabel("Temps")
    ax_curve.set_ylabel("Vitesse (m/s)")
    ax_curve.legend()
    ax_curve.grid()

    # ===============================
    # 5. COURBES INDIVIDUELLES (km/h)
    # ===============================
    for d in data:
        aid = d["id"]
        cls = d["class"]
        times = d["times"]
        speeds_kmh = d["speeds_kmh"]

        color = CLASS_COLORS.get(cls, "black")
        label = f"{CLASS_NAMES.get(cls)} ID={aid}"

        ax_curve_kmh.plot(
            times,
            speeds_kmh,
            label=label,
            color=color,
            linewidth=2
        )

    ax_curve_kmh.set_title("Évolution des vitesses")
    ax_curve_kmh.set_xlabel("Temps")
    ax_curve_kmh.set_ylabel("Vitesse (km/h)")
    ax_curve_kmh.legend()
    ax_curve_kmh.grid()

    plt.tight_layout()
    plt.show()



def compute_ped_cyc_interactions(df, distance_threshold=5.0):
    # pour l'instant, interaction si piéton et cycliste proches l'un de l'autre dans un rayon de 5 m.
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
    Classifie les interactions piétons-cyclistes dans le temps selon l'apparition de la voiture dans un rayon de 5m autour de l'un des 2 VRU.
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

    # ===== voitures actives (InD) =====
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

    # ===== analyse =====
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

        # ===== classification =====
        if has_during_close:
            label = "PENDANT_PROCHE"
        else:
            # fallback temporel simple
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


# def classify_direction_angle(angle):
#     if angle is None:
#         return "None"

#     if angle < 30:
#         return "SAME_DIRECTION"
#     elif angle < 75:
#         return "SLIGHT_DIVERGENCE"
#     elif angle < 105:
#         return "CROSSING" # interaction perpendiculaire
#     elif angle < 150:
#         return "STRONG_DIVERGENCE"
#     else:
#         return "OPPOSITE_DIRECTION"
    
# def classify_direction_angle_interaction(df, ped_id, cyc_id, times, angles):
#     interactions = compute_ped_cyc_interactions_with_time(df)
#     pair = tuple(sorted((ped_id, cyc_id)))

#     frames = interactions.get(pair, [])
#     if len(frames) == 0:
#         return {"label": "no interaction"}

#     intervals = frames_to_intervals(frames)

#     # ===== filtrage pendant interaction =====
#     mask = np.zeros_like(times, dtype=bool)
#     for start, end in intervals:
#         mask |= (times >= start) & (times <= end)

#     angles_inter = angles[mask]

#     if len(angles_inter) == 0:
#         return {"label": "UNKNOWN"}

#     # ===== classification point par point =====
#     angle_classes = [classify_direction_angle(a) for a in angles_inter]

#     # ===== compression de séquence =====
#     sequence = [angle_classes[0]]
#     for c in angle_classes[1:]:
#         if c != sequence[-1]:
#             sequence.append(c)

#     # priorité à la géométrie la plus critique
#     if "OPPOSITE_DIRECTION" in sequence:
#         label_main = "OPPOSITE_DIRECTION"
#     elif "STRONG_CONVERGENCE" in sequence:
#         label_main = "STRONG_CONVERGENCE"
#     elif "CROSSING" in sequence:
#         label_main = "CROSSING"
#     elif "SLIGHT_CONVERGENCE" in sequence:
#         label_main = "SLIGHT_CONVERGENCE"
#     else:
#         label_main = "SAME_DIRECTION"

#     return {
#         "label_main": label_main,
#         "sequence": sequence,
#         "angle_min": np.min(angles_inter),
#         "angle_median": np.median(angles_inter)
#     }


# def classify_approach_angle(angle):
#     if angle is None:
#         return "UNKNOWN"

#     if angle < 30:
#         return "FRONTAL_APPROACH"
#     elif angle < 75:
#         return "OBLIQUE_APPROACH"
#     elif angle < 105:
#         return "CROSSING" # latérale ou perpendiculaire normalement
#     elif angle < 150:
#         return "OBLIQUE_DEPART"
#     else:
#         return "MOVING_AWAY"
    

# def classify_approach_angle_interaction(df, ped_id, cyc_id, times, angles):
#     interactions = compute_ped_cyc_interactions_with_time(df)
#     pair = tuple(sorted((ped_id, cyc_id)))

#     frames = interactions.get(pair, [])
#     if len(frames) == 0:
#         return {"label": "no interaction"}

#     intervals = frames_to_intervals(frames)

#     # filtrage
#     mask = np.zeros_like(times, dtype=bool)
#     for start, end in intervals:
#         mask |= (times >= start) & (times <= end)

#     angles_inter = angles[mask]

#     if len(angles_inter) == 0:
#         return {"label": "UNKNOWN"}

#     # classification point par point
#     angle_classes = [classify_approach_angle(a) for a in angles_inter]

#     # compression
#     sequence = [angle_classes[0]]
#     for c in angle_classes[1:]:
#         if c != sequence[-1]:
#             sequence.append(c)
    
#     # label_main = max(set(sequence), key=sequence.count)

#     #label principal
#     if "FRONTAL_APPROACH" in sequence:
#         label_main = "FRONTAL_APPROACH"
#     elif "OBLIQUE_APPROACH" in sequence:
#         label_main = "OBLIQUE_APPROACH"
#     elif "CROSSING" in sequence:
#         label_main = "CROSSING"
#     else:
#         label_main = sequence[0]

#     return {
#         "label_main": label_main,
#         "sequence": sequence,
#         "angle_min": np.min(angles_inter),
#         "angle_median": np.median(angles_inter)
#     }

def classify_distance_interaction(times, distances, intervals):
    """
    Classifie une interaction en fonction de la distance piéton-cycliste.

    Parameters:
        times : array (temps ou frames)
        distances : array (m)
        intervals : [(t_start, t_end), ...]
    
    Returns:
        dict avec classification + stats
    """

    if len(intervals) == 0:
        return {
            "label": "NO_INTERACTION",
            "risk_level": "NONE",
            "min_distance": None,
            "mean_distance": None
        }

    # ===== filtrer les points pendant interaction =====
    mask = np.zeros_like(times, dtype=bool)

    for (start, end) in intervals:
        mask |= (times >= start) & (times <= end)

    d_inter = distances[mask]

    if len(d_inter) == 0:
        return {
            "label": "NO_VALID_DATA",
            "risk_level": "UNKNOWN"
        }

    d_min = np.min(d_inter)
    d_mean = np.mean(d_inter)

    # ===== classification =====
    if d_min < 1.5:
        label = "VERY_CLOSE"
        risk = "HIGH"

    elif d_min < 3.0:
        label = "CLOSE"
        risk = "MEDIUM"

    elif d_min < 5.0:
        label = "PRETTY_CLOSE"
        risk = "LOW"

    else:
        label = "DISTANT"
        risk = "LOW"

    return {
        "label_main": label,
        "risk_level": risk, # à voir en fonction de la vitesse du cycliste
        "min_distance": float(d_min),
        "mean_distance": float(d_mean)
    }


# def classify_relative_speed(v_rel):
#     # pour des vitesses en km/h
#     if v_rel is None:
#         return "UNKNOWN"

#     if v_rel < 3:
#         return "LOW"
#     elif v_rel < 15:
#         return "MODERATE"
#     elif v_rel < 20:
#         return "HIGH"
#     else:
#         return "VERY_HIGH"

# def classify_relative_speed_interaction(
#     df,
#     ped_id,
#     cyc_id,
#     times,
#     v_rel
# ):
#     # pour des vitesses en km/h
#     interactions = compute_ped_cyc_interactions_with_time(df)
#     pair = tuple(sorted((ped_id, cyc_id)))

#     frames = interactions.get(pair, [])
#     if len(frames) == 0:
#         return {"label": "NO_INTERACTION"}

#     intervals = frames_to_intervals(frames)

#     # ===== filtrage =====
#     mask = np.zeros_like(times, dtype=bool)
#     for start, end in intervals:
#         mask |= (times >= start) & (times <= end)

#     v_inter = v_rel[mask]

#     # enlever NaN / inf
#     v_inter = v_inter[np.isfinite(v_inter)]

#     if len(v_inter) == 0:
#         return {"label": "UNKNOWN"}

    
#     v_max = np.max(v_inter) # le moment le plus dangereux serait le moment où la vitesse est la plus élevée
#     v_mean = np.mean(v_inter)

#     label_max = classify_relative_speed(v_max)
#     labels = [classify_relative_speed(v) for v in v_inter]
#     dominant = max(set(labels), key=labels.count)

#     return {
#         "label_main": dominant, # selon l'occurence des labels
#         "label_max": label_max, # selon la vitesse max
#         "v_max": v_max,
#         "v_mean": v_mean,
#         "labels": labels,
#     }


# def classify_relative_motion(dx_rel, dy_rel, dx_pos, dy_pos): # même cjose que angle d'approche
#     """
#     dx_rel, dy_rel : vitesse relative
#     dx_pos, dy_pos : vecteur position (cycliste -> piéton)
#     """

#     v_rel = np.array([dx_rel, dy_rel])
#     pos_vec = np.array([dx_pos, dy_pos])

#     norm_v = np.linalg.norm(v_rel)
#     norm_p = np.linalg.norm(pos_vec)

#     if norm_v == 0 or norm_p == 0:
#         return "STATIC"

#     cos_theta = np.dot(v_rel, pos_vec) / (norm_v * norm_p)

#     if cos_theta > 0.5:
#         return "APPROACHING"
#     elif cos_theta < -0.5:
#         return "MOVING_AWAY"
#     else:
#         return "LATERAL"


# def classify_pet(pet):
#     """
#     Classification du PET (en secondes). Permet d'évaluer le niveau de risque lors de l'interaction.
#     """

#     if pet is None:
#         return "None"

#     if pet < 1:
#         return "CRITICAL" # quasi-collision
#     elif pet < 2:
#         return "HIGH"
#     elif pet < 4:
#         return "MEDIUM"
#     else:
#         return "LOW"


def classify_ttc(ttc):
    """
    Classification du TTC (en secondes). Mesure le temps dispo pour réagir avant une potentielle collision si usagers ont la même vitesse et la même trajectoire.
    """
    if ttc is None:
        return "None"

    if ttc < 1:
        return "CRITICAL"
    elif ttc < 2:
        return "HIGH"
    elif ttc < 4:
        return "MEDIUM"
    else:
        return "LOW"


def classify_pair_interaction(pet, ttc, approach_result, direction_result, speed_result):
    """
    Combine tous les indicateurs pour classifier une interaction.

    Returns:
        dict:
            type
            risk_level
            score
            details
    """

    approach_class = approach_result["label_main"] if approach_result else None
    direction_class = direction_result["label_main"] if direction_result else None
    speed_class = speed_result["label_main"] if speed_result else None
    pet_val, pet_class = pet
    ttc_min, ttc_class = ttc

    if approach_class == "CROSSING":
        interaction_type = "CROSSING"

    elif approach_class == "FRONTAL_APPROACH": # à redéfinir, pour être plus précis, car même si approche frontale, peut l'esquiver latéralement
        interaction_type = "FRONTAL_APPROACH" # à voir aussi si OPPOSSITE_DIRECTION ou pas par ex

    elif direction_class == "SAME_DIRECTION":
        if speed_class != "LOW":
            interaction_type = "OVERTAKING" # dépassement
        else:
            interaction_type = "FOLLOWING"

    # elif speed_class == "LOW":
    #     interaction_type = "STATIC_INTERACTION" # à redéfinir, parce que c'est pas vraiment static, peut-être un suivi ou juste un ralentissement, voir un arrêt pour laisser passer

    else:
        interaction_type = "UNDEFINED"


    risk_score = 0

    if pet_class is not None:
        if pet_class == "CRITICAL":
            risk_score += 3
        elif pet_class == "HIGH":
            risk_score += 2
        elif pet_class == "MEDIUM":
            risk_score += 1


    if ttc_class is not None:
        if ttc_class == "CRITICAL":
            risk_score += 3
        elif ttc_class == "HIGH":
            risk_score += 2
        elif ttc_class == "MEDIUM":
            risk_score += 1


    if risk_score >= 5:
        risk_level = "CRITICAL"
    elif risk_score >= 3:
        risk_level = "HIGH"
    elif risk_score >= 2:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "type": interaction_type,
        "risk_level": risk_level,
        "risk_score": risk_score,
        "details": {
            "PET": pet,
            "TTC": ttc,
            "approach": approach_class,
            "direction": direction_class,
            "speed": speed_class
        }
    }



# CAS MULTI-AGENT (avec des groupes d'usagers) (1 cycliste VS 1 groupe de piétons, Groupe de cyclistes VS groupe de piétons)

# def hausdorff_distance(A, B):
#     """
#     A, B: arrays de taille (N,2) et (M,2) respectivement. 
#     """

#     def directed(A, B):
#         dists = []
#         for a in A:
#             d = np.min(np.linalg.norm(B - a, axis=1))
#             dists.append(d)
#         return np.max(dists)

#     return max(directed(A, B), directed(B, A))

# def min_distance(A, B): # pour prendre la dist min entre 2 groupes de points (la distance de Hausdorff)
#     min_d = float("inf")
#     for a in A:
#         d = np.min(np.linalg.norm(B - a, axis=1))
#         if d < min_d:
#             min_d = d
#     return min_d


# def compute_group_distances_over_time(df):
#     results = []

#     for t in df[COL_TIME].unique():
#         frame = df[df[COL_TIME] == t]

#         P = frame[frame[COL_CLASS] == 1][["x_m", "y_m"]].values
#         C = frame[frame[COL_CLASS] == 2][["x_m", "y_m"]].values

#         if len(P) == 0 or len(C) == 0:
#             continue

#         h = hausdorff_distance(P, C)
#         dmin = min_distance(P, C)

#         results.append((t, h, dmin))

#     return np.array(results)


# def plot_group_distances(results):
#     times = results[:,0]
#     haus = results[:,1]
#     dmin = results[:,2]

#     plt.figure(figsize=(10,5))
#     plt.plot(times, haus, label="Hausdorff distance")
#     plt.plot(times, dmin, label="Min distance", linestyle="--")

#     plt.xlabel("Temps")
#     plt.ylabel("Distance (m)")
#     plt.title("Distances groupe piétons - groupe cyclistes")
#     plt.legend()
#     plt.grid()
#     plt.show()


def classify_interactions(splits, intrusions, time_window=5):
    """
    Combine les événements pour classifier les interactions.

    Params:
        splits: liste de dicts (events split)
        intrusions: liste de dicts (events hull)
        time_window: tolérance temporelle (frames)

    Returns:
        interactions: liste de dicts
    """

    interactions = []

    for split in splits:
        t_split = split["time"]
        cyc_ids = set(split["cyclists"])

        # chercher intrusion proche dans le temps
        related_intrusions = [
            intr for intr in intrusions
            if abs(intr["time"] - t_split) <= time_window
            and len(cyc_ids.intersection(intr["cyclists"])) > 0
        ]

        if related_intrusions:
            interaction_type = "STRONG_DISRUPTION"
        else:
            interaction_type = "GROUP_AVOIDANCE"

        interactions.append({
            "time": t_split,
            "type": interaction_type,
            "cyclists": list(cyc_ids),
            "details": split
        })

    # traiter intrusions sans split
    for intr in intrusions:
        t_intr = intr["time"]
        cyc_ids = set(intr["cyclists"])

        related_splits = [
            sp for sp in splits
            if abs(sp["time"] - t_intr) <= time_window
            and len(cyc_ids.intersection(sp["cyclists"])) > 0
        ]

        if not related_splits:
            interactions.append({
                "time": t_intr,
                "type": "INTRUSION_WITHOUT_SPLIT",
                "cyclists": list(cyc_ids),
                "details": intr
            })

    return interactions


def summarize_interactions(interactions):
    summary = {}

    for inter in interactions:
        t = inter["type"]
        summary[t] = summary.get(t, 0) + 1

    print("\n===== INTERACTION SUMMARY =====")
    for k, v in summary.items():
        print(f"{k} : {v}")

    return summary


def group_interactions(interactions, max_gap=5):
    grouped = []
    current = None

    for inter in sorted(interactions, key=lambda x: x["time"]):
        if current is None:
            current = inter
            current["end_time"] = inter["time"]
            continue

        if inter["time"] - current["end_time"] <= max_gap and inter["type"] == current["type"]:
            current["end_time"] = inter["time"]
        else:
            grouped.append(current)
            current = inter
            current["end_time"] = inter["time"]

    if current:
        grouped.append(current)

    return grouped




def compute_individual_interaction_features(event, df, fps=None, plot=False):
    ids_A = list(event["ids_ped"])
    ids_B = list(event["ids_cyc"])

    start = event["start"]
    end = event["end"]

    if len(ids_A) != 1 or len(ids_B) != 1:
        return None  # par sécurité

    id_A = ids_A[0]
    id_B = ids_B[0]

    times, distances = compute_distance_series(df, id_A, id_B)

    t_dir, angles_dir = compute_direction_angle_velocity_based(df, id_A, id_B, fps, plot=plot)
    angles_dir_i = None
    if t_dir is not None:
        mask_dir = (t_dir >= start) & (t_dir <= end)
        angles_dir_i = angles_dir[mask_dir]

    t_app, angles_app = compute_approach_angle(df, id_A, id_B, fps, plot=plot)
    angles_app_i = None
    if t_app is not None:
        mask_app = (t_app >= start) & (t_app <= end)
        angles_app_i = angles_app[mask_app]

    t_rel, _, rel_speeds_kmh, _, _ = compute_relative_speed(df, id_A, id_B, fps, plot=plot)
    rel_speeds_kmh_i = None
    if t_rel is not None:
        mask_rel = (t_rel >= start) & (t_rel <= end)
        rel_speeds_kmh_i = rel_speeds_kmh[mask_rel]

    _, dx, dy, _ = compute_relative_position_series(df, id_A, id_B)

    mask = np.zeros_like(times, dtype=bool)
    mask |= (times >= start) & (times <= end)

    ped_df = slice_interaction(df, [id_A], start, end)
    cyc_df = slice_interaction(df, [id_B], start, end)

    reactive_agent, stable_agent = detect_reactive_agent(df, ped_df, cyc_df)

    if reactive_agent == "ped":
        ra = reactive_agent, id_A
        sa = stable_agent, id_B
    else:
        ra = reactive_agent, id_B
        sa = stable_agent, id_A

    return {
        "distances": distances[mask],
        "direction_angle_series": angles_dir_i,
        "approach_angle_series": angles_app_i,
        "relative_speed_series": rel_speeds_kmh_i,
        "relative_position_series": np.column_stack((dx[mask], dy[mask])),
        "PET": compute_pet(df, id_A, id_B, fps, plot=plot),
        "TTC": compute_ttc(df, id_A, id_B, fps, plot=plot),
        "reactive_agent": ra,
        "stable_agent": sa
    }

def slice_interaction(df, ids, t_start, t_end):
    return df[
        (df[COL_ID].isin(ids)) &
        (df[COL_TIME] >= t_start) &
        (df[COL_TIME] <= t_end)
    ]

def compute_group_interaction_features(event, history, df, split_events, hull_events, fps=None, plot=False):

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

    in_hull_flags = []
    split_flags = []

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

        # ===== DISTANCES =====
        pA, pB, dmin = get_closest_points(A, B)
        distances.append(dmin)

        hausdorffs.append(hausdorff_distance(A, B))
        hausdorff_mods.append(modified_hausdorff(A, B))

        # ===== POSITION RELATIVE =====
        rel_positions.append(pB - pA)

        # ===== VITESSE RELATIVE (au point de distance min) =====
        # prendre les agents correspondants
        # aid_A = list(ids_A)[0]
        # aid_B = list(ids_B)[0]

        # vA = compute_group_velocity_at_t(df, ids_A, t, fps)
        # vB = compute_group_velocity_at_t(df, ids_B, t, fps)

        # if vA is not None and vB is not None:
        #     rel_speeds.append(abs(vA - vB))

        # pA, pB, idA, idB = get_closest_points_with_ids(A, B, list(ids_A), list(ids_B))

        # vA = compute_agent_velocity(df, idA, t, fps)
        # vB = compute_agent_velocity(df, idB, t, fps)

        # if vA is not None and vB is not None:
        #     v_rel = np.linalg.norm(vA - vB)  # en m/s
        #     rel_speeds.append(v_rel)

        rel_speeds.append(compute_group_relative_speed(df, ids_A, ids_B, t, fps))

        # ===== ANGLE =====
        dir_A = compute_group_direction_angle(df, ids_A, ids_B, t)
        # vec_AB = pB - pA

        direction_angles.append(dir_A)

        # ===== DENSITÉ =====
        densities_A.append(compute_group_density(A))
        densities_B.append(compute_group_density(B))

        # # ===== CONVEX HULL =====
        hull_events_inter = get_cyclists_in_hull_for_interaction(event, hull_events)
        split_events_inter = get_split_events_for_interaction(event, split_events)

    features= {
        "time": times,
        "distances": distances,
        "hausdorff": hausdorffs,
        "modified_hausdorff": hausdorff_mods,

        "relative_position_series": rel_positions,
        "relative_speed_series": np.array(rel_speeds) * 3.6, # en km/h
        "direction_angle_series": direction_angles,

        "density_A_series": densities_A,
        "density_B_series": densities_B,

        "cyclist_in_hull": len(hull_events_inter) > 0,
        "cyclist_in_hull_events": hull_events_inter,

        "cluster_split": len(split_events_inter) > 0,
        "cluster_split_events": split_events_inter
    }

    if plot:
        plot_group_interaction_features(features, fps)

    return features

# {
#         "time": times,
#         "min_distance": np.min(distances) if distances else None,
#         "hausdorff": np.max(hausdorffs) if hausdorffs else None,
#         "modified_hausdorff": np.mean(hausdorff_mods) if hausdorff_mods else None,

#         "relative_position_series": rel_positions,
#         "relative_speed_series": rel_speeds,
#         "direction_angle_series": direction_angles,

#         "density_A_mean": np.mean(densities_A) if densities_A else None,
#         "density_B_mean": np.mean(densities_B) if densities_B else None,

#         "cyclist_in_hull": len(hull_events_inter) > 0,
#         "cyclist_in_hull_events": hull_events_inter,

#         "cluster_split": len(split_events_inter) > 0,
#         "cluster_split_events":split_events_inter
#     }

def plot_group_interaction_features(features, fps):
    t = np.array(features["time"]) / fps

    def plot_series(y, title, ylabel):
        if y is None or len(y) == 0:
            return

        y = np.array(y)
        tt = t[:len(y)]  # alignement simple

        plt.figure(figsize=(8, 4))
        plt.plot(tt, y)
        plt.title(title)
        plt.xlabel("Temps (s)")
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    

    plot_series(features.get("distances"), "Minimum Distance", "Distance (m)")

    plot_series(features.get("hausdorff"), "Hausdorff Distance", "Distance (m)")

    plot_series(features.get("modified_hausdorff"), "Modified Hausdorff Distance", "Distance (m)")


    plot_series(features.get("relative_speed_series"), "Relative Speed", "Speed (km/h)")

    plot_series(features.get("direction_angle_series"), "Direction Angle", "Angle (deg)")


    plot_series(features.get("density_A_series"), "Density Group Pedestrians", "Density")

    plot_series(features.get("density_B_series"), "Density Group Cyclists", "Density")

    if features.get("cluster_split_events"):
        plt.figure(figsize=(8, 2))
        for e in features["cluster_split_events"]:
            t_event = e["time"] / fps if fps else e["time"]
            plt.axvline(t_event, color="red", linestyle="--")
        plt.title("Cluster Split Events")
        plt.xlabel("Temps (s)")
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
        plt.xlabel("Temps (s)")
        plt.yticks([])
        plt.grid(True)
        plt.show()


def compute_one_interaction_features(df, history, interaction, fps=None, plot=False):
    split_events = detect_split_events_with_cyclists(history)
    hull_events = detect_cyclists_in_hulls(history)

    if is_noise_only_interaction(interaction):
        return compute_individual_interaction_features(interaction, df, fps=fps, plot=plot)
    else:
        return compute_group_interaction_features(interaction, history, df, split_events, hull_events, fps=fps, plot=plot)


def compute_all_interactions_features(df, history, interactions, fps=None, plot=False):

    split_events = detect_split_events_with_cyclists(history)
    hull_events = detect_cyclists_in_hulls(history)

    all_features = []

    for inter in interactions:

        t_start = inter["start"]
        t_end = inter["end"]
        cyc_ids = list(inter["cyc_ids"])
        ped_ids = list(inter["ped_ids"])
        inter_type = inter["type"]

        is_noise_noise = is_noise_only_interaction(inter)

        if is_noise_noise:
            f = compute_individual_interaction_features(inter, df, fps=fps, plot=plot)
        else:
            f = compute_group_interaction_features(inter, history, df, split_events, hull_events, fps=fps, plot=plot)
        
        all_features.append(f)

    return all_features





# def is_group_interaction(interaction):
#     """
#     Détermine si interaction implique un cluster
#     """
#     return (
#         "cluster" in interaction["type_ped"]
#         and "cluster" in interaction["type_cyc"]
#     )


def classify_one_interaction(df, history, inter, fps):
    """
    Classifie une interaction à partir des features.

    Parameters
    ----------
    features : dict
    is_group : bool

    Returns
    -------
    str : label de l'interaction
    """
    features = compute_one_interaction_features(df, history, inter, fps=fps)
    # is_group = is_group_interaction(inter)

    label = {}

    type_A = inter["type_ped"]
    type_B = inter["type_cyc"]

    if "noise" in type_A and "noise" in type_B:
        label = classify_individual_interaction(features)
    
    else: 
        label = classify_group_interaction(features)
    
    return {"interaction": inter,
            "features": features,
            "label": label}


def classify_all_interactions(df, history, interactions, fps):
    """
    Classifie toutes les interactions d'un recording.
    """

    results = []

    for inter in interactions:

        res = classify_one_interaction(df, history, inter, fps)

        results.append(res)
    
    return results



# def classify_individual_interaction(features):
#     """
#     Classification pour noise-noise (interaction paire)
#     """

#     d = features.get("min_distance")
#     angle = features.get("direction_angle")
#     approach = features.get("approach_angle")
#     rel_speed = features.get("relative_speed")
#     pet = features.get("PET")
#     ttc = features.get("ttc_min")

#     if d is None:
#         return "unknown"

#     # Pas interaction
#     if d > 5:
#         return "no_interaction"

#     # Head-on
#     if angle is not None and angle > 150:
#         if rel_speed is not None and rel_speed > 1.5:
#             return "head_on"

#     # Crossing
#     if angle is not None and 60 < angle < 120:
#         return "crossing"

#     # Overtaking
#     if angle is not None and angle < 30:
#         if rel_speed is not None and rel_speed > 0.5:
#             return "overtaking"

#     # Following
#     if angle is not None and angle < 30:
#         if rel_speed is not None and rel_speed < 0.5:
#             return "following"

#     # Avoidance (basé sur TTC)
#     if ttc is not None and ttc < 2:
#         return "avoidance"

#     return "interaction_other"



# def classify_group_interaction(features):
#     """
#     Classification pour interactions impliquant clusters
#     """

#     d_min = np.nanmin(features.get("distances", [np.nan]))
#     angles = features.get("direction_angle_series", [])
#     speeds = features.get("relative_speed_series", [])
#     densities_A = features.get("density_A_series", [])
#     densities_B = features.get("density_B_series", [])

#     in_hull = features.get("cyclist_in_hull", False)
#     split = features.get("cluster_split", False)

#     # Nettoyage
#     angles = np.array([a for a in angles if a is not None])
#     speeds = np.array([s for s in speeds if s is not None])

#     mean_angle = np.nanmean(angles) if len(angles) > 0 else None
#     mean_speed = np.nanmean(speeds) if len(speeds) > 0 else None

#     if d_min is None or np.isnan(d_min):
#         return "unknown"

#     # Pas interaction
#     if d_min > 5:
#         return "no_interaction"

#     # SPLIT = priorité max
#     if split:
#         return "group_split"

#     # PÉNÉTRATION (fort signal)
#     if in_hull:
#         if mean_speed is not None and mean_speed < 2:
#             return "penetration"
#         else:
#             return "cross_group"

#     # Crossing
#     if mean_angle is not None and 60 < mean_angle < 120:
#         return "cross_group"

#     # Bypass (contournement)
#     if mean_angle is not None and mean_angle > 120:
#         return "bypass_group"

#     # Approach
#     if mean_speed is not None and mean_speed > 1.5:
#         return "approach_group"

#     # Avoidance
#     if mean_speed is not None and mean_speed < 0.5:
#         return "group_avoidance"

#     return "group_interaction_other"




def classify_pet(pet):
    """
    Classification du PET (en secondes). Permet d'évaluer le niveau de risque lors de l'interaction.
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


# def clean_series(values):
#     values = np.array(values)
#     return values[np.isfinite(values)]

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

    # discrétisation
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

    # pattern
    if "VERY_CLOSE" in seq:
        label = "CRITICAL_PROXIMITY"
    elif "CLOSE" in seq:
        label = "CLOSE_INTERACTION"
    elif seq == ["FAR"]:
        label = "FAR_INTERACTION"
    else:
        label = "MODERATE_INTERACTION"

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
        elif x < 75:
            return "SLIGHT_DIVERGENCE"
        elif x < 120:
            return "CROSSING" # interaction perpendiculaire
        elif x < 160:
            return "STRONG_DIVERGENCE"
        else:
            return "OPPOSITE_DIRECTION"

    labels = [label_angle(x) for x in a]
    seq = compress_sequence(labels)

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
            return "FRONTAL_APPROACH" # rapprochement
        elif x < 75:
            return "OBLIQUE_APPROACH" # rapprochement en diagonal (vers le piéton)
        elif x < 105:
            return "CROSSING" # interaction perpendiculaire
        elif x < 150:
            return "OBLIQUE_DEPART" # éloignement en diagonal
        else:
            return "MOVING_AWAY" # éloignement

    labels = [label(x) for x in a]
    seq = compress_sequence(labels)

    # pattern clé
    if seq == ["FRONTAL_APPROACH", "CROSSING", "MOVING_AWAY"] or seq == ["FRONTAL_APPROACH", "OBLIQUE_APPROACH", "CROSSING", "OBLIQUE_DEPART", "MOVING_AWAY"] or seq == ["OBLIQUE_APPROACH", "CROSSING", "OBLIQUE_DEPART"]:
        label_main = "AVOIDANCE" # si dans des directions opposées, mais OVERTAKING (dépassement) si même direction

    elif "FRONTAL_APPROACH" in seq and "MOVING_AWAY" in seq:
        label_main = "APPROACH_AND_ESCAPE"

    elif "CROSSING" in seq:
        label_main = "CROSSING"
    
    elif "FRONTAL_APPROACH" in seq:
        label_main = "FRONTAL_APPROACH"

    elif "MOVING_AWAY" in seq or "OBLIQUE_DEPART" in seq:
        label_main = "MOVING_AWAY" # éloignement / évitement
    
    elif "OBLIQUE_APPROACH" in seq:
        label_main = "OBLIQUE_APPROACH"

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
        main = "VERY_DYNAMIC"
    elif "HIGH" in seq:
        main = "DYNAMIC"
    else:
        main = "LOW_DYNAMIC"

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

    # simplification
    front = np.sum((angles > -45) & (angles < 45))
    side = np.sum((angles >= 45) & (angles <= 135))
    back = np.sum((angles > 135) | (angles < -135))

    if front > side and front > back:
        label = "FRONT_INTERACTION"
    elif side > front and side > back:
        label = "SIDE_INTERACTION"
    else:
        label = "REAR_INTERACTION"

    return {
        "label_main": label,
        "angles": angles.tolist()
    }


def classify_individual_interaction(features):

    dist_res = classify_distance_series(features["distances"])

    dir_res = classify_direction_angle_series(features["direction_angle_series"])

    approach_res = classify_approach_angle_series(features["approach_angle_series"])

    speed_res = classify_relative_speed_series(features["relative_speed_series"])

    pos_res = classify_relative_position_series(features["relative_position_series"])

    pet_res = classify_pet(features["PET"])

    # ===== logique finale =====

    print("DIRECTION ANGLE :", dir_res["label_main"])

    if approach_res["label_main"] in ["AVOIDANCE", "APPROACH_AND_ESCAPE"]:
        if dir_res["label_main"] == "SAME_DIRECTION":
            interaction = "OVERTAKING"
        elif dir_res["label_main"] == "OPPOSITE_DIRECTION":
            interaction = "AVOIDANCE"
        elif dir_res["label_main"] == "CROSSING":
            interaction = "CROSSING"
    
    elif approach_res["label_main"] == "CROSSING":
        if dir_res["label_main"] in ["CROSSING", "DIVERGING"]:
            interaction = "CROSSING"

    elif approach_res["label_main"] == "FRONTAL_APPROACH":
        if dir_res["label_main"] == "SAME_DIRECTION":
            interaction = "FOLLOWING"
        elif dir_res["label_main"] == "OPPOSITE_DIRECTION":
            interaction = "AVOIDANCE"
        elif dir_res["label_main"] == "CROSSING":
            interaction = "CROSSING"
            if pet_res == "CRITICAL" or (pet_res == "CRITICAL" and speed_res["label_main"] == "VERY_DYNAMIC") or (pet_res == "CRITICAL" and dist_res["label_main"] == "CRITICAL_PROXIMITY"):
                interaction = "CLOSE_COLLISION"
            elif pet_res == "MEDIUM" or pet_res == "LOW":
                if speed_res["label_main"] == "DYNAMIC" or speed_res["label_main"] == "LOW_DYNAMIC":
                    interaction = "GIVE_WAY"
    
    elif approach_res["label_main"] == "MOVING_AWAY":
        interaction = "MOVING_AWAY" # éloignement / évitement
    
    elif approach_res["label_main"] == "OBLIQUE_APPROACH":
        interaction = "OBLIQUE_APPROACH"
    
    else:
        interaction = "UNDEFINED"

    score = 0

    if pet_res == "CRITICAL":
        score += 3
    elif pet_res == "HIGH":
        score += 2
    elif pet_res == "MEDIUM":
        score += 1
    

    if dist_res["label_main"] == "CRITICAL_PROXIMITY":
        if speed_res["label_main"] == "VERY_DYNAMIC":
            score += 3
        elif speed_res["label_main"] == "DYNAMIC":
            score += 2
        elif speed_res["label_main"] == "LOW_DYNAMIC":
            score += 1
    elif dist_res["label_main"] == "CLOSE_INTERACTION":
        if speed_res["label_main"] == "VERY_DYNAMIC":
            score += 3
        elif speed_res["label_main"] == "DYNAMIC":
            score += 2
    elif dist_res["label_main"] == "MODERATE_INTERACTION":
        if speed_res["label_main"] == "VERY_DYNAMIC":
            score += 2
        elif speed_res["label_main"] == "DYNAMIC":
            score += 1
    elif dist_res["label_main"] == "FAR_INTERACTION":
        if speed_res["label_main"] == "VERY_DYNAMIC":
            score += 1
    

    if score >= 5:
        risk = "CRITICAL"
    elif score >= 3:
        risk = "HIGH"
    elif score >= 2:
        risk = "MEDIUM"
    else:
        risk = "LOW"


    return {
        "interaction_type": interaction,
        "direction": dir_res,
        "approach": approach_res,
        "position": pos_res,
        "distance": dist_res,
        "speed": speed_res,
        "pet": pet_res,
        "risk": risk,
        "risk_score": score
    }



def classify_group_interaction(features):
    """
    features attendu :
    {
        "distances": [],
        "relative_position_series": [],
        "direction_angle_series": [],
        "relative_speed_series": [],
        "cyclist_in_hull": bool,
        "cluster_split": bool
    }

    noise_type = "cyc"
    """

    dist_res = classify_distance_series(features["distances"])
    speed_res = classify_relative_speed_series(features["relative_speed_series"])

    direction_res = classify_direction_angle_series(features["direction_angle_series"])
    pos_res = classify_relative_position_series(features["relative_position_series"])

    in_hull = features["cyclist_in_hull"]
    split = features["cluster_split"]

    # ===== TYPE =====

    # pénétration dans groupe
    if in_hull and split:
        interaction = "GROUP_PENETRATION_AND_SPLIT" # faufilement du cycliste puis split du groupe

    elif in_hull and not split:
        interaction = "GROUP_PENETRATION" # faufilement du cycliste
    
    # cassure du groupe
    elif not in_hull and split:
        interaction = "GROUP_SPLIT"

    # dépassement
    elif (direction_res["label_main"] == "SAME_DIRECTION" or direction_res["label_main"] == "OPPOSITE_DIRECTION") and pos_res["label_main"] == "SIDE_INTERACTION" and (speed_res["label_main"] == "VERY_DYNAMIC" or speed_res["label_main"] == "DYNAMIC"):
        interaction = "OVERTAKING_GROUP"

    # contournement
    elif pos_res["label_main"] == "SIDE_INTERACTION":
        interaction = "BYPASSING_GROUP"

    # croisement
    elif direction_res["label_main"] == "CROSSING":
        interaction = "CROSSING_GROUP"

    else:
        interaction = "WEAK_INTERACTION"

    # ===== RISQUE =====
    score = 0

    if dist_res["label_main"] == "CRITICAL_PROXIMITY":
        if speed_res["label_main"] == "VERY_DYNAMIC":
            score += 3
        elif speed_res["label_main"] == "DYNAMIC":
            score += 2
        elif speed_res["label_main"] == "LOW_DYNAMIC":
            score += 1
    elif dist_res["label_main"] == "CLOSE_INTERACTION":
        if speed_res["label_main"] == "VERY_DYNAMIC":
            score += 3
        elif speed_res["label_main"] == "DYNAMIC":
            score += 2
    elif dist_res["label_main"] == "MODERATE_INTERACTION":
        if speed_res["label_main"] == "VERY_DYNAMIC":
            score += 2
        elif speed_res["label_main"] == "DYNAMIC":
            score += 1
    elif dist_res["label_main"] == "FAR_INTERACTION":
        if speed_res["label_main"] == "VERY_DYNAMIC":
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
        "risk": risk,
        "risk_score": score
    }


def compute_reactivity_score(df_agent):
    """
    Score global de réactivité basé sur variation de direction.
    """

    directions, _ = compute_vector_direction_series(df_agent)

    if len(directions) < 2:
        return 0

    variations = compute_direction_variation(directions)

    # métriques possibles
    mean_var = np.mean(variations)
    max_var = np.max(variations)
    p90_var = np.percentile(variations, 90)

    return {
        "mean_variation": mean_var,
        "max_variation": max_var,
        "p90_variation": p90_var
    }


def detect_reactive_agent(df, ped_df, cyc_df):
    """
    Renvoie reactive_agent, stable_agent"""
    # ped_df = df[df[COL_ID] == ped_id]
    # cyc_df = df[df[COL_ID] == cyc_id]

    ped_score = compute_reactivity_score(ped_df)
    cyc_score = compute_reactivity_score(cyc_df)

    # on compare une métrique robuste, pour ne pas prendre en compte le bruit notamment
    if ped_score["p90_variation"] > cyc_score["p90_variation"]:
        return "ped", "cyc"
    else:
        return "cyc", "ped"