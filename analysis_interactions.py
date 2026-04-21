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


def classify_direction_angle(angle):
    if angle is None:
        return "None"

    if angle < 30:
        return "SAME_DIRECTION"
    elif angle < 75:
        return "SLIGHT_CONVERGENCE"
    elif angle < 105:
        return "CROSSING" # interaction perpendiculaire
    elif angle < 150:
        return "STRONG_CONVERGENCE"
    else:
        return "OPPOSITE_DIRECTION"
    
def classify_direction_angle_interaction(df, ped_id, cyc_id, times, angles):
    interactions = compute_ped_cyc_interactions_with_time(df)
    pair = tuple(sorted((ped_id, cyc_id)))

    frames = interactions.get(pair, [])
    if len(frames) == 0:
        return {"label": "no interaction"}

    intervals = frames_to_intervals(frames)

    # ===== filtrage pendant interaction =====
    mask = np.zeros_like(times, dtype=bool)
    for start, end in intervals:
        mask |= (times >= start) & (times <= end)

    angles_inter = angles[mask]

    if len(angles_inter) == 0:
        return {"label": "UNKNOWN"}

    # ===== classification point par point =====
    angle_classes = [classify_direction_angle(a) for a in angles_inter]

    # ===== compression de séquence =====
    sequence = [angle_classes[0]]
    for c in angle_classes[1:]:
        if c != sequence[-1]:
            sequence.append(c)

    # priorité à la géométrie la plus critique
    if "OPPOSITE_DIRECTION" in sequence:
        label_main = "OPPOSITE_DIRECTION"
    elif "STRONG_CONVERGENCE" in sequence:
        label_main = "STRONG_CONVERGENCE"
    elif "CROSSING" in sequence:
        label_main = "CROSSING"
    elif "SLIGHT_CONVERGENCE" in sequence:
        label_main = "SLIGHT_CONVERGENCE"
    else:
        label_main = "SAME_DIRECTION"

    return {
        "label_main": label_main,
        "sequence": sequence,
        "angle_min": np.min(angles_inter),
        "angle_median": np.median(angles_inter)
    }


def classify_approach_angle(angle):
    if angle is None:
        return "UNKNOWN"

    if angle < 30:
        return "FRONTAL_APPROACH"
    elif angle < 75:
        return "OBLIQUE_APPROACH"
    elif angle < 105:
        return "CROSSING"
    elif angle < 150:
        return "OBLIQUE_DEPART"
    else:
        return "MOVING_AWAY"
    

def classify_approach_angle_interaction(df, ped_id, cyc_id, times, angles):
    interactions = compute_ped_cyc_interactions_with_time(df)
    pair = tuple(sorted((ped_id, cyc_id)))

    frames = interactions.get(pair, [])
    if len(frames) == 0:
        return {"label": "no interaction"}

    intervals = frames_to_intervals(frames)

    # filtrage
    mask = np.zeros_like(times, dtype=bool)
    for start, end in intervals:
        mask |= (times >= start) & (times <= end)

    angles_inter = angles[mask]

    if len(angles_inter) == 0:
        return {"label": "UNKNOWN"}

    # classification point par point
    angle_classes = [classify_approach_angle(a) for a in angles_inter]

    # compression
    sequence = [angle_classes[0]]
    for c in angle_classes[1:]:
        if c != sequence[-1]:
            sequence.append(c)
    
    # label_main = max(set(sequence), key=sequence.count)

    #label principal
    if "FRONTAL_APPROACH" in sequence:
        label_main = "FRONTAL_APPROACH"
    elif "OBLIQUE_APPROACH" in sequence:
        label_main = "OBLIQUE_APPROACH"
    elif "CROSSING" in sequence:
        label_main = "CROSSING"
    else:
        label_main = sequence[0]

    return {
        "label_main": label_main,
        "sequence": sequence,
        "angle_min": np.min(angles_inter),
        "angle_median": np.median(angles_inter)
    }


def classify_relative_speed(v_rel):
    # pour des vitesses en km/h
    if v_rel is None:
        return "UNKNOWN"

    if v_rel < 3:
        return "LOW"
    elif v_rel < 15:
        return "MODERATE"
    elif v_rel < 20:
        return "HIGH"
    else:
        return "VERY_HIGH"

def classify_relative_speed_interaction(
    df,
    ped_id,
    cyc_id,
    times,
    v_rel
):
    # pour des vitesses en km/h
    interactions = compute_ped_cyc_interactions_with_time(df)
    pair = tuple(sorted((ped_id, cyc_id)))

    frames = interactions.get(pair, [])
    if len(frames) == 0:
        return {"label": "NO_INTERACTION"}

    intervals = frames_to_intervals(frames)

    # ===== filtrage =====
    mask = np.zeros_like(times, dtype=bool)
    for start, end in intervals:
        mask |= (times >= start) & (times <= end)

    v_inter = v_rel[mask]

    # enlever NaN / inf
    v_inter = v_inter[np.isfinite(v_inter)]

    if len(v_inter) == 0:
        return {"label": "UNKNOWN"}

    
    v_max = np.max(v_inter) # le moment le plus dangereux serait le moment où la vitesse est la plus élevée
    v_mean = np.mean(v_inter)

    label_max = classify_relative_speed(v_max)
    labels = [classify_relative_speed(v) for v in v_inter]
    dominant = max(set(labels), key=labels.count)

    return {
        "label_main": dominant, # selon l'occurence des labels
        "label_max": label_max, # selon la vitesse max
        "v_max": v_max,
        "v_mean": v_mean,
        "labels": labels,
    }


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


def classify_global_interaction(pet, ttc, approach_result, direction_result, speed_result):
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

def hausdorff_distance(A, B):
    """
    A, B: arrays de taille (N,2) et (M,2) respectivement. 
    """

    def directed(A, B):
        dists = []
        for a in A:
            d = np.min(np.linalg.norm(B - a, axis=1))
            dists.append(d)
        return np.max(dists)

    return max(directed(A, B), directed(B, A))

def min_distance(A, B): # pour prendre la dist min entre 2 groupes de points (la distance de Hausdorff)
    min_d = float("inf")
    for a in A:
        d = np.min(np.linalg.norm(B - a, axis=1))
        if d < min_d:
            min_d = d
    return min_d


def compute_group_distances_over_time(df):
    results = []

    for t in df[COL_TIME].unique():
        frame = df[df[COL_TIME] == t]

        P = frame[frame[COL_CLASS] == 1][["x_m", "y_m"]].values
        C = frame[frame[COL_CLASS] == 2][["x_m", "y_m"]].values

        if len(P) == 0 or len(C) == 0:
            continue

        h = hausdorff_distance(P, C)
        dmin = min_distance(P, C)

        results.append((t, h, dmin))

    return np.array(results)


def plot_group_distances(results):
    times = results[:,0]
    haus = results[:,1]
    dmin = results[:,2]

    plt.figure(figsize=(10,5))
    plt.plot(times, haus, label="Hausdorff distance")
    plt.plot(times, dmin, label="Min distance", linestyle="--")

    plt.xlabel("Temps")
    plt.ylabel("Distance (m)")
    plt.title("Distances groupe piétons - groupe cyclistes")
    plt.legend()
    plt.grid()
    plt.show()