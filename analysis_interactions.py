import numpy as np
import matplotlib.pyplot as plt
from config import *

def compute_speed(g, fps):
    g = g.sort_values(COL_TIME)

    xs = g["x_m"].values
    ys = g["y_m"].values
    times = g[COL_TIME].values

    if len(xs) < 2:
        return None, None

    dx = np.diff(xs)
    dy = np.diff(ys)

    if len(np.unique(times)) > 1:
        dt = np.diff(times)
    else:
        dt = np.ones_like(dx) / fps # à revoir pour éviter les incohérences (dt = 1/fps en s)

    dt[dt == 0] = 1e-6 # pour éviter division par zéro

    speeds = np.hypot(dx, dy) / dt # switch à dist euclidienne au cas où

    return times[1:], speeds


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


def analyze_speeds(
    df,
    cfg,
    agent_ids=None,
    classes=None
):
    import matplotlib.pyplot as plt
    import numpy as np

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

        times, speeds = compute_speed(g, cfg)

        if times is None:
            continue

        data.append({
            "id": aid,
            "class": cls,
            "times": times,
            "speeds": speeds
        })

    if len(data) == 0:
        print("Pas de données exploitables")
        return

    # ===============================
    # 3. FIGURE
    # ===============================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax_hist = axes[0]
    ax_curve = axes[1]

    # ===============================
    # 4. HISTOGRAMMES PAR CLASSE
    # ===============================
    speeds_by_class = {}

    for d in data:
        cls = d["class"]
        speeds_by_class.setdefault(cls, []).extend(d["speeds"])

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
        print(f"Nb mesures : {len(speeds)}")
        print(f"Min : {speeds.min():.2f} m/s")
        print(f"Max : {speeds.max():.2f} m/s")
        print(f"Moy : {speeds.mean():.2f} m/s")

    ax_hist.set_title("Distribution des vitesses")
    ax_hist.set_xlabel("Vitesse (m/s)")
    ax_hist.set_ylabel("Fréquence")
    ax_hist.legend()
    ax_hist.grid()

    # ===============================
    # 5. COURBES INDIVIDUELLES
    # ===============================
    for d in data:
        aid = d["id"]
        cls = d["class"]
        times = d["times"]
        speeds = d["speeds"]

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
                cars = frame[frame[COL_CLASS] == 3]

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
            car_frames = set(df[df[COL_CLASS] == 3][COL_TIME].unique())

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