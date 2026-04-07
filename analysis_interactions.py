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