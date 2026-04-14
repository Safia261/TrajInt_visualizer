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



def compute_agent_direction(
    df,
    agent_id,
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

            ax1.plot(t[1:], angles, label="angle")
            ax1.set_xlabel("Temps")
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