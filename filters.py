import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from config import *
from analysis_interactions import *


def analyze_initial_nb_traj_interactions(df, verbose=True):

    # Comptage par classe
    initial_counts = df.groupby(COL_CLASS)[COL_ID].nunique().to_dict()

    initial_ped = initial_counts.get(1, 0)
    initial_cyc = initial_counts.get(2, 0)
    initial_car = initial_counts.get(3, 0)

    initial_ids = set(df[COL_ID].unique())
    initial_nb_traj = len(initial_ids)

    initial_interactions = compute_ped_cyc_interactions(df)
    initial_nb_interactions = len(initial_interactions)

    if verbose:
        print("\nAnalyse avant filtrage")
        print(f"Nb piétons au total : {initial_ped}")
        print(f"Nb cyclistes au total : {initial_cyc}")
        print(f"Nb voitures au total : {initial_car}")
        print(f"Trajectoires totales : {initial_nb_traj}")
        print(f"Interactions piéton-cycliste (rayon 5m): {initial_nb_interactions}")

    return initial_nb_traj, initial_nb_interactions



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



def filter_spatial_car_influence(df, distance_threshold):
    """
    Supprime des datasets les agents (piétons/cyclistes) qui passent trop près d'une voiture (cercle autour avec un rayon donné).
    Supprime aussi les voitures influentes à la fin.

    Parameters:
        df (DataFrame)
        distance_threshold (float): distance en mètres

    Returns:
        df_filtered
    """

    bad_vru_ids = set()
    bad_car_ids = set()

    # Comptage initial par classe
    initial_counts = df.groupby(COL_CLASS)[COL_ID].nunique().to_dict()

    # Parcours par frame
    for t in df[COL_TIME].unique():
        frame = df[df[COL_TIME] == t]

        cars = frame[frame[COL_CLASS] == 3]
        pedestrians = frame[frame[COL_CLASS] == 1]
        cyclists = frame[frame[COL_CLASS] == 2]

        if len(cars) == 0 or len(pedestrians) == 0 or len(cyclists)==0:
            # on applique le filtre que si voiture + au moins 1 piéton + au moins 1 cyclist présents dans la même frame
            continue

        vrus = frame[frame[COL_CLASS].isin([1, 2])]

        # Comparaison distance voiture - VRU
        for _, vru in vrus.iterrows():
            for _, car in cars.iterrows():
                dx = vru["x_m"] - car["x_m"]
                dy = vru["y_m"] - car["y_m"]
                dist = np.hypot(dx, dy) # hypothénuse triange = dist euclidienne en 2D

                if dist < distance_threshold:
                    bad_vru_ids.add(vru[COL_ID])
                    bad_car_ids.add(car[COL_ID])
                    break
    
    
    # FILTRAGE
    df_filtered = df[~df[COL_ID].isin(bad_vru_ids.union(bad_car_ids))].copy()

    # COMPTAGE
    final_counts = df_filtered.groupby(COL_CLASS)[COL_ID].nunique().to_dict()

    removed_counts = {
        1: initial_counts.get(1, 0) - final_counts.get(1, 0),
        2: initial_counts.get(2, 0) - final_counts.get(2, 0),
        3: initial_counts.get(3, 0) - final_counts.get(3, 0),
    }

    print("\n FILTRAGE SPATIAL")
    print(f"Seuil distance : {distance_threshold} m")
    print(f"Piétons supprimés   : {removed_counts[1]}")
    print(f"Cyclistes supprimés : {removed_counts[2]}")
    print(f"Voitures supprimées : {removed_counts[3]}")

    return df_filtered


def filter_coexisting_with_cars(df):
    initial_counts = df.groupby(COL_CLASS)[COL_ID].nunique().to_dict()

    initial_ped = initial_counts.get(1, 0)
    initial_cyc = initial_counts.get(2, 0)
    initial_car = initial_counts.get(3, 0)


    bad_ids = set()

    # frames avec voiture
    car_frames = set(df[df[COL_CLASS] == 3][COL_TIME].unique())

    # récupérer tous les agents présents dans ces frames (y compris voitures)
    for t in car_frames:
        frame = df[df[COL_TIME] == t]
        ids = frame[COL_ID].unique()

        for i in ids:
            bad_ids.add(i)

    # filtrage
    df_filtered = df[~df[COL_ID].isin(bad_ids)].copy()

    print("\nAnalyse et impact du filtre")

    final_ids = set(df_filtered[COL_ID].unique())
    final_nb_traj = len(final_ids)

    final_counts = df_filtered.groupby(COL_CLASS)[COL_ID].nunique().to_dict()

    final_ped = final_counts.get(1, 0)
    final_cyc = final_counts.get(2, 0)
    final_car = final_counts.get(3, 0)

    final_interactions = compute_ped_cyc_interactions(df_filtered)
    final_nb_interactions = len(final_interactions)

    # IMPACT du filtrage
    initial_nb_traj, initial_nb_interactions = analyze_initial_nb_traj_interactions(df, verbose=False)
    removed_traj = initial_nb_traj - final_nb_traj
    removed_inter = initial_nb_interactions - final_nb_interactions

    removed_ped = initial_ped - final_ped
    removed_cyc = initial_cyc - final_cyc
    removed_car = initial_car - final_car

    print(f"Trajectoires supprimées : {removed_traj} sur {initial_nb_traj} "
          f"({removed_traj / initial_nb_traj * 100:.2f}%)")
    print(f"Trajectoires restantes : {final_nb_traj}")

    if initial_nb_interactions > 0:
            print(f"Interactions piéton-cycliste supprimées : {removed_inter} sur {initial_nb_interactions} "
                f"({removed_inter / initial_nb_interactions * 100:.2f}%)")
    else:
        print("Aucune interaction initiale.")
    print(f"Interactions piéton-cycliste restantes : {final_nb_interactions}")

    print(f"Piétons supprimés   : {removed_ped} sur {initial_ped} "
          f"({(removed_ped / initial_ped * 100 if initial_ped else 0):.2f}%)")

    print(f"Cyclistes supprimés : {removed_cyc} sur {initial_cyc} "
          f"({(removed_cyc / initial_cyc * 100 if initial_cyc else 0):.2f}%)")

    print(f"Voitures supprimées : {removed_car} sur {initial_car} "
          f"({(removed_car / initial_car * 100 if initial_car else 0):.2f}%)")

    return df_filtered, bad_ids





###############################################
# Pour CTV, lissage des trajectoires (filtre de Kalman)
###############################################

def apply_kalman_filter(df, R_value=0.5):
    filtered_data = []

    for object_id, g in df.groupby(COL_ID):
        g = g.sort_values(COL_TIME)

        xs = g["x_m"].values
        ys = g["y_m"].values

        if len(xs) < 2:
            filtered_data.append(g)
            continue

        # xs_f, ys_f = kalman_filter_2d(xs, ys)
        xs_f, ys_f = kalman_filter_2d(xs, ys, R_value)

        g = g.copy()
        g["x_m"] = xs_f
        g["y_m"] = ys_f

        filtered_data.append(g)

    return pd.concat(filtered_data, ignore_index=True)


def kalman_filter_2d(xs, ys, R_value=0.5):
    n = len(xs)

    # État : [x, y, vx, vy]
    x = np.array([xs[0], ys[0], 0, 0], dtype=float)

    # Matrice de transition (modèle vitesse constante)
    dt = 1.0
    F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1,  0],
        [0, 0, 0,  1]
    ])

    # Observation : on observe seulement (x, y)
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    # Bruit (à ajuster !)
    Q = np.eye(4) * 0.01   # bruit du modèle (= à quel point le mouvement peut changer | petit -> mouv rigide + traj lisse | grand -> mouv flexible + traj réactive | vaut mieux petit pour VRUs)
    # R = np.eye(2) * 0.5    # bruit de mesure (petit -> peu de lissage et bruit visible | grand -> fort lissage, trajectoire plus propre)
    R = np.eye(2) * R_value

    # Covariance initiale
    P = np.eye(4)

    xs_f = []
    ys_f = []

    for i in range(n):
        z = np.array([xs[i], ys[i]])

        # prédiction
        x = F @ x
        P = F @ P @ F.T + Q

        # update
        y = z - (H @ x)
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)

        x = x + K @ y
        P = (np.eye(4) - K @ H) @ P

        xs_f.append(x[0])
        ys_f.append(x[1])

    return np.array(xs_f), np.array(ys_f)


def compare_kalman_R(df, R_values, agent_id=None):
    import matplotlib.pyplot as plt

    if agent_id is None:
        agent_id = df[COL_ID].iloc[0]

    g = df[df[COL_ID] == agent_id].sort_values(COL_TIME)

    xs = g["x_m"].values
    ys = g["y_m"].values

    plt.figure(figsize=(10, 8))

    # trajectoire brute
    plt.plot(xs, ys, 'k--', label="Raw", alpha=0.5)

    for R in R_values:
        xs_f, ys_f = kalman_filter_2d(xs, ys, R_value=R)
        # smoothness = compute_smoothness(xs_f, ys_f)
        # print(f"R={R} -> smoothness={smoothness:.4f}")
        plt.plot(xs_f, ys_f, label=f"R={R}")

    plt.legend()
    plt.title(f"Comparaison Kalman - agent {agent_id}")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.gca().invert_yaxis()
    plt.grid()

    plt.show()

# fonction peut-être pas si utile
# def compute_smoothness(xs, ys):
#     # variation de vitesse par rapport aux données intiales
#     dx = np.diff(xs)
#     dy = np.diff(ys)
#     speed = np.hypot(dx, dy)

#     return np.std(speed)


def get_one_agent_per_class(df):
    agents = {}

    for cls in [1, 2]:  # 1=piéton, 2=cycliste
        subset = df[df[COL_CLASS] == cls]

        if len(subset) == 0:
            continue

        # prendre celui avec la trajectoire la plus longue (meilleur pour analyse comparative ?)
        lengths = subset.groupby(COL_ID).size()
        best_id = lengths.idxmax()

        agents[cls] = best_id

    return agents


def compare_kalman_R_two_agents(df, R_values):
    import matplotlib.pyplot as plt

    agents = get_one_agent_per_class(df)

    fig, axes = plt.subplots(1, len(agents), figsize=(14, 6))

    if len(agents) == 1:
        axes = [axes]

    for ax, (cls, agent_id) in zip(axes, agents.items()):
        g = df[df[COL_ID] == agent_id].sort_values(COL_TIME)

        xs = g["x_m"].values
        ys = g["y_m"].values

        # brute
        ax.plot(xs, ys, 'k--', label="Raw", alpha=0.5)

        title = "Pedestrian" if cls == 1 else "Cyclist"
        ax.set_title(f"{title} (ID={agent_id})")
        print(f"\n{title} (ID={agent_id})")
        # smoothness = compute_smoothness(xs, ys)
        # print(f"Raw -> smoothness={smoothness:.4f}")

        for R in R_values:
            xs_f, ys_f = kalman_filter_2d(xs, ys, R_value=R)
            # smoothness = compute_smoothness(xs_f, ys_f)
            # print(f"R={R:<4} -> smoothness={smoothness:.4f}")
            ax.plot(xs_f, ys_f, label=f"R={R}")


        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.invert_yaxis()
        ax.grid()
        ax.legend()

    plt.suptitle("Comparaison Kalman selon R")
    plt.tight_layout()
    plt.show()