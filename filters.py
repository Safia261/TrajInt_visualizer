import numpy as np
import matplotlib.pyplot as plt
from config import *


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