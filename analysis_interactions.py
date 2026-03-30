import numpy as np
from config import *


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