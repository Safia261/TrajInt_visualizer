import os
import glob
import pandas as pd
from config import *

def prepare_data(df, no_cars=False):
    df = df.copy()

    if no_cars:
        df = df[df[COL_CLASS].isin([1, 2])].copy()

    df = df.sort_values([COL_ID, COL_TIME]).reset_index(drop=True)
    df[COL_ID] = df[COL_ID].astype(int)
    df[COL_CLASS] = df[COL_CLASS].astype(int)
    return df



def load_dataset(dataset_name, file_path, args=None):
    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset inconnu : {dataset_name}")

    cfg = DATASETS[dataset_name]

    if cfg["type"] == "vru":
        return load_vru_dataset(cfg, args.vru_type, args.vru_behavior)

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    # ===== LOAD =====
    # df = pd.read_csv(file_path)
    if cfg["type"] == "csv":
        df = pd.read_csv(file_path)

    elif cfg["type"] == "pkl":
        df = pd.read_pickle(file_path)

    else:
        raise ValueError(f"Type non supporté : {cfg['type']}")
    df["__file__"] = os.path.basename(file_path)

    # ===== NORMALISATION =====
    col = cfg["columns"]

    df = df.rename(columns={
        col["time"]: COL_TIME,
        col["id"]: COL_ID
        # col["class"]: COL_CLASS
    })

    if "class" in col:
        df = df.rename(columns={col["class"]: COL_CLASS})
    else:
        df[COL_CLASS] = 1

    # ===== POSITION =====
    if cfg["format"] == "centroid":
        df["x_m"] = df[col["x"]]
        df["y_m"] = df[col["y"]]

    elif cfg["format"] == "bbox":
        df["x_m"] = (df[col["tl_x"]] + df[col["br_x"]]) / 2.0
        df["y_m"] = (df[col["tl_y"]] + df[col["br_y"]]) / 2.0

    # ===== SCALE =====
    if cfg["scale"] == "pixel":
        ppm = cfg["pixels_per_meter"]
        df["x_m"] = df["x_m"] / ppm
        df["y_m"] = df["y_m"] / ppm

    # ===== IMAGE =====
    image_path = cfg["image"]

    if image_path is None:
        folder = cfg["folder"]
        imgs = glob.glob(os.path.join(folder, "*.png"))
        image_path = imgs[0] if imgs else None

    return df, image_path, cfg


# def load_vru_folder(cfg):
#     folder = cfg["folder"]

#     files = glob.glob(os.path.join(folder, "*.csv"))

#     if not files:
#         raise FileNotFoundError(f"Aucun CSV trouvé dans {folder}")

#     all_data = []
#     global_id = 0

#     for f in files:
#         df = pd.read_csv(f)

#         col = cfg["columns"]

#         # ===== rename =====
#         df = df.rename(columns={
#             col["time"]: COL_TIME,
#             col["x"]: "x_m",
#             col["y"]: "y_m"
#         })

#         # ===== ID unique (1 fichier = 1 trajectoire) =====
#         df[COL_ID] = global_id
#         global_id += 1

#         # ===== classe =====
#         df[COL_CLASS] = 1 # A REVOIR!!!!

#         df["__file__"] = os.path.basename(f)

#         all_data.append(df)

#     df_all = pd.concat(all_data, ignore_index=True)

#     return df_all, None, cfg

# def load_vru_dataset(cfg, vru_type, vru_behavior):
#     base_folder = cfg["folder"]

#     types = []
#     if vru_type == "both":
#         types = ["pedestrians", "cyclists"]
#     else:
#         types = [vru_type]

#     behaviors = []
#     if vru_behavior == "all":
#         behaviors = ["starting", "moving", "stopping", "waiting"]
#     else:
#         behaviors = [vru_behavior]

#     all_data = []
#     global_id = 0

#     for t in types:
#         for b in behaviors:
#             folder = os.path.join(base_folder, t, b)
#             files = glob.glob(os.path.join(folder, "*.csv"))

#             for f in files:
#                 df = pd.read_csv(f)

#                 # ===== colonnes (à adapter si besoin) =====
#                 df = df.rename(columns={
#                     "timestamp": COL_TIME,
#                     "x": "x_m",
#                     "y": "y_m"
#                 })

#                 # ===== ID =====
#                 # CAS 1 : fusion comportements -> même agent
#                 # CAS 2 : sinon -> nouvel agent

#                 if vru_behavior == "all":
#                     # on essaye de garder l'id fichier
#                     agent_id = os.path.splitext(os.path.basename(f))[0]
#                 else:
#                     agent_id = global_id
#                     global_id += 1

#                 df[COL_ID] = agent_id

#                 # ===== classe =====
#                 if t == "pedestrians":
#                     df[COL_CLASS] = 1
#                 else:
#                     df[COL_CLASS] = 2

#                 df["behavior"] = b
#                 df["__file__"] = os.path.basename(f)

#                 all_data.append(df)

#     df_all = pd.concat(all_data, ignore_index=True)

#     # Attention si IDs sont strings alors faut peut-être convertir
#     df_all[COL_ID] = df_all[COL_ID].astype("category").cat.codes

#     return df_all, None, cfg


def load_vru_dataset(cfg, vru_type, vru_behavior):
    base_folder = cfg["folder"]

    # ===== choix des types =====
    if vru_type == "both":
        types = ["pedestrians", "cyclists"]
    else:
        types = [vru_type]

    # ===== ordre IMPORTANT =====
    behavior_order = ["starting", "moving", "stopping", "waiting"]

    if vru_behavior == "all":
        behaviors = behavior_order
    else:
        behaviors = [vru_behavior]

    all_data = []
    global_id = 0
    time_offset = 0  # pour enchaîner les comportements

    # =========================================================
    # BOUCLE PRINCIPALE
    # =========================================================
    for b in behavior_order:
        if b not in behaviors:
            continue

        block_data = []

        for t in types:
            folder = os.path.join(base_folder, t, b)
            files = glob.glob(os.path.join(folder, "*.csv"))

            for f in files:
                df = pd.read_csv(f)

                # ===== rename colonnes =====
                df = df.rename(columns={
                    "timestamp": COL_TIME,
                    "x": "x_m",
                    "y": "y_m"
                })

                # ===== ID unique =====
                # df[COL_ID] = global_id
                # global_id += 1

                filename = os.path.splitext(os.path.basename(f))[0]
                # print(filename)
                df[COL_ID] = int(filename)

                # ===== classe =====
                if t == "pedestrians":
                    df[COL_CLASS] = 1
                else:
                    df[COL_CLASS] = 2

                # ===== info debug =====
                df["behavior"] = b
                df["__file__"] = os.path.basename(f)

                block_data.append(df)

        # ===== si aucun fichier dans ce behavior =====
        if not block_data:
            continue

        block_df = pd.concat(block_data, ignore_index=True)

        # =========================================================
        # NORMALISATION TEMPORELLE DU BLOC
        # =========================================================
        t_min = block_df[COL_TIME].min()
        block_df[COL_TIME] = block_df[COL_TIME] - t_min

        # =========================================================
        # DÉCALAGE POUR ENCHAÎNER LES BLOCS
        # =========================================================
        block_df[COL_TIME] += time_offset

        # =========================================================
        # UPDATE OFFSET
        # =========================================================
        t_max = block_df[COL_TIME].max()
        time_offset = t_max + 1  # petit gap entre comportements MAIS à voir si je retire ça ou pas après

        all_data.append(block_df)

    # ===== concat final =====
    if not all_data:
        raise ValueError("Aucune donnée VRU chargée")

    df_all = pd.concat(all_data, ignore_index=True)

    return df_all, None, cfg