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

    if cfg["type"] == "vru": # mettre dataset_name == vru plutôt ??
        return load_vru_dataset(cfg, args.vru_type, args.vru_behavior)
    
    if dataset_name == "stanford2":
        return load_stanford2_dataset(cfg, args.scene, args.video)
    
    if cfg["type"] == "ind":
        return load_ind_dataset(cfg, file_path)

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




def load_stanford2_dataset(cfg, scene=None, video=None):
    root = cfg["folder"]

    # ===== CHECK SCENE =====
    if scene is None:
        scenes = os.listdir(root)
        scene = scenes[0]  # fallback
        print(f"[INFO] Scene par défaut : {scene}")

    scene_path = os.path.join(root, scene)

    if not os.path.isdir(scene_path):
        raise ValueError(f"Scene invalide : {scene}")

    # ===== CHECK VIDEO =====
    if video is None:
        videos = os.listdir(scene_path)
        video = videos[0]
        print(f"[INFO] Video par défaut : {video}")

    video_path = os.path.join(scene_path, video)

    ann_file = os.path.join(video_path, "annotations.txt")
    ref_img = os.path.join(video_path, "reference.jpg")

    if not os.path.exists(ann_file):
        raise FileNotFoundError(f"annotations.txt introuvable dans {video_path}")

    # ===== LOAD =====
    df = pd.read_csv(ann_file, sep=" ", header=None)

    df.columns = [
        "track_id",
        "xmin", "ymin", "xmax", "ymax",
        "frame",
        "lost", "occluded", "generated",
        "label"
    ]

    # filtre
    df = df[df["lost"] == 0]

    # conversion bbox -> centroid
    df["x_m"] = (df["xmin"] + df["xmax"]) / 2
    df["y_m"] = (df["ymin"] + df["ymax"]) / 2

    # normalisation
    df = df.rename(columns={
        "track_id": COL_ID,
        "frame": COL_TIME
    })

    # classification des usagers
    def map_class(label):
        label = str(label).replace('"', '')

        mapping = {
            "Pedestrian": 1,
            "Biker": 2,
            "Car": 3,
            "Cart": 4,
            "Skater": 5,
            "Bus": 6
        }

        return mapping.get(label, 0)


    df[COL_CLASS] = df["label"].apply(map_class)

    df["scene"] = scene
    df["video"] = video
    df["__file__"] = f"{scene}_{video}"

    return df, ref_img, cfg



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

    for b in behavior_order:
        if b not in behaviors:
            continue

        block_data = []

        for t in types:
            folder = os.path.join(base_folder, t, b)
            files = glob.glob(os.path.join(folder, "*.csv"))

            for f in files:
                df = pd.read_csv(f)

                df = df.rename(columns={
                    "timestamp": COL_TIME,
                    "x": "x_m",
                    "y": "y_m"
                })

                # ID unique
                # df[COL_ID] = global_id
                # global_id += 1

                filename = os.path.splitext(os.path.basename(f))[0]
                # print(filename)
                df[COL_ID] = int(filename)

                # classe
                if t == "pedestrians":
                    df[COL_CLASS] = 1
                else:
                    df[COL_CLASS] = 2

                # info debug
                df["behavior"] = b
                df["__file__"] = os.path.basename(f)

                block_data.append(df)

        # si aucun fichier dans ce behavior
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

    # concat final
    if not all_data:
        raise ValueError("Aucune donnée VRU chargée")

    df_all = pd.concat(all_data, ignore_index=True)

    return df_all, None, cfg




def map_ind_class(label):
    mapping = {
        "pedestrian": 1,
        "bicycle": 2,
        "car": 3,
        "truck_bus": 3,
        "truck": 3,
        "bus": 6,
        "van": 3,
        "motorcycle": 7
    }
    return mapping.get(label, 0)


def load_ind_dataset(cfg, recording_id):
    folder = cfg["folder"]

    # ===== récupérer recordingId depuis filename =====
    recording_id = str(recording_id).zfill(2)
    # filename = os.path.basename(file_path) # à changer pour n'entrer que l'ID du recording à la place
    # recording_id = filename.split("_")[0]

    # ===== chemins fichiers =====
    tracks_file = os.path.join(folder, f"{recording_id}_tracks.csv")
    meta_file = os.path.join(folder, f"{recording_id}_tracksMeta.csv")

    if not os.path.exists(tracks_file) or not os.path.exists(meta_file):
        raise FileNotFoundError(f"Fichiers manquants pour recording {recording_id}")
    
    meta_rec_file = os.path.join(folder, f"{recording_id}_recordingMeta.csv")

    if not os.path.exists(meta_rec_file):
        raise FileNotFoundError(f"{meta_rec_file} introuvable")

    df_meta_rec = pd.read_csv(meta_rec_file)

    x_origin = df_meta_rec["xUtmOrigin"].iloc[0]
    y_origin = df_meta_rec["yUtmOrigin"].iloc[0]
    px_to_m = df_meta_rec["orthoPxToMeter"].iloc[0]

    # ===== LOAD =====
    df_tracks = pd.read_csv(tracks_file)
    df_meta = pd.read_csv(meta_file)

    # ===== MERGE =====
    df = pd.merge(
        df_tracks,
        df_meta[["trackId", "class"]],
        on="trackId",
        how="left"
    )

    # ===== NORMALISATION =====
    df = df.rename(columns={
        "frame": COL_TIME,
        "trackId": COL_ID
    })

    df[COL_CLASS] = df["class"].apply(map_ind_class)

    # ===== POSITION =====
    # df["x_m"] = df["xCenter"]
    # df["y_m"] = df["yCenter"]
    df["x_m"] = df["xCenter"] - x_origin
    df["y_m"] = df["yCenter"] - y_origin

    # df["__file__"] = filename
    df["__file__"] = recording_id

    # ===== IMAGE =====
    # que 2 images ont été conservées (une par intersection) pour réduire la taille des dossiers
    rid = int(recording_id)
    image_path = None
    if 0 <= rid <= 6:
        img_path = os.path.join(folder, "03_background.png")
    elif 18 <= rid <= 29:
        img_path = os.path.join(folder, "22_background.png")
   
    if image_path is not None and not os.path.exists(image_path):
        image_path = None

    return df, img_path, cfg