DATASETS = {
    "noname": { # changer le nom en TSS !
        "type": "csv",
        "folder": "trajectory_data",
        "image": "trajectory_data/background.png",
        "file_pattern": "*traj*.csv",

        "columns": {
            "time": "time_step",
            "id": "object_id",
            "class": "user_type",
            "x": "x_coordinate",
            "y": "y_coordinate"
        },

        "format": "centroid",
        "scale": "pixel",
        "pixels_per_meter": 21.185660421977854,
        "fps": 0.5,
        "has_cars": True
    },

    "ind": {
        "type": "ind",
        "folder": "InD",
        "image": None,  # géré dynamiquement lors du dataset load
        "file_pattern": "*_tracks.csv", # à intégrer plus tard pour faciliter les choses

        "columns": {
            "time": "frame",
            "id": "trackId",
            "x": "xCenter",
            "y": "yCenter"
        },

        "format": "centroid",
        "scale": "meter",
        "fps": 25,
        "has_cars": True
    },

    "ctv_area1": {
        "type": "csv",
        "folder": "CTV_Dataset_v2/transformed/area1",
        "image": "CTV_Dataset_v2/transformed/area1/bg_area1.jpg",

        "columns": {
            "time": "frame",
            "id": "id",
            "class": "class",
            "tl_x": "tl_x",
            "tl_y": "tl_y",
            "br_x": "br_x",
            "br_y": "br_y"
        },

        "format": "bbox", # conversion en coordonnées centroïde ensuite
        "scale": "meter",
        "meters_to_pixels": 24,
        "fps": 29.97,
        "recommended_speed": 30,
        "has_cars": False
    },

    "ctv_area2": {
        "type": "csv",
        "folder": "CTV_Dataset_v2/transformed/area2",
        "image": "CTV_Dataset_v2/transformed/area2/bg_area2.jpg",

        "columns": {
            "time": "frame",
            "id": "id",
            "class": "class",
            "tl_x": "tl_x",
            "tl_y": "tl_y",
            "br_x": "br_x",
            "br_y": "br_y"
        },

        "format": "bbox",
        "scale": "meter",
        "meters_to_pixels": 24,
        "fps": 29.97,
        "recommended_speed": 30,
        "has_cars": False
    },

    "stanford": {
        "type": "pkl",
        "folder": "stanford",

        "files": ["train_trajnet.pkl", "test_trajnet.pkl"],

        "image": None,

        "columns": {
            "time": "frame",
            "id": "trackId",
            "x": "x",
            "y": "y",
            "scene": "sceneId",
            "meta": "metaId"
        },

        "format": "centroid", # pas spécifié dans l'article
        "scale": "meter", # pas spécifié dans l'article
        "fps": 0.5, # pas spécifié dans l'article
        "has_cars": True
    },

    "stanford2": { # dataset correct !
        "type": "txt",
        "folder": "stanford_campus_dataset/annotations",

        "image": "reference.jpg",

        "columns": {
            "id": 0,
            "xmin": 1,
            "ymin": 2,
            "xmax": 3,
            "ymax": 4,
            "frame": 5,
            "lost": 6,
            "occluded": 7,
            "generated": 8,
            "label": 9
        },

        "format": "bbox", # pas précisé dans l'article
        "scale": "pixel", # pas précisé dans l'article mais semble cohérent d'après les valeurs
        "pixels_per_meter": 1.0, # pas précisé dans l'article
        "fps": 1, # pas précisé dans l'article
        "has_cars": True
    },

    "vru": {
        "type": "vru",
        "folder": "VRU_dataset",

        "image": None,

        "columns": {
            "time": "timestamp",
            "x": "x",
            "y": "y"
        },

        "format": "centroid",
        "scale": "meter",
        "fps": 1, # pas spécifié dans l'article
        "has_cars": False
    }
}


# Constantes et noms communs pour les colonnes des fichiers
COL_TIME = "time_step"
COL_ID = "object_id"
COL_CLASS = "user_type"

# Classes
CLASS_NAMES = {
    1: "Pedestrian",
    2: "Cyclist",
    3: "Car",
    4: "Cart",
    5: "Skater",
    6: "Bus",
    7: "Motorcycle"
}

# Couleurs des classes
CLASS_COLORS = {
    1: "tab:blue",
    2: "tab:green",
    3: "tab:red",
    4: "tab:orange",
    5: "tab:purple",
    6: "tab:brown",
    7: "tab:yellow"
}

# Classes des autres véhicules (motorisés) pas pris en compte dans le sujet du stage
VEHICLE_CLASSES = {3, 4, 5, 6, 7}

# Affichage statique
STATIC_LINEWIDTH = 1.8
STATIC_ALPHA = 0.85
ID_FONT_SIZE = 8

# Affichage animé
AGENT_MARKER_SIZE = 50
ANIM_LINEWIDTH = 1.8
ANIM_ALPHA = 0.8
DEFAULT_FPS = 25 # frame per second, dans le cas où un dataset n'indique pas son frame rate

TAIL_LENGTH = None # peut-être pas nécessaire, à voir avec les autres datasets

# Pour highlight un ID donné
HIGHLIGHT_COLOR = "red"
HIGHLIGHT_SIZE = 100