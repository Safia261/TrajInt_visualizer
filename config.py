DATASETS = {
    "tss": {
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
        "fps": 2, # because, according to the article, "Each time step lasts 0.5 seconds"
        "has_cars": True
    },

    "ind": {
        "type": "ind",
        "folder": "InD",
        "image": None,  # loaded automatically if there is an image when data is loaded
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

        "format": "bbox", # converted in centroïd afterwards
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

    "stanford": { # old version of SDD dataset
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

        "format": "centroid", # not specified in the article
        "scale": "meter", # not specified in the article
        "fps": 0.5, # not specified in the article
        "has_cars": True
    },

    "sdd": {
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

        "format": "bbox", # not specified in the article
        "scale": "pixel", # not specified in the article but seems coherent with the values
        "pixels_per_meter": 1.0, # not specified in the article
        "fps": 29.97, # not specified in the article but according to the video properties
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
        "fps": 1, # not specified in the article
        "has_cars": False
    }
}


# Constants and common column names for the dataframe files
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

# Class colors
CLASS_COLORS = {
    1: "tab:blue",
    2: "tab:green",
    3: "tab:red",
    4: "tab:orange",
    5: "tab:purple",
    6: "tab:brown",
    7: "tab:yellow"
}

# Classes of other road users not considered in the scope of this project
VEHICLE_CLASSES = {3, 4, 5, 6, 7}

# Static display
STATIC_LINEWIDTH = 1.8
STATIC_ALPHA = 0.85
ID_FONT_SIZE = 8

# Animated display
AGENT_MARKER_SIZE = 50
ANIM_LINEWIDTH = 1.8
ANIM_ALPHA = 0.8
DEFAULT_FPS = 30 # frame per second, in case a dataset doesn't have one

TAIL_LENGTH = None # maybe not necessary

# To highlight a given ID
HIGHLIGHT_COLOR = "red"
HIGHLIGHT_SIZE = 100