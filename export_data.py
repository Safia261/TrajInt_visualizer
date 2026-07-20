import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import *

###############################################
# Exportation des données filtrées
###############################################


def export_filtered_ind_recording(recording_id, bad_ids, dataset_cfg, output_suffix="filtered"):
    """
    Exporte un recording InD filtré (tracks + tracksMeta) compatible avec l'outil officiel.

    Parameters:
        recording_id (str ou int): ex "00", "01", ...
        bad_ids (set): IDs à supprimer (issus du filtre)
        dataset_cfg (dict): config du dataset (DATASETS["ind"])
        output_suffix (str): suffixe pour les fichiers exportés
    """

    folder = dataset_cfg["folder"]
    recording_id = str(recording_id).zfill(2)

    # chemins fichiers originaux
    tracks_file = os.path.join(folder, f"{recording_id}_tracks.csv")
    meta_file   = os.path.join(folder, f"{recording_id}_tracksMeta.csv")
    rec_meta_file = os.path.join(folder, f"{recording_id}_recordingMeta.csv")

    if not os.path.exists(tracks_file) or not os.path.exists(meta_file):
        raise FileNotFoundError(f"Fichiers manquants pour recording {recording_id}")

    # chargement
    tracks = pd.read_csv(tracks_file)
    meta   = pd.read_csv(meta_file)

    # filtrage des données
    tracks_filtered = tracks[~tracks["trackId"].isin(bad_ids)].copy()
    meta_filtered   = meta[~meta["trackId"].isin(bad_ids)].copy()

    # dossier de sortie
    output_dir = os.path.join(folder, "filtered_ind")
    os.makedirs(output_dir, exist_ok=True)

    # noms fichiers
    tracks_out = os.path.join(output_dir, f"{recording_id}_{output_suffix}_tracks.csv")
    meta_out   = os.path.join(output_dir, f"{recording_id}_{output_suffix}_tracksMeta.csv")

    # recordingMeta copié tel quel
    rec_meta_out = os.path.join(output_dir, f"{recording_id}_{output_suffix}_recordingMeta.csv")

    # sauvegarde
    tracks_filtered.to_csv(tracks_out, index=False)
    meta_filtered.to_csv(meta_out, index=False)

    if os.path.exists(rec_meta_file):
        pd.read_csv(rec_meta_file).to_csv(rec_meta_out, index=False)

    # stats
    initial_tracks = tracks["trackId"].nunique()
    final_tracks = tracks_filtered["trackId"].nunique()

    print("\nEXPORT InD FILTRÉ")
    print(f"Recording : {recording_id}")
    print(f"IDs supprimés : {len(bad_ids)}")
    print(f"Trajectoires : {final_tracks} / {initial_tracks}")
    print(f"Fichiers sauvegardés dans : {output_dir}")

    return tracks_out, meta_out, rec_meta_out



def export_filtered_data_original(df_filtered, dataset_type, dataset_folder, raw_csv_name, output_folder):
    """
    Filtre un CSV brut (CTV, noname, etc.) en utilisant le df_filtered comme masque.
    """

    if "ctv" in dataset_type:
        id_col_raw = "id"
        time_col_raw = "frame"

    elif dataset_type == "noname":
        id_col_raw = "object_id"
        time_col_raw = "time_step"

    else:
        raise ValueError(f"Dataset inconnu: {dataset_type}")

    # on load les vraies données
    raw_csv_path = os.path.join(dataset_folder, raw_csv_name)
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Fichier introuvable : {raw_csv_path}")
    raw_df = pd.read_csv(raw_csv_path)

    # IDs à garder
    ids_to_keep = df_filtered[COL_ID].unique()

    # filtrage principal -> on extrait les données des agents dont les trajectoires sont à garder
    df_out = raw_df[raw_df[id_col_raw].isin(ids_to_keep)]

    times_to_keep = df_filtered[COL_TIME].unique()
    df_out = df_out[df_out[time_col_raw].isin(times_to_keep)]

    # tri propre
    df_out = df_out.sort_values([id_col_raw, time_col_raw])

    # export
    base, ext = os.path.splitext(raw_csv_name)
    output_path = os.path.join(output_folder, f"{base}_filtered{ext}")

    df_out.to_csv(output_path, index=False)

    print(f"\nExport terminé : {output_path}")
    print(f"Nb agents : {df_out[id_col_raw].nunique()}")
    print(f"Nb lignes : {len(df_out)}")



def export_filtered_data(df, dataset_name, raw_file_name=None, output_folder="data_filtered", suffix="filtered", file_format="csv"):
    """
    Sauvegarde directement le DataFrame filtré normalisé.
    """

    if file_format not in ["csv", "txt"]:
        raise ValueError("file_format doit être 'csv' ou 'txt'")

    # dataset_output = os.path.join(output_folder, dataset_name)
    os.makedirs(output_folder, exist_ok=True) # créé un dossier s'il n'existe pas, et ne fait rien sinon

    # nom fichier
    if raw_file_name is not None:
        base = os.path.splitext(os.path.basename(raw_file_name))[0]
        filename = f"{base}_{suffix}.{file_format}"
    else:
        filename = f"{dataset_name}_{suffix}.{file_format}"

    # output_path = os.path.join(dataset_output, filename)
    output_path = os.path.join(output_folder, filename)

    # export CSV
    if file_format == "csv":
        df.to_csv(output_path, index=False)
    
    elif file_format == "txt":
        df.to_csv(output_path, index=False, sep="\t")

    print("\nExport DataFrame filtré terminé")
    print(f"Fichier : {output_path}")
    print(f"Nb agents : {df[COL_ID].nunique()}")
    print(f"Nb lignes : {len(df)}")

    return output_path



###############################################
# Exportation des résultats de la classification
###############################################

def export_interactions_to_csv(
    df,
    history,
    interactions,
    fps,
    output_path="interactions.csv"
):
    """
    Exporte les interactions avec :
    - infos de base (type, ids, start, end)
    - features calculées
    - classification finale

    Parameters:
        df : DataFrame trajectoires
        history : output de compute_clusters_and_hulls_over_time
        interactions : output de build_interaction_events
        fps : frames per second
        output_path : chemin du CSV
    """

    from analysis_interactions import classify_one_interaction

    rows = []

    for i, inter in enumerate(interactions):
        res = None

        try:
            # Classification de l'interaction selon features
            res = classify_one_interaction(df, history, inter, fps)

            # res est supposé contenir:
            # {
            #   "interaction": inter,
            #   "features": features,
            #   "label": label
            # }

            features = res.get("features", {})
            main_label = res["label"].get("interaction_type", "unknown")

        except Exception as e:
            print(f"Erreur interaction {i}: {e}")
            features = {}
            label = {}
            main_label = "error"
        
        if res is None:
            res = {"features": {}, "label": {}}

        label = res["label"]

        ttac_data = features.get("TTAC", None)
        if ttac_data is not None:
            _, _, ttac_min = ttac_data
        else:
            ttac_min = None

        # Infos de base interaction
        row = {
            "interaction_id": i,
            "type_ped": inter["type_ped"],
            "type_cyc": inter["type_cyc"],
            "ids_ped": [int(i) for i in inter["ids_ped"]],
            "ids_cyc": [int(i) for i in inter["ids_cyc"]],
            "start_frame": int(inter["start"]),
            "end_frame": int(inter["end"]),
            "start_time_s": inter["start"] / fps,
            "end_time_s": inter["end"] / fps,
            "duration_s": (inter["end"] - inter["start"]) / fps,
            "interaction_label": main_label,
            
            "direction": label.get("direction", {}).get("label_main"),
            "approach": label.get("approach", {}).get("label_main"),       # que pour interaction individu-individu
            "position": label.get("position", {}).get("label_main"),
            "distance": label.get("distance", {}).get("label_main"),
            "dist_mean": label.get("distance", {}).get("mean"),
            "dist_min": label.get("distance", {}).get("min"),
            "speed": label.get("speed", {}).get("label_main"),
            "speed_mean": label.get("speed", {}).get("v_mean"),

            # spécifiques groupe
            "in_hull": label.get("in_hull", None),
            "split": label.get("split", None),

            # spécifique interaction paire
            "pet": label.get("pet", None),
            "pet_val": features.get("PET", None),
            "ttac": label.get("ttac", None),
            "ttac_min": ttac_min,

            # risque
            "risk": label.get("risk", None),
            "risk_score": label.get("risk_score", None),

            # réactivité des agents (si interaction paire)
            "speed_var_ped": label.get("speed_var_ped", None),
            "speed_var_cyc": label.get("speed_var_cyc", None),
            "spatial_var_ped": label.get("spatial_var_ped", None),
            "spatial_var_cyc": label.get("spatial_var_cyc", None),
            "most_reactive_agent": label.get("most_reactive_agent", None)
        }

        rows.append(row)

    # DataFrame + export
    df_out = pd.DataFrame(rows)

    if output_path is not None:
        df_out.to_csv(output_path, index=False)
        print(f"CSV sauvegardé : {output_path}")

    return df_out



def compute_interaction_statistics(csv_path, save_xlsx):

    df = pd.read_csv(csv_path)

    df = df[(df["type_ped"] == "ped_noise") & (df["type_cyc"] == "cyc_noise")]

    # 1. % cycliste plus réactif que piéton

    # reactive_stats = (
    #     df.groupby(["type_ped", "type_cyc", "interaction_label"])
    #     .agg(n=("interaction_id", "count"), pct_cyc_more_reactive=("most_reactive_agent", lambda x: 100 * (x == "cyc").mean())).reset_index()
    # )

    print("\n Agent le plus réactif par interaction:")

    reactive_stats = (
        df.groupby("interaction_label")
        .agg(
            nb_interactions=("interaction_id", "count"),
            pct_cyc_reactive=(
                "most_reactive_agent",
                lambda x: 100 * (x == "cyc").mean()
            ),
            pct_ped_reactive=(
                "most_reactive_agent",
                lambda x: 100 * (x == "ped").mean()
            )
        )
        .reset_index()
    )

    print(reactive_stats.round(2))

    # 2. Variabilité vitesse

    # speed_ped = (
    #     df["speed_var_ped"]
    #     .value_counts(normalize=True)
    #     .mul(100)
    #     .rename("percent")
    #     .reset_index(names="speed_var_ped")
    # )

    # speed_cyc = (
    #     df["speed_var_cyc"]
    #     .value_counts(normalize=True)
    #     .mul(100)
    #     .rename("percent")
    #     .reset_index(names="speed_var_cyc")
    # )

   

    speed_ped_stats = pd.crosstab(
        df["interaction_label"],
        df["speed_var_ped"],
        normalize="index"
    ) * 100

    speed_cyc_stats = pd.crosstab(
        df["interaction_label"],
        df["speed_var_cyc"],
        normalize="index"
    ) * 100
    
    print("\n PED - Variabilité de vitesse par interaction:")
    print(speed_ped_stats.round(1))
    print("\n CYC - Variabilité de vitesse par interaction:")
    print(speed_cyc_stats.round(1))

    # 3. Variabilité spatiale

    # spatial_ped = (
    #     df["spatial_var_ped"]
    #     .value_counts(normalize=True)
    #     .mul(100)
    #     .rename("percent")
    #     .reset_index(names="spatial_var_ped")
    # )

    # spatial_cyc = (
    #     df["spatial_var_cyc"]
    #     .value_counts(normalize=True)
    #     .mul(100)
    #     .rename("percent")
    #     .reset_index(names="spatial_var_cyc")
    # )

    spatial_ped_stats = pd.crosstab(
        df["interaction_label"],
        df["spatial_var_ped"],
        normalize="index"
    ) * 100

    spatial_cyc_stats = pd.crosstab(
        df["interaction_label"],
        df["spatial_var_cyc"],
        normalize="index"
    ) * 100

    print("\n PED - Variabilité spatiale par interaction:")
    print(spatial_ped_stats.round(1))
    print("\n CYC - Variabilité spatiale par interaction:")
    print(spatial_cyc_stats.round(1))

    # 4. Combinaisons vitesse

    # speed_comb = (
    #     pd.crosstab(
    #         df["speed_var_ped"],
    #         df["speed_var_cyc"],
    #         normalize="all"
    #     ) * 100
    # )

    print("\n Combinaisons vitesses par interaction:")
    for interaction in sorted(df["interaction_label"].unique()):

        sub = df[df["interaction_label"] == interaction]

        table = pd.crosstab(
            sub["speed_var_ped"],
            sub["speed_var_cyc"],
            normalize="all"
        ) * 100

        print("\n")
        print("-"*50)
        print(interaction)
        print(table.round(1))

    # 5. Combinaisons spatiales

    # spatial_comb = (
    #     pd.crosstab(
    #         df["spatial_var_ped"],
    #         df["spatial_var_cyc"],
    #         normalize="all"
    #     ) * 100
    # )

    print("\n Combinaisons spatiales par interaction:")
    for interaction in sorted(df["interaction_label"].unique()):
        sub = df[df["interaction_label"] == interaction]

        table = pd.crosstab(
            sub["spatial_var_ped"],
            sub["spatial_var_cyc"],
            normalize="all"
        ) * 100

        print("\n")
        print("-"*50)
        print(interaction)
        print(table.round(1))
    
    if save_xlsx:
        with pd.ExcelWriter("interaction_ctv_statistics.xlsx") as writer:

            reactive_stats.to_excel(
                writer,
                sheet_name="reactive",
                index=False
            )

            speed_ped_stats.to_excel(
                writer,
                sheet_name="speed_ped"
            )

            speed_cyc_stats.to_excel(
                writer,
                sheet_name="speed_cyc"
            )

            spatial_ped_stats.to_excel(
                writer,
                sheet_name="spatial_ped"
            )

            spatial_cyc_stats.to_excel(
                writer,
                sheet_name="spatial_cyc"
            )

    return {
        "reactive": reactive_stats,
        "speed_ped": speed_ped_stats,
        "speed_cyc": speed_cyc_stats,
        "spatial_ped": spatial_ped_stats,
        "spatial_cyc": spatial_cyc_stats
    }


def compute_interaction_statistics_bis(csv_path):
    df = pd.read_csv(csv_path)

    # Filtrer uniquement ped_noise / cyc_noise
    df = df[
        (df["type_ped"] == "ped_noise") &
        (df["type_cyc"] == "cyc_noise")
    ].copy()


    def percentage_table(df, group_col, value_col):
        """
        Pour chaque type d'interaction,
        calcule les pourcentages des valeurs de value_col.
        """
        counts = (
            df.groupby(group_col)[value_col]
            .value_counts(normalize=True)
            .mul(100)
            .rename("percent")
            .reset_index()
        )

        return counts.pivot(
            index=group_col,
            columns=value_col,
            values="percent"
        ).fillna(0).round(2)


    # ==========================
    # Agent le plus réactif
    # ==========================

    reactive = percentage_table(
        df,
        "interaction_label",
        "most_reactive_agent"
    )

    # ==========================
    # Variabilité vitesse
    # ==========================

    speed_ped = percentage_table(
        df,
        "interaction_label",
        "speed_var_ped"
    )

    speed_cyc = percentage_table(
        df,
        "interaction_label",
        "speed_var_cyc"
    )

    # ==========================
    # Variabilité spatiale
    # ==========================

    spatial_ped = percentage_table(
        df,
        "interaction_label",
        "spatial_var_ped"
    )

    spatial_cyc = percentage_table(
        df,
        "interaction_label",
        "spatial_var_cyc"
    )

    # ==========================
    # Combinaisons vitesse
    # ==========================

    df["speed_combination"] = (
        df["speed_var_ped"] + " / " +
        df["speed_var_cyc"]
    )

    speed_comb = percentage_table(
        df,
        "interaction_label",
        "speed_combination"
    )

    # ==========================
    # Combinaisons spatiales
    # ==========================

    df["spatial_combination"] = (
        df["spatial_var_ped"] + " / " +
        df["spatial_var_cyc"]
    )

    spatial_comb = percentage_table(
        df,
        "interaction_label",
        "spatial_combination"
    )

    # ==========================
    # Export Excel
    # ==========================

    with pd.ExcelWriter(
        "stats_inter_ctv.xlsx",
        engine="openpyxl"
    ) as writer:

        reactive.to_excel(writer, sheet_name="reactive_agent")
        speed_ped.to_excel(writer, sheet_name="speed_ped")
        speed_cyc.to_excel(writer, sheet_name="speed_cyc")
        spatial_ped.to_excel(writer, sheet_name="spatial_ped")
        spatial_cyc.to_excel(writer, sheet_name="spatial_cyc")
        speed_comb.to_excel(writer, sheet_name="speed_combinations")
        spatial_comb.to_excel(writer, sheet_name="spatial_combinations")

    print("Excel généré.")


def compute_interaction_statistics_final(csv_path, save_xlsx=True, name_output="res"):

    df = pd.read_csv(csv_path)

    # Seulement les interactions ped_noise / cyc_noise
    df = df[
        (df["type_ped"] == "ped_noise") &
        (df["type_cyc"] == "cyc_noise")
    ].copy()

    ####################################################
    # Agent le plus réactif (selon la variation de vitesse)
    ####################################################

    print("\nAgent le plus réactif par interaction")

    reactive_stats = (
        df.groupby("interaction_label")
        .agg(
            nb_interactions=("interaction_id", "count"),
            pct_cyc_reactive=(
                "most_reactive_agent",
                lambda x: 100 * (x == "cyc").mean()
            ),
            pct_ped_reactive=(
                "most_reactive_agent",
                lambda x: 100 * (x == "ped").mean()
            )
        )
        .reset_index()
    )

    print(reactive_stats.round(2))

    ####################################################
    # Variabilité vitesse
    ####################################################

    speed_ped_stats = pd.crosstab(
        df["interaction_label"],
        df["speed_var_ped"],
        normalize="index"
    ) * 100

    speed_cyc_stats = pd.crosstab(
        df["interaction_label"],
        df["speed_var_cyc"],
        normalize="index"
    ) * 100

    print("\nPED - Variabilité vitesse")
    print(speed_ped_stats.round(1))

    print("\nCYC - Variabilité vitesse")
    print(speed_cyc_stats.round(1))


    ####################################################
    # Distance minimale
    ####################################################
    df["dist_min"] = pd.to_numeric(df["dist_min"], errors='coerce')
    dist_stats = (
        df.groupby("interaction_label")
        .agg(
            mean_dist=("dist_min","mean"),
            median_dist=("dist_min","median"),
            std_dist=("dist_min","std"),
            n=('dist_min','count')
        ).round(2))
    print("\nDistances minimales")
    print(dist_stats)

    ####################################################
    # Variabilité spatiale
    ####################################################

    spatial_ped_stats = pd.crosstab(
        df["interaction_label"],
        df["spatial_var_ped"],
        normalize="index"
    ) * 100

    spatial_cyc_stats = pd.crosstab(
        df["interaction_label"],
        df["spatial_var_cyc"],
        normalize="index"
    ) * 100

    print("\nPED - Variabilité spatiale")
    print(spatial_ped_stats.round(1))

    print("\nCYC - Variabilité spatiale")
    print(spatial_cyc_stats.round(1))

    ####################################################
    # Combinaisons vitesse (piéton/cycliste)
    ####################################################

    speed_comb_dict = {}

    print("\nCombinaisons de vitesse")

    for interaction in sorted(df["interaction_label"].unique()):

        sub = df[df["interaction_label"] == interaction].copy()

        table = (
            pd.crosstab(
                sub["speed_var_ped"],
                sub["speed_var_cyc"],
                normalize="all"
            ) * 100
        ).round(1)

        speed_comb_dict[interaction] = table

        print("\n" + "-" * 50)
        print(interaction)
        print(table)

    ####################################################
    # Combinaisons spatiales (piéton/cycliste)
    ####################################################

    spatial_comb_dict = {}

    print("\nCombinaisons spatiales")

    for interaction in sorted(df["interaction_label"].unique()):

        sub = df[df["interaction_label"] == interaction].copy()

        table = (
            pd.crosstab(
                sub["spatial_var_ped"],
                sub["spatial_var_cyc"],
                normalize="all"
            ) * 100
        ).round(1)

        spatial_comb_dict[interaction] = table

        print("\n" + "-" * 50)
        print(interaction)
        print(table)
        
    ####################################################
    # PET
    ####################################################
    df["pet_val"] = pd.to_numeric(df["pet_val"],errors="coerce")
    pet_stats=(
        df.groupby("interaction_label")
        ['pet_val']
        .describe()
    )
    print("\nAnalyse PET par classe d'intercation:")
    print(pet_stats)
    pet_label_stats = pd.crosstab(df["interaction_label"],df["pet"],normalize="index") * 100
    print("\nPET labels (%)")
    print(pet_label_stats.round(1))

    ####################################################
    # TTAC
    #################################################### 
    df["ttac_min"] = pd.to_numeric(df["ttac_min"], errors="coerce")
    ttac_stats=(
        df.groupby("interaction_label")
        ['ttac_min']
        .describe()
    )
    print("\nAnalyse TTAC par classe d'intercation:")
    print(ttac_stats)
    ttac_label_stats = pd.crosstab(df["interaction_label"],df["ttac"],normalize="index") * 100
    print("\nTTAC labels (%)")
    print(ttac_label_stats.round(1))

    ####################################################
    # Version compacte pour Excel
    ####################################################

    def percentage_table(df, group_col, value_col):

        counts = (
            df.groupby(group_col)[value_col]
            .value_counts(normalize=True)
            .mul(100)
            .rename("percent")
            .reset_index()
        )

        return (
            counts.pivot(
                index=group_col,
                columns=value_col,
                values="percent"
            )
            .fillna(0)
            .round(2)
        )

    reactive_excel = percentage_table(
        df,
        "interaction_label",
        "most_reactive_agent"
    )

    speed_comb_excel = percentage_table(
        df.assign(
            speed_combination=
            df["speed_var_ped"] + " / " +
            df["speed_var_cyc"]
        ),
        "interaction_label",
        "speed_combination"
    )

    spatial_comb_excel = percentage_table(
        df.assign(
            spatial_combination=
            df["spatial_var_ped"] + " / " +
            df["spatial_var_cyc"]
        ),
        "interaction_label",
        "spatial_combination"
    )


    ####################################################
    # Graphes
    ####################################################
    # Heatmap combinaison vitesse
    speed_heat = pd.crosstab(
        df["speed_var_ped"],
        df["speed_var_cyc"],
        normalize='all'
    )*100
    # df_speed = df[
    #     (df["speed_var_ped"] != "UNKNOWN") &
    #     (df["speed_var_cyc"] != "UNKNOWN")
    # ].copy()

    # speed_heat = pd.crosstab(
    #     df_speed["speed_var_ped"],
    #     df_speed["speed_var_cyc"],
    #     normalize='all'
    # ) * 100
    plt.figure(figsize=(6,5))
    sns.heatmap(speed_heat,annot=True,fmt=".1f",cmap='viridis')
    plt.title("Pedestrian and cyclist speed variation combinations")
    plt.show()

    # Heatmap combinaisons variations spatiales
    spatial_heat = pd.crosstab(
        df["spatial_var_ped"],
        df["spatial_var_cyc"],
        normalize='all'
    )*100
    plt.figure(figsize=(6,5))
    sns.heatmap(
        spatial_heat,
        annot=True,
        fmt=".1f",
        cmap='Greens'
    )
    plt.show()

    # Heatmap vitesse / spatial
    df['speed_pair']=(
        df.speed_var_ped
        +" / "+
        df.speed_var_cyc
    )

    df['spatial_pair']=(
        df.spatial_var_ped
        +" / "+
        df.spatial_var_cyc
    )

    speed_spatial_heat = pd.crosstab(
        df['speed_pair'],
        df['spatial_pair'],
        normalize='all'
    )*100
    plt.figure(figsize=(12,8))
    sns.heatmap(
        speed_spatial_heat,
        cmap='Reds',
        annot=True,
        fmt=".1f"
    )
    plt.show()

    # Histogrammes pour les valeurs PET et TTAC
    for inter in df.interaction_label.unique():
        sub=df[
            df.interaction_label==inter
        ]
        plt.figure()
        plt.hist(
            sub['pet_val'],
            bins=20
        )
        plt.title("Histogram PET - " + inter)
        plt.xlabel("PET")
        plt.show()

        plt.figure()
        plt.hist(
            sub['ttac_min'],
            bins=20
        )
        plt.title("Histogram TTAC - " + inter)
        plt.xlabel("TTAC")
        plt.show()

    # Graphe dist min VS déviation spatial
    mapping={
        'LINEAR':0,
        'SLIGHT_DEVIATION':1,
        'MODERATE_DEVIATION':2,
        'HIGH_DEVIATION':3
    }
    df['dev_ped']=df.spatial_var_ped.map(mapping)
    df['dev_cyc']=df.spatial_var_cyc.map(mapping)
    corr_ped=df['dist_min'].corr(df['dev_ped'])
    corr_cyc=df['dist_min'].corr(df['dev_cyc'])
    print("\nCorrelation min distance - ped spatial deviation :\n", corr_ped)
    print("\nCorrelation min distance - cyc spatial deviation :\n", corr_cyc)
    plt.scatter(df['dist_min'], df['dev_cyc'])
    plt.title("Correlation between spatial deviation and minimal inter-agent distance")
    plt.xlabel("Minimal inter-agent distance (m)")
    plt.ylabel("Spatial deviation")
    plt.show()

    ####################################################
    # Export Excel (en plus des print)
    ####################################################

    if save_xlsx:

        with pd.ExcelWriter(
            f"{name_output}.xlsx",
            engine="openpyxl"
        ) as writer:

            reactive_stats.to_excel(
                writer,
                sheet_name="reactive_summary",
                index=False
            )

            reactive_excel.to_excel(
                writer,
                sheet_name="reactive_agent"
            )

            speed_ped_stats.round(2).to_excel(
                writer,
                sheet_name="speed_ped"
            )

            speed_cyc_stats.round(2).to_excel(
                writer,
                sheet_name="speed_cyc"
            )

            spatial_ped_stats.round(2).to_excel(
                writer,
                sheet_name="spatial_ped"
            )

            spatial_cyc_stats.round(2).to_excel(
                writer,
                sheet_name="spatial_cyc"
            )

            speed_comb_excel.to_excel(
                writer,
                sheet_name="speed_combinations"
            )

            speed_heat.to_excel(
                writer,
                sheet_name="speed_heatmap"
            )

            spatial_comb_excel.to_excel(
                writer,
                sheet_name="spatial_combinations"
            )

            spatial_heat.to_excel(
                writer,
                sheet_name="spatial_heatmap"
            )

            dist_stats.to_excel(
                writer,
                sheet_name="distance_stats"
            )

            pet_stats.to_excel(
                writer,
                sheet_name='PET'
            )

            pet_label_stats.to_excel(
                writer,
                sheet_name="pet_label"
            )

            ttac_stats.to_excel(
                writer,
                sheet_name='TTAC'
            )

            ttac_label_stats.to_excel(
                writer,
                sheet_name="ttac_label"
            )

        print(f"\nExcel généré : {name_output}.xlsx")

    return {
        "reactive": reactive_stats,
        "speed_ped": speed_ped_stats,
        "speed_cyc": speed_cyc_stats,
        "spatial_ped": spatial_ped_stats,
        "spatial_cyc": spatial_cyc_stats,
        "speed_comb": speed_comb_excel,
        "spatial_comb": spatial_comb_excel,
        "distance_stats": dist_stats,
        "pet_stats": pet_stats,
        "ttac_stats": ttac_stats
    }

# Pour FlowChain

def export_dataset_by_class(df, dataset_name, raw_file_name=None, output_folder="data_filtered", suffix="filtered", file_format="txt"):
    """
    Sépare un DataFrame en plusieurs fichiers CSV selon la classe des agents (piétons et cyclsites seulement).
    """
    classes_to_save = {
        1: "Pedestrian",
        2: "Cyclist"
    }

    for class_id, class_name in classes_to_save.items():

        # filtrage selon la classe (ped ou cyc)
        df_class = df[df[COL_CLASS] == class_id]

        df_class = df_class.drop(columns=[COL_CLASS])

        # si vide on ignore
        if df_class.empty:
            continue
        
        export_filtered_data(df_class, dataset_name, raw_file_name=raw_file_name, output_folder=output_folder, suffix=suffix + "_" + class_name, file_format=file_format)



def split_txt_by_trajectory(input_file, output_folder, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Sépare un fichier txt de trajectoires pour FlowChain en :
    - train.txt (70% des trajectoires)
    - val.txt (15%)
    - test.txt (15%)

    Le split est effectué par trajectoires (i.e. par object_id).
    """

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Les ratios doivent sommer à 1.")

    os.makedirs(output_folder, exist_ok=True)

    # Lecture du fichier
    df = pd.read_csv(input_file, sep="\t")

    # IDs uniques
    traj_ids = df[COL_ID].unique()

    # Mélange aléatoire (mais peut-être pas nécessaire ?)
    # rng = np.random.default_rng(seed)
    # rng.shuffle(traj_ids)

    n = len(traj_ids)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = traj_ids[:n_train]
    val_ids = traj_ids[n_train:n_train + n_val]
    test_ids = traj_ids[n_train + n_val:]

    # Sous-ensembles
    df_train = df[df[COL_ID].isin(train_ids)]
    df_val = df[df[COL_ID].isin(val_ids)]
    df_test = df[df[COL_ID].isin(test_ids)]

    base_name = os.path.splitext(os.path.basename(input_file))[0]

    train_file = os.path.join(output_folder, f"{base_name}_train.txt")
    val_file = os.path.join(output_folder, f"{base_name}_val.txt")
    test_file = os.path.join(output_folder, f"{base_name}_test.txt")

    # Sauvegarde
    df_train.to_csv(train_file, index=False, header=False, sep="\t") # sep="\t"
    df_val.to_csv(val_file, index=False, header=False, sep="\t")
    df_test.to_csv(test_file, index=False, header=False, sep="\t")

    print(f"\nSplit terminé pour {base_name} ({n} trajectoires au total)")
    print(f"Train : {len(train_ids)} trajectoires ({train_ratio*100}%)")
    print(f"Val   : {len(val_ids)} trajectoires ({val_ratio*100}%)")
    print(f"Test  : {len(test_ids)} trajectoires ({test_ratio*100}%)")

    print(f"Train lignes : {len(df_train)}")
    print(f"Val lignes   : {len(df_val)}")
    print(f"Test lignes  : {len(df_test)}")

    return train_file, val_file, test_file