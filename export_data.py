import os
import ast
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import *

###############################################
# Export of filtered data
###############################################


def export_filtered_ind_recording(recording_id, bad_ids, dataset_cfg, output_suffix="filtered"):
    """
    Exports a filtered InD recording (tracks + tracksMeta) compatible with the official tool.

    Parameters:
        recording_id (str or int): e.g. "00", "01", ...
        bad_ids (set): IDs to remove (from the filter)
        dataset_cfg (dict): dataset configuration (DATASETS["ind"])
        output_suffix (str): suffix for the exported files
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

    print("\nExport inD filtered")
    print(f"Recording : {recording_id}")
    print(f"IDs deleted : {len(bad_ids)}")
    print(f"Trajectories : {final_tracks} / {initial_tracks}")
    print(f"Files saved in : {output_dir}")

    return tracks_out, meta_out, rec_meta_out



def export_filtered_data_original(df_filtered, dataset_type, dataset_folder, raw_csv_name, output_folder):
    """
    Filters a raw CSV (CTV, TSS, etc.) using df_filtered as a mask.
    """

    if "ctv" in dataset_type:
        id_col_raw = "id"
        time_col_raw = "frame"

    elif dataset_type == "tss":
        id_col_raw = "object_id"
        time_col_raw = "time_step"

    else:
        raise ValueError(f"Dataset inconnu: {dataset_type}")

    # load the original data
    raw_csv_path = os.path.join(dataset_folder, raw_csv_name)
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Fichier introuvable : {raw_csv_path}")
    raw_df = pd.read_csv(raw_csv_path)

    # IDs to keep
    ids_to_keep = df_filtered[COL_ID].unique()

    # Main filtering -> extracts the data of agents whose trajectories are to be retained
    df_out = raw_df[raw_df[id_col_raw].isin(ids_to_keep)]

    times_to_keep = df_filtered[COL_TIME].unique()
    df_out = df_out[df_out[time_col_raw].isin(times_to_keep)]

    # clean sorting
    df_out = df_out.sort_values([id_col_raw, time_col_raw])

    # export
    base, ext = os.path.splitext(raw_csv_name)
    output_path = os.path.join(output_folder, f"{base}_filtered{ext}")

    df_out.to_csv(output_path, index=False)

    print(f"\nExport done : {output_path}")
    print(f"Nb agents : {df_out[id_col_raw].nunique()}")
    print(f"Nb lines : {len(df_out)}")



def export_filtered_data(df, dataset_name, raw_file_name=None, output_folder="data_filtered", suffix="filtered", file_format="csv"):
    """
    Saves directly the normalized filtered DataFrame.
    """

    if file_format not in ["csv", "txt"]:
        raise ValueError("file_format doit être 'csv' ou 'txt'")

    os.makedirs(output_folder, exist_ok=True) # creates a folder if it does not exist, otherwise does nothing
    # name file
    if raw_file_name is not None:
        base = os.path.splitext(os.path.basename(raw_file_name))[0]
        filename = f"{base}_{suffix}.{file_format}"
    else:
        filename = f"{dataset_name}_{suffix}.{file_format}"

    output_path = os.path.join(output_folder, filename)

    # export CSV
    if file_format == "csv":
        df.to_csv(output_path, index=False)
    
    elif file_format == "txt":
        df.to_csv(output_path, index=False, sep="\t")

    print("\nExport of the filtered DataFrame done")
    print(f"File : {output_path}")
    print(f"Nb agents : {df[COL_ID].nunique()}")
    print(f"Nb lines : {len(df)}")

    return output_path



###############################################
# Export of classification results
###############################################

def export_interactions_to_csv(df, history, interactions, fps, output_path="interactions.csv"):
    """
    Exports interactions with:
    - basic information (type, IDs, start, end)
    - computed features
    - final classification

    Parameters:
        df : trajectory DataFrame
        history : output of compute_clusters_and_hulls_over_time
        interactions : output of build_interaction_events
        fps : frames per second
        output_path : path to the CSV file
    """

    from analysis_interactions import classify_one_interaction

    rows = []

    for i, inter in enumerate(interactions):
        res = None

        try:
            # interaction classification based on features
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
            "approach": label.get("approach", {}).get("label_main"),       # only for binary interactions
            "position": label.get("position", {}).get("label_main"),
            "distance": label.get("distance", {}).get("label_main"),
            "dist_mean": label.get("distance", {}).get("mean"),
            "dist_min": label.get("distance", {}).get("min"),
            "speed": label.get("speed", {}).get("label_main"),
            "speed_mean": label.get("speed", {}).get("v_mean"),

            # specific to interactions with groups
            "in_hull": label.get("in_hull", None),
            "split": label.get("split", None),
            "density_ped_evolution": label.get("density_ped_res", {}).get("label"),
            "mean_density_ped": label.get("density_ped_res", {}).get("mean"),
            "std_density_ped": label.get("density_ped_res", {}).get("std"),
            "min_density_ped": label.get("density_ped_res", {}).get("min"),
            "max_density_ped": label.get("density_ped_res", {}).get("max"),
            "trend_density_ped": label.get("density_ped_res", {}).get("trend"),
            "density_cyc_evolution": label.get("density_cyc_res", {}).get("label"),
            "mean_density_cyc": label.get("density_cyc_res", {}).get("mean"),
            "std_density_cyc": label.get("density_cyc_res", {}).get("std"),
            "min_density_cyc": label.get("density_cyc_res", {}).get("min"),
            "max_density_cyc": label.get("density_cyc_res", {}).get("max"),
            "trend_density_cyc": label.get("density_cyc_res", {}).get("trend"),

            # specific to binary interaction
            "pet": label.get("pet", None),
            "pet_val": features.get("PET", None),
            "ttac": label.get("ttac", None),
            "ttac_min": ttac_min,

            # risk
            "risk": label.get("risk", None),
            "risk_score": label.get("risk_score", None),

            # agent reactivity (for pairwise interactions)
            "speed_var_ped": label.get("speed_var_ped", None),
            "speed_var_cyc": label.get("speed_var_cyc", None),
            "spatial_var_ped": label.get("spatial_var_ped", None),
            "spatial_var_cyc": label.get("spatial_var_cyc", None),
            "most_reactive_agent": label.get("most_reactive_agent", None)
        }

        rows.append(row)

    df_out = pd.DataFrame(rows)

    if output_path is not None:
        df_out.to_csv(output_path, index=False)
        print(f"Interactions exported to CSV file : {output_path}")

    return df_out



def compute_binary_interaction_statistics(csv_path, save_xlsx=True, name_output="res"):

    df = pd.read_csv(csv_path)

    # only for interactions ped_noise / cyc_noise
    df = df[
        (df["type_ped"] == "ped_noise") &
        (df["type_cyc"] == "cyc_noise")
    ].copy()

    #######
    # Most reactive agent (based on spatial deviation)
    print("\nThe agent the most reactive per interaction")

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

    #######
    # Speed variation
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

    print("\nPED - Speed variation")
    print(speed_ped_stats.round(1))

    print("\nCYC - Speed variation")
    print(speed_cyc_stats.round(1))


    #######
    # Min dist
    df["dist_min"] = pd.to_numeric(df["dist_min"], errors='coerce')
    dist_stats = (
        df.groupby("interaction_label")
        .agg(
            mean_dist=("dist_min","mean"),
            median_dist=("dist_min","median"),
            std_dist=("dist_min","std"),
            n=('dist_min','count')
        ).round(2))
    print("\nMinimal distances")
    print(dist_stats)

    #######
    # Spatial deviation
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

    print("\nPED - Spatial deviation")
    print(spatial_ped_stats.round(1))

    print("\nCYC - Spatial deviation")
    print(spatial_cyc_stats.round(1))

    #######
    # Speed combinations (pedestrian/cyclist)
    speed_comb_dict = {}

    print("\nSpeed combinations")

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

        print(interaction)
        print(table)

    #######
    # Spatial combinations (pedestrian/cyclist)
    spatial_comb_dict = {}

    print("\nSpatial combinations")

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

        print(interaction)
        print(table)


    #######
    # Relative positions by interaction type
    relative_position_stats = pd.crosstab(
        df["interaction_label"],
        df["position"],
        normalize="index"
    ) * 100

    # ensure that all four positions are represented
    for position in ["FRONT", "BEHIND", "LEFT", "RIGHT"]:
        if position not in relative_position_stats.columns:
            relative_position_stats[position] = 0.0

    relative_position_stats = (
        relative_position_stats[
            ["FRONT", "BEHIND", "LEFT", "RIGHT"]
        ]
        .round(1)
    )

    print("\nDistribution of the relative positions per type of interaction (%)")
    print(relative_position_stats)

    #######
    # Mean relative speed
    df["speed_mean"] = pd.to_numeric(
        df["speed_mean"],
        errors="coerce"
    )

    relative_speed_stats = (
        df.groupby("interaction_label")
        .agg(
            mean_relative_speed=("speed_mean", "mean"),
            median_relative_speed=("speed_mean", "median"),
            std_relative_speed=("speed_mean", "std"),
            n=("speed_mean", "count")
        )
        .round(2)
    )

    print("\nMean relative speed per type of interaction")
    print(relative_speed_stats)
        
    #######
    # PET
    df["pet_val"] = pd.to_numeric(df["pet_val"],errors="coerce")
    pet_stats=(
        df.groupby("interaction_label")
        ['pet_val']
        .describe()
    )
    print("\nAnalysis PET per type of intercation:")
    print(pet_stats)
    pet_label_stats = pd.crosstab(df["interaction_label"],df["pet"],normalize="index") * 100
    print("\nPET labels (%)")
    print(pet_label_stats.round(1))

    #######
    # mean PET moyen per interaction type
    pet_mean_stats = (
        df.groupby("interaction_label")
        .agg(
            mean_pet=("pet_val", "mean"),
            median_pet=("pet_val", "median"),
            std_pet=("pet_val", "std"),
            n=("pet_val", "count")
        )
        .round(2)
    )

    print("\nMean PET per type of interaction")
    print(pet_mean_stats)

    #######
    # TTAC
    df["ttac_min"] = pd.to_numeric(df["ttac_min"], errors="coerce")
    ttac_stats=(
        df.groupby("interaction_label")
        ['ttac_min']
        .describe()
    )
    print("\nAnalysis of TTAC per type of intercation:")
    print(ttac_stats)
    ttac_label_stats = pd.crosstab(df["interaction_label"],df["ttac"],normalize="index") * 100
    print("\nTTAC labels (%)")
    print(ttac_label_stats.round(1))

    #######
    # mean min TTAC by intreaction type
    ttac_mean_stats = (
        df.groupby("interaction_label")
        .agg(
            mean_ttac=("ttac_min", "mean"),
            median_ttac=("ttac_min", "median"),
            std_ttac=("ttac_min", "std"),
            n=("ttac_min", "count")
        )
        .round(2)
    )

    print("\nMean min TTAC per type of interaction")
    print(ttac_mean_stats)

    #######
    # Compact version for Excel
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


    #######
    # Graphs
    # Heatmap combinaison vitesse
    speed_heat = pd.crosstab(
        df["speed_var_ped"],
        df["speed_var_cyc"],
        normalize='all'
    )*100
    plt.figure(figsize=(14,10))
    sns.heatmap(speed_heat,annot=True,fmt=".1f",cmap='viridis', annot_kws={"size": 16})
    plt.title("Pedestrian and cyclist speed variation combinations", fontsize=22)
    plt.xlabel("Pedestrian speed variation", fontsize=18)
    plt.ylabel("Cyclist speed variation", fontsize=18)
    plt.xticks(rotation=45, ha="right", fontsize=16)
    plt.yticks(rotation=45, ha="right", fontsize=16)
    plt.tight_layout()
    plt.savefig("../results_analyse_classification/sdd/binary_interactions/speed_variation_heatmap.png", 
                dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # Heatmap combinaisons variations spatiales
    spatial_heat = pd.crosstab(
        df["spatial_var_ped"],
        df["spatial_var_cyc"],
        normalize='all'
    )*100
    plt.figure(figsize=(14,10))
    sns.heatmap(
        spatial_heat,
        annot=True,
        fmt=".1f",
        cmap='Greens',
        annot_kws={"size": 16} 
    )
    plt.title("Pedestrian and cyclist spatial deviation combinations", fontsize=22)
    plt.xlabel("Pedestrian spatial variation", fontsize=18)
    plt.ylabel("Cyclist spatial variation", fontsize=18)
    plt.xticks(rotation=45, ha="right", fontsize=16)
    plt.yticks(rotation=45, ha="right", fontsize=16)
    plt.tight_layout()
    plt.savefig("../results_analyse_classification/sdd/binary_interactions/spatial_variation_heatmap.png", 
                dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

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
    plt.figure(figsize=(14,10))
    sns.heatmap(
        speed_spatial_heat,
        cmap='Reds',
        annot=False,
        fmt=".1f"
    )
    plt.title("Pedestrian and cyclist speed-spatial variation combinations", fontsize=22)
    plt.xlabel("Spatial variation combination (ped / cyc)", fontsize=18)
    plt.ylabel("Speed variation combination (ped / cyc)", fontsize=18)
    plt.xticks(rotation=45, ha="right", fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig("../results_analyse_classification/sdd/binary_interactions/speed_spatial_variation_heatmap.png", 
                dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # Histogrammes pour les valeurs PET et TTAC
    # for inter in df.interaction_label.unique():
    #     sub=df[df.interaction_label==inter]
    #     pet_values = sub["pet_val"].to_numpy()
    #     pet_values = pet_values[np.isfinite(pet_values)]
    #     if len(pet_values) > 0:
    #         plt.figure()
    #         plt.hist(sub['pet_val'], bins=20)
    #         plt.title("Histogram PET - " + inter)
    #         plt.xlabel("PET")
    #         plt.show()
    #     else:
    #         print(f"No PET values for {inter}")

    #     ttac_values = sub["ttac_min"].to_numpy()
    #     ttac_values = ttac_values[np.isfinite(ttac_values)]
    #     if len(ttac_values) > 0:
    #         plt.figure()
    #         plt.hist(sub['ttac_min'], bins=20)
    #         plt.title("Histogram TTAC - " + inter)
    #         plt.xlabel("TTAC")
    #         plt.show()
    #     else:
    #         print(f"No TTAC values for {inter}")

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

    # Pie charts
    def plot_pie(counts, title, save_dir="../results_analyse_classification/sdd/binary_interactions"):

        counts = counts[counts > 0]

        if counts.empty:
            return

        fig, ax = plt.subplots(figsize=(9, 7))

        wedges, _, autotexts = ax.pie(
            counts.values,
            labels=None,
            autopct=lambda pct: f"{pct:.1f}%" if pct > 5 else "",
            startangle=90,
            pctdistance=0.7
        )

        for autotext in autotexts:
            autotext.set_fontsize(16)

        ax.set_title(title, fontsize=22, pad=15)
        ax.legend(
            wedges,
            counts.index,
            title="Categories",
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize=15,
            title_fontsize=16
        )

        plt.tight_layout()

        filename = title.lower().replace(" ", "_") + ".png"
        filepath = os.path.join(save_dir, filename)

        plt.savefig(filepath, dpi=300, bbox_inches="tight")

        plt.show()
        plt.close()


    for interaction in sorted(df["interaction_label"].dropna().unique()):
        # relative pos
        position_counts = (
            df[
                df["interaction_label"] == interaction
            ]["position"]
            .value_counts()
            .reindex(
                [
                    "FRONT",
                    "BEHIND",
                    "LEFT",
                    "RIGHT"
                ],
                fill_value=0
            )
        )

        plot_pie(position_counts, f"Relative position distribution - {interaction}")

       # PET
        pet_counts = (
            df[
                df["interaction_label"] == interaction
            ]["pet"]
            .value_counts()
            .dropna()
        )

        plot_pie(pet_counts, f"PET distribution - {interaction}")

        # TTAC
        ttac_counts = (
            df[
                df["interaction_label"] == interaction
            ]["ttac"]
            .value_counts()
            .dropna()
        )

        plot_pie(ttac_counts, f"TTAC distribution - {interaction}")

        # Speed variation combinations
        if interaction in speed_comb_excel.index:
            speed_comb_counts = (
                speed_comb_excel
                .loc[interaction]
                .sort_values(ascending=False)
            )
        else:
            speed_comb_counts = pd.Series(dtype=float)

        plot_pie(speed_comb_counts, f"Speed variation combinations - {interaction}")

        # Spatial variation combinations
        if interaction in spatial_comb_excel.index:
            spatial_comb_counts = (
                spatial_comb_excel
                .loc[interaction]
                .sort_values(ascending=False)
            )
        else:
            spatial_comb_counts = pd.Series(dtype=float)

        plot_pie(spatial_comb_counts, f"Spatial deviation combinations - {interaction}")

    # Pie chart - Most reactive agent
    reactive_global = (
        df["most_reactive_agent"]
        .value_counts(normalize=True)
        .mul(100)
    )

    reactive_global = reactive_global.reindex(["ped", "cyc"], fill_value=0)

    plt.figure(figsize=(7, 7))

    plt.pie(
        reactive_global.values,
        labels=[
            "Pedestrian reactive",
            "Cyclist reactive"
        ],
        autopct="%.1f%%",
        startangle=90,
        textprops={"fontsize": 16},
        pctdistance=0.7
    )

    plt.title("Global distribution of the most reactive agent", fontsize=22)
    plt.tight_layout()
    plt.show()

    # excel export
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

            reactive_global.to_frame(
                name="percentage"
            ).to_excel(
                writer,
                sheet_name="reactive_global"
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

            relative_position_stats.to_excel(
                writer,
                sheet_name="relative_position"
            )

            relative_speed_stats.to_excel(
                writer,
                sheet_name="relative_speed"
            )

            pet_stats.to_excel(
                writer,
                sheet_name='PET'
            )

            pet_label_stats.to_excel(
                writer,
                sheet_name="pet_label"
            )
            
            pet_mean_stats.to_excel(
                writer,
                sheet_name="PET_mean"
            )

            ttac_stats.to_excel(
                writer,
                sheet_name='TTAC'
            )

            ttac_label_stats.to_excel(
                writer,
                sheet_name="ttac_label"
            )

            ttac_mean_stats.to_excel(
                writer,
                sheet_name="TTAC_min_mean"
            )

        print(f"\nExcel : {name_output}.xlsx")

    return {
        "reactive": reactive_stats,
        "speed_ped": speed_ped_stats,
        "speed_cyc": speed_cyc_stats,
        "spatial_ped": spatial_ped_stats,
        "spatial_cyc": spatial_cyc_stats,
        "speed_comb": speed_comb_excel,
        "spatial_comb": spatial_comb_excel,
        "distance_stats": dist_stats,
        "relative_position": relative_position_stats,
        "relative_speed": relative_speed_stats,
        "pet_stats": pet_stats,
        "pet_mean": pet_mean_stats,
        "ttac_stats": ttac_stats,
        "ttac_mean": ttac_mean_stats
    }


def compute_group_interaction_statistics(csv_path, save_xlsx=True, name_output="group_interaction_results"):
    df = pd.read_csv(csv_path)

    df = df[
        ((df["type_ped"] == "ped_cluster") &
            (df["type_cyc"] == "cyc_noise")) |
        ((df["type_ped"] == "ped_cluster") &
            (df["type_cyc"] == "cyc_cluster")) |
        ((df["type_ped"] == "ped_noise") &
            (df["type_cyc"] == "cyc_cluster"))
    ].copy()

    print("Analysis of interactions with group(s)")
    print("\n Total nb of interactions :", len(df))
    print("\n Type distribution :")
    print(df.groupby(["type_ped", "type_cyc"])["interaction_id"].count())

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

    results = {}
    interaction_types = [
        ("ped_cluster", "cyc_noise"),
        ("ped_cluster", "cyc_cluster"),
        ("ped_noise", "cyc_cluster")
    ]

    for ped_type, cyc_type in interaction_types:

        interaction_name = f"{ped_type}_{cyc_type}"

        sub = df[
            (df["type_ped"] == ped_type) &
            (df["type_cyc"] == cyc_type)
        ].copy()

        print(interaction_name)

        print("\nNumber of interactions :", len(sub))

        density_ped_evolution = percentage_table(sub, "interaction_label", "density_ped_evolution")
        density_cyc_evolution = percentage_table(sub, "interaction_label", "density_cyc_evolution")

        for category in ["STABLE", "DENSER", "DISPERSED"]:
            if category not in density_ped_evolution.columns:
                density_ped_evolution[category] = 0.0

            if category not in density_cyc_evolution.columns:
                density_cyc_evolution[category] = 0.0

        density_ped_evolution = density_ped_evolution[["STABLE", "DENSER", "DISPERSED"]]

        density_cyc_evolution = density_cyc_evolution[["STABLE", "DENSER", "DISPERSED"]]

        print("\nEvolution density Pedestrians (%)")
        print(density_ped_evolution)

        print("\nEvolution density Cyclists (%)")
        print(density_cyc_evolution)

        sub["mean_density_ped"] = pd.to_numeric(sub["mean_density_ped"], errors="coerce")
        sub["mean_density_cyc"] = pd.to_numeric(sub["mean_density_cyc"], errors="coerce")

        density_mean_ped = (
            sub.groupby("interaction_label")
            .agg(
                mean_density_ped=(
                    "mean_density_ped",
                    "mean"
                ),
                median_density_ped=(
                    "mean_density_ped",
                    "median"
                ),
                std_density_ped=(
                    "mean_density_ped",
                    "std"
                ),
                n=("mean_density_ped",
                    "count"
                )
            )
            .round(2)
        )

        density_mean_cyc = (
            sub.groupby("interaction_label")
            .agg(
                mean_density_cyc=(
                    "mean_density_cyc",
                    "mean"
                ),
                median_density_cyc=(
                    "mean_density_cyc",
                    "median"
                ),
                std_density_cyc=(
                    "mean_density_cyc",
                    "std"
                ),
                n=(
                    "mean_density_cyc",
                    "count"
                )
            )
            .round(2)
        )

        print("\nMean density Pedestrians")
        print(density_mean_ped)

        print("\nMean density Cyclistes")
        print(density_mean_cyc)

        sub["speed_mean"] = pd.to_numeric(sub["speed_mean"], errors="coerce")

        relative_speed_stats = (
            sub.groupby("interaction_label")
            .agg(
                mean_relative_speed=(
                    "speed_mean",
                    "mean"
                ),
                median_relative_speed=(
                    "speed_mean",
                    "median"
                ),
                std_relative_speed=(
                    "speed_mean",
                    "std"
                ),
                n=("speed_mean",
                    "count"
                )
            )
            .round(2)
        )

        print("\nRelative speed")
        print(relative_speed_stats)

        sub["dist_min"] = pd.to_numeric( sub["dist_min"], errors="coerce")

        distance_stats = (
            sub.groupby("interaction_label")
            .agg(
                mean_dist=(
                    "dist_min",
                    "mean"
                ),
                median_dist=(
                    "dist_min",
                    "median"
                ),
                std_dist=(
                    "dist_min",
                    "std"
                ),
                n=(
                    "dist_min",
                    "count"
                )
            )
            .round(2)
        )

        print("\nMin distance")
        print(distance_stats)

        relative_position_stats = percentage_table(sub, "interaction_label", "position")

        for position in ["FRONT", "BEHIND", "LEFT", "RIGHT"]:
            if position not in relative_position_stats.columns:
                relative_position_stats[position] = 0.0

        relative_position_stats = relative_position_stats[["FRONT", "BEHIND", "LEFT", "RIGHT"]]

        print("\nrelative positions (%)")
        print(relative_position_stats)


        # Pie charts
        def plot_pie(counts, title, save_dir="../results_analyse_classification/sdd/group_interactions"):
            counts = counts.dropna()
            counts = counts[counts > 0]

            if counts.empty or counts.sum() == 0:
                print(f"No valid data for pie chart: {title}")
                return
    
            fig, ax = plt.subplots(figsize=(9, 7))
            wedges, _, autotexts = ax.pie(
                counts.values,
                labels=None,
                autopct=lambda pct: f"{pct:.1f}%" if pct > 5 else "",
                startangle=90,
                pctdistance=0.7
            )
            for autotext in autotexts:
                autotext.set_fontsize(16)
    
            ax.set_title(title, fontsize=22, pad=15)
            ax.legend(
                wedges,
                counts.index,
                title="Categories",
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                fontsize=15,
                title_fontsize=16
            )
            plt.tight_layout()
            filename = title.lower().replace(" ", "_") + ".png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.show()
            plt.close()


        position_counts = (
            sub["position"]
            .value_counts()
            .reindex(["FRONT",  "BEHIND", "LEFT", "RIGHT"], fill_value=0)
        )

        plot_pie(position_counts, f"Relative position distribution - {interaction_name}")

        density_ped_counts = (
            sub["density_ped_evolution"]
            .value_counts()
            .reindex(["STABLE", "DENSER", "DISPERSED"], fill_value=0)
        )

        plot_pie(density_ped_counts, f"Pedestrian group density evolution - {interaction_name}")

        density_cyc_counts = (
            sub["density_cyc_evolution"]
            .value_counts()
            .reindex(["STABLE", "DENSER", "DISPERSED"], fill_value=0)
        )

        plot_pie(density_cyc_counts, f"Cyclist group density evolution - {interaction_name}")

        results[interaction_name] = {
            "density_ped_evolution":
                density_ped_evolution,

            "density_cyc_evolution":
                density_cyc_evolution,

            "density_mean_ped":
                density_mean_ped,

            "density_mean_cyc":
                density_mean_cyc,

            "relative_speed":
                relative_speed_stats,

            "distance":
                distance_stats,

            "relative_position":
                relative_position_stats
        }

    
    # Here, we intentionally group together:
    # - ped_cluster / cyc_noise
    # - ped_noise / cyc_cluster
    # - ped_cluster / cyc_cluster
    #
    # Distributions are therefore computed only
    # by interaction type (interaction_label).

    # def plot_density_pie(counts, title):
    #     counts = counts[counts > 0]

    #     if counts.empty:
    #         return

    #     plt.figure(figsize=(7, 7))
    #     plt.pie(counts.values, labels=counts.index, autopct="%.1f%%", startangle=90)
    #     plt.title(title)
    #     plt.tight_layout()
    #     plt.show()

    group_df = df.copy()

    for interaction in sorted(group_df["interaction_label"].dropna().unique()):
        sub = group_df[group_df["interaction_label"] == interaction].copy()

        density_ped_counts = (
            sub["density_ped_evolution"]
            .value_counts()
            .reindex(["STABLE", "DENSER", "DISPERSED"], fill_value=0)
        )

        plot_pie(
            density_ped_counts,
            f"Pedestrian group density evolution - {interaction}"
        )

        density_cyc_counts = (
            sub["density_cyc_evolution"]
            .value_counts()
            .reindex(["STABLE", "DENSER", "DISPERSED"], fill_value=0)
        )

        plot_pie(density_cyc_counts, f"Cyclist group density evolution - {interaction}")

        position_counts = (
            sub["position"]
            .value_counts()
            .reindex(["FRONT", "BEHIND", "LEFT", "RIGHT"], fill_value=0)
        )

        plot_pie(
            position_counts,
            f"Relative position distribution - {interaction}"
        )

    interaction_counts = (
        df.groupby(
            ["type_ped", "type_cyc", "interaction_label"]
        )["interaction_id"]
        .size()
        .reset_index(name="n_interactions")
    )

    interaction_counts["interaction_configuration"] = (interaction_counts["type_ped"] + "_" + interaction_counts["type_cyc"])

    interaction_counts = interaction_counts[
        [
            "interaction_configuration",
            "interaction_label",
            "n_interactions"
        ]
    ].sort_values(
        ["interaction_configuration",
        "interaction_label"]
    )

    print("Number of interactions per type")
    print(interaction_counts)


    print("Relative speed and min distance")
    print("All group configurations combined")

    df["speed_mean"] = pd.to_numeric(df["speed_mean"], errors="coerce")

    df["dist_min"] = pd.to_numeric(df["dist_min"], errors="coerce")

    group_global_stats = (
        df.groupby("interaction_label")
        .agg(
            mean_relative_speed=(
                "speed_mean",
                "mean"
            ),
            median_relative_speed=(
                "speed_mean",
                "median"
            ),
            std_relative_speed=(
                "speed_mean",
                "std"
            ),
            n_speed=(
                "speed_mean",
                "count"
            ),
            mean_dist_min=(
                "dist_min",
                "mean"
            ),
            median_dist_min=(
                "dist_min",
                "median"
            ),
            std_dist_min=(
                "dist_min",
                "std"
            ),
            n_dist=(
                "dist_min",
                "count"
            )
        )
        .round(2)
    )

    print("\nMean relative speed")
    print(
        group_global_stats[
            ["mean_relative_speed",
            "median_relative_speed",
            "std_relative_speed",
            "n_speed"]
        ]
    )

    print("\nMin distance")
    print(
        group_global_stats[
            ["mean_dist_min",
            "median_dist_min",
            "std_dist_min",
            "n_dist"]
        ]
    )

    # Export excel
    if save_xlsx:
        with pd.ExcelWriter(f"{name_output}.xlsx", engine="openpyxl") as writer:

            interaction_counts.to_excel(writer, sheet_name="interaction_counts", index=False)

            group_global_stats.to_excel(writer, sheet_name="global_speed_distance")

            for interaction_name, data in results.items():
                if interaction_name == "ped_cluster_cyc_noise":
                    prefix = "pc_cn"
                elif interaction_name == "ped_cluster_cyc_cluster":
                    prefix = "pc_cc"
                elif interaction_name == "ped_noise_cyc_cluster":
                    prefix = "pn_cc"
                else:
                    prefix = "other"

                data["density_ped_evolution"].to_excel(
                    writer,
                    sheet_name=f"{prefix}_dens_ped"
                )

                data["density_cyc_evolution"].to_excel(
                    writer,
                    sheet_name=f"{prefix}_dens_cyc"
                )

                data["density_mean_ped"].to_excel(
                    writer,
                    sheet_name=f"{prefix}_mean_dens_ped"
                )

                data["density_mean_cyc"].to_excel(
                    writer,
                    sheet_name=f"{prefix}_mean_dens_cyc"
                )

                data["relative_speed"].to_excel(
                    writer,
                    sheet_name=f"{prefix}_rel_speed"
                )

                data["distance"].to_excel(
                    writer,
                    sheet_name=f"{prefix}_distance"
                )

                data["relative_position"].to_excel(
                    writer,
                    sheet_name=f"{prefix}_position"
                )

        print(
            f"\nExcel : {name_output}.xlsx"
        )

    return results



def plot_binary_distance_approach_evolution(interaction_csv, original_dfs, fps, n_points=100):
    """ Computes and plots the average evolution of inter-agent distance and approach angle for each binary interaction type.
    Time is normalized from 0 to 100% of the interaction.
    For each interaction type:
        - mean distance
        - distance STD
        - mean angle
        - angle STD
        - minimum of the mean distance curve
        - temporal position of this minimum

    Four plots are generated per interaction type:
        1. Distance + angle with STD
        2. Distance + angle without STD
        3. Distance only with STD
        4. Angle only with STD

    Distance and angle are computed over the interaction period (start_frame -> end_frame).
    The minimum distance is computed from the MEAN DISTANCE CURVE, not from the individual minimum distances.
    """

    interactions_df = pd.read_csv(interaction_csv)
    interactions_df = interactions_df[
        (interactions_df["type_ped"] == "ped_noise") &
        (interactions_df["type_cyc"] == "cyc_noise")
    ].copy()

    print("\nNumber of binary interactions analyzed :", len(interactions_df))

    def extract_single_id(value):
        if pd.isna(value):
            return None
        if isinstance(value, list):
            if len(value) == 0:
                return None
            return int(value[0])

        try:
            parsed = ast.literal_eval(str(value))
            if isinstance(parsed, list):
                if len(parsed) == 0:
                    return None
                return int(parsed[0])
            return int(parsed)

        except Exception:
            return None

    normalized_time = np.linspace(0, 100, n_points)

    from utils import (compute_approach_angle, compute_distance_ped_cyc)
    curves = {}

    for _, interaction in interactions_df.iterrows():
        interaction_id = interaction["interaction_id"]
        interaction_label = (interaction["interaction_label"])
        recording = str(interaction["recording"])
        if recording not in original_dfs:
            print(f"Interaction {interaction_id} ignored : recording '{recording}' not found.")
            continue
        df_original = original_dfs[recording]
        ped_id = extract_single_id(interaction["ids_ped"])
        cyc_id = extract_single_id(interaction["ids_cyc"])
        if ped_id is None or cyc_id is None:
            print(f"Interaction {interaction_id} ignored : IDs not valid.")
            continue

        start_frame = interaction["start_frame"]
        end_frame = interaction["end_frame"]

        if (pd.isna(start_frame) or pd.isna(end_frame)):
            print(f"Interaction {interaction_id} ignorée : intervalle invalide.")
            continue

        start_frame = int(start_frame)
        end_frame = int(end_frame)

        if end_frame <= start_frame:
            continue

        times_angle, angles = compute_approach_angle(
            df_original,
            ped_id,
            cyc_id,
            fps,
            start=start_frame,
            end=end_frame,
            angle_unit="deg",
            plot=False,
            return_class=False
        )

   
        (times_distance, distances,  dist_min) = compute_distance_ped_cyc(df_original, ped_id, cyc_id, fps, plot=False, return_class=False)

        if (times_angle is None or angles is None or len(times_angle) == 0):
            continue

        if (times_distance is None or distances is None or len(times_distance) == 0):
            continue

        angle_mask = ((times_angle >= start_frame) & (times_angle <= end_frame))

        distance_mask = ((times_distance >= start_frame) & (times_distance <= end_frame))

        times_angle = times_angle[ angle_mask]

        angles = angles[angle_mask]

        times_distance = times_distance[distance_mask]

        distances = distances[distance_mask]

        if (len(times_angle) < 2 or len(times_distance) < 2):
            continue

        duration = (end_frame - start_frame)

        time_angle_norm = ((times_angle - start_frame) / duration* 100)

        time_distance_norm = ((times_distance - start_frame) / duration * 100)

        valid_angle = (np.isfinite(time_angle_norm) & np.isfinite(angles))

        valid_distance = (np.isfinite(time_distance_norm) & np.isfinite(distances))

        time_angle_norm = (time_angle_norm[valid_angle])

        angles = (angles[valid_angle])

        time_distance_norm = (time_distance_norm[valid_distance])

        distances = (distances[valid_distance])

        if (len(time_angle_norm) < 2 or len(time_distance_norm) < 2):
            continue

        angle_order = np.argsort(time_angle_norm)

        time_angle_norm = (time_angle_norm[angle_order])

        angles = (angles[angle_order])

        distance_order = np.argsort(time_distance_norm)

        time_distance_norm = (time_distance_norm[distance_order])

        distances = (distances[distance_order])

        time_angle_norm, unique_angle_idx = (np.unique(time_angle_norm, return_index=True))

        angles = (angles[unique_angle_idx])

        time_distance_norm, unique_distance_idx = (np.unique(time_distance_norm, return_index=True))

        distances = (distances[unique_distance_idx])

        angle_interp = np.interp(normalized_time, time_angle_norm, angles, left=np.nan, right=np.nan)

        distance_interp = np.interp(normalized_time, time_distance_norm, distances, left=np.nan, right=np.nan)

        if interaction_label not in curves:
            curves[interaction_label] = {"angles": [], "distances": []}

        curves[interaction_label]["angles"].append(angle_interp)
        curves[interaction_label]["distances"].append(distance_interp)

    if len(curves) == 0:
        print("\nNo usable curve n'a été trouvée.")
        return

    results = {}
    for interaction_label, data in curves.items():
        angles_array = np.asarray(data["angles"], dtype=float)
        distances_array = np.asarray(data["distances"], dtype=float)
        mean_angle = np.nanmean(angles_array, axis=0)
        mean_distance = np.nanmean(distances_array, axis=0)
        std_angle = np.nanstd(angles_array, axis=0)
        std_distance = np.nanstd(distances_array, axis=0)
        n_interactions = len(data["angles"])

        valid_mean_distance = np.isfinite(mean_distance)
        if np.any(valid_mean_distance):
            valid_indices = np.where(valid_mean_distance)[0]
            min_idx = valid_indices[np.argmin(mean_distance[valid_mean_distance])]
            min_distance = (mean_distance[min_idx])
            min_time = (normalized_time[min_idx])
        else:
            min_distance = np.nan
            min_time = np.nan

        mean_std_distance = np.nanmean(std_distance)
        mean_std_angle = np.nanmean(std_angle)

        print(f"{interaction_label}")
        print(f"Nb of interactions : {n_interactions}")
        print(f"Min mean distance : {min_distance:.2f} m")
        print(f"Position of the min dist : {min_time:.2f} %")
        print(f"mean distance STD: {mean_std_distance:.2f} m")
        print(f"mean approach angle STD : {mean_std_angle:.2f}°")

        results[interaction_label] = {
            "mean_distance": mean_distance,
            "std_distance": std_distance,
            "mean_angle": mean_angle,
            "std_angle": std_angle,
            "min_distance": min_distance,
            "min_distance_time": min_time,
            "n": n_interactions
        }

        fig, ax1 = plt.subplots(figsize=(10, 6))

        line_distance = ax1.plot(
            normalized_time,
            mean_distance,
            color="blue",
            linewidth=2,
            label="Mean distance"
        )[0]

        ax1.fill_between(
            normalized_time,
            mean_distance - std_distance,
            mean_distance + std_distance,
            color="blue",
            alpha=0.20,
            label=f"STD distance"
        )

        ax1.set_xlabel("Normalized interaction time (%)")
        ax1.set_ylabel("Distance (m)")
        ax1.grid(True, alpha=0.3)

        if np.isfinite(min_time):
            ax1.axvline(
                min_time,
                color="red",
                linestyle="--",
                linewidth=2,
                label=(
                    f"Minimum mean distance = "
                    f"{min_distance:.2f} m "
                    f"(at {min_time:.1f}%)"
                )
            )

        # angle
        ax2 = ax1.twinx()
        line_angle = ax2.plot(
            normalized_time,
            mean_angle,
            color="orange",
            linewidth=2,
            linestyle="--",
            label="Mean approach angle"
        )[0]

        ax2.fill_between(
            normalized_time,
            mean_angle - std_angle,
            mean_angle + std_angle,
            color="orange",
            alpha=0.20,
            label=f"STD angle"
        )

        ax2.set_ylabel(
            "Approach angle (°)"
        )

        # legend
        handles1, labels1 = (ax1.get_legend_handles_labels())
        handles2, labels2 = (ax2.get_legend_handles_labels())
        ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")
        plt.title(
            "Mean distance and approach angle evolution\n"
            f"{interaction_label} "
            f"(n={n_interactions})"
        )

        plt.tight_layout()
        plt.show()

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Distance
        line_distance = ax1.plot(
            normalized_time,
            mean_distance,
            color="blue",
            linewidth=2,
            label="Mean distance"
        )[0]

        ax1.set_xlabel("Normalized interaction time (%)")
        ax1.set_ylabel("Distance (m)")

        ax1.grid(True, alpha=0.3)

        # Min distance
        if np.isfinite(min_time):
            ax1.axvline(
                min_time,
                color="red",
                linestyle="--",
                linewidth=2,
                label=(
                    f"Minimum mean distance = "
                    f"{min_distance:.2f} m "
                    f"({min_time:.1f}%)"
                )
            )

        # Angle
        ax2 = ax1.twinx()
        line_angle = ax2.plot(
            normalized_time,
            mean_angle,
            color="orange",
            linewidth=2,
            linestyle="--",
            label="Mean approach angle"
        )[0]

        ax2.set_ylabel("Approach angle (°)")

        # Legend
        handles1, labels1 = (ax1.get_legend_handles_labels())
        handles2, labels2 = (ax2.get_legend_handles_labels())
        ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")
        plt.title(
            "Mean distance and approach angle evolution\n"
            f"{interaction_label} "
            f"(n={n_interactions})"
        )

        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))

        plt.plot(
            normalized_time,
            mean_distance,
            color="blue",
            linewidth=2,
            label="Mean distance"
        )

        plt.fill_between(
            normalized_time,
            mean_distance - std_distance,
            mean_distance + std_distance,
            color="blue",
            alpha=0.20,
            label=f"STD distance"
        )

        if np.isfinite(min_time):
            plt.axvline(
                min_time,
                color="red",
                linestyle="--",
                linewidth=2,
                label=(
                    f"Minimum mean distance = "
                    f"{min_distance:.2f} m "
                    f"({min_time:.1f}%)"
                )
            )

        plt.xlabel("Normalized interaction time (%)")
        plt.ylabel("Distance (m)")

        plt.title(
            "Mean inter-agent distance evolution\n"
            f"{interaction_label} "
            f"(n={n_interactions})"
        )

        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))

        plt.plot(
            normalized_time,
            mean_angle,
            color="orange",
            linewidth=2,
            label="Mean approach angle"
        )

        plt.fill_between(
            normalized_time,
            mean_angle - std_angle,
            mean_angle + std_angle,
            color="orange",
            alpha=0.20,
            label="Angle ± STD"
        )

        if np.isfinite(min_time):
            plt.axvline(
                min_time,
                color="red",
                linestyle="--",
                linewidth=2,
                label=(
                    f"Minimum mean distance = "
                    f"{min_distance:.2f} m "
                    f"({min_time:.1f}%)"
                )
            )

        plt.xlabel("Normalized interaction time (%)")
        plt.ylabel("Approach angle (°)")
        plt.title(
            "Mean approach angle evolution\n"
            f"{interaction_label} "
            f"(n={n_interactions})"
        )
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        "normalized_time": normalized_time,
        "results": results
    }



# Pour FlowChain
def export_dataset_by_class(df, dataset_name, raw_file_name=None, output_folder="data_filtered", suffix="filtered", file_format="txt"):
    """
    Splits a DataFrame into multiple CSV files based on agent class (pedestrians and cyclists only).
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
    Splits a FlowChain trajectory TXT file into:
    - train.txt (70% of trajectories)
    - val.txt (15%)
    - test.txt (15%)

    The split is performed by trajectory (i.e. by object_id).
    """

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Les ratios doivent sommer à 1.")

    os.makedirs(output_folder, exist_ok=True)

    # Lecture du fichier
    df = pd.read_csv(input_file, sep="\t")

    # IDs uniques
    traj_ids = df[COL_ID].unique()

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