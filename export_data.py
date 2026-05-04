import os
import pandas as pd


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

    # ===== chemins fichiers originaux =====
    tracks_file = os.path.join(folder, f"{recording_id}_tracks.csv")
    meta_file   = os.path.join(folder, f"{recording_id}_tracksMeta.csv")
    rec_meta_file = os.path.join(folder, f"{recording_id}_recordingMeta.csv")

    if not os.path.exists(tracks_file) or not os.path.exists(meta_file):
        raise FileNotFoundError(f"Fichiers manquants pour recording {recording_id}")

    # ===== chargement =====
    tracks = pd.read_csv(tracks_file)
    meta   = pd.read_csv(meta_file)

    # ===== filtrage =====
    tracks_filtered = tracks[~tracks["trackId"].isin(bad_ids)].copy()
    meta_filtered   = meta[~meta["trackId"].isin(bad_ids)].copy()

    # ===== dossier de sortie =====
    output_dir = os.path.join(folder, "filtered_ind")
    os.makedirs(output_dir, exist_ok=True)

    # ===== noms fichiers =====
    tracks_out = os.path.join(output_dir, f"{recording_id}_{output_suffix}_tracks.csv")
    meta_out   = os.path.join(output_dir, f"{recording_id}_{output_suffix}_tracksMeta.csv")

    # recordingMeta copié tel quel
    rec_meta_out = os.path.join(output_dir, f"{recording_id}_{output_suffix}_recordingMeta.csv")

    # ===== sauvegarde =====
    tracks_filtered.to_csv(tracks_out, index=False)
    meta_filtered.to_csv(meta_out, index=False)

    if os.path.exists(rec_meta_file):
        pd.read_csv(rec_meta_file).to_csv(rec_meta_out, index=False)

    # ===== stats =====
    initial_tracks = tracks["trackId"].nunique()
    final_tracks = tracks_filtered["trackId"].nunique()

    print("\nEXPORT InD FILTRÉ")
    print(f"Recording : {recording_id}")
    print(f"IDs supprimés : {len(bad_ids)}")
    print(f"Trajectoires : {final_tracks} / {initial_tracks}")
    print(f"Fichiers sauvegardés dans : {output_dir}")

    return tracks_out, meta_out, rec_meta_out




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
            # Classification et features de l'interaction

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
            "approach": label.get("approach", {}).get("label_main"),
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

            # risque
            "risk": label.get("risk", None),
            "risk_score": label.get("risk_score", None)
        }

        # Ajout des features ?
        # for k, v in features.items():
        #     row[k] = v

        rows.append(row)

    # DataFrame + export
    df_out = pd.DataFrame(rows)

    df_out.to_csv(output_path, index=False)

    print(f"CSV sauvegardé : {output_path}")

    return df_out