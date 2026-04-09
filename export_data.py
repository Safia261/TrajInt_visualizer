import os
import pandas as pd

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