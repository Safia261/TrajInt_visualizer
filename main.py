import os
import glob
import argparse
from config import DATASETS
from loader import *
from visualisation import *
from filters import *
from export_data import *
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualiseur de trajectoires sur image aérienne."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Nom du dataset à utiliser"
    )

    parser.add_argument(
        "--input-mode",
        type=str,
        choices=["single", "all"],
        default="all",
        help="single = un seul fichier, all = tous les fichiers du dataset un à un"
    )

    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Nom du fichier à visualiser (obligatoire si input-mode=single)"
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["static", "animated"],
        help="Mode d'affichage : static ou animated" 
        # static= toutes les trajectoires affichées
        # animlated= trajectoires tracées en live (=vidéo)
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help=f"FPS pour le mode animé (défaut: {DEFAULT_FPS})"
    )

    parser.add_argument(
        "--use-unique-timestamps",
        action="store_true",
        help="En mode animé, utiliser uniquement les timestamps présents dans les données au lieu d'une interpolation régulière."
    )

    parser.add_argument(
        "--hide-ids",
        action="store_true",
        help="Masquer les IDs des objets."
        # ID affichés par défaut
    )

    parser.add_argument(
        "--highlight-id",
        type=int,
        default=None,
        help="ID de l'agent à mettre en évidence"
    )

    parser.add_argument(
        "--no-cars",
        action="store_true",
        help="Exclure les voitures (classe 3) de l'affichage."
        # à voir si on retire pas cet argument plus tard, après grand tri des datasets
    )

    parser.add_argument(
        "--tail-length",
        type=int,
        default=None,
        help="Longueur max de la trace passée en mode animé (par défaut: toute la trajectoire)."
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Facteur d'accélération temporelle pour le mode animé."
    )

    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Nombre de frames à sauter en mode animé (défaut: 1). Exemple: 3 = une frame affichée sur 3."
    )

    parser.add_argument(
        "--save-video",
        type=str,
        default=None,
        help="Chemin de sortie pour enregistrer l'animation en vidéo, par ex. output.mp4"
        # mettre plus tard un dossier output avec toutes les vidéos sauvegardées dedans automatiquement
    )


    parser.add_argument(
        "--vru-type",
        type=str,
        choices=["cyclists", "pedestrians", "both"],
        default="cyclists",
        help="Type d'agents VRU"
    )

    parser.add_argument(
        "--vru-behavior",
        type=str,
        choices=["starting", "moving", "stopping", "waiting", "all"],
        default="starting",
        help="Type de comportement VRU"
    )

    parser.add_argument(
        "--no-smoothing-kalman",
        action="store_true",
        help="Désactiver le lissage avec filtre de Kalman"
        # Filtre activé par défaut pour CTV
    )

    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Nom de la scène (ex: bookstore, nexus, quad...)"
        # pour Stanford dataset
    )

    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Nom de la vidéo (ex: video0, video1...)"
        # pour Stanford dataset
    )

    return parser.parse_args()



def main():
    args = parse_args()

    cfg = DATASETS[args.dataset]
    folder = cfg["folder"]

    if cfg["type"] == "vru":
        df, image_path, cfg = load_dataset(args.dataset, None, args)
        df = prepare_data(df, no_cars=args.no_cars)
        run_visualization(df, image_path, cfg, args)
        return
    
    if args.dataset == "stanford2":
        df, image_path, cfg = load_dataset(args.dataset, None, args)
        df = prepare_data(df, no_cars=args.no_cars)
        _, _ = analyze_initial_nb_traj_interactions(df)
        # df, _ = filter_coexisting_with_cars(df)
        df, _ = filter_interactions_with_close_cars(df)
        run_visualization(df, image_path, cfg, args)
        return

    # ===== MODE SINGLE =====
    if args.input_mode == "single":
        if args.file is None:
            raise ValueError("Tu dois spécifier --file en mode single")

        if args.dataset == "ind":
            df, image_path, cfg = load_dataset(args.dataset, args.file)
        else:
            file_path = os.path.join(folder, args.file)
            df, image_path, cfg = load_dataset(args.dataset, file_path) # on garde les voitures au début, le filtrage se fait après


        # df, image_path, cfg = load_dataset(args.dataset, file_path)
        df = prepare_data(df, no_cars=args.no_cars)
        _, _ = analyze_initial_nb_traj_interactions(df)

        if not args.no_smoothing_kalman and args.dataset.startswith("ctv"):
            # analyse du filtre de Kalman
            # R_values = [0.1, 0.5, 1.0, 2.0, 5.0]
            # compare_kalman_R_two_agents(df, R_values)
            # print("\nRaw data")
            # analyze_speeds(df, cfg)
            # print("\navant filtre")
            # distances = analyze_cycl_ped_distances(df)
            df = apply_kalman_filter(df, R_value=1.0)
            # print("\naprès filtre")
            # distances = analyze_cycl_ped_distances(df)
            # print("\nFiltred data")

            # times, vrel, vrel_kmh, angles = compute_relative_motion(ped, cyc, cfg["fps"], plot=True)
            # plot_velocity_vectors_over_time(df, 1, cfg["fps"], step=20)
            # compute_agent_direction(df, 1, "angle", "deg", plot = True)

            # g = df[df[COL_ID] == 9].sort_values(COL_TIME)
            # compute_velocity_vectors(g, cfg["fps"], plot=True)
            # t, a, c_da = compute_direction_angle_velocity_based(df, 2, 9, cfg["fps"], plot=True, return_class=True)
            # print("\nDirection angle : ", c_da)
            # t, a, c_aa = compute_approach_angle(df, 2, 9, cfg["fps"], plot=True, return_class=True)
            # print("\nApproach angle : ", c_aa)
            # times, rel_speeds, rel_speeds_kmh, angles, distances, c_v = compute_relative_speed(df, 2, 9, cfg["fps"], plot=True, return_class=True)
            # print("\nRelative speed : ", c_v)
            # pet = compute_pet(df, 2, 9, cfg["fps"], plot=True, return_class=True)
            # print("\nPET : ", pet)
            # times, ttc_val, ttc_min, c_ttc = compute_ttc(df, 2, 9, cfg["fps"], plot=True, return_class=True)    
            # print("\n TTC : ", ttc_min, c_ttc)
            # res = classify_pair_interaction(pet, (ttc_min, c_ttc), c_aa, c_da, c_v)
            # print("\nFinal results : ", res)
            # i_fps, i_sec, duree_sec = compute_interaction_intervals(df, 2, 9, cfg["fps"])
            # print("Intervalle interaction :", i_fps, i_sec, duree_sec)
            # _, _, _, c_dist = compute_distance_ped_cyc(df, 2, 9, cfg["fps"], plot=True, return_class=True)
            # print("\nDist classification : ", c_dist)

            
            history_hc = compute_clusters_and_hulls_over_time(df, plot=True, fps=cfg["fps"])

            r_splits = detect_split_events_with_cyclists(history_hc)
            # print("\nSpliting de clusters : ", r_splits)
            r_insertion = detect_cyclists_in_hulls(history_hc)
            # print("\nDétection faufilement : ", r_insertion)

            inter_clusters = detect_cluster_interactions(history_hc, df)
            # print("\nInteractions clusters ", inter_clusters)
            interactions = build_interaction_events(history_hc)
            # print("\n INTERACTIONS ", interactions)
            # res_inter = compute_one_interaction_features(df, history_hc, interactions[17], fps=cfg["fps"], plot=True)
            # print("\n Interaction 17 : ", interactions[17])
            # print("\n Interaction 3 : ", interactions[3])
            # # print("\n FEATURES INTERACTION", res_inter)
            # res_class = classify_one_interaction(df, history_hc, interactions[17], cfg["fps"])
            # print("\n CLASSIFICATION : ", res_class["label"])
            # compute_cluster_distances_tracked(history_hc, fps=cfg["fps"], plot=True)
            export_interactions_to_csv(df, history_hc, interactions, fps=cfg["fps"], output_path="interactions2.csv")


        if cfg.get("has_cars", False):
            # distances = analyze_car_vru_distances(df)
            if args.dataset == "ind":
                # df, bad_ids = filter_spatial_car_influence(df, distance_threshold=5.0, ind = True)
                df, bad_ids = filter_interactions_with_close_cars(df, ind = True)
                # print(bad_ids)
                analyze_speeds(df, cfg)
                # distances = analyze_car_vru_distances(df)
                # export_filtered_ind_recording(args.file, bad_ids, cfg)
            elif args.dataset == "noname":
                df, _ = filter_coexisting_with_cars(df)

        run_visualization(df, image_path, cfg, args)

    # ===== MODE ALL (un par un) =====
    elif args.input_mode == "all":
        # files = glob.glob(os.path.join(folder, "*.csv"))
        pattern = cfg.get("file_pattern", "*.csv")
        # files = glob.glob(os.path.join(folder, pattern))
        if "files" in cfg:
            files = [os.path.join(folder, f) for f in cfg["files"]]
        else:
            pattern = cfg.get("file_pattern", "*.csv")
            files = glob.glob(os.path.join(folder, pattern))

        if not files:
            raise FileNotFoundError(f"Aucun CSV trouvé dans {folder}")

        try:
            for f in files:
                print(f"\n===== Lecture de {os.path.basename(f)} =====")

                df, image_path, cfg = load_dataset(args.dataset, f)
                df = prepare_data(df, no_cars=args.no_cars)
                _, _ = analyze_initial_nb_traj_interactions(df)

                # filtrage
                if not args.no_smoothing_kalman and args.dataset.startswith("ctv"):
                    # df = apply_kalman_filter(df)
                    # R_values = [0.1, 0.5, 1.0, 2.0, 5.0]
                    # compare_kalman_R(df, R_values, agent_id=2)
                    # compare_kalman_R_two_agents(df, R_values)
                    # print("\nRaw data")
                    # analyze_speeds(df, cfg)
                    df = apply_kalman_filter(df, R_value=1.0)
                    # print("\nFiltred data")
                    # analyze_speeds(df, cfg)
                    
                if cfg.get("has_cars", False):
                    # distances = analyze_car_vru_distances(df)
                    # df = filter_spatial_car_influence(df, distance_threshold=5.0)
                    df, _ = filter_coexisting_with_cars(df)

                run_visualization(df, image_path, cfg, args)

        except KeyboardInterrupt:
            print("\nArrêt demandé par l'utilisateur. Fin du programme.")


def run_visualization(df, image_path, cfg, args):
    # if args.speed is not None:
    #     speed = args.speed
    # else:
    #     speed = cfg.get("recommended_speed", 1.0)

    if args.use_unique_timestamps:
        speed = 1.0  # temps réel strict (pour ne pas appliquer recommended_speed quand cet arg est utilisé)
    else:
        if args.speed is not None:
            speed = args.speed
        else:
            speed = cfg.get("recommended_speed", 1.0)

    if args.mode == "static":
        plot_static_trajectories(
            df,
            image_path,
            cfg,
            show_ids=not args.hide_ids
        )

    elif args.mode == "animated":
        animate_trajectories(
            df,
            image_path,
            cfg,
            show_ids=not args.hide_ids,
            fps=args.fps,
            use_unique_timestamps=args.use_unique_timestamps,
            tail_length=args.tail_length,
            speed=speed,
            frame_step=args.frame_step,
            save_video=args.save_video,
            highlight_id=args.highlight_id
        )


if __name__ == "__main__":
    main()