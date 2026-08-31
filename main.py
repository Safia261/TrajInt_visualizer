import os
import glob
import argparse
from config import DATASETS
from loader import *
from visualisation import *
from filters import *
from export_data import *
from utils import *
from validation import *


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualizer of trajectories on an aerial image."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name"
    )

    parser.add_argument(
        "--input-mode",
        type=str,
        choices=["single", "all"],
        default="all",
        help="single = a single file, all = all the files of a dataset one by one"
    )

    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Name of the file to visualize (only for input-mode=single)"
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["static", "animated"],
        help="Display mode : static ou animated" 
        # static= all the trajectories are displayed on an image
        # animlated= trajectories are displayed according to time (= video)
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help=f"FPS for the animated mode (default: {DEFAULT_FPS})"
    )

    parser.add_argument(
        "--use-unique-timestamps",
        action="store_true",
        help="In animated mode, use only the timestamps present in the data instead of regular interpolation"
    )

    parser.add_argument(
        "--hide-ids",
        action="store_true",
        help="Hide the users IDs"
        # ID affichés par défaut
    )

    parser.add_argument(
        "--highlight-id",
        type=int,
        default=None,
        help="Highlight the ID of a user"
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
        help="Maximum length of the past trajectory displayed in animated mode (default: the entire trajectory)."
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Temporal acceleration factor for animated mode"
    )

    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Number of frames to skip in animated mode (default: 1). Ex: 3 = display one frame out of every 3"
    )

    parser.add_argument(
        "--save-video",
        type=str,
        default=None,
        help="Output path for saving the animation as a video (.gif or .mp4)"
        # mettre plus tard un dossier output avec toutes les vidéos sauvegardées dedans automatiquement
    )


    parser.add_argument(
        "--vru-type",
        type=str,
        choices=["cyclists", "pedestrians", "both"],
        default="cyclists",
        help="Road user type (for VRU dataset)"
    )

    parser.add_argument(
        "--vru-behavior",
        type=str,
        choices=["starting", "moving", "stopping", "waiting", "all"],
        default="starting",
        help="Behavior type (for VRU dataset)"
    )

    parser.add_argument(
        "--no-smoothing-kalman",
        action="store_true",
        help="Deactivate kalman fiter for CTV dataset (deafult: activated)"
    )

    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Name of the scene for SDD dataset (ex: bookstore, nexus, quad...)"
        # pour Stanford dataset
    )

    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Name of the video for SDD dataset (ex: video0, video1...)"
        # pour Stanford dataset
    )

    parser.add_argument(
        "--print-interactions",
        action="store_true",
        help="Print all classified interactions of one video (in single mode only)"
    )

    parser.add_argument(
        "--analyze-interaction",
        action="store_true",
        help="Interaction ID to analyze spatio-temporal metrics of a detected and classified interaction (in single mode only)"
    )

    parser.add_argument(
        "--export-interactions",
        action="store_true",
        help="Export detected and classifiedinteractions of one or several videos in a CSV file"
    )

    return parser.parse_args()



def print_interactions_summary(df, history_hc, interactions, fps):

    classified_interactions = []

    print("Detected and classified interactions : ")

    if not interactions:
        print("No interaction detected.")
        return classified_interactions

    for i, interaction in enumerate(interactions):

        # Classification
        res_class = classify_one_interaction(df, history_hc, interaction, fps)

        # Interaction ID
        interaction_id = interaction.get("interaction_id", i)

        # Label
        interaction_label = res_class["label"].get("interaction_type", "unnkown")

        # IDs of roas users involved in the interaction
        ped_ids = [int(x) for x in interaction.get("ids_ped", [])]
        cyc_ids = [int(x) for x in interaction.get("ids_cyc", [])]

        # Type of users (cluster, noise=individual)
        ped_type = ("ped_cluster" if len(ped_ids) > 1 else "ped_noise")

        cyc_type = ("cyc_cluster" if len(cyc_ids) > 1 else "cyc_noise")

        result = {
            "interaction_id": interaction_id,
            "interaction_label": interaction_label,
            "ped_type": ped_type,
            "ped_ids": ped_ids,
            "cyc_type": cyc_type,
            "cyc_ids": cyc_ids,
            "interaction": interaction,
            "classification": res_class
        }

        classified_interactions.append(result)

        print(
            f"{interaction_id} | "
            f"{interaction_label} | "
            f"{ped_type} {ped_ids} | "
            f"{cyc_type} {cyc_ids}"
        )


    return classified_interactions



def main():
    args = parse_args()

    # classification_validation("test_ctv/manual_annotated_inter.csv", "res_validation")

    cfg = DATASETS[args.dataset]
    folder = cfg["folder"]

    if cfg["type"] == "vru":
        df, image_path, cfg = load_dataset(args.dataset, None, args)
        df = prepare_data(df, no_cars=args.no_cars)
        run_visualization(df, image_path, cfg, args)
        return

    # compute_binary_interaction_statistics("test_ctv/final_inter_sdd.csv", True, "stats_sdd_binary_final")
    # compute_group_interaction_statistics("test_ctv/final_inter_sdd.csv", True, "stats_sdd_group_final")

    #################################
    # MODE SINGLE
    #################################
    if args.input_mode == "single":
        if args.dataset != "sdd" and args.file is None:
            raise ValueError("Tu dois spécifier --file en mode single")

        if args.dataset == "ind":
            df, image_path, cfg = load_dataset(args.dataset, args.file)
        elif args.dataset == "sdd":
                df, image_path, cfg = load_dataset(args.dataset, None, args)
        else:
            file_path = os.path.join(folder, args.file)
            df, image_path, cfg = load_dataset(args.dataset, file_path) # on garde les voitures au début, le filtrage se fait après


        # df, image_path, cfg = load_dataset(args.dataset, file_path)
        df = prepare_data(df, no_cars=args.no_cars) # à retirer car inutile grâce au filtrage
        _, _ = analyze_initial_nb_traj_interactions(df) # ajouter arg flag pour l'analyse
       

        if not args.no_smoothing_kalman and args.dataset.startswith("ctv"):
            # Kalman filter analysis
            # analyze_speeds(df, cfg, cfg["fps"], agent_ids=[1,14])
            # R_values = [0.1, 0.5, 1.0, 2.0, 5.0]
            # compare_kalman_R_two_agents(df, R_values)
            df = apply_kalman_filter(df, R_value=1.0)
            # analyze_speeds(df, cfg, cfg["fps"], agent_ids=[1,14])
            # df = resample_dataset(df, cfg["fps"],target_dt=0.4)
            # export_filtered_data_original(df, args.dataset, cfg["folder"], args.file, "CTV_filtered")
            # export_filtered_data(df, args.dataset, args.file, "data_filtered/ctv_josue", suffix="filtered_23111")
            # export_dataset_by_class(df, args.dataset, args.file, "data_filtered/ctv_flowchain", file_format="txt")

        if cfg.get("has_cars", False):

            # distances = analyze_car_vru_distances(df)
            if args.dataset == "ind":
                df, bad_ids = filter_interactions_with_close_cars(df, ind = True)
                # export_filtered_ind_recording(args.file, bad_ids, cfg)

            elif args.dataset == "tss":
                df, _ = filter_coexisting_with_cars(df)
                # export_filtered_data_original(df, args.dataset, cfg["folder"], args.file, "TSS_filtered")
                # export_filtered_data(df, args.dataset, args.file, "data_filtered/ctv_flowchain")
                # export_dataset_by_class(df, args.dataset, args.file, "data_filtered/ind_filt", file_format="txt")

            elif args.dataset == "sdd":
                df, _ = filter_interactions_with_close_cars(df)

        if args.print_interactions or args.analyze_interaction or args.export_interactions:
            history_hc = compute_clusters_and_hulls_over_time(df, plot=False, fps=cfg["fps"]) # to see DBSCAN + convex hulls frame per frame, write plot=True
            interactions = build_interaction_events(history_hc, fps=cfg["fps"])
            print_interactions_summary(df, history_hc, interactions, cfg["fps"])

            if args.analyze_interaction:
                if not interactions:
                    print("No interaction to analyze.")
                else:
                    try:
                        interaction_id = int(input("\n Enter the interaction ID to analyze : "))
                    except ValueError:
                        print("ID not valid.")
                        return
                    if interaction_id < 0 or interaction_id >= len(interactions):
                        print("No interaction with ID ", interaction_id)
                        return

                    print("\nAnalysis of the interaction ", interaction_id)
                    res_inter = compute_one_interaction_features(df, history_hc, interactions[interaction_id], fps=cfg["fps"], plot=True)
                    analyze_speeds(df, cfg, cfg["fps"], [2, 11], start=interactions[interaction_id]["start"], end=interactions[interaction_id]["end"])

            if args.export_interactions:
                if not interactions:
                    print("No interactions to export.")
                else:
                    output_path = f"{os.path.splitext(args.file)[0]}_interactions.csv"
                export_interactions_to_csv(df, history_hc, interactions, fps=cfg["fps"], output_path=output_path)


        run_visualization(df, image_path, cfg, args)

    #################################
    # MODE ALL (un par un)
    #################################
    elif args.input_mode == "all":
        all_dfs = []
        all_histories = {}
        all_interactions = []
        all_interaction_dfs = []
        id_offset = 0
        frame_offset = 0

        # Statistiques globales CTV
        total_initial_traj = 0
        total_initial_interactions = 0

        total_filtered_traj = 0
        total_filtered_interactions = 0

        total_dbscan_interactions = 0

        if args.dataset == "sdd":
            videos = []
            for scene in sorted(os.listdir(folder)):
                scene_path = os.path.join(folder, scene)
                if not os.path.isdir(scene_path):
                    continue
    
                for video in sorted(os.listdir(scene_path)):
                    video_path = os.path.join(scene_path, video)
                    if not os.path.isdir(video_path):
                        continue
    
                    ann_file = os.path.join(video_path, "annotations.txt")
                    if os.path.exists(ann_file):
                        videos.append((scene, video))
            files = videos

        else:
            pattern = cfg.get("file_pattern", "*.csv")
            if "files" in cfg:
                files = [os.path.join(folder, f) for f in cfg["files"]]
            else:
                pattern = cfg.get("file_pattern", "*.csv")
                files = glob.glob(os.path.join(folder, pattern))

        if not files:
            raise FileNotFoundError(f"Aucun CSV trouvé dans {folder}")
        
        # split_txt_by_trajectory("flowchain_data/ctv_new/ctv_Pedestrian.txt", "flowchain_data/ctv_new")
        # split_txt_by_trajectory("flowchain_data/ctv_new/ctv_Cyclist.txt", "flowchain_data/ctv_new")

        # global_speeds = []
        # global_accelerations = []
        # total_cyclists = 0

        original_dfs = {}
        try:
            for f in files:
                # print(f"\nLecture de {os.path.basename(f)}")

                if args.dataset == "ind":
                    recording_id = os.path.basename(f).split("_")[0]

                    df, image_path, cfg = load_dataset(args.dataset, recording_id)
                    print(f"\nLecture de {os.path.basename(f)}")

                elif args.dataset == "sdd":
                    scene, video = f
                    args.scene = scene
                    args.video = video
                    df, image_path, cfg = load_dataset(args.dataset, None, args)
                    print(f"\nLecture {scene}/{video}")

                else:
                    df, image_path, cfg = load_dataset(args.dataset, f)
                    print(f"\nLecture de {os.path.basename(f)}")

                df = prepare_data(df, no_cars=args.no_cars)
                # df = resample_dataset(df, cfg["fps"],target_dt=0.4)

                initial_nb_traj, initial_nb_interactions = analyze_initial_nb_traj_interactions(df, verbose=False)
                total_initial_traj += initial_nb_traj
                total_initial_interactions += initial_nb_interactions

                # filtrage
                if not args.no_smoothing_kalman and args.dataset.startswith("ctv"):
                    df = apply_kalman_filter(df, R_value=1.0)
                    # export_filtered_data_original(df, args.dataset, cfg["folder"], os.path.basename(f), "CTV_filtered")
                    # export_filtered_data(df, args.dataset, os.path.basename(f), "data_filtered/ctv_filt")
                    # export_dataset_by_class(df, args.dataset, args.file, "data_filtered/ctv_flowchain", file_format="txt")
                    
                    filtered_nb_traj, filtered_nb_interactions = analyze_initial_nb_traj_interactions(df, verbose=False)
                    total_filtered_traj += filtered_nb_traj
                    total_filtered_interactions += filtered_nb_interactions
                    print(
                        f"Après Kalman : "
                        f"{total_filtered_traj} trajectoires, "
                        f"{total_filtered_interactions} interactions"
                    )
                    
                if cfg.get("has_cars", False):
                    if args.dataset == "ind":
                        df, _ = filter_interactions_with_close_cars(df, ind = True)
                        filtered_nb_traj, filtered_nb_interactions = analyze_initial_nb_traj_interactions(df, verbose=False)
                        total_filtered_traj += filtered_nb_traj
                        total_filtered_interactions += filtered_nb_interactions
                        print(
                            f"Après filtrage : "
                            f"{filtered_nb_traj} trajectoires, "
                            f"{filtered_nb_interactions} interactions"
                        )


                    elif args.dataset == "tss":
                        df, _ = filter_coexisting_with_cars(df)
                        # export_filtered_data_original(df, args.dataset, cfg["folder"], os.path.basename(f), "TSS_filtered")
                        # export_filtered_data(df, args.dataset, os.path.basename(f), "data_filtered/noname_filt")
                        # export_dataset_by_class(df, args.dataset, args.file, "data_filtered/ctv_flowchain", file_format="txt")
                        # split_txt_by_trajectory("flowchain_data/tss/noname_Pedestrian.txt", "flowchain_data/tss")
                        
                        filtered_nb_traj, filtered_nb_interactions = analyze_initial_nb_traj_interactions(df, verbose=False)
                        total_filtered_traj += filtered_nb_traj
                        total_filtered_interactions += filtered_nb_interactions
                        print(
                            f"Après filtrage : "
                            f"{filtered_nb_traj} trajectoires, "
                            f"{filtered_nb_interactions} interactions"
                        )

                    elif args.dataset == "sdd":
                        df, _ = filter_interactions_with_close_cars(df)
                        filtered_nb_traj, filtered_nb_interactions = analyze_initial_nb_traj_interactions(df, verbose=False)
                        total_filtered_traj += filtered_nb_traj
                        total_filtered_interactions += filtered_nb_interactions
                        print(
                            f"Après filtrage : "
                            f"{filtered_nb_traj} trajectoires, "
                            f"{filtered_nb_interactions} interactions"
                        )

                # df[COL_ID] += id_offset
                # df[COL_TIME] += frame_offset
                # all_dfs.append(df)

                if args.dataset == "sdd":
                    recording_name = f"{scene}_{video}"
                else:
                    recording_name = os.path.basename(f)
                original_dfs[recording_name] = df.copy()

                # POUR LES ALL CTV STATS
                if args.export_interactions:
                    history_hc = compute_clusters_and_hulls_over_time(df, fps=cfg["fps"])
                    interactions = build_interaction_events(history_hc, fps=cfg["fps"])
                    df_inter = export_interactions_to_csv(df, history_hc, interactions, fps=cfg["fps"], output_path=None)
                    if args.dataset == "sdd":
                        df_inter["recording"] = f"{scene}_{video}"
                    else:
                        df_inter["recording"] = os.path.basename(f)
                    all_interaction_dfs.append(df_inter)
                    total_dbscan_interactions += len(interactions)
                    print(f"Interactions detected with DBSCAN : {len(interactions)}")

                    all_histories.update(history_hc)
                    all_interactions.extend(interactions)
                    # id_offset = df[COL_ID].max() + 1
                    # frame_offset = df[COL_TIME].max() + 1

                # speeds, accs, n_cyclists = collect_cyclist_statistics(df, cfg["fps"])
                # global_speeds.extend(speeds)
                # global_accelerations.extend(accs)
                # total_cyclists += n_cyclists

                # run_visualization(df, image_path, cfg, args)
            
            # FOR DATA FORMAT FOR FLOWCHAIN
            # if all_dfs:
            #     # print(f"id offset = {id_offset}, frame offset = {frame_offset}")
            #     final_df = pd.concat(all_dfs, ignore_index=True)
    
            #     # export_interactions_to_csv(final_df, all_histories, all_interactions, fps=cfg["fps"], output_path="final_inter_ctv2.csv")
            #     export_filtered_data(final_df, args.dataset, output_folder="data_filtered/ctv_trackid_df")
            #     # export_dataset_by_class(final_df, args.dataset, args.file, "flowchain_data/ind_new", file_format="txt")
            #     # split_txt_by_trajectory("flowchain_data/ind_new/ind_filtered_Pedestrian.txt", "flowchain_data/ind_new")
            #     # split_txt_by_trajectory("flowchain_data/ind_new/ind_filtered_Cyclist.txt", "flowchain_data/ind_new")

            if args.export_interactions:
                if all_interaction_dfs:
                    final_csv = pd.concat(all_interaction_dfs, ignore_index=True)
                    # renumérotation globale des interactions
                    # final_csv["interaction_id"] = range(len(final_csv))
                    output_path = f"{args.dataset}_interactions.csv"
                    final_csv.to_csv(output_path, index=False)
                    print(f"{len(final_csv)} interactions exported.")


                # Numbers of trajectories, interactions, pedestrians, cyclistes, etc
                removed_interactions = (total_initial_interactions - total_filtered_interactions)
                removed_traj = (total_initial_traj - total_filtered_traj)
                print("Global summary:")
                print(f"Trajectories before filtering : {total_initial_traj}")
                print(f"Trajectories after filtering : {total_filtered_traj}")
                print("Suppressed trajectories : "
                    f"{removed_traj} "
                    f"({removed_traj / total_initial_traj * 100:.2f}%)"
                    if total_initial_traj > 0 else
                    "Suppressed trajectories : 0"
                )

                print(f"\nInteractions before filtering : {total_initial_interactions}")
                print(f"Interactions after filtering : {total_filtered_interactions}")
                if total_initial_interactions > 0:
                    print("Suppressed interactions : "
                        f"{removed_interactions} "
                        f"({removed_interactions / total_initial_interactions * 100:.2f}%)"
                    )

                print(f"\nInteractions dectected with DBSCAN : {total_dbscan_interactions}")


            # global_speeds = np.asarray(global_speeds)
            # global_accelerations = np.asarray(global_accelerations)
            # print_cyclist_statistics(global_speeds, global_accelerations, total_cyclists)
            # plot_cyclist_speed_histogram(global_speeds, total_cyclists)

            # plot_binary_distance_approach_evolution("test_ctv/final_inter_ctv.csv",  original_dfs, cfg["fps"], n_points=100)

        except KeyboardInterrupt:
            print("\nEnd of the program by user.")


def run_visualization(df, image_path, cfg, args):
    if args.use_unique_timestamps:
        speed = 1.0  # temps réel strict (pour ne pas appliquer recommended_speed quand cet arg est utilisé)
    else:
        if args.speed is not None:
            speed = args.speed
        else:
            speed = cfg.get("recommended_speed", 1.0)

    if args.mode == "static":
        plot_static_trajectories(df, image_path, cfg, show_ids=not args.hide_ids)

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
            highlight_id=args.highlight_id,
            video_name=args.file,
            dataset_name=args.dataset
        )


if __name__ == "__main__":
    main()