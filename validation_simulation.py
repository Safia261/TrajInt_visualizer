import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from matplotlib.lines import Line2D

from pathlib import Path

from config import *

# columns of simulation files
SIM_COLS = [
    "simulation_id",
    "seq",
    "time",
    "cyclist_id",
    "pedestrian_id",
    "cyclist_x",
    "cyclist_y",
    "pedestrian_x",
    "pedestrian_y",
    "cyclist_vx",
    "cyclist_vy",
    "cyclist_speed",
    "pedestrian_vx",
    "pedestrian_vy",
    "pedestrian_speed",
    "cyclist_heading",
    "distance",
    "app_angle",
]

fixed_vmaxs = {
    "overtaking": 6.5, # same for overtaking2 
    "avoidance": 6.5, #same for avoidance2
    "intersection_right": 4.0,
    "intersection_left": 5.5,
    "avoiding_group": 4.0,
    "overtaking_group": 5.0
}


# load simulation data
def load_simulation_csv(path):
    df = pd.read_csv(path)

    missing = [c for c in SIM_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} : missing coulmns : {missing}")

    # time relative to 0 (in simulation data, time is machine time)
    df["time_rel"] = df["time"] - df["time"].min()

    # inverse y coordinates (EXCEPT for avoidance for calibration)
    df["cyclist_y"] = -df["cyclist_y"]
    df["pedestrian_y"] = -df["pedestrian_y"]
    df["cyclist_vy"] = -df["cyclist_vy"]
    df["pedestrian_vy"] = -df["pedestrian_vy"]

    # relative speed
    df["relative_speed"] = np.sqrt(
        (df["cyclist_vx"] - df["pedestrian_vx"]) ** 2
        + (df["cyclist_vy"] - df["pedestrian_vy"]) ** 2
    )

    # cyclist vmax in each simulation
    df["cyclist_speed_max"] = df.groupby("simulation_id")["cyclist_speed"].transform("max")

    # mean cyclist speed
    df["cyclist_speed_mean_sim"] = df.groupby("simulation_id")["cyclist_speed"].transform("mean")

    dt = df["time_rel"].diff()

    df["cyclist_delta_v"] = df["cyclist_speed"].diff()
    df["pedestrian_delta_v"] = df["pedestrian_speed"].diff()

    df["cyclist_acceleration"] = (df["cyclist_delta_v"] / dt)
    df["pedestrian_acceleration"] = (df["pedestrian_delta_v"] / dt)

    return df


def load_real_csv(path):
    """
    Load the real trajectories CSV (so the df should already be exported).
    """
    df = pd.read_csv(path)

    required = ["time_step", "object_id", "user_type", "x_m", "y_m"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Real data missing : {missing}")

    return df


def compute_real_min_distance(df):
    """
    Computes the real minimum cyclist-pedestrian distance
    and the approach angle associated with this minimum distance.

    For multiple pedestrians, the pedestrian associated with the
    globally minimum distance is selected, consistently with the
    simulation analysis.
    """

    cyclists = df[df["user_type"] == 2].copy()
    pedestrians = df[df["user_type"] == 1].copy()

    if cyclists.empty or pedestrians.empty:
        return {
            "distance_min": np.nan,
            "angle_at_min": np.nan,
            "cyclist_id": np.nan,
            "pedestrian_id": np.nan
        }

    cyclist_id = cyclists["object_id"].iloc[0]
    cyclist = cyclists[cyclists["object_id"] == cyclist_id]

    pair_results = []

    for pedestrian_id in pedestrians["object_id"].unique():

        pedestrian = pedestrians[pedestrians["object_id"] == pedestrian_id]
        g = prepare_cyclist_pedestrian_pair(cyclist, pedestrian)

        if g.empty:
            continue

        # Minimum distance
        min_idx = g["distance"].idxmin()
        distance_min = g.loc[min_idx, "distance"]

        # Approach angle at the exact same timestep
        angle_at_min = g.loc[min_idx, "app_angle"]

        pair_results.append({
            "cyclist_id": cyclist_id,
            "pedestrian_id": pedestrian_id,
            "distance_min": distance_min,
            "angle_at_min": angle_at_min
        })

    if not pair_results:
        return {
            "distance_min": np.nan,
            "angle_at_min": np.nan,
            "cyclist_id": cyclist_id,
            "pedestrian_id": np.nan
        }

    pair_results = pd.DataFrame(pair_results)

    # Select the cyclist-pedestrian pair with the smallest distance
    min_idx = pair_results["distance_min"].idxmin()
    selected = pair_results.loc[min_idx]

    return {
        "distance_min": selected["distance_min"],
        "angle_at_min": selected["angle_at_min"],
        "cyclist_id": selected["cyclist_id"],
        "pedestrian_id": selected["pedestrian_id"]
    }


def identify_configuration(path):
    """
    Identify :
        SFM modele = NES / Moussaid
        cyclist vmax = random / fixed
        SFM = with / without
    """
    name = path.name.lower()

    is_nes = "nes" in name
    is_random = "random" in name
    without_sfm = "wth" in name

    model = "NES" if is_nes else "Moussaid"
    speed_type = "Random" if is_random else "Fixed"
    sfm = "Without SFM" if without_sfm else "SFM"

    return model, speed_type, sfm


def find_simulation_files(sim_root, scenario):
    """
    Find simulation files corresponding to a specific scenario.
    """

    scenario_dir = Path(sim_root) / scenario

    if not scenario_dir.exists():
        raise FileNotFoundError(f"Folder with simulations files not found : {scenario_dir}")

    files = list(scenario_dir.rglob("simulation_data_*.csv"))
    return files


def load_all_simulations(sim_root, scenario):
    """
    Load all data files of a scenario and add metadata.
    """

    files = find_simulation_files(sim_root, scenario)
    dfs = []

    for path in files:
        try:
            df = load_simulation_csv(path)
        except Exception as e:
            print(f"Warning: {e}")
            continue

        model, speed_type, sfm = identify_configuration(path)

        df["model"] = model
        df["speed_type"] = speed_type
        df["sfm"] = sfm
        df["source_file"] = path.name

        dfs.append(df)

    if not dfs:
        raise ValueError(f"No valid simulation file for scenario {scenario}")

    return pd.concat(dfs, ignore_index=True)



################################


# TRAJECTORIES
def plot_trajectories(real_df, sim_df, scenario, output_dir, fixed_vmax=None):
    """
    Plot all trajectories (real + simulations).

    One figure per configuration:
        NES/Moussaid
        Fixed/Random
        SFM/Without SFM
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configurations = sim_df[["model", "speed_type", "sfm"]].drop_duplicates()

    for _, config in configurations.iterrows():
        model = config["model"]
        speed_type = config["speed_type"]
        sfm = config["sfm"]

        subset = sim_df[
            (sim_df["model"] == model) &
            (sim_df["speed_type"] == speed_type) &
            (sim_df["sfm"] == sfm)
        ]

        fig, ax = plt.subplots(figsize=(10, 8))

        # Real trajectories
        start_label_added = False
        end_label_added = False

        for user_type, group in real_df.groupby("user_type"):

            if user_type == 1:
                color = "blue"
                label = "Real pedestrian"
                lw = 2
            elif user_type == 2:
                color = "black"
                label = "Real cyclist"
                lw = 3
            else:
                continue
            

            for obj_id, traj in group.groupby("object_id"):
                traj = traj.sort_values("time_step")

                ax.plot(traj["x_m"], traj["y_m"],
                    linewidth=lw,
                    alpha=0.8,
                    label=label if obj_id == group["object_id"].iloc[0] else None,
                    color=color)

                # Start
                ax.scatter(traj["x_m"].iloc[0], traj["y_m"].iloc[0],
                    marker="o",
                    s=45,
                    zorder=10,
                    label="Real start" if not start_label_added else None,
                    color="green"
                )

                # End
                ax.scatter(traj["x_m"].iloc[-1], traj["y_m"].iloc[-1],
                    marker="X",
                    s=55,
                    zorder=10,
                    label="Real end" if not end_label_added else None,
                    color="red"
                )

                start_label_added = True
                end_label_added = True


        # Simulated trajectories
        if speed_type.lower() == "random":

            # to print cyclist vmax when RANDOM
            vmax_values = (subset.groupby("simulation_id")["cyclist_speed_max"].first())
            vmin = vmax_values.min()
            vmax = vmax_values.max()
            if vmin == vmax:
                vmin -= 0.1
                vmax += 0.1

            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.cm.plasma

        sim_start_label_added = False
        sim_end_label_added = False

        for sim_id, sim in subset.groupby("simulation_id"):
            sim = sim.sort_values("time_rel")

            sim_vmax = sim["cyclist_speed_max"].iloc[0]
            if speed_type.lower() == "random":
                cyclist_color = cmap(norm(sim_vmax))
            else:
                cyclist_color = "tab:green"

            # Cyclist
            ax.plot(sim["cyclist_x"], sim["cyclist_y"],
                color=cyclist_color,
                alpha=0.45,
                linewidth=1.2,
                label="Simulated cyclist"
                if not sim_start_label_added else None
            )

            # Start cyclist
            ax.scatter(sim["cyclist_x"].iloc[0], sim["cyclist_y"].iloc[0],
                color=cyclist_color,
                marker="o",
                s=30,
                alpha=0.7,
                zorder=8,
                label="Simulated start cyclist" if not sim_start_label_added else None)

            # End cyclist
            ax.scatter(sim["cyclist_x"].iloc[-1], sim["cyclist_y"].iloc[-1],
                color=cyclist_color,
                marker="X",
                s=40,
                alpha=0.7,
                zorder=8,
                label="Simulated end cyclist" if not sim_end_label_added else None)

            n_pedestrians = sim["pedestrian_id"].nunique()
            if n_pedestrians == 1:
                # Pedestrian
                ax.plot(sim["pedestrian_x"], sim["pedestrian_y"],
                    color="tab:blue",
                    alpha=0.45,
                    linewidth=1.2,
                    label="Simulated pedestrian" if not sim_start_label_added else None)

                # Start pedestrian
                ax.scatter(sim["pedestrian_x"].iloc[0], sim["pedestrian_y"].iloc[0],
                    color="tab:blue",
                    marker="o",
                    s=30,
                    alpha=0.7,
                    zorder=8,
                    label="Simulated start pedestrian" if not sim_start_label_added else None)

                # End pedestrian
                ax.scatter(sim["pedestrian_x"].iloc[-1], sim["pedestrian_y"].iloc[-1],
                    color="tab:blue",
                    marker="X",
                    s=40,
                    alpha=0.7,
                    zorder=8,
                    label="Simulated end pedestrian" if not sim_end_label_added else None)

            sim_start_label_added = True
            sim_end_label_added = True

        # Colorbar for RANDOM
        if speed_type == "Random":
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, pad=0.02)
            cbar.set_label("Cyclist maximum speed (m/s)")

        if speed_type == "Fixed" and fixed_vmax is not None:
            title = (f"{scenario} — {model} — {speed_type} vmax = {fixed_vmax}m/s — {sfm} -\nTrajectories (20 simulations)")
        else:
            title = (f"{scenario} — {model} — {speed_type} — {sfm} -\nTrajectories (20 simulations)")

        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.set_xlabel("X (m)", fontsize=15)
        ax.set_ylabel("Y (m)", fontsize=15)
        ax.tick_params(axis="both", labelsize=13)
        ax.invert_yaxis()
        ax.axis("equal")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=14)
        fig.tight_layout()
        filename = (f"{scenario}_{model}_{speed_type}_{sfm}".replace(" ", "_") + "_trajectories.png")
        fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)



# Position heatmap
def plot_trajectory_heatmaps(
    sim_df,
    scenario,
    output_dir,
    bins=50,
    fixed_vmax=None
):
    """
    Density heatmaps of occupied positions during trajectories.

    - If there is one pedestrian:
        -> cyclist heatmap only

    - If there are multiple pedestrians:
        -> cyclist heatmap
        -> pedestrian heatmap containing all pedestrians

    A common horizontal colorbar is displayed below the plots.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configurations = sim_df[
        ["model", "speed_type", "sfm"]
    ].drop_duplicates()

    for _, config in configurations.iterrows():

        model = config["model"]
        speed_type = config["speed_type"]
        sfm = config["sfm"]

        subset = sim_df[
            (sim_df["model"] == model) &
            (sim_df["speed_type"] == speed_type) &
            (sim_df["sfm"] == sfm)
        ].copy()

        if subset.empty:
            continue

        if "pedestrian_id" in subset.columns:
            n_pedestrians = subset["pedestrian_id"].nunique()
        else:
            n_pedestrians = 1

        if n_pedestrians == 1:
            fig, ax = plt.subplots(figsize=(8, 7))
            axes = [ax]
        else:
            fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        # Some space for the colorbar
        fig.subplots_adjust(top=0.40, bottom=0.15, wspace=0.25)

        h_cyc = None
        h_ped = None

        # Cyclist
        x_cyc = subset["cyclist_x"].dropna()
        y_cyc = subset["cyclist_y"].dropna()

        if len(x_cyc) > 0 and len(y_cyc) > 0:

            h_cyc = axes[0].hist2d(
                x_cyc,
                y_cyc,
                bins=bins,
                cmap="YlOrRd",
                vmin=0,
                vmax=fixed_vmax
            )

            axes[0].set_title("Cyclist")
            axes[0].set_xlabel("X (m)")
            axes[0].set_ylabel("Y (m)")

            axes[0].invert_yaxis()

            # little zoom
            range_x = x_cyc.max() - x_cyc.min()
            range_y = y_cyc.max() - y_cyc.min()

            margin_x = max(range_x * 0.10, 0.5)
            margin_y = max(range_y * 0.10, 0.5)

            axes[0].set_xlim(
                x_cyc.min() - margin_x,
                x_cyc.max() + margin_x
            )

            axes[0].set_ylim(
                y_cyc.max() + margin_y,
                y_cyc.min() - margin_y
            )

            axes[0].set_aspect(
                "equal",
                adjustable="box"
            )

        # Pedestrians
        if n_pedestrians > 1:

            x_ped = subset["pedestrian_x"].dropna()
            y_ped = subset["pedestrian_y"].dropna()

            if len(x_ped) > 0 and len(y_ped) > 0:

                h_ped = axes[1].hist2d(
                    x_ped,
                    y_ped,
                    bins=bins,
                    cmap="YlOrRd",
                    vmin=0,
                    vmax=fixed_vmax
                )

                axes[1].set_title(f"Pedestrians ({n_pedestrians})")
                axes[1].set_xlabel("X (m)")
                axes[1].set_ylabel("Y (m)")
                axes[1].invert_yaxis()

                # little zoom
                range_x = x_ped.max() - x_ped.min()
                range_y = y_ped.max() - y_ped.min()

                margin_x = max(range_x * 0.10, 0.5)
                margin_y = max(range_y * 0.10, 0.5)

                axes[1].set_xlim(
                    x_ped.min() - margin_x,
                    x_ped.max() + margin_x
                )

                axes[1].set_ylim(
                    y_ped.max() + margin_y,
                    y_ped.min() - margin_y
                )

                axes[1].set_aspect(
                    "equal",
                    adjustable="box"
                )


        # horizontal common colobar
        heatmap = h_cyc if h_cyc is not None else h_ped
        if heatmap is not None:
            # Position [left, bottom, width, height]
            cbar_ax = fig.add_axes(
                [0.25, 0.075, 0.50, 0.025]
            )
            cbar = fig.colorbar(
                heatmap[3],
                cax=cbar_ax,
                orientation="horizontal"
            )
            cbar.set_label("Number of observations")

        # title
        if speed_type == "fixed" and fixed_vmax is not None:
            title = (
                f"Trajectories for {scenario} — "
                f"{model} model — "
                f"{speed_type} vmax = {fixed_vmax}m/ — "
                f"{sfm} - Position heatmap (20 simulations)"
            )
        else:
            title = (
                f"Trajectories for {scenario} — "
                f"{model} model — "
                f"{speed_type} vmax — "
                f"{sfm} - Position heatmap (20 simulations)"
            )

        fig.suptitle(title, y=0.50)

        filename = (
            f"{scenario}_{model}_{speed_type}_{sfm}"
            .replace(" ", "_")
            + "_heatmap.png"
        )

        fig.savefig(
            output_dir / filename,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)


#################################


# Stats per simulation
def compute_simulation_metrics(sim_df):

    results = []

    for sim_id, g in sim_df.groupby("simulation_id"):

        g = g.sort_values("time_rel")

        # min dist
        idx_min = g["distance"].idxmin()
        distance_min = g.loc[idx_min, "distance"]
        angle_at_min = g.loc[idx_min, "app_angle"]

        # mean relative spead
        relative_speed_mean = g["relative_speed"].mean()

        # mean speeds
        cyclist_speed_mean = g["cyclist_speed"].mean()
        pedestrian_speed_mean = g["pedestrian_speed"].mean()

        # cyclist vmax
        cyclist_speed_max = g["cyclist_speed"].max()

        # mean approach angle
        angle_mean = g["app_angle"].mean()

        # mean distance
        distance_mean = g["distance"].mean()

        # mean direction of the cyclist
        # Careful : circular mean because rad unit
        heading_rad = np.deg2rad(g["cyclist_heading"])
        mean_heading = np.rad2deg(np.arctan2(np.mean(np.sin(heading_rad)), np.mean(np.cos(heading_rad))))

        # speed variation
        cyclist_dv = g["cyclist_delta_v"].dropna()
        pedestrian_dv = g["pedestrian_delta_v"].dropna()

        cyclist_acc = g["cyclist_acceleration"].dropna()
        pedestrian_acc = g["pedestrian_acceleration"].dropna()

        cyclist_delta_v_mean = cyclist_dv.mean()
        pedestrian_delta_v_mean = pedestrian_dv.mean()

        cyclist_delta_v_abs_mean = cyclist_dv.abs().mean()
        pedestrian_delta_v_abs_mean = pedestrian_dv.abs().mean()

        cyclist_acc_mean = cyclist_acc.mean()
        pedestrian_acc_mean = pedestrian_acc.mean()

        cyclist_acc_mean_positive = cyclist_acc[cyclist_acc > 0].mean()
        pedestrian_acc_mean_positive = pedestrian_acc[pedestrian_acc > 0].mean()

        cyclist_dec_mean = cyclist_acc[cyclist_acc < 0].mean()
        pedestrian_dec_mean = pedestrian_acc[pedestrian_acc < 0].mean()

        results.append({
            "simulation_id": sim_id,
            "distance_min": distance_min,
            "angle_at_min": angle_at_min,
            "relative_speed_mean": relative_speed_mean,
            "cyclist_speed_mean": cyclist_speed_mean,
            "pedestrian_speed_mean": pedestrian_speed_mean,
            "cyclist_speed_max": cyclist_speed_max,
            "distance_mean": distance_mean,
            "angle_mean": angle_mean,
            "heading_mean": mean_heading,

            "cyclist_delta_v_mean": cyclist_delta_v_mean,
            "pedestrian_delta_v_mean": pedestrian_delta_v_mean,

            "cyclist_delta_v_abs_mean": cyclist_delta_v_abs_mean,
            "pedestrian_delta_v_abs_mean": pedestrian_delta_v_abs_mean,

            "cyclist_acc_mean": cyclist_acc_mean,
            "pedestrian_acc_mean": pedestrian_acc_mean,

            "cyclist_acc_mean_positive": cyclist_acc_mean_positive,
            "pedestrian_acc_mean_positive": pedestrian_acc_mean_positive,

            "cyclist_dec_mean": cyclist_dec_mean,
            "pedestrian_dec_mean": pedestrian_dec_mean,
        })

    return pd.DataFrame(results)



# PLOT : MEAN distance and approcah angle PER configuration AND for ALL configurations
def plot_distance_min_angle(metrics, scenario, output_dir, real_result=None):
    """
    Analyzes minimum distance / approach angle.

    1. One plot per configuration:
    - 20 simulations
    - mean of the 20 simulations

    2. One global plot:
    - one mean per configuration
    - comparison of all configurations in the scenario
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configurations = metrics[["model", "speed_type", "sfm"]].drop_duplicates()

    # Pour stocker les moyennes de chaque configuration
    configuration_means = []

    # ONE GRAPH PER CONFIGURATION
    for _, config in configurations.iterrows():

        model = config["model"]
        speed_type = config["speed_type"]
        sfm = config["sfm"]
        subset = metrics[(metrics["model"] == model) & (metrics["speed_type"] == speed_type) & (metrics["sfm"] == sfm)].copy()

        # Mean of 20 simualtion
        mean_distance = subset["distance_min"].mean()
        mean_angle = subset["angle_at_min"].mean()

        configuration_means.append({
            "model": model,
            "speed_type": speed_type,
            "sfm": sfm,
            "mean_distance_min": mean_distance,
            "mean_angle_at_min": mean_angle
        })

        fig, ax = plt.subplots(figsize=(8, 6))

        # 20 simulations
        ax.scatter(subset["angle_at_min"], subset["distance_min"], s=180, alpha=0.7, label="20 simulations")

        # Mean
        ax.scatter(
            mean_angle,
            mean_distance,
            s=220,
            marker="X",
            linewidth=1.5,
            label=(f"Mean (angle = {mean_angle:.1f}°, distance = {mean_distance:.2f} m)"))

        
        if (real_result is not None and not np.isnan(real_result["distance_min"]) and not np.isnan(real_result["angle_at_min"]) ): 
            real_distance = real_result["distance_min"] 
            real_angle = real_result["angle_at_min"] 
            ax.scatter(real_angle, real_distance, 
                       s=280, marker="*",
                       color="red",
                       linewidths=1.5, 
                       zorder=20, 
                       label=( f"Real " f"(angle = {real_angle:.1f}°, " f"distance = {real_distance:.2f} m)" ))
            ax.annotate( f"Real\n" f"{real_angle:.1f}°\n" f"{real_distance:.2f} m", 
                        xy=(real_angle, real_distance), 
                        xytext=(12, 12), 
                        textcoords="offset points", 
                        fontsize=12,
                        fontweight="bold", 
                        color="red", 
                        ha="left", 
                        va="bottom", 
                        bbox=dict( 
                            boxstyle="round,pad=0.3",
                            facecolor="white", 
                            edgecolor="red", 
                            alpha=0.9))

        # Area of lateral approcah angle
        ax.axvspan(75,105,alpha=0.15, label="75–105° area", color="deepskyblue")
        ax.set_xlabel("Approach angle at minimal distance (°)", fontsize=15)
        ax.set_ylabel("Minimal distance (m)", fontsize=15)
        ax.set_title(f"{scenario} — {model} — {speed_type} — {sfm} - \nMin distance and approach angle", fontsize=15, fontweight="bold")
        ax.tick_params(axis="both", labelsize=13)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=12)
        fig.tight_layout()
        filename = (f"{scenario}_{model}_{speed_type}_{sfm}".replace(" ", "_") + "_distance_min_angle.png")
        fig.savefig(output_dir / filename, dpi=300)
        plt.close(fig)

        print(f"{model} | {speed_type} | {sfm} : "
            f"mean distance_min = {mean_distance:.3f} m, "
            f"mean angle_at_min = {mean_angle:.2f}°")


    # GLOBAL GRAPH WITH MEANS
    means_df = pd.DataFrame(configuration_means)
    fig, ax = plt.subplots(figsize=(11, 8))

    model_markers = {
        "NES": "s",
        "Moussaid": "o"
    }

    sfm_colors = {
        "SFM": "royalblue",
        "Without SFM": "tab:orange"
    }

    for _, row in means_df.iterrows():
        model = row["model"]
        speed_type = row["speed_type"]
        sfm = row["sfm"]

        marker = model_markers.get(model, "o")
        color = sfm_colors.get(sfm, "black")

        # Fixed = filled
        # Random = empty
        if speed_type == "Fixed":
            facecolor = color
        else:
            facecolor = "white"

        ax.scatter(
            row["mean_angle_at_min"],
            row["mean_distance_min"],
            s=300,
            marker=marker,
            facecolors=facecolor,
            edgecolors=color,
            linewidths=2,
            zorder=5,
            alpha=0.85
        )

        # Values
        annotation = (
            f"{row['mean_angle_at_min']:.1f}°\n"
            f"{row['mean_distance_min']:.2f} m"
        )

        ax.annotate(
            annotation,
            (row["mean_angle_at_min"], row["mean_distance_min"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=13,
            fontweight="bold",
            ha="left",
            va="bottom"
        )

    if (real_result is not None and not np.isnan(real_result["distance_min"]) and not np.isnan(real_result["angle_at_min"])): 
        real_distance = real_result["distance_min"] 
        real_angle = real_result["angle_at_min"] 
        ax.scatter(real_angle, real_distance, 
                   s=320, 
                   marker="*", 
                   color="red",
                   linewidths=1.8,
                   zorder=20, 
                   label="Real scenario") 
        ax.annotate(f"Real\n" f"{real_angle:.1f}°\n" f"{real_distance:.2f} m", 
                    xy=(real_angle, real_distance), 
                    xytext=(12, 12), 
                    textcoords="offset points", 
                    fontsize=13, 
                    fontweight="bold", 
                    color="red", 
                    ha="left", 
                    va="bottom", 
                    bbox=dict(boxstyle="round,pad=0.3", 
                              facecolor="white", 
                              edgecolor="red", 
                              alpha=0.9))

    ax.axvspan(75, 105,alpha=0.15,label="75–105° area", color="deepskyblue")

    legend_elements = [
        # Model
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=12,
            label="Moussaid"
        ),

        Line2D(
            [0], [0],
            marker="s",
            linestyle="None",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=12,
            label="NES"
        ),

        # Speed
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=12,
            label="Fixed"
        ),

        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=12,
            label="Random"
        ),

        # SFM
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor="royalblue",
            markeredgecolor="royalblue",
            markersize=12,
            label="SFM"
        ),

        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor="tab:orange",
            markeredgecolor="tab:orange",
            markersize=12,
            label="Without SFM"
        ),

        # Real
        Line2D(
            [0], [0], 
            marker="*", 
            linestyle="None", 
            markerfacecolor="red",
            markeredgecolor="red",
            markersize=15, 
            label="Real scenario"),

        # Area of lateral approach angle
        Line2D(
            [0], [0],
            color="deepskyblue",
            linewidth=8,
            alpha=0.3,
            label="75–105° area"
        )
    ]

    ax.legend(handles=legend_elements,loc="best",title="Configuration", fontsize=12, title_fontsize=13)
    ax.set_xlabel("Mean approach angle at minimal distance (°)", fontsize=15)
    ax.set_ylabel("Mean minimal distance (m)", fontsize=15)
    ax.set_title(f"{scenario} —\nMean min distance / approach angle by configuration (20 simulations)", fontsize=15, fontweight="bold")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{scenario}_distance_min_angle_configuration_means.png",dpi=300)
    ax.tick_params(axis="both", labelsize=13)
    plt.close(fig)
    means_df.to_csv(output_dir / "distance_min_angle_configuration_means.csv",index=False)
    return means_df



def plot_mean_distance_angle_all_configurations(sim_df, scenario, output_dir):
    """
    Plots the mean distance and approach angle curves for all configurations
    of a scenario on the same graph.

    For each configuration:
        - mean over the 20 simulations
        - mean distance: left Y-axis
        - mean angle: right Y-axis
        - same color for distance and angle
        - vertical line at the minimum of the mean distance
        - display of the angle at this position
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configurations = (
        sim_df[["model", "speed_type", "sfm"]]
        .drop_duplicates()
        .sort_values(["model", "speed_type", "sfm"])
    )

    fig, ax_dist = plt.subplots(figsize=(14, 8))
    ax_angle = ax_dist.twinx()

    # Figure 2 : means + STD
    fig_std, ax_dist_std = plt.subplots(figsize=(14, 8))
    ax_angle_std = ax_dist_std.twinx()

    # different colors for ech configuration
    colors = plt.cm.tab10(np.linspace(0, 1, len(configurations)))

    annotation_offsets = [
        (10, 12),     # 1 : haut-droite
        (-45, 12),    # 2 : haut-gauche
        (10, -25),    # 3 : bas-droite
        (-45, -25),   # 4 : bas-gauche
        (25, 28),     # 5 : haut-droite éloigné
        (-60, 25),    # 6 : haut-gauche éloigné
        (40, -45),    # 7 : bas-droite éloigné
        (-70, -40),   # 8 : bas-gauche éloigné
    ]

    for i, (color, (_, config)) in enumerate(zip(colors, configurations.iterrows())):

        model = config["model"]
        speed_type = config["speed_type"]
        sfm = config["sfm"]

        subset = sim_df[
            (sim_df["model"] == model) &
            (sim_df["speed_type"] == speed_type) &
            (sim_df["sfm"] == sfm)
        ]

        distance_curves = interpolate_simulations(subset, "distance", n_points=100)
        angle_curves = interpolate_simulations(subset, "app_angle", n_points=100)

        if len(distance_curves) == 0 or len(angle_curves) == 0:
            continue

        # Mean of the 20 simulations
        distance_mean = distance_curves.mean(axis=0)
        angle_mean = angle_curves.mean(axis=0)

        distance_std = distance_curves.std(axis=0)
        angle_std = angle_curves.std(axis=0)

        x = np.linspace(0, 100, len(distance_mean))

        config_name = f"{model} / {speed_type} / {sfm}"

        min_idx = np.argmin(distance_mean)
        x_min = x[min_idx]
        distance_min = distance_mean[min_idx]
        angle_at_min = angle_mean[min_idx]

        # Distance curve
        ax_dist.plot(
            x,
            distance_mean,
            color=color,
            linewidth=2.5,
            linestyle="-",
            label=f"{config_name} — Distance = {distance_min:.2f} m"
        )

        # Anle curve
        ax_angle.plot(
            x,
            angle_mean,
            color=color,
            linewidth=2.0,
            linestyle="--",
            label=f"{config_name} — Angle = {angle_at_min:.2f}°"
        )

        # Vertical line going through min dist
        ax_dist.axvline(
            x=x_min,
            color=color,
            linestyle=":",
            linewidth=1.5,
            alpha=0.8
        )

        ax_dist.scatter(
            x_min,
            distance_min,
            color=color,
            s=55,
            zorder=10
        )

        # associated angle
        ax_angle.scatter(
            x_min,
            angle_at_min,
            color=color,
            s=55,
            marker="s",
            zorder=10
        )

        # annotations
        angle_offset = annotation_offsets[i % len(annotation_offsets)]
        distance_offset = (angle_offset[0], -angle_offset[1])

        ax_angle.annotate(
            f"{angle_at_min:.1f}°",
            xy=(x_min, angle_at_min),
            xytext=angle_offset,
            textcoords="offset points",
            fontsize=9,
            color=color,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor=color,
                alpha=0.85
            ),
            arrowprops=dict(
                arrowstyle="-",
                color=color,
                alpha=0.6
            )
        )

        ax_dist.annotate(
            f"{distance_min:.2f} m",
            xy=(x_min, distance_min),
            xytext=distance_offset,
            textcoords="offset points",
            fontsize=9,
            color=color,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor=color,
                alpha=0.85
            ),
            arrowprops=dict(
                arrowstyle="-",
                color=color,
                alpha=0.6
            )
        )

        print(
            f"{scenario} | {config_name} : "
            f"distance min moyenne = {distance_min:.2f} m "
            f"à {x_min:.1f}% de l'interaction | "
            f"angle = {angle_at_min:.1f}°"
        )

        # Figures with STD
        ax_dist_std.plot(
            x,
            distance_mean,
            color=color,
            linewidth=2.5,
            linestyle="-",
            label=f"{config_name} — Distance = {distance_min:.2f} m"
        )

        ax_angle_std.plot(
            x,
            angle_mean,
            color=color,
            linewidth=2.0,
            linestyle="--",
            label=f"{config_name} — Angle = {angle_at_min:.2f}°"
        )

        ax_dist_std.fill_between(
            x,
            distance_mean - distance_std,
            distance_mean + distance_std,
            color=color,
            alpha=0.15
        )

        ax_angle_std.fill_between(
            x,
            angle_mean - angle_std,
            angle_mean + angle_std,
            color=color,
            alpha=0.10
        )

        ax_dist_std.axvline(
            x=x_min,
            color=color,
            linestyle=":",
            linewidth=1.5,
            alpha=0.8
        )

        ax_dist_std.scatter(
            x_min,
            distance_min,
            color=color,
            s=55,
            zorder=10
        )

        ax_angle_std.scatter(
            x_min,
            angle_at_min,
            color=color,
            s=55,
            marker="s",
            zorder=10
        )

        ax_angle_std.annotate(
            f"{angle_at_min:.1f}°",
            xy=(x_min, angle_at_min),
            xytext=angle_offset,
            textcoords="offset points",
            fontsize=9,
            color=color,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor=color,
                alpha=0.85
            )
        )

        ax_dist_std.annotate(
            f"{distance_min:.2f} m",
            xy=(x_min, distance_min),
            xytext=distance_offset,
            textcoords="offset points",
            fontsize=9,
            color=color,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor=color,
                alpha=0.85
            )
        )


    ax_dist.set_xlabel("Normalized interaction time (%)")
    ax_dist.set_ylabel("Mean distance (m)")
    ax_angle.set_ylabel("Mean approach angle (°)")

    ax_dist.set_xlim(0, 100)

    ax_dist.set_title(
        f"{scenario} — Mean distance and approach angle "
        f"across configurations"
    )

    ax_dist.grid(alpha=0.3)

    ax_dist_std.set_xlabel("Normalized interaction time (%)")
    ax_dist_std.set_ylabel("Mean distance (m)")
    ax_angle_std.set_ylabel("Mean approach angle (°)")

    ax_dist_std.set_xlim(0, 100)

    ax_dist_std.set_title(
        f"{scenario} — Mean distance and approach angle "
        f"across configurations ± STD"
    )

    ax_dist_std.grid(alpha=0.3)
    handles_dist_std, labels_dist_std = (ax_dist_std.get_legend_handles_labels())
    handles_angle_std, labels_angle_std = (ax_angle_std.get_legend_handles_labels())

    ax_dist_std.legend(
        handles_dist_std + handles_angle_std,
        labels_dist_std + labels_angle_std,
        loc="upper left",
        fontsize=9
    )

    handles_dist, labels_dist = ax_dist.get_legend_handles_labels()
    handles_angle, labels_angle = ax_angle.get_legend_handles_labels()

    ax_dist.legend(
        handles_dist + handles_angle,
        labels_dist + labels_angle,
        loc="upper left",
        fontsize=9
    )

    fig.tight_layout()

    filename = (
        f"{scenario}_mean_distance_angle_all_configurations.png"
        .replace(" ", "_")
    )

    fig.savefig(
        output_dir / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


    handles_dist_std, labels_dist_std = ax_dist_std.get_legend_handles_labels()
    handles_angle_std, labels_angle_std = ax_angle_std.get_legend_handles_labels()

    ax_dist_std.legend(
        handles_dist_std + handles_angle_std,
        labels_dist_std + labels_angle_std,
        loc="upper left",
        fontsize=9
    )

    filename_std = (
        f"{scenario}_mean_distance_angle_all_configurations_std.png"
        .replace(" ", "_")
    )

    fig_std.tight_layout()

    fig_std.savefig(
        output_dir / filename_std,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig_std)




def interpolate_simulations(subset, variable, n_points=100):
    """
    Interpolate the 20 simulations on a temporal scale normalized from 0 to 1.
    To calculate mean curve despite different simulation durations.
    """

    curves = []
    for sim_id, g in subset.groupby("simulation_id"):
        g = g.sort_values("time_rel")
        if len(g) < 3:
            continue

        t = g["time_rel"].values
        y = g[variable].values
        if t[-1] == t[0]:
            continue

        t_norm = (t - t[0]) / (t[-1] - t[0])
        t_new = np.linspace(0, 1, n_points)
        y_new = np.interp(t_new, t_norm, y)
        curves.append(y_new)

    return np.array(curves)


# Per configuration
def plot_distance_angle_curve(subset, scenario, model, speed_type, sfm, output_path):
    """
    Plot the distance and the approach angle curves for 1 configuration.
    """

    distance_curves = interpolate_simulations(subset, "distance", n_points=100)

    angle_curves = interpolate_simulations(subset, "app_angle", n_points=100)

    if len(distance_curves) == 0 or len(angle_curves) == 0:
        return

    x = np.linspace(0, 100, distance_curves.shape[1])

    # Means and STD 
    distance_mean = distance_curves.mean(axis=0)
    distance_std = distance_curves.std(axis=0)

    angle_mean = angle_curves.mean(axis=0)
    angle_std = angle_curves.std(axis=0)

    # mean min dist
    min_idx = np.argmin(distance_mean)

    x_min = x[min_idx]
    distance_min = distance_mean[min_idx]
    angle_at_min = angle_mean[min_idx]

    fig, ax_dist = plt.subplots(figsize=(11, 7))
    ax_angle = ax_dist.twinx()

    # dist
    for curve in distance_curves:
        ax_dist.plot(
            x, curve,
            alpha=0.15,
            linewidth=1
        )

    ax_dist.plot(
        x,
        distance_mean,
        linewidth=2.5,
        label="Mean distance"
    )

    ax_dist.fill_between(
        x,
        distance_mean - distance_std,
        distance_mean + distance_std,
        alpha=0.2,
        label="Distance STD"
    )

    # angle
    for curve in angle_curves:
        ax_angle.plot(
            x, curve,
            alpha=0.15,
            linewidth=1,
            linestyle="--"
        )

    ax_angle.plot(
        x,
        angle_mean,
        linewidth=2.5,
        linestyle="--",
        label="Mean approach angle",
        color="tab:orange"
    )

    ax_angle.fill_between(
        x,
        angle_mean - angle_std,
        angle_mean + angle_std,
        alpha=0.15,
        label="Angle STD",
        color="tab:orange"
    )

    # min position
    ax_dist.axvline(
        x_min,
        linestyle=":",
        linewidth=2,
        alpha=0.8,
        color="red"
    )

    # point distance
    ax_dist.scatter(
        x_min,
        distance_min,
        s=70,
        zorder=10,
        color = "black"
    )

    # point angle
    ax_angle.scatter(
        x_min,
        angle_at_min,
        s=70,
        marker="s",
        zorder=10,
        color="black"
    )

    # annotations
    ax_dist.annotate(
        f"{distance_min:.2f} m",
        xy=(x_min, distance_min),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            alpha=0.85
        )
    )

    ax_angle.annotate(
        f"{angle_at_min:.1f}°",
        xy=(x_min, angle_at_min),
        xytext=(8, -25),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            alpha=0.85
        )
    )

    ax_dist.set_xlabel("Normalized interaction time (%)")
    ax_dist.set_ylabel("Distance (m)")
    ax_angle.set_ylabel("Approach angle (°)")

    ax_dist.set_xlim(0, 100)

    ax_dist.set_title(
        f"{scenario} — {model} — {speed_type} — {sfm}\n"
        f"Distance and approach angle"
    )

    ax_dist.grid(alpha=0.3)

    handles_dist, labels_dist = ax_dist.get_legend_handles_labels()
    handles_angle, labels_angle = ax_angle.get_legend_handles_labels()

    ax_dist.legend(
        handles_dist + handles_angle,
        labels_dist + labels_angle,
        loc="best",
        fontsize=9
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



# Mean curves
def plot_mean_curve(subset, variable, ylabel, title, output_path):

    curves = interpolate_simulations(subset, variable) # on a normalized time scale
    if len(curves) == 0:
        return

    mean = curves.mean(axis=0)
    std = curves.std(axis=0)

    x = np.linspace(0, 100, len(mean))

    fig, ax = plt.subplots(figsize=(9, 6))

    # all simulations
    for curve in curves:
        ax.plot(x, curve, alpha=0.15)

    # mean
    ax.plot(x, mean, linewidth=3,label="Mean trajectory")

    # with std
    ax.fill_between(x, mean - std, mean + std, alpha=0.2, label=f"STD")
    ax.set_xlabel("Normalized interaction time (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_all_mean_curves(sim_df, scenario, output_dir):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configurations = (
        sim_df[["model", "speed_type", "sfm"]]
        .drop_duplicates()
    )

    for _, config in configurations.iterrows():

        model = config["model"]
        speed_type = config["speed_type"]
        sfm = config["sfm"]

        subset = sim_df[
            (sim_df["model"] == model) &
            (sim_df["speed_type"] == speed_type) &
            (sim_df["sfm"] == sfm)
        ].copy()

        prefix = (
            f"{scenario}_{model}_{speed_type}_{sfm}"
            .replace(" ", "_")
        )

        # distance + angle curves
        plot_distance_angle_curve(
            subset,
            scenario,
            model,
            speed_type,
            sfm,
            output_dir / f"{prefix}_distance_angle_mean.png"
        )

        # relative speed
        plot_mean_curve(
            subset,
            "relative_speed",
            "Relative speed (m/s)",
            f"{scenario} — {model} — {speed_type} — {sfm}",
            output_dir / f"{prefix}_relative_speed_mean.png"
        )

        # agents' speeds
        # max number of pedestrians present in the simulation
        if "pedestrian_id" in subset.columns:
            n_pedestrians = subset["pedestrian_id"].nunique()
        elif "n_pedestrians" in subset.columns:
            n_pedestrians = subset["n_pedestrians"].max()
        else:
            n_pedestrians = 1

        # If 1 pedetsrian -> plot cyclist and pedestrian results on the same graph
        config_name = f"{scenario} — {model} — {speed_type} — {sfm}"

        if n_pedestrians <= 1:
            cyclist_curves = interpolate_simulations(subset, "cyclist_speed")
            pedestrian_curves = interpolate_simulations(subset, "pedestrian_speed")

            if len(cyclist_curves) == 0:
                continue

            x = np.linspace(0, 100, cyclist_curves.shape[1])
            fig, ax = plt.subplots(figsize=(9, 6))

            # cyclist
            cyclist_mean = cyclist_curves.mean(axis=0)
            cyclist_std = cyclist_curves.std(axis=0)

            for curve in cyclist_curves:
                ax.plot(
                    x, curve,
                    alpha=0.15
                )

            ax.plot(
                x,
                cyclist_mean,
                linewidth=3,
                label="Cyclist mean"
            )

            ax.fill_between(
                x,
                cyclist_mean - cyclist_std,
                cyclist_mean + cyclist_std,
                alpha=0.2,
                label="Cyclist STD"
            )

            # pedestrian 
            if len(pedestrian_curves) > 0:
                pedestrian_mean = pedestrian_curves.mean(axis=0)
                pedestrian_std = pedestrian_curves.std(axis=0)

                for curve in pedestrian_curves:
                    ax.plot(
                        x, curve,
                        alpha=0.15,
                        linestyle="--"
                    )

                ax.plot(
                    x,
                    pedestrian_mean,
                    linewidth=3,
                    linestyle="--",
                    label="Pedestrian mean"
                )

                ax.fill_between(
                    x,
                    pedestrian_mean - pedestrian_std,
                    pedestrian_mean + pedestrian_std,
                    alpha=0.2,
                    label="Pedestrian STD"
                )

            ax.set_xlabel("Normalized interaction time (%)")
            ax.set_ylabel("Speed (m/s)")
            ax.set_title(f"{config_name} — Agent speeds")
            ax.grid(alpha=0.3)
            ax.legend()

            fig.tight_layout()
            fig.savefig(
                output_dir / (
                    f"{scenario}_{model}_{speed_type}_{sfm}"
                    "_agent_speeds_mean.png"
                ),
                dpi=300
            )

            plt.close(fig)

        # If several pedestrians -> plot only cyclists curves
        else:
            plot_mean_curve(
                subset,
                "cyclist_speed",
                "Cyclist speed (m/s)",
                config_name,
                output_dir / (
                    f"{scenario}_{model}_{speed_type}_{sfm}"
                    "_cyclist_speed_mean.png"
                )
            )




# Distance of the start of the avoidance (increase of the approach angle)
def find_angle_onset(
    g,
    smooth_window=7,
    min_angle_increase=5.0,
    persistence=9
):
    """
    Estimates the distance at which the approach angle begins
    to increase significantly and persistently.

    Parameters:
        smooth_window : smoothing window
        min_angle_increase : minimum angle increase (°)
        persistence : number of consecutive points required
        -> calibrated empirically !
    """

    g = g.sort_values("time_rel").reset_index(drop=True).copy()

    if len(g) < smooth_window + persistence:
        return np.nan

    angle_smooth = (g["app_angle"].rolling(smooth_window,  center=True, min_periods=1).mean().values)

    min_pos = g["distance"].idxmin()

    if min_pos <= 1:
        return np.nan

    # We consider the angle evolution over multiple points
    # rather than relying only on the instantaneous derivative.
    for i in range(1, min_pos):

        end = min(i + persistence, min_pos)
        angle_change = (angle_smooth[end] - angle_smooth[i])

        # sufficiently large increase in the window
        if angle_change >= min_angle_increase:
            local_diff = np.diff(angle_smooth[i:end + 1])
            positive_ratio = np.mean(local_diff > 0)
            if positive_ratio >= 0.6:
                return g["distance"].iloc[i]

    return np.nan


def compute_angle_onset(sim_df):
    """
    Computes the distance at which the approach angle begins to increase for each simulation.

    For simulations with multiple pedestrians:
        - the onset is computed separately for each cyclist-pedestrian pair;
        - the pair with the globally minimum distance is selected;
        - the returned angle onset corresponds to that same pair.

    This ensures that the logic is consistent with that used for real-world scenarios.
    """

    results = []
    for (model, speed_type, sfm, sim_id), sim in sim_df.groupby(["model", "speed_type", "sfm", "simulation_id"]):
        pair_results = []
        # for each pedestrian-cyclist pair
        for (cyclist_id, pedestrian_id), g in sim.groupby(["cyclist_id", "pedestrian_id"]):
            g = g.sort_values("time_rel").reset_index(drop=True).copy()
            if g.empty:
                continue

            # min dist for this pair
            min_idx = g["distance"].idxmin()
            distance_min = g.loc[min_idx, "distance"]

            # Onset angle for this pair
            distance_onset = find_angle_onset(g)

            pair_results.append({
                "cyclist_id": cyclist_id,
                "pedestrian_id": pedestrian_id,
                "distance_min": distance_min,
                "angle_increase_distance": distance_onset
            })

        # if not valid pair
        if not pair_results:
            results.append({
                "model": model,
                "speed_type": speed_type,
                "sfm": sfm,
                "simulation_id": sim_id,
                "cyclist_id": np.nan,
                "pedestrian_id": np.nan,
                "distance_min": np.nan,
                "angle_increase_distance": np.nan
            })
            continue

        pair_results = pd.DataFrame(pair_results)

        # selection of the pair having the smallest distance
        min_idx = pair_results["distance_min"].idxmin()
        selected = pair_results.loc[min_idx]

        results.append({
            "model": model,
            "speed_type": speed_type,
            "sfm": sfm,
            "simulation_id": sim_id,
            "cyclist_id": selected["cyclist_id"],
            "pedestrian_id": selected["pedestrian_id"],
            "distance_min": selected["distance_min"],
            "angle_increase_distance": selected["angle_increase_distance"]
        })
    return pd.DataFrame(results)


def summarize_angle_onset(onset):
    """
    Computes statistics for the distance at which the approach angle
    starts to increase for each configuration.

    One row = one configuration.
    """

    summary = (
        onset
        .groupby(["model", "speed_type", "sfm"])
        ["angle_increase_distance"]
        .agg(["mean", "std", "median", "min", "max", "count"])
        .reset_index()
    )

    summary = summary.rename(columns={
        "mean": "mean_angle_increase_distance",
        "std": "std_angle_increase_distance",
        "median": "median_angle_increase_distance",
        "min": "min_angle_increase_distance",
        "max": "max_angle_increase_distance",
        "count": "n_simulations"
    })

    return summary


def plot_mean_approach_distance_by_configuration(
    onset_summary,
    scenario,
    output_dir,
    real_result=None
):
    """
    Plot the mean approach distance for each configuration.

    For each configuration:
        - mean distance at approach-angle increase
        - std
        - number of valid simulations

    The real scenario approach distance is also shown for comparison.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = onset_summary.copy()

    # Keep only configurations with valid values
    summary = summary.dropna(subset=["mean_angle_increase_distance"])

    if summary.empty:
        print("No valid approach-distance data to plot.")
        return

    summary["configuration"] = (summary["model"] + " / " + summary["speed_type"] + " / " + summary["sfm"])

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(summary))
    means = summary["mean_angle_increase_distance"].values
    stds = summary["std_angle_increase_distance"].fillna(0).values
    # Colors according to SFM
    sfm_colors = {
        "SFM": "royalblue",
        "Without SFM": "tab:orange"
    }

    # Markers according to model
    model_markers = {
        "NES": "s",
        "Moussaid": "o"
    }

    # Fixed = filled
    # Random = empty
    for i, (_, row) in enumerate(summary.iterrows()):
        model = row["model"]
        speed_type = row["speed_type"]
        sfm = row["sfm"]

        color = sfm_colors.get(sfm, "black")
        marker = model_markers.get(model, "o")

        if speed_type == "Fixed":
            facecolor = color
        else:
            facecolor = "white"

        # Mean + STD
        ax.errorbar(
            i,
            row["mean_angle_increase_distance"],
            yerr=row["std_angle_increase_distance"],
            fmt=marker,
            markersize=11,
            markerfacecolor=facecolor,
            markeredgecolor=color,
            markeredgewidth=2,
            ecolor=color,
            elinewidth=1.5,
            capsize=5,
            zorder=5
        )

        # Mean value annotation
        ax.annotate(
            f"{row['mean_angle_increase_distance']:.2f} m",
            (i, row["mean_angle_increase_distance"]),
            xytext=(12, 0),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor="none",
                alpha=0.85
            )
        )

    # Real scenario
    if (real_result is not None and not np.isnan(real_result["angle_increase_distance"])):
        real_distance = real_result["angle_increase_distance"]
        ax.axhline(
            real_distance,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Real scenario = {real_distance:.2f} m",
            zorder=2
        )

        ax.text(
            1.01,
            real_distance,
            f"{real_distance:.2f} m",
            transform=ax.get_yaxis_transform(),
            color="red",
            va="center",
            ha="left",
            fontsize=11,
            fontweight="bold"
        )

    legend_elements = [
        # Model
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=10,
            label="Moussaid"
        ),

        Line2D(
            [0], [0],
            marker="s",
            linestyle="None",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=10,
            label="NES"
        ),

        # Speed
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=10,
            label="Fixed"
        ),

        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=10,
            label="Random"
        ),

        # SFM
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor="royalblue",
            markeredgecolor="royalblue",
            markersize=10,
            label="SFM"
        ),

        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor="tab:orange",
            markeredgecolor="tab:orange",
            markersize=10,
            label="Without SFM"
        ),

        # Real
        Line2D(
            [0], [0],
            linestyle="--",
            linewidth=1.5,
            color="red",
            label="Real scenario"
        ),

        # STD
        Line2D(
            [0], [0],
            color="black",
            linewidth=1.5,
            label="Standard deviation"
        )
    ]

    ax.legend(
        handles=legend_elements,
        loc="best",
        title="Configuration",
        fontsize=10,
        title_fontsize=11
    )

    ax.set_xticks(x)
    ax.set_xticklabels(summary["configuration"], rotation=30, ha="right")
    ax.set_ylabel("Mean approach distance (m)", fontsize=15)
    ax.set_xlabel("Configuration", fontsize=15)

    ax.set_title(
        f"{scenario} —\nMean approach distance by configuration (20 simulations)",
        fontsize=15,
        fontweight="bold",
        pad=10
    )

    ax.tick_params(axis="both", labelsize=12)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    filename = (f"{scenario}_mean_approach_distance_by_configuration.png".replace(" ", "_"))
    fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # print(f"\n{scenario} — Mean approach distance by configuration:")

    # for _, row in summary.iterrows():
    #     print(
    #         f"{row['model']} | "
    #         f"{row['speed_type']} | "
    #         f"{row['sfm']} : "
    #         f"{row['mean_angle_increase_distance']:.3f} m "
    #         f"± {row['std_angle_increase_distance']:.3f} m "
    #         f"(n={int(row['n_simulations'])})"
    #     )

    # if (real_result is not None and not np.isnan(real_result["angle_increase_distance"])):
    #     print(f"Real scenario : {real_result['angle_increase_distance']:.3f} m")

    return summary


##################################
# For REAL SCENARIOS
def compute_approach_angle(g, fps):
    g = g.sort_values("time_step").reset_index(drop=True).copy()

    vx_c = g["cyc_x"].diff() * fps
    vy_c = g["cyc_y"].diff() * fps

    vx_p = g["ped_x"].diff() * fps
    vy_p = g["ped_y"].diff() * fps

    rx = g["ped_x"] - g["cyc_x"]
    ry = g["ped_y"] - g["cyc_y"]

    vrx = vx_c - vx_p
    vry = vy_c - vy_p
    norm_r = np.sqrt(rx**2 + ry**2)
    norm_v = np.sqrt(vrx**2 + vry**2)

    valid = (
        (norm_r > 0) &
        (norm_v > 0)
    )

    app_angle = np.full(len(g), np.nan)

    cos_theta = (
        (vrx * rx + vry * ry)
        / (norm_r * norm_v)
    )

    cos_theta = np.clip(
        cos_theta,
        -1.0,
        1.0
    )

    app_angle[valid] = np.degrees(
        np.arccos(cos_theta[valid])
    )

    g["app_angle"] = app_angle

    return g



def prepare_cyclist_pedestrian_pair(cyclist, pedestrian):
    """
    Builds the common trajectory of a cyclist and a pedestrian.

    The two agents are associated only when they are
    present at the same time step.
    """

    cyclist = cyclist[["time_step", "object_id", "x_m", "y_m"]].copy()

    pedestrian = pedestrian[["time_step", "object_id", "x_m", "y_m"]].copy()

    cyclist = cyclist.rename(columns={
        "object_id": "cyclist_id",
        "x_m": "cyc_x",
        "y_m": "cyc_y"
    })

    pedestrian = pedestrian.rename(columns={
        "object_id": "pedestrian_id",
        "x_m": "ped_x",
        "y_m": "ped_y"
    })

    # Association on the same timestep
    g = pd.merge(
        cyclist,
        pedestrian,
        on="time_step",
        how="inner"
    )

    if g.empty:
        return g

    # Cyclist-pedetsrian dist
    g["distance"] = np.sqrt(
        (g["ped_x"] - g["cyc_x"]) ** 2 +
        (g["ped_y"] - g["cyc_y"]) ** 2
    )

    # Relative time
    g = g.sort_values("time_step").reset_index(drop=True)

    g["time_rel"] = (g["time_step"] - g["time_step"].iloc[0])

    # Approach angle
    g = compute_approach_angle(g, 29.97)

    return g


def find_angle_onset_real_scenario(
    g,
    smooth_window=7,
    min_angle_increase=5.0,
    persistence=9
):
    
    g = g.sort_values("time_rel").reset_index(drop=True).copy()

    if len(g) < smooth_window + persistence:
        return np.nan

    angle_smooth = (g["app_angle"].rolling(smooth_window, center=True, min_periods=1).mean().values)

    # Min dist position
    min_pos = g["distance"].idxmin()

    if min_pos <= 1:
        return np.nan

    for i in range(1, min_pos):
        end = min(i + persistence, min_pos)
        angle_change = (angle_smooth[end] - angle_smooth[i])

        if angle_change >= min_angle_increase:
            local_diff = np.diff(angle_smooth[i:end + 1])
            positive_ratio = np.mean(local_diff > 0)
            if positive_ratio >= 0.6:
                return g["distance"].iloc[i]

    return np.nan


def compute_real_angle_onset_for_scenario(
    df,
    smooth_window=7,
    min_angle_increase=5.0,
    persistence=9
):
    cyclists = df[df["user_type"] == 2].copy()
    pedestrians = df[df["user_type"] == 1].copy()

    if cyclists.empty or pedestrians.empty:
        return {
            "distance_min": np.nan,
            "angle_increase_distance": np.nan,
            "cyclist_id": np.nan,
            "pedestrian_id": np.nan
        }
    
    # We assume one cyclist per scenario.
    # If multiple cyclists are present, the first one is selected.
    cyclist_ids = cyclists["object_id"].unique()

    if len(cyclist_ids) == 0:
        return {
            "distance_min": np.nan,
            "angle_increase_distance": np.nan,
            "cyclist_id": np.nan,
            "pedestrian_id": np.nan
        }

    cyclist_id = cyclist_ids[0]

    cyclist = cyclists[cyclists["object_id"] == cyclist_id]

    pair_results = []
    for pedestrian_id in pedestrians["object_id"].unique():
        pedestrian = pedestrians[pedestrians["object_id"] == pedestrian_id]
        g = prepare_cyclist_pedestrian_pair(cyclist, pedestrian)

        if g.empty:
            continue

        min_idx = g["distance"].idxmin()
        distance_min = g.loc[min_idx, "distance"]

        distance_onset = find_angle_onset_real_scenario(
            g,
            smooth_window=smooth_window,
            min_angle_increase=min_angle_increase,
            persistence=persistence
        )

        pair_results.append({
            "cyclist_id": cyclist_id,
            "pedestrian_id": pedestrian_id,
            "distance_min": distance_min,
            "angle_increase_distance": distance_onset
        })

    if not pair_results:
        return {
            "distance_min": np.nan,
            "angle_increase_distance": np.nan,
            "cyclist_id": cyclist_id,
            "pedestrian_id": np.nan
        }

    pair_results = pd.DataFrame(pair_results)

    # pair with the smallest distance (like for the simulated scenarios)
    min_idx = pair_results["distance_min"].idxmin()
    selected = pair_results.loc[min_idx]

    return {
        "distance_min": selected["distance_min"],
        "angle_increase_distance": selected[
            "angle_increase_distance"
        ],
        "cyclist_id": selected["cyclist_id"],
        "pedestrian_id": selected["pedestrian_id"]
    }


def summarize_real_angle_onset(results):
    summary = (
        results[
            "angle_increase_distance"
        ]
        .agg([
            "mean",
            "std",
            "median",
            "min",
            "max",
            "count"
        ])
        .to_frame()
        .T
    )

    summary = summary.rename(columns={
        "mean": "mean_angle_increase_distance",
        "std": "std_angle_increase_distance",
        "median": "median_angle_increase_distance",
        "min": "min_angle_increase_distance",
        "max": "max_angle_increase_distance",
        "count": "n_scenarios"
    })

    return summary




########################

# Analysis per vmax value

def speed_bins_analysis(metrics, output_dir):
    """
    Analyzes metric variations as a function of the cyclist's maximum speed.

    The analysis is performed only on Random simulations,
    as their maximum speed varies between simulations.

    Separate analysis for each configuration:
        - model: NES / Moussaid
        - sfm: SFM / Without SFM

    One observation = one simulation.

    Fixed speed bins:
        < 3 m/s
        3-4 m/s
        4–5 m/s
        5–6 m/s
        6–7 m/s
        > 7 m/s
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = metrics[metrics["speed_type"] == "Random"].copy()

    if metrics.empty:
        print("No Random simulations available for speed analysis.")
        return pd.DataFrame()

    variables = [
        "distance_min",
        "angle_at_min",
        "relative_speed_mean",
        "cyclist_speed_mean",
    ]

    bins = [0, 3, 4, 5, 6, 7, np.inf]
    labels = [
        "< 3 m/s",
        "3-4 m/s",
        "4–5 m/s",
        "5–6 m/s",
        "6–7 m/s",
        "> 7 m/s"
    ]

    metrics["speed_bin"] = pd.cut(
        metrics["cyclist_speed_max"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    all_results = []

    # One analysis per model + SFM
    for (model, sfm), subset in metrics.groupby(
        ["model", "sfm"]
    ):

        subset = subset.copy()
        # stats per bin
        grouped = (
            subset
            .groupby(
                "speed_bin",
                observed=True
            )[variables]
            .agg([
                "mean",
                "std",
                "median",
                "min",
                "max",
                "count"
            ])
            .reset_index()
        )

        grouped["model"] = model
        grouped["sfm"] = sfm

        all_results.append(grouped)
        filename = (f"{model}_{sfm}_speed_bins_statistics.csv".replace(" ", "_"))
        grouped.to_csv(output_dir / filename, index=False)

        # graphs
        for variable in variables:
            fig, ax = plt.subplots(figsize=(9, 6))

            data = []
            valid_labels = []

            for speed_bin in labels:
                values = subset[subset["speed_bin"].astype(str) == speed_bin][variable].dropna()

                if len(values) > 0:
                    data.append(values.values)
                    valid_labels.append(speed_bin)

            if not data:
                plt.close(fig)
                continue

            bp = ax.boxplot(
                data,
                labels=valid_labels,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(
                    color="darkorange",
                    linewidth=2
                ),
                whiskerprops=dict(
                    linewidth=1.2
                ),
                capprops=dict(
                    linewidth=1.2
                ),
                boxprops=dict(
                    linewidth=1.2
                )
            )

            colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(data)))

            for box, color in zip(bp["boxes"], colors):
                box.set_facecolor(color)
                box.set_alpha(0.75)

            # mean per bin
            means = [np.mean(values) for values in data]
            positions = np.arange(1, len(data) + 1)

            ax.scatter(
                positions,
                means,
                marker="D",
                s=55,
                color="black",
                edgecolor="black",
                linewidth=0.8,
                label="Mean",
                zorder=4
            )

            ax.set_xlabel("Cyclist desired speed (vmax)", fontsize=15)
            ax.set_ylabel(variable, fontsize=15)
            ax.tick_params(axis="both", labelsize=13)

            ax.set_title(
                f"{model} — {sfm} — Random speed\n"
                f"{variable} according to cyclist desired speed (vmax)",
                fontsize=15,
                fontweight="bold"
            )
            ax.grid(alpha=0.3)
            ax.legend(fontsize=12)
            fig.tight_layout()
            filename = (f"{model}_{sfm}_{variable}_speed_bins_random.png".replace(" ", "_"))
            fig.savefig(output_dir / filename, dpi=300)
            plt.close(fig)

    if all_results:
        all_results_df = pd.concat(all_results, ignore_index=True)
        all_results_df.to_csv(output_dir / "speed_bins_random_all_configurations.csv", index=False)
    else:
        all_results_df = pd.DataFrame()

    return all_results_df


# comparison of variables according to fixed / random vmax
def compare_speed_types(sim_df, scenario, output_dir, fixed_vmax):
    """
    Compares Fixed vs Random cyclist speed.

    Comparison is performed separately for each:
        - model (NES / Moussaid)
        - sfm (SFM / Without SFM)

    For each configuration:
        - 20 Fixed simulations
        - 20 Random simulations

    Statistics are computed across simulations,
    not across time-series rows.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variables = [
        "distance_min",
        "angle_at_min",
        "relative_speed_mean",
        "distance_mean",
        "cyclist_speed_mean",
        "pedestrian_speed_mean",
    ]

    # per simulation
    all_metrics = []

    for (model, speed_type, sfm), subset in sim_df.groupby(["model", "speed_type", "sfm"]):
        metrics = compute_simulation_metrics(subset)
        metrics["model"] = model
        metrics["speed_type"] = speed_type
        metrics["sfm"] = sfm
        all_metrics.append(metrics)

    metrics = pd.concat(all_metrics, ignore_index=True)

    summary = (
        metrics
        .groupby(["model", "sfm", "speed_type"])[variables]
        .agg(["mean", "std", "median", "min", "max", "count"])
    )

    summary.to_csv(output_dir / f"{scenario}_fixed_random_statistics.csv")

    # boxplots
    for model in metrics["model"].unique():

        for sfm in metrics["sfm"].unique():

            subset = metrics[(metrics["model"] == model) & (metrics["sfm"] == sfm)].copy()

            # verify if Fixed and Random files exist
            if not {"Fixed", "Random"}.issubset(set(subset["speed_type"].unique())):
                continue

            for variable in variables:
                fig, ax = plt.subplots(figsize=(8, 6))
                subset.boxplot(
                    column=variable,
                    by="speed_type",
                    ax=ax,
                    positions=[1, 2]
                )

                ax.set_xlabel("Cyclist speed type")
                ax.set_ylabel(variable)
                ax.set_title(
                    f"{scenario} — {model} — {sfm}\n"
                    f"Fixed ({fixed_vmax} m/s) vs Random"
                )
                ax.grid(alpha=0.3)
                plt.suptitle("")
                fig.tight_layout()

                filename = (
                    f"{scenario}_{model}_{sfm}_"
                    f"{variable}_fixed_random.png"
                    .replace(" ", "_")
                )

                fig.savefig(output_dir / filename, dpi=300)
                plt.close(fig)

    mean_comparison = (
        metrics
        .groupby(["model", "sfm", "speed_type"])[variables]
        .mean()
        .reset_index()
    )

    mean_comparison.to_csv(
        output_dir /
        f"{scenario}_fixed_random_means.csv",
        index=False
    )

    return metrics, summary, mean_comparison


# heatmaps acceleration / decelration according to position
def plot_speed_variation_heatmaps_old(
    sim_df,
    scenario,
    output_dir,
    bins=50
):
    """
    Spatial heatmaps of speed variation.

    One figure per configuration:
        - cyclist
        - pedestrian

    Color = mean acceleration/deceleration
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configurations = sim_df[["model", "speed_type", "sfm"]].drop_duplicates()

    for _, config in configurations.iterrows():
        model = config["model"]
        speed_type = config["speed_type"]
        sfm = config["sfm"]

        subset = sim_df[
            (sim_df["model"] == model) &
            (sim_df["speed_type"] == speed_type) &
            (sim_df["sfm"] == sfm)
        ].copy()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # cyclist
        h1 = axes[0].hist2d(
            subset["cyclist_x"],
            subset["cyclist_y"],
            bins=bins,
            weights=subset["cyclist_acceleration"]
        )

        axes[0].set_title("Cyclist")
        axes[0].set_xlabel("X (m)")
        axes[0].set_ylabel("Y (m)")
        axes[0].axis("equal")

        fig.colorbar(
            h1[3],
            ax=axes[0],
            label="Acceleration (m/s^2)"
        )

        # pedestrian
        h2 = axes[1].hist2d(
            subset["pedestrian_x"],
            subset["pedestrian_y"],
            bins=bins,
            weights=subset["pedestrian_acceleration"]
        )

        axes[1].set_title("Pedestrian")
        axes[1].set_xlabel("X (m)")
        axes[1].set_ylabel("Y (m)")
        axes[1].axis("equal")

        fig.colorbar(
            h2[3],
            ax=axes[1],
            label="Acceleration (m/s^2)"
        )

        fig.suptitle(
            f"{scenario} — {model} — "
            f"{speed_type} — {sfm}"
        )

        fig.tight_layout()

        filename = (
            f"{scenario}_{model}_{speed_type}_{sfm}"
            .replace(" ", "_")
            + "_speed_variation_heatmap.png"
        )

        fig.savefig(output_dir / filename,  dpi=300)
        plt.close(fig)


def plot_speed_variation_heatmaps(
    sim_df,
    scenario,
    output_dir,
    bins=50
):
    """
    Spatial heatmap of cyclist speed variation for interactions
    involving exactly one pedestrian.

    The cyclist acceleration is represented as a spatial heatmap.
    The mean pedestrian trajectory is superimposed on the same plot.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configurations = sim_df[["model", "speed_type", "sfm"]].drop_duplicates()
    for _, config in configurations.iterrows():
        model = config["model"]
        speed_type = config["speed_type"]
        sfm = config["sfm"]

        subset = sim_df[
            (sim_df["model"] == model) &
            (sim_df["speed_type"] == speed_type) &
            (sim_df["sfm"] == sfm)
        ].copy()

        # keep the interactions with 1 pedestrian
        n_pedestrians = subset["pedestrian_id"].nunique()

        if n_pedestrians != 1:
            continue

        mean_acc, xedges, yedges = mean_heatmap(
            subset["cyclist_x"].values,
            subset["cyclist_y"].values,
            subset["cyclist_acceleration"].values,
            bins=bins
        )

        fig, ax = plt.subplots(figsize=(9, 7))

        mesh = ax.pcolormesh(
            xedges,
            yedges,
            mean_acc.T,
            shading="auto"
        )

        fig.colorbar(mesh, ax=ax, label="Cyclist acceleration (m/s^2)")

        pedestrian_mean = (
            subset
            .groupby("time_rel")[
                ["pedestrian_x", "pedestrian_y"]
            ]
            .mean()
            .sort_index()
        )

        ax.plot(
            pedestrian_mean["pedestrian_x"],
            pedestrian_mean["pedestrian_y"],
            linewidth=3,
            label="Mean pedestrian trajectory",
            color="red"
        )

        ax.set_title(
            f"{scenario} — {model} — "
            f"{speed_type} — {sfm}"
        )

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.axis("equal")
        ax.legend()

        fig.tight_layout()

        filename = (
            f"{scenario}_{model}_{speed_type}_{sfm}"
            .replace(" ", "_")
            + "_cyclist_speed_variation_heatmap.png"
        )

        fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight")

        plt.close(fig)


def mean_heatmap(x, y, values, bins=50):
    mean, xedges, yedges = np.histogram2d(x, y, bins=bins, weights=values)
    count, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])
    mean = np.divide(
        mean,
        count,
        out=np.full_like(mean, np.nan, dtype=float),
        where=count > 0
    )
    return mean, xedges, yedges


# heatmap combination speed variation pedestrian-cyclist
def plot_joint_speed_variation_heatmap(
    sim_df,
    scenario,
    output_dir,
    bins=40
):
    """
    Heatmap of pedestrian vs cyclist speed variation.

    X = cyclist acceleration
    Y = pedestrian acceleration
    Color = number of observations
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configurations = sim_df[["model", "speed_type", "sfm"]].drop_duplicates()

    for _, config in configurations.iterrows():

        model = config["model"]
        speed_type = config["speed_type"]
        sfm = config["sfm"]

        subset = sim_df[
            (sim_df["model"] == model) &
            (sim_df["speed_type"] == speed_type) &
            (sim_df["sfm"] == sfm)
        ].copy()

        subset = subset.dropna(subset=["cyclist_acceleration", "pedestrian_acceleration"])

        fig, ax = plt.subplots(figsize=(8, 7))

        h = ax.hist2d(
            subset["cyclist_acceleration"],
            subset["pedestrian_acceleration"],
            bins=bins
        )

        ax.axvline(0, linestyle="--", linewidth=1)
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_xlabel("Cyclist acceleration (m/s^2)")
        ax.set_ylabel("Pedestrian acceleration (m/s^2)")
        ax.set_title(
            f"{scenario} — {model} — "
            f"{speed_type} — {sfm}"
        )

        fig.colorbar(
            h[3],
            ax=ax,
            label="Number of observations"
        )

        fig.tight_layout()

        filename = (
            f"{scenario}_{model}_{speed_type}_{sfm}"
            .replace(" ", "_")
            + "_joint_speed_variation_heatmap.png"
        )

        fig.savefig(output_dir / filename, dpi=300)
        plt.close(fig)


# export metrics
def save_metrics(sim_df, output_dir):

    output_dir = Path(output_dir)

    all_metrics = []

    for config, subset in sim_df.groupby(["model", "speed_type", "sfm"]):

        model, speed_type, sfm = config
        metrics = compute_simulation_metrics(subset)

        metrics["model"] = model
        metrics["speed_type"] = speed_type
        metrics["sfm"] = sfm

        all_metrics.append(metrics)

    metrics = pd.concat(all_metrics, ignore_index=True)

    onset = compute_angle_onset(sim_df)
    metrics = metrics.merge(
        onset[
            ["model",
            "speed_type",
            "sfm",
            "simulation_id",
            "angle_increase_distance"]
        ],
        on=["model",
            "speed_type",
            "sfm",
            "simulation_id"],
        how="left"
    )

    metrics.to_csv(output_dir / "simulation_metrics.csv", index=False)

    summary = (metrics.groupby(["model", "speed_type", "sfm"])
        [[
            "distance_min",
            "angle_at_min",
            "angle_increase_distance",
            "relative_speed_mean",
            "cyclist_speed_mean",
            "pedestrian_speed_mean",
            "cyclist_speed_max",
            "distance_mean",
            "angle_mean",
        ]].agg(["mean", "std", "median"]))

    summary.to_csv(output_dir / "simulation_summary.csv")

    onset_summary = summarize_angle_onset(onset)
    onset_summary.to_csv( output_dir / "angle_onset_summary.csv", index=False)


    return metrics, summary, onset, onset_summary


#####################

# ANOVA study

def run_two_way_anova(
    metrics,
    output_dir,
    scenario,
    variables=None
):
    """
    Performs a two-way ANOVA for each metric.

    Factors:
        - model: Moussaid / NES
        - speed_type: Fixed / Random

    Model:
        metric ~ C(model) + C(speed_type) + C(model):C(speed_type)

    One observation = one simulation.

    This ANOVA is appropriate for scenarios where SFM is constant
    (i.e. group scenarios with only SFM and no Without SFM).
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if variables is None:
        variables = [
            "distance_min",
            "angle_at_min",
            "angle_increase_distance",
            "relative_speed_mean",
            "cyclist_speed_mean"
        ]

    required_columns = [
        "model",
        "speed_type",
        "simulation_id"
    ]

    missing = [
        col
        for col in required_columns
        if col not in metrics.columns
    ]

    if missing:
        raise ValueError(f"Missing columns for two-way ANOVA: {missing}")

    data = metrics.copy()

    if data.empty:
        print(f"No data available for ANOVA — {scenario}.")
        return pd.DataFrame()

    print(f"\nRunning TWO-WAY ANOVA — {scenario}")

    print("Factors:")
    print("Model : Moussaid / NES")
    print("Speed type : Fixed / Random")

    all_results = []

    for variable in variables: # one ANOVA per metric
        if variable not in data.columns:
            print(f"Warning: {variable} not found in metrics, Skipping.")
            continue

        subset = data[
            ["simulation_id",
            "model",
            "speed_type",
            variable]].dropna(subset=[variable]).copy()

        if subset.empty:
            print(f"{variable}: no valid observations.")
            continue

        subset["model"] = pd.Categorical(subset["model"],
            categories=["Moussaid", "NES"], ordered=True)

        subset["speed_type"] = pd.Categorical(subset["speed_type"],
            categories=["Fixed", "Random"], ordered=True)

        # Remove rows with unknown factor values (just in case)
        subset = subset.dropna(subset=["model", "speed_type"])
        n_models = subset["model"].nunique()
        n_speed_types = subset["speed_type"].nunique()

        if n_models < 2:
            print(f"{variable}: not enough model levels for two-way ANOVA.")
            continue

        if n_speed_types < 2:
            print(f"{variable}: not enough speed-type levels for two-way ANOVA.")
            continue

        combinations = (subset.groupby(["model", "speed_type"], observed=True).size())
        expected_combinations = 4

        if len(combinations) < expected_combinations:
            print(
                f"Warning: {variable}: only "
                f"{len(combinations)}/"
                f"{expected_combinations} "
                f"model x speed combinations are present."
            )
            print(combinations)

        if len(subset) < 4:
            print(f"{variable}: not enough observations for two-way ANOVA.")
            continue

        formula = (f"{variable} ~ C(model) + C(speed_type) + C(model):C(speed_type)")

        try:
            model_fit = ols(formula, data=subset).fit()
            anova_table = sm.stats.anova_lm(model_fit, typ=2)

        except Exception as e:
            print(f"ANOVA failed for {variable}: {e}")
            continue

        for effect, row in anova_table.iterrows():
            p_value = row["PR(>F)"]
            all_results.append({
                "scenario": scenario,
                "variable": variable,
                "effect": effect,
                "sum_sq": row["sum_sq"],
                "df": row["df"],
                "F": row["F"],
                "p_value": p_value,
                "significant": (p_value < 0.05 if not pd.isna(p_value) else False),
                "n": len(subset),
                "n_moussaid": ((subset["model"] == "Moussaid").sum()),
                "n_nes": ((subset["model"] == "NES").sum()),
                "n_fixed": ((subset["speed_type"] == "Fixed").sum()),
                "n_random": ((subset["speed_type"] == "Random").sum())
            })

    results = pd.DataFrame(all_results)
    if results.empty:
        print("No ANOVA results.")
        return results
    results.to_csv(output_dir / f"{scenario}_anova_two_way_model_speed.csv", index=False)

    significant = results[results["significant"]].copy()
    significant.to_csv(output_dir / f"{scenario}_anova_two_way_model_speed_significant.csv", index=False)

    for variable in results["variable"].unique():
        print(variable)
        variable_results = results[results["variable"] == variable]
        for _, row in variable_results.iterrows():
            effect = row["effect"]
            p_value = row["p_value"]
            F_value = row["F"]
            if pd.isna(p_value):
                p_text = "p = NaN"
            else:
                p_text = (f"p = {p_value:.4g}")

            if pd.isna(F_value):
                f_text = "F = NaN"
            else:
                f_text = (f"F = {F_value:.3f}")
            print(
                f"  {effect:<35} "
                f"{f_text:<15} "
                f"{p_text}"
            )

            if row["significant"]:
                print("-> Significant (p < 0.05)")
            else:
                print("-> Not significant")

    print("SIGNIFICANT EFFECTS: ")
    if significant.empty:
        print("No statistically significant effects (p < 0.05).")
    else:
        for _, row in significant.iterrows():
            print(
                f"  {row['variable']} — "
                f"{row['effect']} : "
                f"F = {row['F']:.3f}, "
                f"p = {row['p_value']:.4g}"
            )
    return results


def run_three_way_anova(
    metrics,
    output_dir,
    scenario,
    variables=None
):
    """
    Performs a three-way ANOVA for each simulation-level metric.

    Factors:
        - model: NES / Moussaid
        - sfm: SFM / Without SFM
        - speed_type: Fixed / Random

    Full factorial model:
        metric ~ C(model) * C(sfm) * C(speed_type)

    This includes:
        - Main effect of model
        - Main effect of SFM
        - Main effect of speed type
        - Model x SFM interaction
        - Model x speed type interaction
        - SFM x speed type interaction
        - Model x SFM x speed type interaction

    One observation = one simulation.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if variables is None:
        variables = [
            "distance_min",
            "angle_at_min",
            "angle_increase_distance",
            "relative_speed_mean",
            "cyclist_speed_mean"
        ]

    data = metrics.copy()
    required_columns = [
        "simulation_id",
        "model",
        "sfm",
        "speed_type"
    ]

    missing = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns for three-way ANOVA: {missing}"
        )

    data = data.dropna(
        subset=[
            "simulation_id",
            "model",
            "sfm",
            "speed_type"
        ]
    ).copy()

    if data.empty:
        print("No data available for three-way ANOVA.")
        return pd.DataFrame()

    data["model"] = data["model"].astype("category")
    data["sfm"] = data["sfm"].astype("category")
    data["speed_type"] = data["speed_type"].astype("category")

    if data["model"].nunique() < 2:
        raise ValueError("Three-way ANOVA requires at least two model levels.")

    if data["sfm"].nunique() < 2:
        raise ValueError("Three-way ANOVA requires at least two SFM levels.")

    if data["speed_type"].nunique() < 2:
        raise ValueError("Three-way ANOVA requires at least two speed_type levels.")

    
    design = (data.groupby(
            ["model", "sfm", "speed_type"],
            observed=True
        ).size().reset_index(name="n"))

    design.to_csv(output_dir / f"{scenario}_anova_three_way_design.csv", index=False)

    # ANOVA
    results = []
    for variable in variables:
        if variable not in data.columns:
            print(f"Warning: {variable} not found in metrics. Skipping.")
            continue

        subset = data[["simulation_id",
                        "model",
                        "sfm",
                        "speed_type",
                        variable]].copy()

        subset = subset.dropna(subset=[variable])
        if subset.empty:
            print(f"{variable}: no valid observations. Skipping.")
            continue

        if subset["model"].nunique() < 2:
            print(f"{variable}: not enough model levels. Skipping.")
            continue

        if subset["sfm"].nunique() < 2:
            print(f"{variable}: not enough SFM levels. Skipping.")
            continue

        if subset["speed_type"].nunique() < 2:
            print(f"{variable}: not enough speed_type levels. Skipping.")
            continue

        formula = (
            f"{variable} ~ "
            f"C(model) * "
            f"C(sfm) * "
            f"C(speed_type)"
        )

        print(variable)
        try:
            model_fit = ols(formula, data=subset).fit()
            anova_table = sm.stats.anova_lm(model_fit, typ=2)

        except Exception as e:
            print(f"ANOVA failed for {variable}: {e}")
            continue

        for effect, row in anova_table.iterrows():
            p_value = row["PR(>F)"]
            results.append({
                "scenario": scenario,
                "variable": variable,
                "effect": effect,
                "sum_sq": row["sum_sq"],
                "df": row["df"],
                "F": row["F"],
                "p_value": p_value,
                "significant": (p_value < 0.05 if not pd.isna(p_value) else False),
                "n": len(subset)
            })

        safe_variable = (variable.replace(" ", "_").replace("/", "_"))
        individual_table = anova_table.reset_index()
        individual_table.to_csv(output_dir / f"{scenario}_anova_three_way_{safe_variable}.csv", index=False)

        print(anova_table[["sum_sq", "df", "F", "PR(>F)"]])

    results = pd.DataFrame(results)
    if results.empty:
        print("\nNo three-way ANOVA results.")
        return results

    results = results.rename(columns={"p_value": "p_value"})

    results.to_csv(output_dir / f"{scenario}_anova_three_way.csv", index=False)

    significant = results[results["significant"]].copy()
    significant.to_csv(output_dir / f"{scenario}_anova_three_way_significant.csv", index=False)

    print(f"THREE-WAY ANOVA — {scenario}")

    for variable in results["variable"].unique():
        print(f"\n{variable}")
        print("-" * len(variable))
        variable_results = results[results["variable"] == variable]
        for _, row in variable_results.iterrows():
            effect = row["effect"]
            F = row["F"]
            p = row["p_value"]
            if pd.isna(F) or pd.isna(p):
                print(
                    f"  {effect:<35} "
                    f"F = NaN, "
                    f"p = NaN"
                )
            else:
                significance = ("SIGNIFICANT" if p < 0.05 else "not significant")
                print(
                    f"  {effect:<35} "
                    f"F = {F:.3f}, "
                    f"p = {p:.4g} "
                    f"→ {significance}"
                )

    print(f"\nComplete results saved to:")
    print(output_dir / f"{scenario}_anova_three_way.csv")
    print(f"\nSignificant effects saved to:")
    print(output_dir / f"{scenario}_anova_three_way_significant.csv")
    return results



###############################


# COMPARISON WITH REAL SCENARIO

def compute_real_speed_metrics(real_df, fps=29.97):
    """
    Computes the real-scenario speed metrics used for comparison with the simulations.
    """

    cyclists = real_df[real_df["user_type"] == 2].copy()
    pedestrians = real_df[real_df["user_type"] == 1].copy()

    if cyclists.empty:
        return {"cyclist_speed_mean": np.nan, "relative_speed_mean": np.nan}

    # Cyclist mean speed
    cyclist_speeds = []
    for cyclist_id, g in cyclists.groupby("object_id"):
        g = g.sort_values("time_step").copy()
        vx = g["x_m"].diff() * fps
        vy = g["y_m"].diff() * fps
        speed = np.sqrt(vx**2 + vy**2)
        cyclist_speeds.extend(speed.dropna().values)

    if cyclist_speeds:
        cyclist_speed_mean = np.mean(cyclist_speeds)
    else:
        cyclist_speed_mean = np.nan

    relative_speeds = []
    # Pair each cyclist with each pedestrian
    for cyclist_id, cyclist in cyclists.groupby("object_id"):
        for pedestrian_id, pedestrian in pedestrians.groupby("object_id"):
            g = prepare_cyclist_pedestrian_pair(cyclist, pedestrian)
            if g.empty:
                continue

            g = g.sort_values("time_rel").reset_index(drop=True)
            vx_c = g["cyc_x"].diff() * fps
            vy_c = g["cyc_y"].diff() * fps
            vx_p = g["ped_x"].diff() * fps
            vy_p = g["ped_y"].diff() * fps
            vrx = vx_c - vx_p
            vry = vy_c - vy_p
            relative_speed = np.sqrt(vrx**2 + vry**2)
            relative_speeds.extend(relative_speed.dropna().values)

    if relative_speeds:
        relative_speed_mean = np.mean(relative_speeds)
    else:
        relative_speed_mean = np.nan

    return {"cyclist_speed_mean": cyclist_speed_mean, "relative_speed_mean": relative_speed_mean}


def compare_simulations_to_real(
    metrics,
    real_min_distance_angle,
    real_angle_onset,
    real_speed_metrics,
    scenario,
    output_dir
):
    """
    Compares simulation results with the real scenario and computes errors (signed, absolute, relative).
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    real_values = {
        "distance_min": real_min_distance_angle.get(
            "distance_min",
            np.nan
        ),

        "angle_at_min": real_min_distance_angle.get(
            "angle_at_min",
            np.nan
        ),

        "angle_increase_distance": real_angle_onset.get(
            "angle_increase_distance",
            np.nan
        ),

        "cyclist_speed_mean": real_speed_metrics.get(
            "cyclist_speed_mean",
            np.nan
        ),

        "relative_speed_mean": real_speed_metrics.get(
            "relative_speed_mean",
            np.nan
        )
    }

    variables = [
        "distance_min",
        "angle_at_min",
        "angle_increase_distance",
        "cyclist_speed_mean",
        "relative_speed_mean"
    ]

    required_columns = [
        "model",
        "sfm",
        "speed_type",
        "simulation_id"
    ]

    missing = [
        col
        for col in required_columns
        if col not in metrics.columns
    ]

    if missing:
        raise ValueError(f"Missing columns for real/simulation comparison: {missing}")

    comparison_results = []

    model_order = ["Moussaid", "NES"]
    sfm_order = ["SFM", "Without SFM"]
    speed_order = ["Fixed", "Random"]

    configurations = (metrics[["model", "sfm", "speed_type"]].drop_duplicates().copy())

    configurations["model"] = pd.Categorical(
        configurations["model"],
        categories=model_order,
        ordered=True
    )

    configurations["sfm"] = pd.Categorical(
        configurations["sfm"],
        categories=sfm_order,
        ordered=True
    )

    configurations["speed_type"] = pd.Categorical(
        configurations["speed_type"],
        categories=speed_order,
        ordered=True
    )

    configurations = configurations.sort_values(["model", "sfm", "speed_type"])

    for _, config in configurations.iterrows():
        model = config["model"]
        sfm = config["sfm"]
        speed_type = config["speed_type"]

        subset = metrics[
            (metrics["model"] == model) &
            (metrics["sfm"] == sfm) &
            (metrics["speed_type"] == speed_type)
        ].copy()

        for variable in variables:
            if variable not in subset.columns:
                print(f"Warning: {variable} not found in metrics. Skipping.")
                continue

            values = subset[variable].dropna()
            if values.empty:
                continue

            sim_mean = values.mean()
            sim_std = values.std()
            sim_median = values.median()
            n = len(values)

            real_value = real_values[variable]

            # errors
            if pd.isna(real_value):
                difference = np.nan
                absolute_difference = np.nan
                relative_error = np.nan
            else:
                difference = sim_mean - real_value
                absolute_difference = abs(difference)
                if real_value != 0:
                    relative_error = (absolute_difference / abs(real_value) * 100)
                else:
                    relative_error = np.nan

            comparison_results.append({
                "scenario": scenario,
                "model": model,
                "sfm": sfm,
                "speed_type": speed_type,
                "variable": variable,

                "simulation_mean": sim_mean,
                "simulation_std": sim_std,
                "simulation_median": sim_median,

                "real_value": real_value,

                "difference_sim_minus_real": difference,
                "absolute_difference": absolute_difference,
                "relative_error_percent": relative_error,

                "n_simulations": n
            })

    comparison = pd.DataFrame(comparison_results)
    if comparison.empty:
        print("No valid simulation results available for comparison with the real scenario.")
        return comparison

    comparison["rank"] = (comparison.groupby("variable")["relative_error_percent"].rank(method="min", ascending=True))
    comparison = comparison.sort_values(["variable", "relative_error_percent"])

    comparison.to_csv(
        output_dir /
        f"{scenario}_simulation_vs_real.csv",
        index=False
    )

    comparison.to_csv(
        output_dir /
        f"{scenario}_simulation_vs_real_ranked.csv",
        index=False
    )

    print(f"SIMULATION vs REAL — {scenario}")
    for variable in variables:
        subset = comparison[comparison["variable"] == variable].copy()
        if subset.empty:
            continue
        print(f"\n{variable}")
        print("-" * len(variable))

        real_value = real_values[variable]
        if not pd.isna(real_value):
            print(f"  Real scenario = {real_value:.3f}")

        for _, row in subset.iterrows():
            error = row["relative_error_percent"]
            error_text = (
                f"{error:.2f}%"
                if not pd.isna(error)
                else "N/A"
            )

            print(
                f"  {row['model']} | "
                f"{row['sfm']} | "
                f"{row['speed_type']} : "
                f"{row['simulation_mean']:.3f} "
                f"± {row['simulation_std']:.3f} "
                f"(n={int(row['n_simulations'])}) | "
                f"error = {error_text}"
            )


    print("BEST CONFIGURATIONS")
    best_results = []
    for variable in variables:
        subset = comparison[
            comparison["variable"] == variable
        ].dropna(
            subset=["relative_error_percent"]
        )

        if subset.empty:
            continue

        best = subset.loc[subset["relative_error_percent"].idxmin()]
        best_results.append({
            "scenario": scenario,
            "variable": variable,

            "model": best["model"],
            "sfm": best["sfm"],
            "speed_type": best["speed_type"],

            "simulation_mean": best["simulation_mean"],
            "simulation_std": best["simulation_std"],

            "real_value": best["real_value"],
            "absolute_difference": best["absolute_difference"],
            "relative_error_percent": best["relative_error_percent"],

            "n_simulations": best["n_simulations"]
        })

        print(f"\n{variable}:")
        print(
            f"  Best configuration = "
            f"{best['model']} | "
            f"{best['sfm']} | "
            f"{best['speed_type']}"
        )

        print(
            f"  Simulation = "
            f"{best['simulation_mean']:.3f} "
            f"± {best['simulation_std']:.3f}"
        )

        print(f"  Real = {best['real_value']:.3f}")

        print(
            f"  Relative error = "
            f"{best['relative_error_percent']:.2f}%"
        )

    best_results = pd.DataFrame(best_results)

    best_results.to_csv(
        output_dir /
        f"{scenario}_simulation_vs_real_best.csv",
        index=False
    )

    # GRAPHS (per config)
    sfm_colors = {
        "SFM": "royalblue",
        "Without SFM": "tab:orange"
    }

    model_markers = {
        "Moussaid": "o",
        "NES": "s"
    }

    model_order = [
        "Moussaid",
        "NES"
    ]

    sfm_order = [
        "SFM",
        "Without SFM"
    ]

    speed_order = [
        "Fixed",
        "Random"
    ]

    # Create the complete canonical configuration order
    configuration_order = [
        (model, sfm, speed_type)
        for model in model_order
        for sfm in sfm_order
        for speed_type in speed_order
    ]

    configuration_rank = {
        config: i
        for i, config in enumerate(configuration_order)
    }

    comparison["configuration_rank"] = comparison.apply(
        lambda row: configuration_rank.get(
            (
                row["model"],
                row["sfm"],
                row["speed_type"]
            ),
            1000
        ),
        axis=1
    )

    ylabel_dict = {
        "distance_min": "Minimum distance (m)",
        "angle_at_min": "Approach angle at minimum distance (°)",
        "angle_increase_distance": "Distance at approach-angle increase (m)",
        "cyclist_speed_mean": "Mean cyclist speed (m/s)",
        "relative_speed_mean": "Mean relative speed (m/s)"
    }

    for variable in variables:
        subset = comparison[comparison["variable"] == variable].copy()
        subset = subset.dropna(subset=["simulation_mean"])
        if subset.empty:
            continue

        subset = subset.sort_values("configuration_rank").reset_index(drop=True)

        subset["configuration"] = (
            subset["model"]
            + " / "
            + subset["sfm"]
            + " / "
            + subset["speed_type"]
        )

        x = np.arange(len(subset))

        fig, ax = plt.subplots(figsize=(13, 7))

        for i, (_, row) in enumerate(subset.iterrows()):
            model = row["model"]
            sfm = row["sfm"]
            speed_type = row["speed_type"]

            color = sfm_colors.get(
                sfm,
                "black"
            )

            marker = model_markers.get(
                model,
                "o"
            )

            # Fixed = filled
            # Random = empty
            if speed_type == "Fixed":
                facecolor = color
            else:
                facecolor = "white"

            # Mean + STD
            ax.errorbar(
                i,
                row["simulation_mean"],
                yerr=(
                    row["simulation_std"]
                    if not pd.isna(row["simulation_std"])
                    else 0
                ),
                fmt=marker,
                markersize=11,
                markerfacecolor=facecolor,
                markeredgecolor=color,
                markeredgewidth=2,
                ecolor=color,
                elinewidth=1.5,
                capsize=5,
                zorder=5
            )

            # Simulation value
            ax.annotate(
                f"{row['simulation_mean']:.2f}",
                (
                    i,
                    row["simulation_mean"]
                ),
                xytext=(12, 0),
                textcoords="offset points",
                fontsize=12,
                fontweight="bold",
                ha="left",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.85
                )
            )

            # Relative error
            error = row["relative_error_percent"]
            if not pd.isna(error):
                ax.annotate(
                    f"error: {error:.1f}%",
                    (
                        i,
                        row["simulation_mean"]
                    ),
                    xytext=(12, -24),
                    textcoords="offset points",
                    fontsize=12,
                    color="dimgray",
                    ha="left",
                    va="center",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.85
                    )
                )

        # REAL SCENARIO
        real_value = real_values[variable]
        if not pd.isna(real_value):
            ax.axhline(
                real_value,
                color="red",
                linestyle="--",
                linewidth=2,
                label=(
                    f"Real scenario = "
                    f"{real_value:.2f}"
                ),
                zorder=2
            )

            ax.text(
                1.01,
                real_value,
                f"{real_value:.2f}",
                transform=ax.get_yaxis_transform(),
                color="red",
                va="center",
                ha="left",
                fontsize=11,
                fontweight="bold"
            )

        legend_elements = [
            # MODEL
            Line2D(
                [0], [0],
                marker="o",
                linestyle="None",
                markerfacecolor="black",
                markeredgecolor="black",
                markersize=10,
                label="Moussaid model"
            ),

            Line2D(
                [0], [0],
                marker="s",
                linestyle="None",
                markerfacecolor="black",
                markeredgecolor="black",
                markersize=10,
                label="NES model"
            ),

            # SPEED
            Line2D(
                [0], [0],
                marker="o",
                linestyle="None",
                markerfacecolor="black",
                markeredgecolor="black",
                markersize=10,
                label="Fixed cyclist speed"
            ),

            Line2D(
                [0], [0],
                marker="o",
                linestyle="None",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=10,
                label="Random cyclist speed"
            ),

            # SFM
            Line2D(
                [0], [0],
                marker="o",
                linestyle="None",
                markerfacecolor="royalblue",
                markeredgecolor="royalblue",
                markersize=10,
                label="SFM"
            ),

            Line2D(
                [0], [0],
                marker="o",
                linestyle="None",
                markerfacecolor="tab:orange",
                markeredgecolor="tab:orange",
                markersize=10,
                label="Without SFM"
            ),

            # REAL
            Line2D(
                [0], [0],
                linestyle="--",
                linewidth=1.5,
                color="red",
                label="Real scenario"
            ),

            # STD
            Line2D(
                [0], [0],
                color="black",
                linewidth=1.5,
                label="Standard deviation"
            ),

            # ERROR
            Line2D(
                [0], [0],
                linestyle="None",
                marker=None,
                label=(
                    "Relative err = |simulation mean − real| / |real| x 100"
                )
            )
        ]

        ax.legend(
            handles=legend_elements,
            loc="best",
            title="Configuration",
            fontsize=9,
            title_fontsize=11
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            subset["configuration"],
            rotation=30,
            ha="right"
        )

        ax.set_ylabel(
            ylabel_dict.get(
                variable,
                variable
            ),
            fontsize=15
        )

        ax.set_xlabel(
            "Simulation configuration",
            fontsize=15
        )

        ax.set_title(
            f"{scenario} — "
            f"{ylabel_dict.get(variable, variable)}\n"
            "Simulation vs real scenario",
            fontsize=15,
            fontweight="bold",
            pad=10
        )

        ax.tick_params(axis="both", labelsize=12)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        filename = (f"{scenario}_{variable}_simulation_vs_real.png".replace(" ", "_"))
        fig.savefig(
            output_dir / filename,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)


    # RELATIVE ERROR GRAPH
    valid_comparison = comparison.dropna(subset=["relative_error_percent"]).copy()

    if not valid_comparison.empty:
        valid_comparison["configuration"] = (
            valid_comparison["model"]
            + " / "
            + valid_comparison["sfm"]
            + " / "
            + valid_comparison["speed_type"]
        )

        valid_comparison["configuration_rank"] = (
            valid_comparison.apply(
                lambda row: configuration_rank.get(
                    (
                        row["model"],
                        row["sfm"],
                        row["speed_type"]
                    ),
                    999
                ),
                axis=1
            )
        )

        valid_comparison = valid_comparison.sort_values(
            "configuration_rank"
        )

        pivot = valid_comparison.pivot(
            index="configuration",
            columns="variable",
            values="relative_error_percent"
        )

        ordered_labels = []
        for model, sfm, speed_type in configuration_order:
            label = (
                f"{model} / "
                f"{sfm} / "
                f"{speed_type}"
            )

            if label in pivot.index:
                ordered_labels.append(label)

        pivot = pivot.reindex(ordered_labels)

        fig, ax = plt.subplots(figsize=(14, 7))

        pivot.plot(kind="bar", ax=ax)
        for container in ax.containers:
            labels = []
            for bar in container:
                height = bar.get_height()
                if np.isnan(height):
                    labels.append("")
                else:
                    labels.append(f"{height:.1f}%")

            ax.bar_label(
                container,
                labels=labels,
                fontsize=11,
                padding=3,
                fontweight="bold"
            )

        ax.set_ylabel("Relative error (%)", fontsize=14)
        ax.set_xlabel("Simulation configuration", fontsize=14)
        ax.set_title(
            f"{scenario} — "
            "Relative error compared with real scenario",
            fontsize=15,
            fontweight="bold"
        )

        ax.tick_params(
            axis="x",
            labelrotation=30,
            labelsize=10
        )

        ax.grid(axis="y", alpha=0.3)
        ax.legend(title="Metric", fontsize=10)

        ax.text(
            0.5,
            -0.22,
            "Relative error = "
            "|simulation mean − real value| "
            "/ |real value| x 100",
            transform=ax.transAxes,
            ha="center",
            fontsize=10
        )

        fig.tight_layout()

        fig.savefig(
            output_dir /
            f"{scenario}_simulation_vs_real_relative_error.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

    comparison = comparison.drop(columns=["configuration_rank"], errors="ignore")
    return comparison


# validation pipeline
def analyze_scenario(scenario, real_csv, sim_root, output_root, fixed_vmax):

    output_dir = (Path(output_root) / scenario)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"SCENARIO : {scenario}")

    # load data
    real_df = load_real_csv(real_csv)
    sim_df = load_all_simulations(sim_root, scenario)

    # print(f"Loaded simulations: {sim_df['simulation_id'].nunique()}")

    print("Trajectoires...")
    plot_trajectories(real_df, sim_df, scenario, output_dir, fixed_vmax=fixed_vmax)

    print("Heatmaps...")
    plot_trajectory_heatmaps(sim_df, scenario, output_dir, fixed_vmax=fixed_vmax)

    print("Speed variation heatmaps...")
    plot_speed_variation_heatmaps(sim_df, scenario, output_dir)

    print("Combination speed variation heatmaps...")
    plot_joint_speed_variation_heatmap(sim_df, scenario, output_dir)

    print("Mean curves...")
    plot_all_mean_curves(sim_df, scenario, output_dir)

    print("Metrics...")
    metrics, summary, onset, onset_summary = save_metrics(sim_df, output_dir)

    print("\nSummary :")
    print(summary)

    real_min_distance_angle = compute_real_min_distance(real_df) 
    print("\nReal scenario — minimum distance / approach angle:") 
    print(f"distance_min = {real_min_distance_angle['distance_min']:.3f} m | " 
          f"angle_at_min = {real_min_distance_angle['angle_at_min']:.2f}°" )

    plot_distance_min_angle(metrics, scenario, output_dir, real_result=real_min_distance_angle)

    print("Mean distance / angle curves...")
    plot_mean_distance_angle_all_configurations(sim_df, scenario, output_dir)

    # onset = compute_angle_onset(sim_df)
    # Résultats individuels des simulations
    # onset.to_csv(output_dir / "angle_onset.csv", index=False)
    # Moyenne par configuration
    # onset_summary = summarize_angle_onset(onset)
    # onset_summary.to_csv(output_dir / "angle_onset_summary.csv", index=False)
    real_angle_onset = compute_real_angle_onset_for_scenario(real_df, smooth_window=7, min_angle_increase=5.0, persistence=9)
    # plot_mean_approach_distance_by_configuration(onset_summary, scenario, output_dir, real_result=real_angle_onset)

    print("\nDistance at increasing approach angle — per configuration:")
    print(onset_summary)

    print("\nDistance at increasing approach angle :")
    print(onset.describe())

    speed_bins_analysis(metrics, output_dir)

    real_speed_metrics = compute_real_speed_metrics(real_df, fps=29.97)

    print("\nReal scenario — speed metrics:")
    print(
        f"Mean cyclist speed = "
        f"{real_speed_metrics['cyclist_speed_mean']:.3f} m/s"
    )
    print(
        f"Mean relative speed = "
        f"{real_speed_metrics['relative_speed_mean']:.3f} m/s"
    )

    # compare_speed_types(sim_df, scenario, output_dir, fixed_vmax)
    print("\nANOVA...")
    if scenario in ["avoidance", "avoidance2", "overtaking", "overtaking2"]:
        anova_results = run_three_way_anova(metrics, output_dir, scenario)
    else:
        anova=reseults = run_two_way_anova(metrics, output_dir, scenario)

    print("\nComparison simulations vs real scenario...")
    comparison_real = compare_simulations_to_real(
        metrics=metrics,
        real_min_distance_angle=real_min_distance_angle,
        real_angle_onset=real_angle_onset,
        real_speed_metrics=real_speed_metrics,
        scenario=scenario,
        output_dir=output_dir
    )

    print(f"\nResults saved in : {output_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scenario",
        required=True,
        help="Scenario name, ex : overtaking"
    )

    parser.add_argument(
        "--real-csv",
        required=True,
        help="CSV with real trajectories"
    )

    parser.add_argument(
        "--sim-root",
        default="simulation_res",
        help="Folder simulation results"
    )

    parser.add_argument(
        "--output",
        default="comparison_results",
        help="Output folder"
    )

    args = parser.parse_args()

    fixed_vmax = fixed_vmaxs[args.scenario]

    df = pd.read_csv(args.real_csv)
    print("Distance minimale parmi plusieurs piétons ")
    print(compute_real_min_distance(df))

    print("Angle onset distance in real scenario : ")
    print(compute_real_angle_onset_for_scenario(pd.read_csv(args.real_csv),
                    smooth_window=7,
                    min_angle_increase=5.0,
                    persistence=9
                ))

    analyze_scenario(
        scenario=args.scenario,
        real_csv=args.real_csv,
        sim_root=args.sim_root,
        output_root=args.output,
        fixed_vmax=fixed_vmax
    )