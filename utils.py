import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.path import Path
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.spatial.distance import cdist, directed_hausdorff
from scipy.stats import norm
from config import *
from statsmodels.nonparametric.smoothers_lowess import lowess


def frames_to_intervals(frames):
    """
    Converts a list of frames into continuous intervals (start, end)
    """
    if len(frames) == 0:
        return []

    frames = sorted(frames)
    intervals = []
    start = frames[0]
    prev = frames[0]

    for t in frames[1:]:
        if t == prev + 1:
            # continuité
            prev = t
        else:
            intervals.append((start, prev))
            start = t
            prev = t

    intervals.append((start, prev))
    return intervals


def add_time_markers(ax, intervals):
    for i, (start, end) in enumerate(intervals):
        ax.axvspan(start, end, color="grey", alpha=0.2, label=f"Interaction ({start:.2f} - {end:.2f}, duration={end-start:.2f}s)")

def add_spatial_markers(ax, df, ped_id, intervals):
    ped = df[df[COL_ID] == ped_id]
    for i, (start, end) in enumerate(intervals):
        p_start = ped[ped[COL_TIME] == start]
        p_end = ped[ped[COL_TIME] == end]

        if not p_start.empty:
            p_start = p_start.iloc[0]
            ax.scatter(
                p_start["x_m"], p_start["y_m"],
                color="green", s=80, marker="o",
                label="Début interaction" if i == 0 else ""
            )

        if not p_end.empty:
            p_end = p_end.iloc[0]
            ax.scatter(
                p_end["x_m"], p_end["y_m"],
                color="red", s=80, marker="x",
                label="Fin interaction" if i == 0 else ""
            )


###############################################
# Spatio-temporal criteria for individual-individual interactions
###############################################

def compute_speed(g, fps):
    g = g.sort_values(COL_TIME)

    xs = g["x_m"].values
    ys = g["y_m"].values
    times = g[COL_TIME].values

    if len(xs) < 2:
        return None, None

    dx = np.diff(xs)
    dy = np.diff(ys)

    if len(np.unique(times)) > 1: # si y a au moins 2 timestamps différents
        dt = np.diff(times) / fps
    else:
        dt = np.ones_like(dx) / fps # (c'est exactement équivalent à dt = 1/fps en s)

    dt[dt <= 0] = 1 / fps # pour éviter division par zéro
    speeds = np.hypot(dx, dy) / dt

    # conversion km/h
    speeds_kmh = speeds * 3.6

    return times[1:], speeds, speeds_kmh


def compute_acceleration(g, fps, sp=None, t=None):
    speeds = sp
    times = t
    if speeds is None:
        times, speeds, _ = compute_speed(g, fps)

    if speeds is None or len(speeds) < 2:
        return None, None

    threshold = np.percentile(speeds, 99)
    mask = ((speeds <= threshold) & (speeds >= 0.5))

    speeds_clean = speeds[mask]
    times_clean = times[mask]

    dt = np.diff(times_clean) / fps
    dt[dt <= 0] = 1 / fps

    accelerations = np.diff(speeds_clean) / dt

    return times[1:], accelerations


def collect_cyclist_statistics(df, fps):
    """
    Returns all cyclist speeds and accelerations from a video.
    """
    cyclist_df = df[df[COL_CLASS] == 2]

    all_speeds = []
    all_accelerations = []
    cyclist_count = 0

    for _, g in cyclist_df.groupby(COL_ID):

        if len(g) < 2:
            continue

        cyclist_count += 1
        times, speeds, _ = compute_speed(g, fps)
        if speeds is not None:
            all_speeds.extend(speeds)

        _, acc = compute_acceleration(g, fps, sp=speeds, t=times)
        if acc is not None:
            all_accelerations.extend(acc)

    return (np.asarray(all_speeds), np.asarray(all_accelerations), cyclist_count)


def plot_cyclist_speed_histogram(all_speeds, cyclist_count):
    """
    Histogramme des vitesses de tous les cyclistes.
    """

    if len(all_speeds) == 0:
        print("No speed available.")
        return

    print("Nb of speed (of cyclists) :", len(all_speeds))
    print("Min speed :", np.min(all_speeds))
    print("Max speed :", np.max(all_speeds))
    print("Mean speed :", np.mean(all_speeds))
    print("Std speed :", np.std(all_speeds))
    print("Unique values :", len(np.unique(all_speeds)))

    q99 = np.percentile(all_speeds, 99)
    speeds_plot = all_speeds[(all_speeds <= q99) & (all_speeds >= 0.5)]

    mean = np.mean(speeds_plot)
    std = np.std(speeds_plot)

    # Histo avec densité de proba
    plt.figure(figsize=(10,6))
    counts, bins, _ = plt.hist(
        speeds_plot,
        bins=30,
        density=True,
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
        label="Observed speeds"
    )
    x = np.linspace(bins[0], bins[-1], 300)
    plt.plot(x, norm.pdf(x, mean, std), 'r', lw=2.5, label="Normal distribution")
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Probability density")
    plt.title("Cyclist speed distribution")
    text = (
        f"N cyclists = {cyclist_count}\n"
        f"N samples = {len(all_speeds)}\n"
        f"Mean = {mean:.2f} m/s\n"
        f"Std = {std:.2f} m/s"
    )
    plt.text(
        0.98,
        0.98,
        text,
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        bbox=dict(facecolor="white", alpha=0.8)
    )
    plt.legend()
    plt.tight_layout()
    plt.show()

    # histo avec nombre d'observations
    plt.figure(figsize=(10,6))
    counts, bins, _ = plt.hist(
        speeds_plot,
        bins=30,
        density=False,
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
        label="Observed speeds"
    )
    bin_width = bins[1] - bins[0]
    x = np.linspace(bins[0], bins[-1], 300)
    plt.plot(
        x,
        norm.pdf(x, mean, std) * len(speeds_plot) * bin_width,
        'r',
        lw=2.5,
        label="Normal distribution"
    )
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Number of observations")
    plt.title("Cyclist speed distribution")
    text = (
        f"N cyclists = {cyclist_count}\n"
        f"N samples = {len(all_speeds)}\n"
        f"Mean = {mean:.2f} m/s\n"
        f"Std = {std:.2f} m/s"
    )
    plt.text(
        0.98,
        0.98,
        text,
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        bbox=dict(facecolor="white", alpha=0.8)
    )
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Histo avec en Y la freq en %
    weights = np.ones_like(speeds_plot) * 100 / len(speeds_plot)
    plt.figure(figsize=(10,6))
    counts, bins, _ = plt.hist(
        speeds_plot,
        bins=30,
        weights=weights,
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
        label="Observed speeds"
    )
    bin_width = bins[1] - bins[0]
    x = np.linspace(bins[0], bins[-1], 300)
    plt.plot(
        x,
        norm.pdf(x, mean, std) * 100 * bin_width,
        'r',
        lw=2.5,
        label="Normal distribution"
    )
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Frequency (%)")
    plt.title("Cyclist speed distribution")
    text = (
        f"N cyclists = {cyclist_count}\n"
        f"N samples = {len(all_speeds)}\n"
        f"Mean = {mean:.2f} m/s\n"
        f"Std = {std:.2f} m/s"
    )
    plt.text(
        0.98,
        0.98,
        text,
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        bbox=dict(facecolor="white", alpha=0.8)
    )
    plt.legend()
    plt.tight_layout()
    plt.show()



def print_cyclist_statistics(all_speeds, all_accelerations, cyclist_count):
    print("\nCyclist stats")
    print(f"Number of cyclists : {cyclist_count}")
    print(f"Speed samples : {len(all_speeds)}")
    print(f"Acceleration samples : {len(all_accelerations)}")
    print(f"Mean speed: {np.mean(all_speeds):.2f} m/s")
    print(f"Speed std : {np.std(all_speeds):.2f} m/s")
    print(f"Mean acceleration : {np.mean(all_accelerations):.2f} m/s2")
    print(f"Acceleration std : {np.std(all_accelerations):.2f} m/s2")


def compute_distance_ped_cyc(df, ped_id, cyc_id, fps, distance_threshold=5.0, plot=False, return_class=False):
    """
    Compute and plot the distance between a pedestrian and a cyclist.
    """
    # extraction
    ped = df[df[COL_ID] == ped_id].sort_values(COL_TIME)
    cyc = df[df[COL_ID] == cyc_id].sort_values(COL_TIME)

    # synchronization
    common_times = np.intersect1d(
        ped[COL_TIME].values,
        cyc[COL_TIME].values
    )

    if len(common_times) == 0:
        return None, None

    distances = []
    valid_times = []

    for t in common_times:
        p = ped[ped[COL_TIME] == t].iloc[0]
        c = cyc[cyc[COL_TIME] == t].iloc[0]

        dx = p["x_m"] - c["x_m"]
        dy = p["y_m"] - c["y_m"]

        dist = np.hypot(dx, dy)

        distances.append(dist)
        valid_times.append(t)

    distances = np.array(distances)
    valid_times = np.array(valid_times)

    dist_min = distances.min()
    idx_min = np.nan
    t_min = np.nan
    t_min_plot = np.nan
    if len(distances) != 0:
        idx_min = np.argmin(distances) 
        t_min = valid_times[idx_min]
        t_min_plot = t_min / fps
    

    # interaction intervals
    from analysis_interactions import compute_ped_cyc_interactions_with_time
    interactions = compute_ped_cyc_interactions_with_time(df, distance_threshold)
    key = tuple(sorted((ped_id, cyc_id)))
    frames = interactions.get(key, [])

    intervals = frames_to_intervals(frames) if len(frames) > 0 else []

    intervals_plot = [(s / fps, e / fps) for s, e in intervals]

    # plot
    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(valid_times / fps, distances)
        plt.axhline(distance_threshold, color="red", linestyle="--", label="Spatial interaction limit (5m)")
        add_time_markers(plt.gca(), intervals_plot)
        if not np.isnan(t_min_plot):
            plt.scatter(t_min_plot, dist_min, color="red", zorder=5, label=f"Min distance = {dist_min:.2f}m (at {t_min_plot:.2f}s)")
        plt.xlabel("Time (s)", fontsize=15)
        plt.ylabel("Distance (m)", fontsize=15)
        plt.tick_params(axis="both", labelsize=13)
        plt.title(f"Distance between pedestrian {ped_id} and cyclist {cyc_id} accross time", fontsize=15, fontweight="bold")
        plt.legend(fontsize=13)
        plt.grid()

        plt.tight_layout()
        plt.show()
    
    if return_class:
        from analysis_interactions import classify_distance_interaction
        return valid_times, distances, dist_min, classify_distance_interaction(valid_times, distances, intervals)

    return valid_times, distances, dist_min


def find_closest_ped_cyc(df):
    min_dist = float("inf")
    best_pair = (None, None)

    for t in df[COL_TIME].unique():
        frame = df[df[COL_TIME] == t]

        peds = frame[frame[COL_CLASS] == 1]
        cycs = frame[frame[COL_CLASS] == 2]

        if len(peds) == 0 or len(cycs) == 0:
            continue

        for _, ped in peds.iterrows():
            for _, cyc in cycs.iterrows():
                dx = ped["x_m"] - cyc["x_m"]
                dy = ped["y_m"] - cyc["y_m"]
                dist = np.hypot(dx, dy)

                if dist < min_dist:
                    min_dist = dist
                    best_pair = (ped[COL_ID], cyc[COL_ID])

    return best_pair



def compute_velocity_vectors(g, fps, plot=False):
    """
    Computes the velocity vectors (vx, vy) for a given agent.
    """
    g = g.sort_values(COL_TIME)

    xs = g["x_m"].values
    ys = g["y_m"].values
    times = g[COL_TIME].values

    if len(xs) < 2:
        return None, None, None

    dx = np.diff(xs)
    dy = np.diff(ys)

    if len(np.unique(times)) > 1:
        dt = np.diff(times) / fps
    else:
        dt = np.ones_like(dx) / fps

    dt[dt <= 0] = 1 / fps

    vx = dx / dt
    vy = dy / dt

    if plot:
        plt.figure(figsize=(6, 6))
        plt.plot(xs, ys, 'k--', alpha=0.5, label="Trajectoire")
        plt.quiver(
            xs[:-1], ys[:-1],
            vx, vy,
            angles='xy', scale_units='xy', scale=1,
            color="blue"
        )
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title(f"Velocity vectors of agent {g[COL_ID].iloc[0]}")
        plt.gca().invert_yaxis()
        plt.grid()
        plt.legend()
        plt.axis("equal")

        plt.show()

    return times[1:], vx, vy


def compute_direction_angle_velocity_based(df, ped_id, cyc_id, fps, start=None, end=None, angle_unit="deg", plot=False, return_class=False):
    """
    Computes the direction angle between the pedestrian's and cyclist's velocity vectors.
    Indicates the users' movement directions during the interaction.
    """

    ped = df[df[COL_ID] == ped_id].sort_values(COL_TIME)
    cyc = df[df[COL_ID] == cyc_id].sort_values(COL_TIME)

    # velocity vectors
    t_p, vx_p, vy_p = compute_velocity_vectors(ped, fps)
    t_c, vx_c, vy_c = compute_velocity_vectors(cyc, fps)

    if t_p is None or t_c is None:
        return None, None

    # timestamps synchronization
    common_times = np.intersect1d(t_p, t_c)

    angles = []
    valid_times = []

    for t in common_times:
        i_p = np.where(t_p == t)[0][0]
        i_c = np.where(t_c == t)[0][0]

        v_p = np.array([vx_p[i_p], vy_p[i_p]])
        v_c = np.array([vx_c[i_c], vy_c[i_c]])

        norm_p = np.linalg.norm(v_p)
        norm_c = np.linalg.norm(v_c)

        if norm_p == 0 or norm_c == 0:
            continue

        cos_theta = np.dot(v_p, v_c) / (norm_p * norm_c)
        cos_theta = np.clip(cos_theta, -1.0, 1.0) # en radians

        angle = np.arccos(cos_theta)

        if angle_unit == "deg":
            angle = np.degrees(angle)

        angles.append(angle)
        valid_times.append(t)

    angles = np.array(angles)
    valid_times = np.array(valid_times)

    # plot
    if plot and len(angles) > 0:
        intervals_sec = []
        if start is not None and end is not None:
            intervals_sec = [(start / fps, end / fps)]
        else:
            from analysis_interactions import compute_ped_cyc_interactions_with_time
            interactions = compute_ped_cyc_interactions_with_time(df)
            frames = interactions.get((ped_id, cyc_id), [])
            intervals = frames_to_intervals(frames)
            intervals_sec = [(s / fps, e / fps) for (s, e) in intervals]

        plt.figure(figsize=(10, 4))
        plt.plot(valid_times/fps, angles)
        plt.xlabel("Time (s)", fontsize=15)
        plt.ylabel(f"Angle ({angle_unit})", fontsize=15)
        plt.tick_params(axis="both", labelsize=13)
        plt.title(f"Direction angle (ped {ped_id} - cyc {cyc_id})", fontsize=15, fontweight="bold")
        if angle_unit == "deg":
            plt.axhline(0, linestyle="--", color="black", alpha=0.5, label="Same direction (0°)")
            plt.axhline(90, linestyle="--", color="orange", label="Perpendicular (90°)") # croisement latéral ou perpendiculaire mais pas forcément de collision
            plt.axhline(180, linestyle="--", color="purple", label="Opposite (180°)")
        plt.grid()
        add_time_markers(plt.gca(), intervals_sec)
        smooth = lowess(angles, valid_times/fps,frac=0.05)
        plt.plot(smooth[:,0], smooth[:,1], linewidth=2, color ="red", label="Smoothed angle", alpha=0.5)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()

    if return_class:
        from analysis_interactions import classify_direction_angle_interaction
        return valid_times, angles, classify_direction_angle_interaction(df, ped_id, cyc_id, valid_times, angles)
    
    return valid_times, angles


def compute_approach_angle(
    df,
    ped_id,
    cyc_id,
    fps,
    start=None,
    end=None,
    angle_unit="deg",
    plot=False,
    return_class=False):
    """
    Computes the approach angle between a cyclist and a pedestrian (between the line of sight and the relative velocity vector).

    Based on:
        - relative position vector
        - relative velocity vector
    """

    ped = df[df[COL_ID] == ped_id].sort_values(COL_TIME)
    cyc = df[df[COL_ID] == cyc_id].sort_values(COL_TIME)

    # vecteurs vitesses
    t_p, vx_p, vy_p = compute_velocity_vectors(ped, fps)
    t_c, vx_c, vy_c = compute_velocity_vectors(cyc, fps)

    if t_p is None or t_c is None:
        return None, None

    # positions
    t_pos = np.intersect1d(ped[COL_TIME].values, cyc[COL_TIME].values)
    t_vel = np.intersect1d(t_p, t_c)

    common_times = np.intersect1d(t_pos, t_vel)

    angles = []
    valid_times = []

    for t in common_times:
        # indices positions
        i_p_pos = np.where(ped[COL_TIME].values == t)[0][0]
        i_c_pos = np.where(cyc[COL_TIME].values == t)[0][0]

        x_p, y_p = ped.iloc[i_p_pos][["x_m", "y_m"]]
        x_c, y_c = cyc.iloc[i_c_pos][["x_m", "y_m"]]

        # indices vitesses
        i_p_vel = np.where(t_p == t)[0][0]
        i_c_vel = np.where(t_c == t)[0][0]

        v_p = np.array([vx_p[i_p_vel], vy_p[i_p_vel]])
        v_c = np.array([vx_c[i_c_vel], vy_c[i_c_vel]])

        # vceteurs importants
        r = np.array([x_p - x_c, y_p - y_c])   # position relative cycliste par rapp au piéton
        v_rel = v_c - v_p                      # vitesse relative

        norm_r = np.linalg.norm(r)
        norm_v = np.linalg.norm(v_rel)

        if norm_r == 0 or norm_v == 0:
            continue

        cos_theta = np.dot(v_rel, r) / (norm_r * norm_v)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        angle = np.arccos(cos_theta)

        if angle_unit == "deg":
            angle = np.degrees(angle)

        angles.append(angle)
        valid_times.append(t)

    angles = np.array(angles)
    valid_times = np.array(valid_times)

    # plot
    if plot and len(angles) > 0:
        intervals_sec = []
        if start is not None and end is not None:
            intervals_sec = [(start / fps, end / fps)]
        else:
            from analysis_interactions import compute_ped_cyc_interactions_with_time
            interactions = compute_ped_cyc_interactions_with_time(df)
            pair = tuple(sorted((ped_id, cyc_id)))
            frames = interactions.get(pair, [])
            intervals = frames_to_intervals(frames)
            intervals_sec = [(s / fps, e / fps) for (s, e) in intervals]
        
        plt.figure(figsize=(10, 4))
        plt.plot(valid_times/fps, angles)

        plt.xlabel("Time (s)", fontsize=15)
        plt.ylabel(f"Angle ({angle_unit})", fontsize=15)
        plt.tick_params(axis="both", labelsize=13)
        plt.title(f"Approach angle (ped {ped_id} - cyc {cyc_id})", fontsize=15, fontweight="bold")
        if angle_unit == "deg":
            plt.axhline(0, linestyle="--", color="black", alpha=0.5, label="Frontal (0°)")
            plt.axhline(90, linestyle="--", color="orange", label="Crossing (90°)") # croisement latéral ou perpendiculaire mais pas forcément de collision
            plt.axhline(180, linestyle="--", color="purple", label="Opposition (180°)")
        plt.grid()
        add_time_markers(plt.gca(), intervals_sec)
        smooth = lowess(angles, valid_times/fps,frac=0.05)
        plt.plot(smooth[:,0], smooth[:,1], linewidth=2, color ="red", label="Smoothed angle", alpha=0.5)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()
    
    if return_class:
        from analysis_interactions import classify_approach_angle_interaction
        return valid_times, angles, classify_approach_angle_interaction(df, ped_id, cyc_id, valid_times, angles)

    return valid_times, angles


def compute_relative_speed(df, ped_id, cyc_id, fps, start=None, end=None,
                            angle_unit="deg", return_distance=True, distance_threshold=5.0, 
                            plot=False, return_class=False):
    """
    Computes the relative motion between a pedestrian and a cyclist:
    - relative speed (m/s)
    - relative speed (km/h)
    - approach angle (rad)
    - distance (optional)
    """
    ped = df[df[COL_ID] == ped_id].sort_values(COL_TIME)
    cyc = df[df[COL_ID] == cyc_id].sort_values(COL_TIME)

    if len(ped) < 2 or len(cyc) < 2:
        return None

    # vecteurs vitesses
    t_p, vx_p, vy_p = compute_velocity_vectors(ped, fps)
    t_c, vx_c, vy_c = compute_velocity_vectors(cyc, fps)

    # synchronisation timestamps
    common_times = np.intersect1d(t_p, t_c)

    if len(common_times) == 0:
        print("Pas de temps communs")
        return None

    # index correspondants
    idx_p = np.isin(t_p, common_times)
    idx_c = np.isin(t_c, common_times)

    vx_p = vx_p[idx_p]
    vy_p = vy_p[idx_p]

    vx_c = vx_c[idx_c]
    vy_c = vy_c[idx_c]

    times = common_times

    # vitesse relative
    speed_p = np.hypot(vx_p, vy_p)
    speed_c = np.hypot(vx_c, vy_c)

    dot = vx_p * vx_c + vy_p * vy_c
    norm_p = np.maximum(speed_p, 1e-8)
    norm_c = np.maximum(speed_c, 1e-8)

    cos_theta = dot / (norm_p * norm_c)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    theta = np.arccos(cos_theta)

    rel_speeds = np.sqrt(speed_p**2 + speed_c**2 - 2 * speed_p * speed_c * cos_theta)
    rel_speeds = np.maximum(rel_speeds, 0)
    rel_speeds_kmh = rel_speeds * 3.6

    # angle d'approche
    angles = theta
    if angle_unit == "deg":
        angles = np.degrees(angles)

    if plot:
        fig, ax1 = plt.subplots(figsize=(12, 10))

        intervals_sec = []
        if start is not None and end is not None:
            intervals_sec = [(start / fps, end / fps)]
        else:
            from analysis_interactions import compute_ped_cyc_interactions_with_time
            interactions = compute_ped_cyc_interactions_with_time(df)
            pair = tuple(sorted((ped_id, cyc_id)))
            frames = interactions.get(pair, [])
            intervals = frames_to_intervals(frames)
            intervals_sec = [(s / fps, e / fps) for (s, e) in intervals]

        # vitesse m/s
        ax1.plot(common_times/fps, rel_speeds)
        ax1.set_xlabel("Time (s)", fontsize=15)
        ax1.set_ylabel("Relative speed (m/s)", fontsize=15)
        ax1.tick_params(axis="both", labelsize=13)
        ax1.set_title(f"Cyclist {cyc_id}'s relative speed compared to pedestrian {ped_id}", fontsize=15, fontweight="bold")
        ax1.grid()

        # vitesse km/h
        ax2 = ax1.twinx()
        ax2.plot(common_times/fps, rel_speeds_kmh)
        ax2.tick_params(axis="both", labelsize=13)
        smooth = lowess(rel_speeds_kmh, common_times/fps,frac=0.05)
        ax2.plot(smooth[:,0], smooth[:,1], linewidth=2, color ="red", label="Smoothed speed", alpha=0.5)
        ax2.set_ylabel("Relative speed (km/h)", fontsize=15)
        add_time_markers(ax1, intervals_sec)

        # légende fusionnée
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        by_label = dict(zip(labels1 + labels2, handles1 + handles2))
        ax1.legend(by_label.values(), by_label.keys(), fontsize=12)
        plt.tight_layout()
        plt.show()

    # distance (optionnel)
    distances = None

    if return_distance:
        ped_pos = ped[ped[COL_TIME].isin(times)]
        cyc_pos = cyc[cyc[COL_TIME].isin(times)]

        dx = cyc_pos["x_m"].values - ped_pos["x_m"].values
        dy = cyc_pos["y_m"].values - ped_pos["y_m"].values

        distances = np.hypot(dx, dy)

    # Résultats
    if return_distance:
        if return_class:
            from analysis_interactions import classify_relative_speed_interaction
            return times, rel_speeds, rel_speeds_kmh, angles, distances, classify_relative_speed_interaction(df, ped_id, cyc_id, times, rel_speeds_kmh)
        else:
            return times, rel_speeds, rel_speeds_kmh, angles, distances
    else:
        if return_class:
            from analysis_interactions import classify_relative_speed_interaction
            return times, rel_speeds, rel_speeds_kmh, angles, classify_relative_speed_interaction(df, ped_id, cyc_id, times, rel_speeds_kmh)
        else:
            return times, rel_speeds, rel_speeds_kmh, angles


def compute_pet(df, ped_id, cyc_id, fps, distance_threshold=1.0, plot=False, return_class=False):
    """
    Computes the PET (Post-Encroachment Time) between a pedestrian and a cyclist.
    """

    ped = df[df[COL_ID] == ped_id].sort_values(COL_TIME)
    cyc = df[df[COL_ID] == cyc_id].sort_values(COL_TIME)

    # 1. Find the conflict point (minimum distance)
    min_dist = float("inf")
    conflict_point = None

    for _, p in ped.iterrows():
        for _, c in cyc.iterrows():
            dx = p["x_m"] - c["x_m"]
            dy = p["y_m"] - c["y_m"]
            dist = np.hypot(dx, dy)

            if dist < min_dist:
                min_dist = dist
                conflict_point = ((p["x_m"] + c["x_m"]) / 2,
                                  (p["y_m"] + c["y_m"]) / 2)

    if conflict_point is None:
        return None

    cx, cy = conflict_point

    # 2. Find the passage times near the conflict point
    def get_crossing_time(agent_df):
        for _, row in agent_df.iterrows():
            dx = row["x_m"] - cx
            dy = row["y_m"] - cy
            dist = np.hypot(dx, dy)

            if dist < distance_threshold:
                return row[COL_TIME], row["x_m"], row["y_m"]

        return None, None, None

    t_ped, x_ped, y_ped = get_crossing_time(ped)
    t_cyc, x_cyc, y_cyc = get_crossing_time(cyc)

    if t_ped is None or t_cyc is None:
        return None

    # 3. PET
    pet = abs(t_ped - t_cyc) / fps

    if t_ped <= t_cyc:
        first_out = "Ped"
        last_in = "Cyc"
    else:
        first_out = "Cyc"
        last_in = "Ped"

    if plot:
        plt.figure(figsize=(8, 8))

        # trajectories
        plt.plot(ped["x_m"], ped["y_m"],
                 color=CLASS_COLORS.get(1, "blue"),
                 label=f"Ped {ped_id}")

        plt.plot(cyc["x_m"], cyc["y_m"],
                 color=CLASS_COLORS.get(2, "green"),
                 label=f"Cyc {cyc_id}")

        # conflict point
        plt.scatter(cx, cy,
                    color="red",
                    s=120,
                    marker="X",
                    label="Conflict point")

        plt.text(cx + 2.0, cy + 2.0, f"First out: {first_out}\nLast in: {last_in}", bbox=dict(facecolor="white", alpha=0.8))

        # conflict zone circle
        circle = plt.Circle(
            (cx, cy),
            distance_threshold,
            color="red",
            fill=False,
            linestyle="--",
            label="Conflict zone (1m radius)"
        )
        plt.gca().add_patch(circle)

        # annotation PET
        plt.text(
            cx + 1.0, cy - 1.0,
            f"PET = {pet:.2f}s",
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.8)
        )

        plt.title(f"Post-Encroachment Time (PET) (ped {ped_id} / cyc {cyc_id})", fontsize=15, fontweight="bold")
        plt.xlabel("x (m)", fontsize=15)
        plt.ylabel("y (m)", fontsize=15)
        plt.tick_params(axis="both", labelsize=13)
        plt.gca().invert_yaxis()
        plt.grid()
        plt.legend(fontsize=12)
        plt.axis("equal")
        plt.tight_layout()
        plt.show()

    if return_class:
        from analysis_interactions import classify_pet
        return pet, classify_pet(pet)
    
    return pet


# def compute_ttc(df, ped_id, cyc_id, fps, distance_threshold=5.0, plot=False, return_class=False):
#     """
#     Calcule le TTC (Time-To-Collision) entre un piéton et un cycliste.

#     Returns:
#         times, ttc_values
#     """

#     ped = df[df[COL_ID] == ped_id].sort_values(COL_TIME)
#     cyc = df[df[COL_ID] == cyc_id].sort_values(COL_TIME)

#     # vitesses
#     t_p, vx_p, vy_p = compute_velocity_vectors(ped, fps)
#     t_c, vx_c, vy_c = compute_velocity_vectors(cyc, fps)

#     if t_p is None or t_c is None:
#         return None, None

#     # positions (alignées sur t_p[1:])
#     ped_pos = ped[["x_m", "y_m"]].values[1:]
#     cyc_pos = cyc[["x_m", "y_m"]].values[1:]

#     # synchronisation temporelle
#     common_times = np.intersect1d(t_p, t_c)

#     ttc_values = []
#     valid_times = []

#     for t in common_times:
#         i_p = np.where(t_p == t)[0][0]
#         i_c = np.where(t_c == t)[0][0]

#         # vecteurs 
#         # (peut-être que ça ne fonctionne pas là ? car TTC généralement quand usagers dans même direction et l'un devant l'autre d'après Josué)
#         r = cyc_pos[i_c] - ped_pos[i_p]
#         v = np.array([vx_c[i_c] - vx_p[i_p],
#                       vy_c[i_c] - vy_p[i_p]])

#         v_norm_sq = np.dot(v, v)

#         if v_norm_sq == 0:
#             continue

#         dot = np.dot(r, v)

#         # condition approche 
#         if dot >= 0:
#             continue

#         ttc = - dot / v_norm_sq

#         if ttc < 0:
#             continue

#         ttc_values.append(ttc)
#         valid_times.append(t)

#     ttc_values = np.array(ttc_values)
#     valid_times = np.array(valid_times)

#     if len(ttc_values) == 0:
#         return valid_times, ttc_values, None

#     # TTC min
#     idx_min = np.argmin(ttc_values)
#     ttc_min = ttc_values[idx_min]
#     ttc_min_time = valid_times[idx_min]

#     if plot and len(ttc_values) > 0:

#         from analysis_interactions import compute_ped_cyc_interactions_with_time

#         interactions = compute_ped_cyc_interactions_with_time(df)
#         frames = interactions.get((ped_id, cyc_id), [])
#         intervals = frames_to_intervals(frames)

#         times_d, rel_speeds, rel_speeds_kmh, a = compute_relative_speed(
#             df, ped_id, cyc_id, fps, return_distance=False
#         )

#         t, angles = compute_approach_angle(df, ped_id, cyc_id, fps)

#         common_times = times_d
#         distances = []

#         for t in common_times:
#             p = ped[ped[COL_TIME] == t]
#             c = cyc[cyc[COL_TIME] == t]

#             if p.empty or c.empty:
#                 distances.append(np.nan)
#                 continue

#             dx = p.iloc[0]["x_m"] - c.iloc[0]["x_m"]
#             dy = p.iloc[0]["y_m"] - c.iloc[0]["y_m"]
#             distances.append(np.hypot(dx, dy))

#         distances = np.array(distances)

#         mask = np.zeros_like(valid_times, dtype=bool)
#         for start, end in intervals:
#             mask |= (valid_times >= start) & (valid_times <= end)
#         ttc_inter = ttc_values[mask]
#         if len(ttc_inter) == 0:
#             return None
#         ttc_min_inter = np.min(ttc_inter)

#         # plt.figure(figsize=(10, 4))
#         fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
#         ax_dist, ax_ttc, ax_angle = axes

#         ax_ttc.plot(valid_times/fps, ttc_values, color="purple", label="TTC")
#         ax_ttc.axhline(2, linestyle="--", color="red", label="Seuil critique (2s)")
#         ax_ttc.scatter(
#             ttc_min_time/fps,
#             ttc_min,
#             color="red",
#             s=80,
#             label=f"TTC min = {ttc_min:.2f}s"
#         )
#         ax_ttc.set_ylabel("TTC (s)")
#         ax_ttc.set_title("Time-To-Collision")
#         ax_ttc.grid()
#         ax_ttc.legend()
#         ax_ttc.text(
#             0.02, 0.95,
#             f"TTC min = {ttc_min:.2f}s\n(t = {ttc_min_time:.2f})",
#             transform=plt.gca().transAxes,
#             bbox=dict(facecolor="white", alpha=0.8)
#         )

#         ax_dist.plot(common_times/fps, distances, color="blue", label="Distance")
#         ax_dist.axhline(distance_threshold, linestyle="--", color="red", label="Seuil interaction (5m)")
#         ax_dist.set_ylabel("Distance (m)")
#         ax_dist.set_title("Distance piéton-cycliste")
#         ax_dist.grid()
#         ax_dist.legend()

#         ax_angle.plot(common_times/fps, angles, color="green", label="Angle")
#         ax_angle.axhline(0, linestyle="--", color="black", alpha=0.5, label="Frontal (0°)")
#         ax_angle.axhline(90, linestyle="--", color="orange", label="Croisement (90°)") # croisement latéral ou perpendiculaire mais pas forcément de collision
#         ax_angle.axhline(180, linestyle="--", color="red", label="Opposition (180°)")
#         ax_angle.set_ylabel("Angle (deg)")
#         ax_angle.set_xlabel("Temps (s)")
#         ax_angle.set_title("Angle d'approche")
#         ax_angle.grid()
#         ax_angle.legend()

#         intervals_sec = [(s / fps, e / fps) for (s, e) in intervals]
#         for ax in axes:
#             add_time_markers(ax, intervals_sec)

#         plt.tight_layout()
#         # plt.legend()
#         plt.show()
    
#     if return_class:
#         from analysis_interactions import classify_ttc
#         return valid_times, ttc_values, ttc_min, classify_ttc(ttc_min_inter)

#     return valid_times, ttc_values, ttc_min


def compute_ttac(df, id_A, id_B, fps, start=None, end=None, plot=False):
    """
    Computes the TTAC (Time-To-Avoided-Collision point) between two agents.
    TTAC = difference in arrival times at the conflict point (intersection of the trajectories).
    """
    # retrieve aligned data
    gA = df[df[COL_ID] == id_A].sort_values(COL_TIME) # généralement le piéton (même si pas obligatoire)
    gB = df[df[COL_ID] == id_B].sort_values(COL_TIME) # généralement le cycliste

    merged = gA.merge(gB, on=COL_TIME, suffixes=("_A", "_B"))

    if len(merged) < 2:
        return None, None, None

    times = merged[COL_TIME].values

    ttac_values = []

    for i in range(len(merged) - 1):

        # positions
        pA = np.array([merged.iloc[i]["x_m_A"], merged.iloc[i]["y_m_A"]])
        pB = np.array([merged.iloc[i]["x_m_B"], merged.iloc[i]["y_m_B"]])

        # compute TTAC only when the users are interacting (both within a 5 m radius)
        distance = np.linalg.norm(pA - pB)
        if distance > 5.0:
            ttac_values.append(None)
            continue

        # speeds
        pA_next = np.array([merged.iloc[i+1]["x_m_A"], merged.iloc[i+1]["y_m_A"]])
        pB_next = np.array([merged.iloc[i+1]["x_m_B"], merged.iloc[i+1]["y_m_B"]])

        vA = (pA_next - pA) * fps
        vB = (pB_next - pB) * fps

        # avoid degenerate cases
        if np.linalg.norm(vA) < 1e-3 or np.linalg.norm(vB) < 1e-3:
            ttac_values.append(None)
            continue

        # Compute the trajectory intersection point = conflict point (CP), assuming constant speeds and directions, with a 1 m radius around the CP.
        # solve : pA + tA*vA = pB + tB*vB

        A_mat = np.column_stack((vA, -vB))
        b_vec = pB - pA

        if np.linalg.matrix_rank(A_mat) < 2:
            ttac_values.append(None)
            continue

        try:
            t_vals = np.linalg.solve(A_mat, b_vec)
            tA, tB = t_vals

            # first and second agents to reach the CP
            t_first = min(tA, tB)
            t_second = max(tA, tB)

            if t_first < 0: # if the first agent to reach the CP has already passed the CP, do not compute TTAC
                ttac_values.append(None)
                continue

            # ttac = abs(tA - tB)
            ttac_values.append(t_second) # according to the article, TTAC = max(t1, t2) (i.e., the second agent to reach the CP)
        except:
            ttac_values.append(None)

    ttac_values = np.array(ttac_values, dtype=float)

    # filter out None values before computing the minimum
    valid = ttac_values[np.isfinite(ttac_values)]

    if len(valid) == 0:
        return times[:-1], ttac_values, None

    ttac_min = np.min(valid)
    min_idx = np.where(ttac_values == ttac_min)[0] # index du min
    first_time = times[:-1][min_idx[0]] / fps

    # plot
    if plot:
        intervals_sec = []
        if start is not None and end is not None:
            intervals_sec = [(start/fps, end/fps)]
        else:
            from analysis_interactions import compute_ped_cyc_interactions_with_time
            interactions = compute_ped_cyc_interactions_with_time(df)
            frames = interactions.get((id_A, id_B), [])
            intervals = frames_to_intervals(frames)
            intervals_sec = [(float(s/fps), float(e/fps)) for s, e in intervals]

        plt.figure(figsize=(8, 4))
        plt.plot(times[:-1]/fps, ttac_values)
        # plt.axhline(ttac_min, linestyle="--", label=f"TTAC min={ttac_min:.2f}")
        plt.scatter(times[:-1][min_idx]/fps, ttac_values[min_idx], color="red", s=60, zorder=5, label=f"TTAC min={ttac_min:.2f}s (at {first_time:.2f}s)")
        plt.title(f"Time To Avoided Collision Point (TTAC) during the interaction (ped {id_A} - cyc {id_B})", fontsize=15, fontweight="bold")
        plt.xlabel("Time (s)", fontsize=15)
        plt.ylabel("TTAC (s)", fontsize=15)
        plt.tick_params(axis="both", labelsize=13)
        add_time_markers(plt.gca(), intervals_sec)
        plt.legend(fontsize=12)
        plt.grid()
        plt.tight_layout()
        plt.show()

    return times[:-1], ttac_values, ttac_min


###############################################
# Spatio-temporal criteria for group–individual and group–group interactions
###############################################

def compute_vector_direction_series(df_agent):
    """
    Returns a series of normalized direction vectors between each pair of consecutive frames.
    """
    df_agent = df_agent.sort_values(COL_TIME)

    directions = []
    times = df_agent[COL_TIME].values

    for i in range(1, len(df_agent)):
        dx = df_agent["x_m"].iloc[i] - df_agent["x_m"].iloc[i - 1]
        dy = df_agent["y_m"].iloc[i] - df_agent["y_m"].iloc[i - 1]

        norm = np.hypot(dx, dy)

        if norm == 0:
            directions.append(np.array([0.0, 0.0]))
        else:
            directions.append(np.array([dx / norm, dy / norm]))

    return np.array(directions), times[1:]


def compute_directions_for_ids(df, ids, t, max_gap_frame=2):
    # interprétation: [1, 0]-> droite, [0, 1] -> haut, [-1, 0] -> gauche
    dirs = []
    valid_ids = []

    # impossible de calculer la direction à la 1ère frame
    # mais pq pas essayer avce t+1 (en pensant à inverser dx et dy) ?
    if t == 0:
        return np.empty((0, 2)), np.array([], dtype=int)

    for aid in ids:
        traj = df[df[COL_ID] == aid].sort_values(COL_TIME)

        curr = traj[traj[COL_TIME] == t]

        if curr.empty:
            # agent inexistant à cette frame
            continue

        prev = None

        # recherche de la dernière observation dispo (pour contourner le pbm de frames manquantes)
        for gap in range(1, max_gap_frame + 1):
            prev_candidate = traj[traj[COL_TIME] == t-gap]

            if not prev_candidate.empty:
                prev = prev_candidate
                break

        if prev is None:
            continue

        dx = curr.iloc[0]["x_m"] - prev.iloc[0]["x_m"]
        dy = curr.iloc[0]["y_m"] - prev.iloc[0]["y_m"]

        norm = np.hypot(dx, dy)
        if norm == 0:
            continue

        dirs.append([dx / norm, dy / norm])
        valid_ids.append(aid)

    return np.array(dirs), np.array(valid_ids)


def compute_direction_variation_old(directions):
    """
    Calcule les variations angulaires (en radians) entre directions successives.
    Pour déterminer ensuite quel agent est plus récatif que l'autre dans une interaction.
    """

    variations = []

    for i in range(1, len(directions)):
        v1 = directions[i - 1]
        v2 = directions[i]

        dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle = np.arccos(dot)  # radians

        variations.append(angle)

    return np.array(variations)


def compute_distance_series(df, id_A, id_B, distance_threshold=5.0, fps=None, start=None, end=None, plot=False, return_seconds=False):
    """
    Distance between 2 agents.
    """

    A = df[df[COL_ID] == id_A].sort_values(COL_TIME)
    B = df[df[COL_ID] == id_B].sort_values(COL_TIME)

    common_times = np.intersect1d(
        A[COL_TIME].values,
        B[COL_TIME].values
    )

    if len(common_times) == 0:
        return None, None

    A_sync = A[A[COL_TIME].isin(common_times)]
    B_sync = B[B[COL_TIME].isin(common_times)]

    dx = A_sync["x_m"].values - B_sync["x_m"].values
    dy = A_sync["y_m"].values - B_sync["y_m"].values

    distances = np.hypot(dx, dy)
    dist_min = distances.min()
    idx_min = np.nan
    t_min = np.nan
    t_min_plot = np.nan
    if len(distances) != 0:
        idx_min = np.argmin(distances) 
        t_min = common_times[idx_min]
        t_min_plot = t_min / fps

    if plot and start is not None and end is not None:
        plt.figure(figsize=(10, 4))
        plt.plot(common_times / fps, distances)
        plt.axhline(distance_threshold, color="red", linestyle="--", label="Spatial interaction limit (5m)")
        intervals_sec = [(start/fps, end/fps)]
        add_time_markers(plt.gca(), intervals_sec)
        if not np.isnan(t_min_plot):
            plt.scatter(t_min_plot, dist_min, color="red", zorder=5, label=f"Min distance = {dist_min:.2f}m (at {t_min_plot:.2f}s)")
        plt.xlabel("Time (s)", fontsize=15)
        plt.ylabel("Distance (m)", fontsize=15)
        plt.tick_params(axis="both", labelsize=13)
        plt.title(f"Distance between pedestrian {id_A} and cyclist {id_B} accross time", fontsize=15, fontweight="bold")
        plt.legend(fontsize=12)
        plt.grid()
        plt.tight_layout()
        plt.show()
        

    if return_seconds and fps is not None:
        return common_times / fps, distances

    return common_times, distances


def compute_relative_position_series(df, id_A, id_B, fps=None, start=None, end=None, plot=False, return_seconds=False):
    """
    Relative position of cyclist compared to pedestrian (B - A) (in A's reference frame, rather than in the global reference frame)
    """

    A = df[df[COL_ID] == id_A].sort_values(COL_TIME)
    B = df[df[COL_ID] == id_B].sort_values(COL_TIME)

    common_times = np.intersect1d(A[COL_TIME].values, B[COL_TIME].values)

    if len(common_times) == 0:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([])
        )

    start_idx = None
    end_idx = None
    if start is not None and end is not None:
        start_idx = np.where(common_times == start)[0][0]
        end_idx   = np.where(common_times == end)[0][0]

    A_sync = A[A[COL_TIME].isin(common_times)]
    B_sync = B[B[COL_TIME].isin(common_times)]

    Ax = A_sync["x_m"].to_numpy()
    Ay = -A_sync["y_m"].to_numpy()

    Bx = B_sync["x_m"].to_numpy()
    By = -B_sync["y_m"].to_numpy()

    x_rel = []
    y_rel = []
    distances = []

    for i in range(len(common_times)):
        # position du cycliste par rapport au piéton
        rel = np.array([Bx[i] - Ax[i], By[i] - Ay[i]])

        distances.append(np.linalg.norm(rel))

        # pedestrian direction
        if i < len(common_times) - 1:
            heading = np.array([Ax[i+1] - Ax[i], Ay[i+1] - Ay[i]])
        else:
            heading = np.array([Ax[i] - Ax[i-1], Ay[i] - Ay[i-1]])

        norm = np.linalg.norm(heading)

        if norm < 1e-6:
            # nearly stationary pedestrian: use the previous direction
            if i == 0:
                forward = np.array([1.0, 0.0])
            else:
                forward = forward_prev
        else:
            forward = heading / norm

        forward_prev = forward

        # left axis
        left = np.array([-forward[1], forward[0]])

        # projection into the pedestrian's local reference frame (rather than the global reference frame)
        x_rel.append(np.dot(rel, forward))
        y_rel.append(np.dot(rel, left))

    x_rel = np.array(x_rel)
    y_rel = np.array(y_rel)
    distances = np.array(distances)

    if plot:
        plt.figure(figsize=(7, 7))
        # relative trajectory
        plt.plot(x_rel, y_rel, "-o", markersize=3, linewidth=1.5, alpha=0.8)


        plt.axhline(0, color="red", linestyle="--", linewidth=2.0, label="Left (+) / Right (-) [m]")
        plt.axvline(0, color="black", linestyle="--", linewidth=2.0, label="Front (+) / Rear (-) [m]")

        # start and end if the interaction
        if start is not None and end is not None:
            plt.scatter(x_rel[start_idx], y_rel[start_idx], color="green", s=180, label=f"Start interaction (at {(start/fps):.2f}s)", zorder=1000)
            plt.scatter(x_rel[end_idx], y_rel[end_idx], color="orange", s=180, label=f"End interaction (at {(end/fps):.2f}s)", zorder=1000)
        plt.xlabel("Relative x (m)", fontsize=15)
        plt.ylabel("Relative y (m)", fontsize=15)
        plt.tick_params(axis="both", labelsize=13)
        plt.title(f"Relative position of cyclist {id_B} with respect to pedestrian {id_A} (in pedestrian local frame)", fontsize=15, fontweight="bold")
        plt.axis("equal")
        # plt.gca().invert_yaxis()
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()

    if return_seconds and fps is not None:
        return common_times / fps, x_rel, y_rel, distances

    return common_times, x_rel, y_rel, distances



def compute_clusters_and_hulls_over_time(df, min_samples=2,
                                         eps_dir=0.3,
                                         plot=False, fps=None,
                                         save_gif=False, output_path="clusters.gif",
                                         highlight_id=None):
    """
    Computes DBSCAN clustering and convex hulls frame by frame.

    Two DBSCAN algorithms are applied:
        1. Directional DBSCAN (clustering based on agents' movement directions)
        2. Spatial DBSCAN (clustering based on the distance between agents
        moving in the same direction)
    """

    history = {}

    for t in sorted(df[COL_TIME].unique()):
        frame = df[df[COL_TIME] == t]
        history[t] = {}

        for cls, name in [(1, "ped"), (2, "cyc")]:
            if name == "ped":
                eps = 1.5
            else:
                eps = 2.5

            sub = frame[frame[COL_CLASS] == cls]

            if len(sub) == 0:
                history[t][name] = None
                continue

            points = sub[["x_m", "y_m"]].values
            ids = sub[COL_ID].values

            clusters = []
            clusters_ids = []
            hulls = []

            noise_points = []
            noise_ids = []

            # =========================================================
            # 1) DBSCAN DIRECTION
            # =========================================================
            dirs, valid_ids_all = compute_directions_for_ids(df, ids, t)

            if len(dirs) < 2:
                # pas assez de direction -> fallback spatial direct
                # print(f"{t}: Pas assez de direction ! ", dirs, valid_ids_all)
                clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
                labels = clustering.labels_
                valid_groups = [(labels, ids, points)]
            else:
                # print(f"{t}: Directions : ", dirs, valid_ids_all)
                clustering_dir = DBSCAN(eps=eps_dir, min_samples=2, metric="cosine").fit(dirs)
                dir_labels = clustering_dir.labels_

                valid_groups = []

                for dlab in set(dir_labels):
                    mask_dir = dir_labels == dlab
                    sub_ids = valid_ids_all[mask_dir]

                    if dlab == -1:
                        # bruit directionnel -> bruit final direct
                        noise_pts = df[
                            (df[COL_TIME] == t) &
                            (df[COL_ID].isin(sub_ids))
                        ][["x_m", "y_m"]].values

                        noise_points.append(noise_pts)
                        noise_ids.extend(sub_ids)
                        continue

                    sub_pts = df[
                        (df[COL_TIME] == t) &
                        (df[COL_ID].isin(sub_ids))
                    ][["x_m", "y_m"]].values

                    if len(sub_pts) > 0:
                        valid_groups.append((None, sub_ids, sub_pts))

            # =========================================================
            # 2) DBSCAN SPATIAL (on each direction cluster)
            # =========================================================
            for _, group_ids, group_pts in valid_groups:

                if len(group_pts) < min_samples:
                    noise_points.append(group_pts)
                    noise_ids.extend(group_ids)
                    continue

                spatial_labels = DBSCAN(
                    eps=eps,
                    min_samples=min_samples
                ).fit_predict(group_pts)

                for lab in set(spatial_labels):

                    mask = spatial_labels == lab
                    pts = group_pts[mask]
                    ids_cluster = np.array(list(group_ids))[mask]

                    if lab == -1:
                        noise_points.append(pts)
                        noise_ids.extend(ids_cluster.tolist())
                        continue

                    clusters.append(pts)
                    clusters_ids.append(set(ids_cluster))

                    if len(pts) >= 3:
                        hulls.append((pts, ConvexHull(pts)))

            # concat bruit
            if len(noise_points) > 0:
                noise_points = np.vstack(noise_points)
            else:
                noise_points = np.empty((0, 2))

            history[t][name] = {
                "points": points,
                "ids": ids,
                "clusters": clusters,
                "clusters_ids": clusters_ids,
                "hulls": hulls,
                "noise": noise_points,
                "noise_ids": np.array(noise_ids),
                "n_clusters": len(clusters),
                "n_noise": len(noise_ids)
            }

    # =========================================================
    # VISUALISATION
    # =========================================================
    if plot:
        times = sorted(history.keys())
        fig, ax = plt.subplots(figsize=(6, 6))

        def update(i):
            ax.clear()

            from matplotlib.lines import Line2D

            handles = [
                Line2D([], [], marker='o', color='blue', linestyle='None',
                    markersize=8, label='Pedestrian cluster'),
                Line2D([], [], marker='s', color='green', linestyle='None',
                    markersize=8, label='Cyclist cluster'),
                Line2D([], [], color='black', linewidth=2, label='Convex hull'),
                Line2D([], [], marker='x', color='cyan', linestyle='None',
                    markersize=8, label='Pedestrian noise'),
                Line2D([], [], marker='x', color='lime', linestyle='None',
                    markersize=8, label='Cyclist noise'),
            ]

            if highlight_id is not None:
                handles.append(
                    Line2D([], [], marker='o',
                        markerfacecolor='none',
                        markeredgecolor='red',
                        linestyle='None',
                        markersize=10,
                        label=f'Highlighted agent ({highlight_id})')
                )

            ax.legend(handles=handles, loc="upper right", fontsize=12)

            t = times[i]
            data = history[t]

            legend = {}

            for name, color, noise_color, marker in [
                ("ped", "blue", "cyan", "o"),
                ("cyc", "green", "lime", "s")
            ]:

                if data[name] is None:
                    continue

                # clusters
                for idx, pts in enumerate(data[name]["clusters"]):
                    sc = ax.scatter(
                        pts[:, 0], pts[:, 1],
                        c=color,
                        marker=marker,
                        label=f"{name} cluster {idx}",
                        s=60
                    )
                    legend[f"{name} cluster {idx}"] = sc

                # noise
                if len(data[name]["noise"]) > 0:
                    sc = ax.scatter(
                        data[name]["noise"][:, 0],
                        data[name]["noise"][:, 1],
                        c=noise_color,
                        marker="x",
                        label=f"{name} noise",
                        s=60
                    )
                    legend[f"{name} noise"] = sc

                # hulls
                for pts, hull in data[name]["hulls"]:
                    for simplex in hull.simplices:
                        ln, = ax.plot(
                            pts[simplex, 0],
                            pts[simplex, 1],
                            color=color,
                            linewidth=2
                        )

                # ids
                for (x, y), aid in zip(data[name]["points"], data[name]["ids"]):
                    if aid == highlight_id:
                        ax.scatter(x, y, s=200, facecolors="none", edgecolors="red", linewidths=2, zorder=20)
                        ax.text(x + 0.2, y - 0.2, str(aid), fontsize=12, color=color, zorder=21)
                    else:
                        ax.text(x + 0.2, y - 0.2, str(aid), fontsize=12, color=color)

            ax.set_title(f"DBSCAN clusters and convex hulls\nFrame {t} ({t/fps:.2f}s)", fontsize=15, fontweight="bold")
            ax.set_xlim(df["x_m"].min(), df["x_m"].max())
            ax.set_ylim(df["y_m"].max(), df["y_m"].min())
            # ax.set_xlim(1, 70)
            # ax.set_ylim(df["y_m"].max(), 1)
            ax.set_xlabel("x (m)", fontsize=15)
            ax.set_ylabel("y (m)", fontsize=15)
            ax.tick_params(axis="both", labelsize=13)
            ax.grid()

        ani = FuncAnimation(fig, update, frames=len(times), interval=1000 / fps)

        if save_gif:
            ani.save(output_path, writer=PillowWriter(fps=fps))
            print(f"Saved: {output_path}")
        fig.tight_layout()
        plt.show()

    return history


# def compute_clusters_and_hulls_over_time_single_dbscan(
#         df,
#         min_samples=2,
#         eps_ped=1.5,
#         eps_cyc=2.5,
#         lambda_dir=1.63,
#         max_gap_frame=2,
#         plot=False,
#         fps=None,
#         save_gif=False,
#         output_path="clusters.gif",
#         highlight_id=None):
#     """
#     Calcule DBSCAN + convex hull frame par frame avec un seul DBSCAN.
#     """

#     history = {}

#     for t in sorted(df[COL_TIME].unique()):
#         frame = df[df[COL_TIME] == t]
#         history[t] = {}

#         for cls, name, eps in [(1, "ped", eps_ped),(2, "cyc", eps_cyc)]:
#             sub = frame[frame[COL_CLASS] == cls]

#             if len(sub) == 0:
#                 history[t][name] = None
#                 continue

#             points = sub[["x_m", "y_m"]].values
#             ids = sub[COL_ID].values

#             # =========================================================
#             # Calcul des directions
#             # =========================================================
#             dirs, valid_ids = compute_directions_for_ids(df, ids, t, max_gap_frame=max_gap_frame)

#             # Certains agents peuvent ne pas avoir de direction
#             # (première apparition, trajectoire trop courte...)
#             # On garde leur position mais direction nulle.

#             features = []

#             feature_ids = []
#             feature_points = []


#             for i, aid in enumerate(ids):
#                 point = points[i]

#                 # recherche direction correspondante
#                 idx = np.where(valid_ids == aid)[0]

#                 if len(idx) > 0:
#                     direction = dirs[idx[0]]
#                 else:
#                     direction = np.array([0.0, 0.0])

#                 feature = np.array([
#                     point[0], point[1],
#                     lambda_dir * direction[0],
#                     lambda_dir * direction[1]])

#                 features.append(feature)
#                 feature_ids.append(aid)
#                 feature_points.append(point)

#             features = np.array(features)
#             feature_ids = np.array(feature_ids)
#             feature_points = np.array(feature_points)

#             # =========================================================
#             # DBSCAN (unique -> spatial + directionnel)
#             # =========================================================
#             labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(features)

#             clusters = []
#             clusters_ids = []
#             hulls = []
#             noise_points = []
#             noise_ids = []

#             for lab in set(labels):
#                 mask = labels == lab
#                 pts = feature_points[mask]
#                 ids_cluster = feature_ids[mask]

#                 if lab == -1:
#                     noise_points.append(pts)
#                     noise_ids.extend(ids_cluster.tolist())
#                     continue

#                 clusters.append(pts)
#                 clusters_ids.append(set(ids_cluster))

#                 # convex hull seulement si >=3 points
#                 if len(pts) >= 3:
#                     try:
#                         hulls.append((pts,ConvexHull(pts)))

#                     except Exception:
#                         pass

#             if len(noise_points) > 0:
#                 noise_points = np.vstack(noise_points)
#             else:
#                 noise_points = np.empty((0,2))

#             history[t][name] = {
#                 "points": points,
#                 "ids": ids,
#                 "clusters": clusters,
#                 "clusters_ids": clusters_ids,
#                 "hulls": hulls,
#                 "noise": noise_points,
#                 "noise_ids": np.array(noise_ids),
#                 "n_clusters": len(clusters),
#                 "n_noise": len(noise_ids)
#             }

#     # =========================================================
#     # VISUALISATION
#     # =========================================================
#     if plot:
#         times = sorted(history.keys())
#         fig, ax = plt.subplots(figsize=(6,6))

#         def update(i):
#             ax.clear()
#             t = times[i]
#             data = history[t]

#             for name, color, noise_color, marker in [
#                 ("ped", "blue", "cyan", "o"),
#                 ("cyc", "green", "lime", "s")
#             ]:

#                 if data[name] is None:
#                     continue

#                 # clusters
#                 for idx, pts in enumerate(data[name]["clusters"]):
#                     ax.scatter(pts[:,0], pts[:,1], c=color, marker=marker, label=f"{name} cluster {idx}")

#                 # bruit
#                 if len(data[name]["noise"]) > 0:
#                     ax.scatter(data[name]["noise"][:,0],data[name]["noise"][:,1], 
#                         c=noise_color, marker="x", label=f"{name} noise")

#                 # hulls
#                 for pts, hull in data[name]["hulls"]:
#                     for simplex in hull.simplices:
#                         ax.plot(pts[simplex,0], pts[simplex,1], color=color, linewidth=2)


#                 # ids
#                 for (x,y), aid in zip(data[name]["points"], data[name]["ids"]):
#                     if aid == highlight_id:
#                         ax.scatter(x, y, s=200, facecolors="none", edgecolors="red", linewidths=2, zorder=20)
#                     ax.text(x+0.2, y-0.2, str(aid), fontsize=8)

#             ax.set_title(f"DBSCAN clusters and convex hulls - Frame {t} ({t/fps:.2f}s)")
#             ax.set_xlim(df["x_m"].min(), df["x_m"].max())
#             ax.set_ylim(df["y_m"].max(), df["y_m"].min())
#             ax.grid()

#         ani = FuncAnimation(fig,update,frames=len(times),interval=1000/fps)

#         if save_gif:
#             ani.save(output_path, writer=PillowWriter(fps=fps))
#             print(f"Saved: {output_path}")

#         plt.show()

#     return history


def match_clusters(prev_clusters, curr_clusters):
    """
    Associates clusters between t-1 and t based on point overlap.

    prev_clusters / curr_clusters = list of sets of IDs
    """

    matches = {}

    for i, prev in enumerate(prev_clusters):
        matches[i] = []

        for j, curr in enumerate(curr_clusters):
            overlap = len(prev.intersection(curr))

            if overlap > 0:
                matches[i].append(j)

    return matches


def detect_cluster_splits(prev_clusters, curr_clusters):
    """
    Detects if a cluster has split.
    Returns: splits = [(prev_id, [new_ids])]
    """
    matches = match_clusters(prev_clusters, curr_clusters)
    splits = []
    for prev_id, curr_ids in matches.items():
        if len(curr_ids) >= 2:
            splits.append((prev_id, curr_ids))

    return splits


def is_point_in_hull(point, hull_pts):
    """
    Verifies if a road user (e.g. a cyclist) is in a convex hull.
    """
    path = Path(hull_pts)
    return path.contains_point(point)


def is_cyclist_near_cluster(cluster_pts, cyclists_pts, threshold=2.5):
    """
    Checks whether a cyclist is close to a cluster.
    Used to determine whether a cluster splitting into two is caused by a cyclist passing through it.
    """
    for c in cyclists_pts:
        dists = np.linalg.norm(cluster_pts - c, axis=1)
        if np.min(dists) < threshold:
            return True
    return False


def detect_split_events_with_cyclists(history, distance_threshold=2.5):
    """
    Detects pedestrian cluster splits and checks whether a cyclist is involved.
    """

    events = []
    times = sorted(history.keys())
    prev_clusters = None

    for t in times:
        data = history[t]
        if data["ped"] is None:
            prev_clusters = None
            continue

        curr_clusters = data["ped"]["clusters_ids"]
        ped_points = data["ped"]["points"]
        ped_ids = data["ped"]["ids"]

        # cyclists
        cyc_pts = []
        cyc_ids = []

        if data["cyc"] is not None:
            cyc_pts = data["cyc"]["points"]
            cyc_ids = data["cyc"]["ids"]

        if prev_clusters is not None:

            splits = detect_cluster_splits(prev_clusters, curr_clusters)

            for prev_id, new_ids in splits:

                prev_ids = list(prev_clusters[prev_id])

                # retrieve the cluster positions BEFORE the split
                mask = np.isin(ped_ids, prev_ids)
                prev_pts = ped_points[mask]

                if len(prev_pts) == 0:
                    continue

                # verify cyclist
                involved = False
                involved_cyclists = []

                if len(cyc_pts) > 0:
                    involved = is_cyclist_near_cluster(prev_pts, cyc_pts, threshold=distance_threshold)
                
                for c_pt, c_id in zip(cyc_pts, cyc_ids):
                    dists = np.linalg.norm(prev_pts - c_pt, axis=1)

                    if np.min(dists) < distance_threshold:
                        involved_cyclists.append(c_id)

                events.append({
                    "time": t,
                    "type": "SPLIT",
                    "parent_cluster": prev_id,
                    "child_clusters": new_ids,
                    # "cyclist_involved": involved,
                    "cyclist_involved": len(involved_cyclists) > 0,
                    "cyclist_ids": [int(c) for c in involved_cyclists]
                })

        prev_clusters = curr_clusters

    return events


def detect_cyclists_in_hulls(history):
    results = []

    for t, data in history.items():

        if data["ped"] is None or data["cyc"] is None:
            continue

        cyc_pts = data["cyc"]["points"]
        cyc_ids = data["cyc"]["ids"]

        for i, (pts, hull) in enumerate(data["ped"]["hulls"]):
            hull_pts = pts[hull.vertices]
            cyclists_inside = []

            for c_pt, c_id in zip(cyc_pts, cyc_ids):
                if is_point_in_hull(c_pt, hull_pts):
                    cyclists_inside.append(c_id)

            if len(cyclists_inside) > 0:
                results.append({
                    "time_frame": t,
                    # "time_s": t / fps,
                    "event": "CYCLIST_IN_HULL",
                    "hull_id": i,
                    "cyclist_ids": [int(c) for c in cyclists_inside]
                })

    return results


def min_distance_inter_agent(A, B):
    """
    A, B: arrays (N, 2) and (M, 2). Minimum inter-agent distance (cyclist vs. pedestrian or cyclist vs. group).
    """
    if len(A) == 0 or len(B) == 0:
        return np.nan

    dists = cdist(A, B)
    return np.min(dists)


def min_distance_point_cluster(point, cluster_pts):
    d = np.linalg.norm(cluster_pts - point, axis=1)
    return np.min(d)


def min_distance_clusters(a_pts, b_pts):
    # Computes the minimum distance between two clusters, i.e. the distance between the closest pair of points, one from each cluster.
    d = np.linalg.norm(a_pts[:, None, :] - b_pts[None, :, :], axis=2)
    return np.min(d)


def hausdorff_distance(A, B):
    if len(A) == 0 or len(B) == 0:
        return np.nan

    d_ab = directed_hausdorff(A, B)[0]
    d_ba = directed_hausdorff(B, A)[0]
    return max(d_ab, d_ba)


def modified_hausdorff(A, B):
    if len(A) == 0 or len(B) == 0:
        return np.nan

    dists = cdist(A, B)
    mean_ab = np.mean(np.min(dists, axis=1))
    mean_ba = np.mean(np.min(dists, axis=0))
    return max(mean_ab, mean_ba)


def compute_cluster_distances(history, fps=None, plot=False):
    results = {}

    for t, data in history.items():

        if data["ped"] is None or data["cyc"] is None:
            continue

        cyc = data["cyc"]["points"]
        ped_clusters = data["ped"]["clusters"]

        results[t] = []

        for i, cluster in enumerate(ped_clusters):

            res = {
                "cluster_id": i,
                "min_dist": min_distance_inter_agent(cluster, cyc),
                "hausdorff": hausdorff_distance(cluster, cyc),
                "mod_hausdorff": modified_hausdorff(cluster, cyc)
            }

            results[t].append(res)
    
    if plot:
        # retrieve all existing cluster_ids
        cluster_ids = set()
        for t in results:
            for r in results[t]:
                cluster_ids.add(r["cluster_id"])

        cluster_ids = sorted(cluster_ids)

        times = sorted(results.keys())
        times_arr = np.array(times) / fps if fps else np.array(times)

        plt.figure(figsize=(10,5))

        for cid in cluster_ids:

            min_vals = []
            haus_vals = []
            mod_vals = []

            for t in times:
                r_list = results.get(t, [])

                r = next((r for r in r_list if r["cluster_id"] == cid), None)

                if r is None:
                    min_vals.append(np.nan)
                    haus_vals.append(np.nan)
                    mod_vals.append(np.nan)
                else:
                    min_vals.append(r["min_dist"])
                    haus_vals.append(r["hausdorff"])
                    mod_vals.append(r["mod_hausdorff"])

            # plot
            plt.plot(times_arr, min_vals, label=f"Cluster {cid} - min")
            plt.plot(times_arr, haus_vals, linestyle="--", label=f"Cluster {cid} - haus")
            plt.plot(times_arr, mod_vals, linestyle=":", label=f"Cluster {cid} - mod")

        plt.xlabel("Time (s)" if fps else "Frame")
        plt.ylabel("Distance (m)")
        plt.title("Distances cluster pedetsrians - cyclists")
        plt.legend(fontsize=8)
        plt.grid()
        plt.show()
    return results



def detect_ped_cluster_interactions(history, df, threshold=5.0):
    frame_events = {}

    for t, data in history.items():

        if data["ped"] is None or data["cyc"] is None:
            continue

        ped_clusters = data["ped"]["clusters_ids"]
        cyc_points = df[(df[COL_TIME] == t) & (df[COL_CLASS] == 2)]

        if len(cyc_points) == 0:
            continue

        events = []

        for _, cyc in cyc_points.iterrows():
            cyc_id = cyc[COL_ID]
            cyc_pos = np.array([cyc["x_m"], cyc["y_m"]])

            for ped_cluster in ped_clusters:

                ped_pts = df[
                    (df[COL_TIME] == t) &
                    (df[COL_ID].isin(ped_cluster))
                ][["x_m", "y_m"]].values

                if len(ped_pts) == 0:
                    continue

                d = min_distance_point_cluster(cyc_pos, ped_pts)

                if d <= threshold:
                    events.append({
                        "cyclist": cyc_id,
                        "ped_cluster": ped_cluster,
                        "distance": d,
                        "type": "CYCLIST_INDIVIDUAL - PEDESTRIAN_CLUSTER"
                    })

        frame_events[t] = events

    return frame_events


def detect_cluster_interactions(history, df, threshold=5.0):
    frame_events = {}

    for t, data in history.items():

        if data["ped"] is None or data["cyc"] is None:
            continue

        ped_clusters = data["ped"]["clusters_ids"]
        cyc_clusters = data["cyc"]["clusters_ids"]

        events = []

        for cyc_cluster in cyc_clusters:
            cyc_pts = df[
                (df[COL_TIME] == t) &
                (df[COL_ID].isin(cyc_cluster))
            ][["x_m", "y_m"]].values

            if len(cyc_pts) == 0:
                continue

            for ped_cluster in ped_clusters:
                ped_pts = df[
                    (df[COL_TIME] == t) &
                    (df[COL_ID].isin(ped_cluster))
                ][["x_m", "y_m"]].values

                if len(ped_pts) == 0:
                    continue

                d = min_distance_clusters(cyc_pts, ped_pts)

                if d <= threshold:
                    events.append({
                        "cyclist_cluster": cyc_cluster,
                        "ped_cluster": ped_cluster,
                        "distance": d,
                        "type": "CYCLIST_CLUSTER - PEDESTRIAN_CLUSTER"
                    })

        frame_events[t] = events

    return frame_events


def extract_entities(frame_data):
    """
    Retourne les entités avec leurs IDs :
    - clusters (avec ids)
    - noise (avec ids)
    """

    entities = []

    for cls_name in ["ped", "cyc"]:
        data = frame_data[cls_name]
        if data is None:
            continue

        # clusters
        for pts, ids in zip(data["clusters"], data["clusters_ids"]):
            entities.append({
                "type": cls_name + "_cluster",
                "points": pts,
                "ids": set(ids)
            })

        # noise
        if len(data["noise"]) > 0:
            # chaque point bruit = entité individuelle
            for pt, aid in zip(data["noise"], data["noise_ids"]):
                entities.append({
                    "type": cls_name + "_noise",
                    "points": np.array([pt]),
                    "ids": {aid}
                })

    return entities


def min_distance(A, B):
    return np.min(cdist(A, B))


def detect_interactions_at_frame(frame_data, threshold=5.0):
    """
    Single-frame interactions with IDs
    """

    entities = extract_entities(frame_data)
    interactions = []

    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):

            A = entities[i]
            B = entities[j]

            d = min_distance(A["points"], B["points"])

            if d <= threshold:
                if A["type"].startswith("ped") and B["type"].startswith("cyc"):
                    interactions.append({
                        "type_ped": A["type"],
                        "type_cyc": B["type"],
                        "ids_ped": A["ids"],
                        "ids_cyc": B["ids"]
                    })

                elif A["type"].startswith("cyc") and B["type"].startswith("ped"):
                    interactions.append({
                        "type_ped": B["type"],
                        "type_cyc": A["type"],
                        "ids_ped": B["ids"],
                        "ids_cyc": A["ids"]
                    })

    return interactions


def same_interaction(inter, active_inter):
    """
    Checks whether an interaction matches an active interaction.
    Criterion: ID overlap.
    """

    if inter["type_ped"] != active_inter["type_ped"]:
        return False
    if inter["type_cyc"] != active_inter["type_cyc"]:
        return False

    overlap_A = len(inter["ids_ped"].intersection(active_inter["ids_ped"])) > 0
    overlap_B = len(inter["ids_cyc"].intersection(active_inter["ids_cyc"])) > 0

    return overlap_A and overlap_B


def build_interaction_events(history, threshold=5.0, fps=None):
    """
    Builds interactions (including clusters and noise) with:
    - start
    - end
    - involved IDs (union over time)
    Final function to be used to capture all interactions from a filtered set of trajectories.
    """

    max_gap = 2 # 2 frames manquantes max par interaction
    active_events = []
    finished_events = []

    for t in sorted(history.keys()):

        frame_data = history[t]
        interactions = detect_interactions_at_frame(frame_data, threshold)

        updated = [False] * len(active_events)

        for inter in interactions:

            matched = False

            for i, act in enumerate(active_events):

                if same_interaction(inter, act):
                    # update event
                    act["end"] = t
                    act["ids_ped"].update(inter["ids_ped"])
                    act["ids_cyc"].update(inter["ids_cyc"])
                    act["missing_frames"] = 0 # interaction retrouvée, pas de frame manquante

                    updated[i] = True
                    matched = True
                    break

            if not matched:
                # nouvelle interaction
                active_events.append({
                    "type_ped": inter["type_ped"],
                    "type_cyc": inter["type_cyc"],
                    "ids_ped": set(inter["ids_ped"]),
                    "ids_cyc": set(inter["ids_cyc"]),
                    "start": t,
                    "end": t,
                    "missing_frames": 0
                })
                updated.append(True)

        # fermer celles non vues 2 frames de suite (i.e. interactions terminées)
        new_active = []
        for i, act in enumerate(active_events):
            if i < len(updated) and updated[i]:
                new_active.append(act) # interaction toujours en cours
            else:
                act["missing_frames"] += 1

                if(act["missing_frames"] <= max_gap):
                    # interaction potentiellement toujours en cours
                    new_active.append(act)
                else:
                    # interaction non observée dans plus de 2 frames consécutives -> terminée
                    del act["missing_frames"]
                    finished_events.append(act)

        active_events = new_active

    # fermer les interactions restantes
    for act in active_events:
        if "missing_frames" in act:
            del act["missing_frames"]
        # finished_events.append(act)
    finished_events.extend(active_events)

    allowed_pairs = {
        frozenset(["cyc_noise", "ped_cluster"]),
        frozenset(["ped_cluster", "cyc_noise"]),
        frozenset(["ped_cluster", "cyc_cluster"]),
        frozenset(["cyc_cluster", "ped_cluster"]),
        frozenset(["cyc_cluster", "ped_noise"]),
        frozenset(["ped_noise", "cyc_cluster"]),
        frozenset(["cyc_noise", "ped_noise"]),
        frozenset(["ped_noise", "cyc_noise"])
    }

    # print("Toutes les interactions : ", finished_events)
    filtered = []

    for e in finished_events:
        pair = frozenset([e["type_ped"], e["type_cyc"]])

        if pair not in allowed_pairs:
            continue
            
        # filtre durée 
        # (une interaction doit durer au moins 0.1 sec sinon pas assez de données et erreur)
        if fps is not None:
            duration_frames = e["end"] - e["start"]
            duration_s = duration_frames / fps

            if duration_s < 0.1:
                continue

        filtered.append(e)


    return filtered


def get_closest_points_with_ids(A_pts, B_pts, A_ids, B_ids):
    min_dist = np.inf
    best = None

    for i, a in enumerate(A_pts):
        for j, b in enumerate(B_pts):
            d = np.linalg.norm(a - b)
            if d < min_dist:
                min_dist = d
                best = (a, b, A_ids[i], B_ids[j])

    return best  # (pA, pB, idA, idB)


def compute_agent_velocity(df, agent_id, t, fps):
    traj = df[df[COL_ID] == agent_id].sort_values(COL_TIME)

    if t not in traj[COL_TIME].values:
        return None

    idx = traj[traj[COL_TIME] == t].index[0]

    if idx == traj.index.min():
        return None

    prev = traj.loc[idx - 1]

    curr = traj.loc[idx]

    dt_frames = curr[COL_TIME] - prev[COL_TIME]
    dt = dt_frames / fps

    if dt == 0:
        return None

    vx = (curr["x_m"] - prev["x_m"]) / dt
    vy = (curr["y_m"] - prev["y_m"]) / dt

    return np.array([vx, vy])


def compute_velocity(df_agent):
    df_agent = df_agent.sort_values(COL_TIME)

    dx = np.diff(df_agent["x_m"])
    dy = np.diff(df_agent["y_m"])
    dt = np.diff(df_agent[COL_TIME])

    speeds = np.hypot(dx, dy) / np.maximum(dt, 1e-6)
    return np.mean(speeds) if len(speeds) > 0 else 0


def compute_group_velocity_at_t(df, ids, t, fps):
    speeds = []

    for aid in ids:
        v = compute_agent_velocity(df, aid, t, fps)
        if v is not None:
            speeds.append(np.linalg.norm(v))  # norme

    if len(speeds) == 0:
        return None
    
    return np.mean(speeds) if speeds else None


def compute_group_relative_speed(df, ped_ids, cyc_ids, t, fps):
    v_ped = compute_group_velocity_at_t(df, ped_ids, t, fps)
    v_cyc = compute_group_velocity_at_t(df, cyc_ids, t, fps)

    if v_ped is None or v_cyc is None:
        return None

    return abs(v_cyc - v_ped)


def compute_group_diameter(points):
    if len(points) < 2:
        return 0
    return np.max(cdist(points, points))


def compute_group_density(points):
    if len(points) < 3:
        return 0

    hull = ConvexHull(points)
    area = hull.volume  # en 2D = aire

    return len(points) / area if area > 0 else 0


def compute_group_direction_angle(df, ped_ids, cyc_ids, t):
    """
    Angle between the mean directions of the two groups (in degrees), based on temporal motion (t-1 -> t)
    """

    def group_velocity(ids):
        df_now = df[(df[COL_TIME] == t) & (df[COL_ID].isin(ids))]
        df_prev = df[(df[COL_TIME] == t - 1) & (df[COL_ID].isin(ids))]

        if len(df_now) == 0 or len(df_prev) == 0:
            return None

        # centroid à t
        c_now = df_now[["x_m", "y_m"]].mean().values
        c_prev = df_prev[["x_m", "y_m"]].mean().values

        v = c_now - c_prev
        norm = np.linalg.norm(v)

        if norm == 0:
            return None

        return v / norm

    v_ped = group_velocity(ped_ids)
    v_cyc = group_velocity(cyc_ids)

    if v_ped is None or v_cyc is None:
        return None

    dot = np.clip(np.dot(v_ped, v_cyc), -1.0, 1.0)
    angle = np.degrees(np.arccos(dot))

    return angle


def angle_between(v1, v2):
    dot = np.dot(v1, v2)
    n = np.linalg.norm(v1) * np.linalg.norm(v2)

    if n == 0:
        return 0

    return np.degrees(np.arccos(np.clip(dot / n, -1, 1)))


def get_points_from_ids(frame_data, ids, cls_name):
    data = frame_data[cls_name]
    if data is None:
        return None

    points = []
    for (pt, aid) in zip(data["points"], data["ids"]):
        if aid in ids:
            points.append(pt)

    return np.array(points) if points else None



def is_noise_only_interaction(event):
    return ("noise" in event["type_ped"] and "noise" in event["type_cyc"])


def get_entity(frame, ids, cls_name):
    data = frame[cls_name]
    if data is None:
        return None, None

    pts = []
    valid_ids = []

    for p, aid in zip(data["points"], data["ids"]):
        if aid in ids:
            pts.append(p)
            valid_ids.append(aid)

    return np.array(pts), set(valid_ids)


def get_closest_points(A, B):
    D = cdist(A, B)
    i, j = np.unravel_index(np.argmin(D), D.shape)
    return A[i], B[j], D[i, j]


def get_cyclists_in_hull_for_interaction(interaction, hull_events):
    t_start = interaction["start"]
    t_end = interaction["end"]

    cyc_ids = set(interaction["ids_cyc"])

    relevant = []

    for ev in hull_events:
        t = ev["time_frame"]

        if t_start <= t <= t_end:
            # intersection des cyclistes
            if any(c in cyc_ids for c in ev["cyclist_ids"]):
                relevant.append(ev)

    return relevant


def get_split_events_for_interaction(interaction, split_events):
    t_start = interaction["start"]
    t_end = interaction["end"]

    cyc_ids = set(interaction["ids_cyc"])

    relevant = []

    for ev in split_events:
        t = ev["time"]

        if t_start <= t <= t_end:
            if ev["cyclist_involved"]:
                # vérifier que c’est un cycliste de l’interaction
                if any(c in cyc_ids for c in ev["cyclist_ids"]):
                    relevant.append(ev)

    return relevant


################################
# Functions for evaluating agent reactivity during interactions
################################

def compute_speed_variation_with_ref(df_agent, inter_start, inter_end, fps):
    """
    Speed variation relative to a reference (10 frames before the interaction). Used to detect abrupt changes
    """

    df = df_agent.sort_values(COL_TIME).copy()

    _, speed_ms, _ = compute_speed(df, fps)

    df = df.copy()
    df["speed_ms"] = np.nan
    df.loc[df.index[1:], "speed_ms"] = speed_ms

    # valeur de référence sur 10 frames avant le début de l'interaction
    pre_inter = df[(df[COL_TIME] < inter_start) & (df[COL_TIME] >= inter_start - 10)].copy()
    # v_ref = pre_inter["speed_ms"].mean()

    inter = df[(df[COL_TIME] >= inter_start) & (df[COL_TIME] <= inter_end)].copy()
    inter = inter.dropna(subset=["speed_ms"])

    if len(inter) < 2:
        return None

    speeds_ms = inter["speed_ms"].values
    times = inter[COL_TIME].values

    window = min(5, len(speeds_ms))
    speed_smooth = (pd.Series(speeds_ms).rolling(window=window, center=True, min_periods=1).mean().to_numpy())

    delta_v_ms = np.abs(np.diff(speed_smooth))

    # normalisation
    delta_v_norm_ms = delta_v_ms / (speed_smooth[:-1] + 1e-6)

    # calcul de la dérivée de la vitesse
    dt = 1 / fps
    dv = np.gradient(speed_smooth, dt)

    # moyenne de la dérivée (prendre eps=0.05 ? et voir si mv < -0.05 -> décélération ou mv > 0.05 -> accélération)
    mean_dv = np.nanmean(dv)

    # accélération brute (réactions rapides/brusques)
    dv_abs = np.abs(dv)

    fast_reaction = np.nanmax(dv_abs)   # pic de réaction

    # accélération soutenue (trend) (si négatif, décélération)
    t = np.arange(len(speed_smooth)) / fps

    # régression linéaire simple
    slope = np.polyfit(t, speed_smooth, 1)[0]  # km/h/s

    # séparation accel / decel (pour quantifier et voir si accélère plus que décélération)
    accel_part = dv[dv > 0]
    decel_part = dv[dv < 0]

    mean_accel = np.nanmean(accel_part) if len(accel_part) > 0 else 0
    mean_decel = np.nanmean(decel_part) if len(decel_part) > 0 else 0

    # par rapport à la valeur de référence
    pre_inter = pre_inter.dropna(subset=["speed_ms"])
    v_ref = np.nan
    mean_speed_change = np.nan
    mean_relative_change = np.nan
    if len(pre_inter) >= 5:
        v_ref = pre_inter["speed_ms"].mean()
        speed_change = speed_smooth - v_ref
        mean_speed_change = np.nanmean(speed_change) # positif -> accélération, négatif -> décélération

        relative_speed_change = (speed_smooth - v_ref) / (v_ref + 1e-6)

        mean_relative_change = np.nanmean(relative_speed_change)

    return {
        "times": times[1:],
        # "speeds": speeds_kmh[1:],
        # "delta_v": delta_v_kmh,
        # "delta_v_norm": delta_v_norm_kmh,
        "mean_delta": np.nanmean(delta_v_norm_ms),
        "max_delta": np.nanmax(delta_v_norm_ms),
        # "dv": dv,
        "mean_dv": mean_dv, # en m/s^2
        # "speed_smooth": speed_smooth
        "max_abs_acceleration": fast_reaction,

        # tendance globale
        "trend_slope": slope,

        # structure comportementale
        "mean_acceleration": mean_accel,
        "mean_deceleration": mean_decel,

        # indicateur global de la réactivité de l'agent
        "reactivity_score": np.nanmean(dv_abs),

        "mean_speed": np.nanmean(speed_smooth),
        "reference_speed": v_ref, # en m/s
        "mean_speed_change": mean_speed_change, # écart moyen à la vitesse relative
        "mean_relative_speed_change": mean_relative_change
    }



# def compute_direction_variation(df_agent, inter_start, inter_end, fps):
#     """
#     Calcule la variation de direction pendant une interaction.

#     Méthode :
#     - direction via vecteurs unitaires (dx, dy)
#     - lissage des vecteurs (= moyenne glissante sur fenêtre de 3, à voir si pas 5 plutôt)
#     - dérivée angulaire entre directions successives
#     """

#     from sklearn.linear_model import LinearRegression

#     df = df_agent.sort_values(COL_TIME).copy()

#     # calcul direction brute
#     # dx = df["x_m"].diff()
#     # dy = df["y_m"].diff()
#     k = 2
#     dx = df["x_m"].shift(-k) - df["x_m"].shift(k)
#     dy = df["y_m"].shift(-k) - df["y_m"].shift(k)

#     vec = np.stack([dx, dy], axis=1)

#     norms = np.linalg.norm(vec, axis=1)
#     norms[norms == 0] = np.nan

#     dir_vec = vec / norms[:, None]

#     df["dx"] = dir_vec[:, 0]
#     df["dy"] = dir_vec[:, 1]

#     # lissage des vecteurs via moyenne glissante
#     df["dx_smooth"] = (pd.Series(df["dx"]).rolling(window=5, center=True, min_periods=1).mean())

#     df["dy_smooth"] = (pd.Series(df["dy"]).rolling(window=5, center=True, min_periods=1).mean())

#     # renormalisation
#     smooth_vec = np.stack([df["dx_smooth"], df["dy_smooth"]], axis=1)
#     smooth_norm = np.linalg.norm(smooth_vec, axis=1)
#     smooth_norm[smooth_norm == 0] = np.nan

#     df["dx_smooth"] = df["dx_smooth"] / smooth_norm
#     df["dy_smooth"] = df["dy_smooth"] / smooth_norm

#     # extraction interaction
#     inter = df[(df[COL_TIME] >= inter_start) & (df[COL_TIME] <= inter_end)].copy()

#     if len(inter) < 2:
#         return None

#     v = inter[["dx_smooth", "dy_smooth"]].values
#     times = inter[COL_TIME].values

#     # direction inertielle de référence (25% des frmaes de l'interaction)
#     n_ref = max(2, int(len(v) * 0.25))
#     ref_vec = np.nanmean(v[:n_ref], axis=0)
#     ref_norm = np.linalg.norm(ref_vec)
#     if ref_norm == 0 or np.isnan(ref_norm):
#         return None

#     v_ref = ref_vec / ref_norm

#     # dérivée directionnelle (angle entre vect, variation angulaire par frame) VARIATION LOCALE
#     dtheta = [] # en radians
#     omega_series = [] # vitesse angulaire (= rapidité de réaction dans le changement de direction)
#     dt = 1 / fps

#     for i in range(len(v) - 1):
#         v1 = v[i]
#         v2 = v[i + 1]

#         norm1 = np.linalg.norm(v1)
#         norm2 = np.linalg.norm(v2)

#         if norm1 == 0 or norm2 == 0:
#             dtheta.append(np.nan)
#             continue

#         cosang = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
#         angle = np.arccos(cosang)

#         omega = angle / dt # en rad / s
#         omega_series.append(omega)

#         dtheta.append(angle)

#     dtheta = np.array(dtheta)

#     mean_dtheta = np.nanmean(dtheta)
#     max_dtheta = np.nanmax(dtheta)

#     # stabilité directionnelle
#     stability = 1.0 / (1.0 + np.nanstd(dtheta))

#     # dérivation globale par rapp à la direction inertielle
#     dtheta_ref = [] # en degrés
#     for vt in v:

#         normt = np.linalg.norm(vt)

#         if normt == 0 or np.isnan(normt):
#             dtheta_ref.append(np.nan)
#             continue

#         cosang = np.clip(np.dot(v_ref, vt), -1.0, 1.0)

#         angle = np.arccos(cosang)

#         dtheta_ref.append(np.degrees(angle)) 

#     dtheta_ref = np.array(dtheta_ref)

#     return {
#         "times": inter[COL_TIME].values[1:],
#         # "direction_vectors": v,
#         # "dtheta": dtheta,
#         "mean_dtheta": mean_dtheta,
#         "max_dtheta": max_dtheta,
#         "direction_stability": stability,
#         # "omega": omega_series,
#         "mean_omega": np.mean(np.array(omega_series)), # peut-être pas si utile
#         "cum_dtheta": np.nansum(dtheta), # rotation/dérivation cumulative
#         # "dtheta_ref": dtheta_ref,
#         "mean_dtheta_ref": np.nanmean(dtheta_ref),
#         "max_dtheta_ref": np.nanmax(dtheta_ref),
#         "std_ref": np.nanstd(dtheta_ref),
#         "cum_dtheta_ref": np.nansum(dtheta_ref),
#         "stability": 1 / (1 + np.nanstd(dtheta_ref)) # division inversement proportionnelle avec écart-type

#     }



def compute_spatial_deviation(df_agent, inter_start, inter_end):
    """
    Global spatial deviation from the inertial trajectory.
    Measures the distance between the actual positions and the inertial line.
    """

    df = df_agent.sort_values(COL_TIME).copy()

    inter = df[
        (df[COL_TIME] >= inter_start) &
        (df[COL_TIME] <= inter_end)
    ].copy()

    if len(inter) < 2:
        return None

    pts = inter[["x_m", "y_m"]].values

    # ligne inertielle (devrait être droite, si pas d'obstacle) (et si courbée de base ?)

    p0 = pts[0]
    p1 = pts[-1]

    direction = p1 - p0

    norm = np.linalg.norm(direction)

    if norm == 0:
        return None

    direction = direction / norm

    # distance entre les points de position réels et la ligne -> déviation
    deviations = []

    for p in pts:

        vec = p - p0

        proj_len = np.dot(vec, direction)

        proj = p0 + proj_len * direction

        dist = np.linalg.norm(p - proj)

        deviations.append(dist)

    deviations = np.array(deviations)


    mean_dev = np.mean(deviations)

    max_dev = np.max(deviations)

    cum_dev = np.sum(deviations)

    std_dev = np.std(deviations)

    stability = 1 / (1 + std_dev)

    return {
        "times": inter[COL_TIME].values,

        # "deviation_series": deviations,

        "mean_deviation": mean_dev,

        "max_deviation": max_dev,

        "cum_deviation": cum_dev,

        "std_deviation": std_dev,

        "trajectory_stability": stability,

        "inertial_line_start": p0,

        "inertial_line_end": p1
    }


# Fonction pour plot plusieurs critères spatio-temporels en même temps
def plot_interaction_series(
    series,
    title="Interaction metrics",
    xlabel="Time (s)",
    ylabel_left=None,
    ylabel_right=None,
    interaction_intervals=None,
    invert_y=False,
    figsize=(10, 5),
    grid=True,
    vertical_lines=None
):
    """
    Affiche plusieurs séries temporelles synchronisées.

    Paramètres
    ----------
    series : list of dict
        Chaque élément décrit une courbe.

        Exemple :
        [
            {
                "times": times_dist,
                "values": distances,
                "label": "Distance",
                "color": "royalblue",
                "axis": "left"
            },
            {
                "times": times_angle,
                "values": angles,
                "label": "Approach angle",
                "color": "darkorange",
                "axis": "right"
            }
        ]

    interaction_intervals : list[(t_start, t_end)], optional
        Intervalles d'interaction à mettre en évidence.

    vertical_lines : list of dict, optional
        Lignes verticales supplémentaires.

        Exemple :
        [
            {
                "x": 2.5,
                "label": "Event",
                "color": "red"
            }
        ]

        Si aucune ligne n'est fournie et qu'une série "Distance"
        et une série "Approach angle" sont présentes, la fonction
        ajoute automatiquement une ligne au niveau de la distance
        minimale.

    invert_y : bool
        Inverse l'axe Y gauche si nécessaire.
    """
    fig, ax_left = plt.subplots(figsize=figsize)
    ax_right = None
    handles = []
    labels = []

    distance_times = None
    distance_values = None

    angle_times = None
    angle_values = None

    for s in series:

        label = s.get("label", "")

        if label.lower() == "distance":
            distance_times = np.asarray(s["times"])
            distance_values = np.asarray(s["values"])

        elif label.lower() == "approach angle":
            angle_times = np.asarray(s["times"])
            angle_values = np.asarray(s["values"])

    for s in series:

        axis = s.get("axis", "left")

        if axis == "left":
            ax = ax_left

        else:
            if ax_right is None:
                ax_right = ax_left.twinx()

            ax = ax_right

        line, = ax.plot(
            s["times"],
            s["values"],
            color=s.get("color", None),
            linewidth=s.get("linewidth", 2),
            linestyle=s.get("linestyle", "-"),
            marker=s.get("marker", None),
            label=s.get("label", "")
        )

        handles.append(line)
        labels.append(s.get("label", ""))

    min_distance = None
    min_distance_time = None
    angle_at_min_distance = None

    if (
        distance_times is not None
        and distance_values is not None
    ):

        # Retirer les valeurs invalides
        valid = (
            np.isfinite(distance_times)
            & np.isfinite(distance_values)
        )

        if np.any(valid):

            valid_times = distance_times[valid]
            valid_distances = distance_values[valid]

            # Minimum de distance
            min_idx = np.argmin(valid_distances)

            min_distance = valid_distances[min_idx]
            min_distance_time = valid_times[min_idx]
            if (angle_times is not None and angle_values is not None):
                valid_angle = (np.isfinite(angle_times) & np.isfinite(angle_values))
                if np.any(valid_angle):
                    angle_t = angle_times[valid_angle]
                    angle_v = angle_values[valid_angle]

                    # Cherche l'observation d'angle
                    # temporellement la plus proche
                    angle_idx = np.argmin(np.abs(angle_t - min_distance_time))
                    angle_at_min_distance = angle_v[angle_idx]

    if interaction_intervals is not None:
        for start, end in interaction_intervals:
            ax_left.axvspan(
                start,
                end,
                color="grey",
                alpha=0.12,
                zorder=0
            )

    if min_distance_time is not None:
        ax_left.axvline(
            x=min_distance_time,
            color="crimson",
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            zorder=2
        )

        ax_left.scatter(
            min_distance_time,
            min_distance,
            color="royalblue",
            s=60,
            edgecolor="white",
            linewidth=1.3,
            zorder=5
        )

        # Annotation distance
        ax_left.annotate(
            # f"$d_{{min}}$ = {min_distance:.2f} m",
            f"{min_distance:.2f} m",
            xy=(
                min_distance_time,
                min_distance
            ),
            xytext=(10, 12),
            textcoords="offset points",
            fontsize=12,
            color="royalblue",
            weight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="royalblue",
                linewidth=1,
                alpha=0.9
            )
        )

        if (angle_at_min_distance is not None and ax_right is not None):
            ax_right.scatter(
                min_distance_time,
                angle_at_min_distance,
                color="darkorange",
                s=60,
                edgecolor="white",
                linewidth=1.3,
                zorder=5
            )

            # Annotation angle
            ax_right.annotate(
                # f"$\\theta$ = "
                f"{angle_at_min_distance:.1f}°",
                xy=(
                    min_distance_time,
                    angle_at_min_distance
                ),
                xytext=(10, -30),
                textcoords="offset points",
                fontsize=12,
                color="darkorange",
                weight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor="darkorange",
                    linewidth=1,
                    alpha=0.9
                )
            )

    ax_left.set_title(
        title,
        fontsize=15,
        fontweight="bold",
        pad=12
    )

    ax_left.set_xlabel(
        xlabel,
        fontsize=15
    )

    ax_left.tick_params(axis="both", labelsize=13)

    if ylabel_left is not None:

        ax_left.set_ylabel(
            ylabel_left,
            fontsize=15
        )

    if (
        ax_right is not None
        and ylabel_right is not None
    ):

        ax_right.set_ylabel(
            ylabel_right,
            fontsize=15
        )
    ax_right.tick_params(axis="both", labelsize=13)

    if invert_y:
        ax_left.invert_yaxis()

    if grid:

        ax_left.grid(
            True,
            which="major",
            axis="both",
            alpha=0.2,
            linestyle="-",
            linewidth=0.8
        )

        ax_left.set_axisbelow(True)

    if vertical_lines:

        for line in vertical_lines:

            x = line["x"]

            ax_left.axvline(
                x=x,
                color=line.get(
                    "color",
                    "red"
                ),
                linestyle=line.get(
                    "linestyle",
                    "--"
                ),
                linewidth=line.get(
                    "linewidth",
                    1.3
                ),
                alpha=line.get(
                    "alpha",
                    0.7
                )
            )

    if ax_right is not None:
        legend = ax_right.legend(
            handles,
            labels,
            loc="best",
            frameon=True,
            fancybox=True,
            framealpha=1.0,
            fontsize=12
        )
    else:
        legend = ax_left.legend(
            handles,
            labels,
            loc="best",
            frameon=True,
            fancybox=True,
            framealpha=1.0,
            fontsize=12
        )

    legend.set_zorder(1000)

    ax_left.spines["top"].set_visible(False)
    if ax_right is not None:
        ax_right.spines["top"].set_visible(False)

    fig.tight_layout()
    plt.show()

    return fig, ax_left, ax_right