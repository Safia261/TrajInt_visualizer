import pandas as pd
import numpy as np
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.lines import Line2D
from config import *


def load_background_image(image_path, cfg):
    img = mpimg.imread(image_path)
    h, w = img.shape[:2]

    if cfg["scale"] == "pixel":
        meters_per_pixel = 1.0 / cfg["pixels_per_meter"]

    elif cfg["scale"] == "meter":
        if "meters_to_pixels" in cfg:
            meters_per_pixel = 1.0 / cfg["meters_to_pixels"]
        else:
            meters_per_pixel = 1.0 # coordonnées déjà en mètres

    img_w_m = w * meters_per_pixel
    img_h_m = h * meters_per_pixel

    return img, img_w_m, img_h_m


def add_background(ax, img, img_w_m, img_h_m):
    ax.imshow(
        img,
        extent=[0, img_w_m, img_h_m, 0],
        origin="upper",
        alpha=0.95
    )
    ax.set_xlim(0, img_w_m)
    ax.set_ylim(img_h_m, 0)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.25)


def add_legend(ax, df):
    # legend_elements = [
    #     Line2D([0], [0], color=CLASS_COLORS[1], lw=2, label="Pedestrian"),
    #     Line2D([0], [0], color=CLASS_COLORS[2], lw=2, label="Cyclist"),
    # ]

    # if include_cars:
    #     legend_elements.append(
    #         Line2D([0], [0], color=CLASS_COLORS[3], lw=2, label="Car")
    #     )

    # ax.legend(handles=legend_elements, loc="upper right")

    classes_present = sorted(df[COL_CLASS].unique())

    legend_elements = []

    for c in classes_present:
        if c == 0:
            continue

        name = CLASS_NAMES.get(c, f"Class {c}")
        color = CLASS_COLORS.get(c, "yellow")

        legend_elements.append(
            Line2D([0], [0], color=color, lw=2, label=name)
        )

    ax.legend(handles=legend_elements, loc="upper right")


def add_mouse_coordinates(fig, ax):
    coord_text = ax.text(
        0.01, 0.99, "",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray")
    )

    def on_mouse_move(event):
        if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
            coord_text.set_text(f"x = {event.xdata:.2f} m\ny = {event.ydata:.2f} m")
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)
    return coord_text


# ============================================================
# MODE 1 : AFFICHAGE STATIQUE
# ============================================================

def plot_static_trajectories(df, image_path, cfg, show_ids=True):
    fig, ax = plt.subplots(figsize=(12, 8))
    # img, img_w_m, img_h_m = load_background_image(image_path, cfg)
    if image_path is not None:
        img, img_w_m, img_h_m = load_background_image(image_path, cfg)
        add_background(ax, img, img_w_m, img_h_m)
    else:
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(True, alpha=0.25)

        ax.set_xlim(df["x_m"].min(), df["x_m"].max())
        ax.set_ylim(df["y_m"].max(), df["y_m"].min())

    # include_cars = 3 in set(df[COL_CLASS].unique())

    # fig, ax = plt.subplots(figsize=(12, 8))
    # add_background(ax, img, img_w_m, img_h_m)
    add_legend(ax, df)
    add_mouse_coordinates(fig, ax)

    grouped = df.groupby(COL_ID)

    for object_id, traj in grouped:
        traj = traj.sort_values(COL_TIME)

        agent_class = int(traj[COL_CLASS].iloc[0])
        color = CLASS_COLORS.get(agent_class, "yellow")

        x = traj["x_m"].values
        y = traj["y_m"].values

        ax.plot(
            x, y,
            color=color,
            linewidth=STATIC_LINEWIDTH,
            alpha=STATIC_ALPHA
        )

        if len(x) > 0:
            ax.scatter(x[0], y[0], color=color, s=22, marker="o", edgecolors="black", zorder=3)
            ax.scatter(x[-1], y[-1], color=color, s=28, marker="x", zorder=3)

        if show_ids and len(x) > 0:
            ax.text(
                x[0], y[0],
                f"{object_id}",
                color=color,
                fontsize=ID_FONT_SIZE,
                weight="bold",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.5)
            )

    ax.set_title("Trajectoires - affichage statique")
    plt.tight_layout()
    plt.show()


# ============================================================
# MODE 2 : AFFICHAGE ANIME
# ============================================================
def build_time_grid(df, dataset_fps, use_unique_timestamps, speed=1.0):
    times = np.sort(df[COL_TIME].unique())

    if use_unique_timestamps:
        if len(times) == 0:
            return times
        t_min = times.min()
        return t_min + (times - t_min) * speed

    t_min = df[COL_TIME].min()
    t_max = df[COL_TIME].max()

    # plus speed est grand, plus on avance vite dans le temps simulé
    dt = speed / dataset_fps
    return np.arange(t_min, t_max + dt, dt)


def interpolate_agent_at_time(agent_df, t):
    times = agent_df[COL_TIME].values
    xs = agent_df["x_m"].values
    ys = agent_df["y_m"].values
    # behaviors = agent_df["behavior"].values 
    has_behavior = "behavior" in agent_df.columns

    if has_behavior:
        behaviors = agent_df["behavior"].values

    if t < times[0] or t > times[-1]:
        return None

    idx_exact = np.where(times == t)[0]
    if len(idx_exact) > 0:
        i = idx_exact[0]
        return xs[i], ys[i]

    idx_right = np.searchsorted(times, t)
    if idx_right == 0 or idx_right >= len(times):
        return None

    idx_left = idx_right - 1
    t0, t1 = times[idx_left], times[idx_right]
    x0, x1 = xs[idx_left], xs[idx_right]
    y0, y1 = ys[idx_left], ys[idx_right]

    if has_behavior:
        if behaviors[idx_left] != behaviors[idx_right]:
            return None

    alpha = (t - t0) / (t1 - t0)
    x = x0 + alpha * (x1 - x0)
    y = y0 + alpha * (y1 - y0)

    return x, y


def get_past_trajectory(agent_df, t, tail_length=None):
    # past = agent_df[agent_df[COL_TIME] <= t][["x_m", "y_m", COL_TIME]].copy()
    has_behavior = "behavior" in agent_df.columns
    if has_behavior:
        current_rows = agent_df[agent_df[COL_TIME] <= t]
        
        if len(current_rows) == 0:
            return None, None

        current_behavior = current_rows["behavior"].iloc[-1]

        past = current_rows[current_rows["behavior"] == current_behavior][
            ["x_m", "y_m", COL_TIME]
        ].copy()
    else:
        past = agent_df[agent_df[COL_TIME] <= t][["x_m", "y_m", COL_TIME]].copy()

    interp = interpolate_agent_at_time(agent_df, t)
    if interp is None:
        return None, None

    x_t, y_t = interp

    if len(past) == 0 or past[COL_TIME].iloc[-1] != t:
        extra = pd.DataFrame([{"x_m": x_t, "y_m": y_t, COL_TIME: t}])
        past = pd.concat([past, extra], ignore_index=True)

    if tail_length is not None and len(past) > tail_length:
        past = past.iloc[-tail_length:]

    return past["x_m"].values, past["y_m"].values


def animate_trajectories(
    df,
    image_path,
    cfg,
    show_ids=True,
    fps=25,
    use_unique_timestamps=False,
    tail_length=None,
    speed=1.0,
    frame_step=1,
    save_video=None,
    highlight_id=None
):
    fig, ax = plt.subplots(figsize=(12, 8))
    # img, img_w_m, img_h_m = load_background_image(image_path, cfg)
    if image_path is not None:
        img, img_w_m, img_h_m = load_background_image(image_path, cfg)
        add_background(ax, img, img_w_m, img_h_m)
    else:
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(True, alpha=0.25)

        ax.set_xlim(df["x_m"].min(), df["x_m"].max())
        ax.set_ylim(df["y_m"].max(), df["y_m"].min())

    dataset_fps = cfg.get("fps", DEFAULT_FPS)

    # time_grid = build_time_grid(df, fps=fps, use_unique_timestamps=use_unique_timestamps)
    time_grid = build_time_grid(
        df,
        dataset_fps=dataset_fps,
        use_unique_timestamps=use_unique_timestamps,
        speed=speed
    )

    # include_cars = 3 in set(df[COL_CLASS].unique()) # à revoir pour ça

    agents = {}
    for object_id, g in df.groupby(COL_ID):
        g = g.sort_values(COL_TIME).reset_index(drop=True)
        agent_class = int(g[COL_CLASS].iloc[0])
        # agents[object_id] = {
        #     "df": g,
        #     "class": agent_class,
        #     "color": CLASS_COLORS.get(agent_class, "yellow")
        # }

        base_color = CLASS_COLORS.get(agent_class, "yellow")

        if highlight_id is not None and object_id == highlight_id:
            color = HIGHLIGHT_COLOR
        else:
            color = base_color

        agents[object_id] = {
            "df": g,
            "class": agent_class,
            "color": color
        }

    # fig, ax = plt.subplots(figsize=(12, 8))
    # add_background(ax, img, img_w_m, img_h_m)
    add_legend(ax, df)
    coord_text = add_mouse_coordinates(fig, ax)

    ax.set_title("Trajectoires - affichage animé")

    time_text = ax.text(
        0.01, 0.89,
        "",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray")
    )

    status_text = ax.text(
        0.01, 0.83,
        "PLAY",
        transform=ax.transAxes,
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.8)
    )

    if cfg["type"] == "vru":

        behavior_text = ax.text(
            0.01, 0.75,
            "",
            transform=ax.transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.8)
        )
    else:
        behavior_text = None

    trail_artists = {}
    point_artists = {}
    id_artists = {}

    for object_id, info in agents.items():
        color = info["color"]

        line, = ax.plot([], [], color=color, linewidth=ANIM_LINEWIDTH, alpha=ANIM_ALPHA)
        # point = ax.scatter([], [], s=AGENT_MARKER_SIZE, color=color, edgecolors="black", zorder=5)
        is_highlight = (highlight_id is not None and object_id == object_id)

        point = ax.scatter(
            [],
            [],
            s=HIGHLIGHT_SIZE if object_id == highlight_id else AGENT_MARKER_SIZE,
            color=color,
            edgecolors="black",
            zorder=5
        )

        txt = ax.text(
            0, 0, "",
            fontsize=8,
            color=color,
            weight="bold",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.2),
            visible=False
        )

        trail_artists[object_id] = line
        point_artists[object_id] = point
        id_artists[object_id] = txt

    # paused = {"value": False}
    # current_frame = {"idx": 0}
    paused = {"value": False}
    current_frame = {"idx": 0}
    manual_step = {"value": False}

    def on_click(event):
        paused["value"] = not paused["value"]

    fig.canvas.mpl_connect("button_press_event", on_click)


    def on_key(event):
        if event.key == " ":
            paused["value"] = not paused["value"]

        elif event.key == "right":
            paused["value"] = True
            current_frame["idx"] = min(current_frame["idx"] + 1, len(time_grid) - 1)
            manual_step["value"] = True

        elif event.key == "left":
            paused["value"] = True
            current_frame["idx"] = max(current_frame["idx"] - 1, 0)
            manual_step["value"] = True

    fig.canvas.mpl_connect("key_press_event", on_key)


    # def update(frame_idx):
    #     if paused["value"]:
    #         frame_idx = current_frame["idx"]
    #     else:
    #         current_frame["idx"] = frame_idx

    def update(frame_idx):
        if manual_step["value"]:
            frame_idx = current_frame["idx"]
            manual_step["value"] = False

        elif paused["value"]:
            frame_idx = current_frame["idx"]

        else:
            current_frame["idx"] = frame_idx

        t = time_grid[frame_idx]
        time_text.set_text(f"Temps = {t:.3f}")

        status_text.set_text("PAUSE" if paused["value"] else "PLAY")

        current_rows = df[df[COL_TIME] <= t]

        if cfg["type"] == "vru" and len(current_rows) > 0:
            current_behavior = current_rows["behavior"].iloc[-1]
            behavior_text.set_text(f"Behavior: {current_behavior}")

        for object_id, info in agents.items():
            g = info["df"]

            pos = interpolate_agent_at_time(g, t)

            if pos is None:
                trail_artists[object_id].set_data([], [])
                point_artists[object_id].set_offsets(np.empty((0, 2)))
                id_artists[object_id].set_visible(False)
                continue

            x, y = pos

            tx, ty = get_past_trajectory(g, t, tail_length=tail_length)
            if tx is not None and ty is not None:
                trail_artists[object_id].set_data(tx, ty)
            else:
                trail_artists[object_id].set_data([], [])

            point_artists[object_id].set_offsets(np.array([[x, y]]))

            if show_ids:
                id_artists[object_id].set_position((x, y))
                id_artists[object_id].set_text(str(object_id))
                id_artists[object_id].set_visible(True)
            else:
                id_artists[object_id].set_visible(False)

        artists = (
            list(trail_artists.values())
            + list(point_artists.values())
            + list(id_artists.values())
            + [time_text, coord_text, status_text]
        )
        return artists

    # ani = FuncAnimation(
    #     fig,
    #     update,
    #     frames=len(time_grid),
    #     interval=1000 / fps,
    #     blit=False,
    #     repeat=True
    # )

    frame_indices = range(0, len(time_grid), max(1, frame_step))

    ani = FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        interval=1000 / fps, # fps != dataset_fps, c'est le frame rate de l'animation
        blit=False,
        repeat=True
    )

    if save_video is not None:
        print("Téléchargement vidéo en cours...")
        # writer = FFMpegWriter(fps=25)
        # ani.save("video.mp4", writer=writer)
        # ani.save(save_video, writer="ffmpeg", fps=fps, dpi=150)
        if shutil.which("ffmpeg") is not None:
            ani.save(save_video, writer="ffmpeg", fps=fps, dpi=150)
        else:
            print("ffmpeg non trouvé, export en GIF")
            gif_path = save_video.replace(".mp4", ".gif")
            ani.save(gif_path, writer="pillow", fps=fps)
        print("Téléchargement vidéo terminé.")
    # ani.save("trajectoires.mp4", writer="ffmpeg", fps=fps, dpi=150)

    plt.tight_layout()
    plt.show()
    return ani