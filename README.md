# Trajectories and Interactions Visualizer – Analysis of Pedestrian-Cyclist Interactions in Shared Spaces

## Project Description

This project provides a Python tool for the **visualization and analysis of road-user trajectories**, with a particular focus on **pedestrian-cyclist interactions**.

The objectives are to:
- visualize trajectories from different datasets,
- filter data to isolate pedestrian-cyclist interactions in shared spaces,
- analyze and **classify interactions based on spatio-temporal criteria** between pedestrians and cyclists.

This code was developed as part of a Master's internship focused on the **analysis and modeling of pedestrian-micromobility interactions in shared spaces**.

<p align="center">

| <img src="img/ctv.gif" width="450"> | <img src="img/clusters_2sep.gif" width="300"> |
|:----------------------------------:|:--------------------------------------------:|
| Trajectory visualization from a CTV video | DBSCAN clusters and convex hulls for a CTV video |

</p>

<p align="center">
  <img src="img/noname_D1_nofilter.gif" width="600" />
</p>

---

## Main Features

- Visualization:
  - **static** (complete trajectories)
  - **animated** (evolution over time)

- Support for multiple datasets:
  - **CTV**: reference benchmark containing multiple pedestrian and cyclist trajectories and interactions (Germany).
  - **Trajectory Shared Space (TSS)**: pedestrian, cyclist, and vehicle trajectories recorded on a straight bidirectional road near a university campus in Germany.
  - **Stanford Drone Dataset (SDD)**: trajectories of road users (pedestrians, cyclists, cars, and skaters) at 8 locations on the Stanford University campus.
  - **inD**: pedestrian, cyclist, and motor-vehicle trajectories recorded at 4 intersections in Germany.
  - **VRU**: pedestrian and cyclist trajectories recorded at an intersection in Germany (this dataset was then discarded).

- Support for different road users:
  - pedestrians
  - cyclists
  - vehicles

- Advanced data filtering

- Trajectory smoothing using a **Kalman filter**

- Interaction analysis:
  - inter-agent distance
  - relative speed
  - approach and heading angles
  - interaction classification

---

## Project Structure

```bash
.
├── main.py                     # Entry point (CLI)
├── loader.py                   # Dataset loading and normalization
├── visualisation.py            # Visualization (static + animation)
├── filters.py                  # Data filtering
├── config.py                   # Dataset configuration
├── utils.py                    # Utility functions (spatio-temporal criteria)
├── analysis_interactions.py    # Pedestrian-cyclist interaction analysis and classification
├── export_data.py              # Export data (classified interactions, filtered dataframe)
├── validation.py               # Launch the validation (confusion matrix, etc) of the automatic classification on TSS, inD and SDD
├── validation_simulation.py    # Launch the validation of SPACiSS simulation results in comparison with real CTV scenarios
│
├── trajectory_data/            # TSS dataset
├── CTV_Dataset_v2/             # CTV dataset
├── stanford_campus_dataset/    # SDD dataset
├── VRU_dataset/                # VRU dataset
```

---

## Installation and Usage

- Clone the repository:

```bash
git clone <repo_url>
cd TrajInt_visualizer
```

- Install the dependencies:

```bash
pip install -r requirements.txt
```

- Minimal command to launch a visualization:

```bash
python main.py --dataset <dataset> --file <file> --input-mode <input-mode> --mode <mode>
```

- Example command to visualize a CTV video:

```bash
python main.py --dataset ctv_area1 --input-mode single --mode animated --file P2_03_01_07.csv --use-unique-timestamps
```

---

## Command-Line Options

| Command line | Description | Default value | Possible values |
|--------------|-------------|---------------|-----------------|
| `--dataset` | Dataset to use | Required | `tss`, `ctv_area1`, `ctv_area2`, `ind`, `sdd`, `vru` |
| `--mode` | Visualization mode | Required | `static` or `animated` |
| `--input-mode` | Load all files or a single file | `all` | `all` or `single` |
| `--file` | Specific file to load (if `--input-mode single`) | `None` | See file names in each dataset |
| `--speed` | Temporal acceleration factor | `1` | |
| `--frame-step` | Frame step (reduces the number of displayed frames) | `1` | |
| `--highlight-id` | ID of the agent to highlight | `None` | |
| `--hide-ids` | Hide agent IDs | `False` | |
| `--save-video` | Save the animation as a video | `False` | `name.gif` or `name.mp4` |
| `--use-unique-timestamps` | Use only existing timestamps (no interpolation) | `False` | |
| `--no-smoothing-kalman` | Disable Kalman filtering for CTV | `False` | |
| `--vru-type` | VRU type: `pedestrians`, `cyclists`, `both` | `cyclists` | |
| `--vru-behavior` | VRU behavior: `starting`, `moving`, `stopping`, `waiting`, `all` | `starting` | |
| `--scene` | Scene ID (Stanford dataset) | `None` | e.g. `hyang` |
| `--video` | Video ID (Stanford dataset) | `None` | e.g. `video1` |
| `--print-interactions` | Print in the terminal the detected and classified interactions of a video | `False` | |
| `--analyze-interaction` | Analyze one interaction of a video (the user enters the interaction ID when asked in the terminal) | `False` | |
| `--export-interactions` | Export in a CSV file the detected and classified interactions of a video | `False` | |

---

## Datasets

This project uses four pedestrian/cyclist trajectory datasets.

| Dataset | Location | Road users | Data type | Source | Data used for this project | Filtering |
|---------|----------|------------|-----------|--------|----------------------------|-----------|
| [**CTV**](https://www.ifi-mec.tu-clausthal.de/ctv-dataset) | Germany | Pedestrians, cyclists, vehicles | Videos + trajectories | [Paper](https://ieeexplore.ieee.org/document/10422465) | area 1 (P2, P6), area 2 (P5_02) | Kalman filter |
| [**TSS**](https://www.ifi-mec.tu-clausthal.de/ctv-dataset) | Germany | Pedestrians, cyclists, vehicles | Trajectories | [Paper](https://ieeexplore.ieee.org/document/8813849) | D1, D2 | Trajectories are removed whenever a vehicle is present in the same frame as both a pedestrian and a cyclist.|
| [**inD**](https://gitlab.tu-clausthal.de/pka20/Trajectory-Prediction-Pedestrian) | Germany | Pedestrians, cyclists, vehicles | Trajectories | [Paper](https://ieeexplore.ieee.org/document/9304839) | Intersection Neuköllner Strasse (recordings 0-6), Intersection Frakenburg (recordings 18-29) | Pedestrian and cyclist trajectories are removed whenever a vehicle is present within a 5 m radius of their interaction.|
| [**SDD**](https://cvgl.stanford.edu/projects/uav_data/) | USA | Pedestrians, cyclists, vehicles, skaters | Videos + trajectories | [Paper](https://link.springer.com/chapter/10.1007/978-3-319-46484-8_33) | bookstore (videos 0-6), coupa (videos 0-3), gates (videos 0, 2, 7), hyang (videos 0-7, 10-14), nexus (videos 1, 6-10), quad (videos 0-3) | Pedestrian and cyclist trajectories are removed whenever a vehicle is present within a 5 m radius of their interaction.|

---


## Spatio-Temporal Criteria

| Criterion | Definition | Purpose for Classification | Interpretation |
|-----------|------------|----------------------------|----------------|
| **DBSCAN** | Detection of clusters and noise | Detect groups of users (including group splits) and individuals outside groups according to their direction and their distance | **Cluster**: at least 2 users of the same type, moving in the same direction and within a radius of at most 2 m. **Noise**: individual outside a cluster. |
| **Convex Hull** | Convex envelope containing all points of a cluster | Detect another type of user inside the envelope | Weaving |
| **Inter-agent distance** | Euclidean distance between two agents at each time step | Detect proximity and interaction start/end | The smaller the distance, the stronger the interaction |
| **Minimum distance** | Smallest distance reached during the interaction | Identify critical situations | Used to characterize the level of risk |
| **Relative speed** | Difference in speed between two agents | Detect convergence or divergence | High relative speed = dynamic interaction |
| **Direction** | Angle between the agents' velocity vectors | Characterize interaction geometry | Same direction, opposite directions, crossing (perpendicular interaction) |
| **Approach angle** | Angle between the agents' movement directions | Characterize the cyclist's approach toward the pedestrian | Frontal approach, crossing, moving away |
| **Relative position** | Position of one agent relative to the other (ahead, behind, lateral) | Understand the spatial configuration | Helps distinguish overtaking from following |
| **PET (Post-Encroachment Time)** | Estimated time between one agent leaving a conflict zone (intersection of trajectories) and another entering it | Detect risky situations | Low PET = potential danger |
| **TTAC (Time-To-Avoided-Collision-Point)** | Remaining time for the agents to avoid each other during the interaction | Identify the critical moment | Helps anticipate the interaction |
| **Interaction duration** | Total duration of the interaction | Classify short vs. long interactions | Long duration = structured interaction |

---

## Classification of Pedestrian-Cyclist Interactions in Shared Spaces

### Three Main Interaction Categories

- Individual–Individual
- Group–Individual
- Group–Group

### Main Classes

- Avoidance
- Overtaking
- Following
- Crossing
- Near-collision
- Give way
- Moving away
- Weaving and group splitting
- Weaving
- Group splitting
- Overtaking, avoiding and crossing group
- Weak interaction (too far apart and low risk)
