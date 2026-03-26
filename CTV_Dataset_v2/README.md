## Description
The accompanied dataset (Introduced in IEEE International Conference on Intelligent Transportation Systems with the title "CTV-Dataset: A Shared Space Drone Dataset for Cyclist Road User Interaction Derived from Campus Experiments"), consists of different scenarios for cyclists interactions with different road users, like pedestrians and cars. For more information please refer to the dataset webspage https://www.ifi-mec.tu-clausthal.de/ctv-dataset


### Update 2.0:
The dataset version 2.0 includes now 2 different formats of the dataset:

#### (1) 4K Videos: 
stored in "scenarios" folder, it has the scenarios 4K videos, appended with the trajectory labels in pixels. This dataset is an improved version of the original dataset, we keep the same idea of puting the 4K videos of the scenarios along with the trajectories, but the videos and the trajecttories are now stabilzied.

#### (2) transformed trajectories:
stored in "transformed" folder, this dataset is new in this version. The CTV dataset contains road user tracks obtained from a high-altitude drone recording the same space for different camera positions. Hence the videos containing different perspectives are mapped videos to a common reference viewpoint. 

All scenarios are brought back and distributed based on the two recording areas (cf. the paper to understand the experiment areas). In this ready to use format:
1- the dataset is splitted into two folders based on the two areas
2- all trajectories are transformed to meters
3- each area has a background image in FHD resolution
4- to draw trajectories on the background, use the scale mtopx = 24. see the metadata.csv file


### Trajectories entries:
The trajectories are extracted and stored in txt as well as csv files. The trajectories has the following data:

frame - id - class - tl_x - tl_y - br_x - br_y

frame: frame number, related to the video
id: object id
class: road user type, please refer to the classes.csv file
tl_x - tl_y: object label top-left point
br_x - br_y: object label bottom-right point


## Citation
If you use this dataset in your work, please cite the dataset as follows:
[A. Mukbil, Y. Yousif, S. Hossain and J. P. Müller, "CTV-Dataset: A Shared Space Drone Dataset for Cyclist-Road User Interaction Derived from Campus Experiments," 2023 IEEE 26th International Conference on Intelligent Transportation Systems (ITSC), Bilbao, Spain, 2023, pp. 3186-3191, doi: 10.1109/ITSC57777.2023.10422465]

## License:
This dataset is distributed under Creative Commons Attribution 4.0 International (CC BY 4.0) License.