The format for the dataset D1, D2, and D3
#################################################################
Tables for trajectories contain
Header:
time_step,
object_id,
x_coordinate,
y_coordinate,
user_type.

Each time step lasts 0.5 seconds

user type: 
1 -> pedestrian, 
2 -> cyclist, 
3 -> vehicle.

Ratio from pixel to meter
21.185660421977854:1


Tables for user info contain
Header:
dataset_id,
object_id,
user_type,
ingroup,
group_id,
gender,
age,
disable,
cellphone,
talking,
extra_luggage.

#################################################################
Data source:
Please cite
Cheng, Hao, Li, Yao, and Sester, Monika. "Pedestrian Group Detection in Shared Space" 2019 IEEE Intelligent Vehicles Symposium (IV). IEEE, 2019.

Background image
please cite
Imagery ©2019 Google, Map data ©2019 GeoBasis-DE/BKG(©2009), Google  
