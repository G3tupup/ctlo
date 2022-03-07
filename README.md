# CTLO：Continuous Time Lidar Odometry

CTLO is a light-weight Continuous Time Lidar Odometry for lidar data collected by UGVs. As a pure lidar odometry, It can achieve 40-50hz with high accuracy.

<div  align="center">  
<img src="image/hdl_400.png" width = "540" align=center />
</div>
<center>hdl_400</center>

<div  align="center">  
<img src="image/Deutsches Museum.png" width = "540" align=center />
</div>
<center>Deutsches Museum</center>


# How to Run

```bash
mkdir -p ctlo_ws/src
cd ctlo_ws/src/
git clone https://github.com/G3tupup/ctlo.git
cd ..
catkin_make
source devel/setup.bash
roslaunch ctlo lidar_odometry.launch
```

## Dependency

CTLO is tested in Ubuntu 16.04, 18.04 and 20.04. Please install the following libraries before compilation.

- [ROS](http://wiki.ros.org/ROS/Installation)
- [Ceres Solver](http://www.ceres-solver.org/installation.html)
- Eigen
- PCL(only used to display pointcloud in rviz)
- OpenMP(optional)

## Config launch file

- lidar_topic: the topic of pointcloud in your rosbag. (Do not mix up "/xxx" with "xxx", which may cause error.)
- rosbag_folder_name: the path of your rosbag. (Do not forget "/" at the end.)
- rosbag_file_name: the name of your rosbag.
- accumulate_count: the number of pointcloud which belongs in one frame. (1 for default and 75 for cartographer datasets)

## Tips for higher speed

- set OPENMPTHREAD in active_feature_map.hpp larger if your cpu have more cores.
- set image_width_sample_step_ in feature_processor.hpp to 2/3/4 to reduce feature number.(This may reduce a bit of accuracy and robustness.)

# Datasets

VLP16 data collected by a UGV is highly recommended. Here are some tested public datasets: 

- [hdl_400](https://github.com/koide3/hdl_graph_slam)
- [same_position](https://github.com/RobustFieldAutonomyLab/jackal_dataset_20170608)
- [cartographer](https://google-cartographer-ros.readthedocs.io/en/latest/data.html#id4)

# Keystones
- feature extraction proposed in [LeGO-LOAM](https://github.com/RobustFieldAutonomyLab/LeGO-LOAM) and improve it
- continuous time solver proposed in [CT-ICP](https://github.com/jedeschaud/ct_icp)
- iterative ndt in voxels for fast data assosiation and O(1) sliding window map update

# Known Issues

- Default parameters in feature_processor.hpp are for VLP16 data, and you need to modify some of them for other kinds of lidar.
- Feature extraction is similar with LeGO-LOAM, so it may not behave well for UAVs datasets.
