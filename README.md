# ROS Wrapper for TransNet on visual-based transparent object robot manipulation

## Prerequisite

Install [ROS](http://wiki.ros.org/ROS/Installation) and [TransNet](https://github.com/huijieZH/TransNet)

Install ros_numpy 

`sudo apt-get install ros-$release-ros-numpy`

Install numpy, open3d, opencv-python, scipy, absl using TransNet python environment

(optional) Install [g2opy](https://github.com/uoip/g2opy) and [dt-apriltags](https://pypi.org/project/dt-apriltags/) from  if you need to do camera calibration using AprilTags.

Create catkin workspace

```
mkdir -p catkin_ws/src && cd catkin_ws/src
git clone https://github.com/cxt98/transnet_ros
cd ..
catkin_make -DPYTHON_EXECUTABLE=`which python` # within TransNet python environment
source devel/setup.bash
ln -s {PATH_TO_TransNet} ${pwd}/src/transnet_ros/src
```

## Run Perception rosnodes

### Publish camera RGB-D data

e.g. for [Intel RealSense L515](https://github.com/leggedrobotics/realsense-ros-rsl) 

```
roslaunch realsense2_camera rs_camera.launch align_depth:=true
```

### (optional) Camera Calibration

Specify camera channel to `--cam1_rgb_channel, --cam1_info_channel, --cam2_rgb_channel, --cam2_info_channel`, output channel `--output_channel`, see scripts/camera_calibrator.py for more details

```
rosrun transnet_ros camera_calibrator.py
```

### TransNet Object Pose Estimation

TODO: complete this after merging mask-rcnn detection with transnet pose estimation