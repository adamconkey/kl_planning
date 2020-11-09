# Planning under Uncertainty to Goal Distributions

This repository hosts code associated with our paper _Planning under Uncertainty to Goal Distributions_, currently under review. Link to arxiv paper coming soon.

## Installation
Currently supported:
  - Ubuntu 18.04
  - ROS Melodic
  - Python 3.6

The environments use rviz for rendering and ROS for communication between different running nodes, so you will need to [install ROS](http://wiki.ros.org/melodic/Installation/Ubuntu). You should clone this repository into a catkin workspace.

It's highly recommended you use a conda environment. Assuming you have conda installed on your system, you can quickly create the environment from the root of this repository with

    conda env create -f config/conda/conda_env.yaml
    
That will create a conda environment named `kl` that you can activate with

    conda activate kl
    
There are two dependencies needed to render the environments in rviz, both should be checked out into the same catkin workspace you use for this repository:
  - [rviz_textured_quads](https://github.com/lucasw/rviz_textured_quads) - Projects images into rviz
  - [vision_opencv](https://github.com/ros-perception/vision_opencv) - Needed to use OpenCV ROS bridge with Python3. Note I hope to get rid of this dependency moving to Ubuntu 20.04 and ROS Noetic which supports Python3.
  
You will need to setup catkin to build against you conda Python3 for the opencv bridge to work. The command on my system looks like this, you'll need to adapt to the path to your conda Python version:
    
    source /opt/ros/melodic/setup.bash
    catkin config -DPYTHON_EXECUTABLE=/home/adam/.miniconda3/envs/kl/bin/python3 -DPYTHON_INCLUDE_DIR=/home/adam/.miniconda3/envs/kl/include/python3.6m -DPYTHON_LIBRARY=/home/adam/.miniconda3/envs/kl/lib/libpython3.6m.so

Once that is setup you can do (assuming your workspace is named `catkin_ws`):

    catkin build
    source ~/catkin_ws/devel/setup.bash
    
You should be all set! Again I am hoping to simplify this soon to move to Ubuntu 20.04/ROS Noetic with native Python3, so hopefully the opencv build can be omitted. I will also provide Docker containers so you can avoid this setup entirely.

## Usage
