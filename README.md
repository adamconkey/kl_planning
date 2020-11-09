# Planning under Uncertainty to Goal Distributions

This repository hosts code associated with our paper _Planning under Uncertainty to Goal Distributions_, currently under review. Link to arxiv paper coming soon.

If you try to use this code and have troubles installing or using it, or have a question about the paper, please raise an issue or email me at [adam.conkey@gmail.com](adam.conkey@gmail.com) and I will get back to you.

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
There are currently two environments offered, one for 2D navigation and one for planning with a 7-DOF arm.
### 2D Navigation
First launch the rviz environment for visualization. You will need to specify a scene configuration which are YAML configuration files that specify everything needed to run a planning session. All available configs are contained in `config/scenes/nav_2d`. The command to launch the environment is

    roslaunch kl_planning nav_2d_env.launch scene:=SCENE_NAME
    
where SCENE_NAME should be replaced by one of the available scenes:

| Scene                                |
|--------------------------------------|
| `one_obstacle`                       |
| `three_room_dirac_left`              |
| `three_room_gaussian_left`           |
| `three_room_gmm_even-weight_gmm-cem` |
| `three_room_gmm_even-weight_Iproj`   |
| `three_room_gmm_even-weight_Mproj`   |
| `three_room_gmm_right-higher-weight` |
| `three_room_uniform_left`            |

The scene name gets loaded to the ROS parameter server so that you can run the associated run script in `src/kl_planning/planning` with the default config values automatically:

    python run_nav_2d_planner.py
    
You can also override some parameters that can make tuning the CEM parameters more easy, for example:

    python run_nav_2d_planner.py --n_candidates 1000 --n_elite 50 --belief_dynamics_noise 0.05 --cpu
    
You can run this command to see all the command line args you can pass in:

    python run_nav_2d_planner.py -h
    
### 7-DOF Arm
A similar structure holds for the arm environment as the 2D navigation. Right now there is only one scene offered:

    roslaunch kl_planning arm_env.launch scene:=pole_obstacle
    
There is an associated run script in `src/kl_planning/planning`:

    python run_arm_planner.py -h
