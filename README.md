# Planning under Uncertainty to Goal Distributions

This repository hosts code associated with our paper _Planning under Uncertainty to Goal Distributions_, currently under review. Link to arxiv paper coming soon.

If you try to use this code and have troubles installing or using it, or have a question about the paper, please raise an issue or email me at [adam.conkey@gmail.com](adam.conkey@gmail.com) and I will get back to you.

## Installation
Currently supported:
  - Ubuntu 20.04
  - ROS Noetic
  - Python 3.8

The environments use rviz for rendering and ROS for communication between different running nodes, so you will need to [install ROS](http://wiki.ros.org/noetic/Installation/Ubuntu). You should clone this repository into a catkin workspace:

    cd ~
    mkdir -p catkin_ws/src
    cd catkin_ws/src
    git clone https://github.com/adamconkey/kl_planning.git

It's highly recommended you use a conda environment to avoid versioning problems with your system install. Assuming you have conda installed on your system, you can quickly create the environment:

    cd ~/catkin_ws/src/config/conda
    conda env create -f conda_env.yaml
    
That will create a conda environment named `kl` that you can activate with

    conda activate kl
    
There are two dependencies needed to render the environments in rviz:
  - MoveIt - Has rviz plugins needed for arm visualizations. `sudo apt install ros-noetic-moveit`
  - [rviz_textured_quads](https://github.com/lucasw/rviz_textured_quads) - Projects images into rviz. Check this out into the same catkin workspace. For now you need to make one small edit to the `CMakeLists.txt` file in that project. On [Line 42](https://github.com/lucasw/rviz_textured_quads/blob/image_topic/CMakeLists.txt#L42), simply remove the word `EXACT` from where it finds Qt.

The source/build procedure is as follows (order matters):

    conda activate kl
    source /opt/ros/noetic/setup.bash
    cd ~/catkin_ws
    catkin init
    catkin build
    source ~/catkin_ws/devel/setup.bash
    
You should be all set! I may also provide a Docker container so you can avoid this setup entirely if that's desired.

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
