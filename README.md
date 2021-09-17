# Robot Controller written in Python
- [Robot Controller written in Python](#robot-controller-written-in-python)
- [Description](#description)
  - [Reinforcement Learning Formulation](#reinforcement-learning-formulation)
  - [Tools used](#tools-used)
- [Setting up the environment](#setting-up-the-environment)
  - [Prerequisites](#prerequisites)
  - [Set up](#set-up)
- [Running the environments](#running-the-environments)
  - [For the Teresa Robot](#for-the-teresa-robot)
    - [Prerequisites](#prerequisites-1)
    - [Executing the simulation ](#executing-the-simulation)
- [Author and Advisor](#author-and-advisor)
- [Note about this project](#note-about-this-project)

# Description
This directory contains the Teresa project which aims to allow the robot to follow a person autonomously thanks to the image rendering. 
Note : the ROS library allows us to generalize this project with ease to many other robots with similar movements.
## Reinforcement Learning Formulation of the Project
 ![reinforcement learning formulation](https://github.com/saad2050lahlali/Teresa_Robot/blob/master/images/rl_map.png)

## Tools used
One library it is used to connect with the ROS Server, ```roslibpy``` ([See docs here](https://roslibpy.readthedocs.io/en/latest/reference/index.html)).
# Setting up the environment
## Prerequisites
You must have installed:
- Python 3
- Virtualenv, you can install it with the following command: ```sudo apt-get install virtualenv```

## Set up
1) Clone this repository.
2) Create a folder inside the project called ```venv```
3) Create a virtual environment with the ```virtualenv``` command. Example: ```virtualenv --python=python3 venv/```
4) Activate the environment. ```source venv/bin/activate```
5) Install the dependencies of the Robot library: ```pip install -r requirements.txt```

# Running the environments
## For the Teresa Robot
### Prerequisites
The only prerequisite is to have your virtual environment activated before you run the python project: ```source venv/bin/activate``` .

### Executing the simulation 
For this you have to install all the dependencies manually, this includes:
1) [ROS Melodic](http://wiki.ros.org/melodic/Installation/Ubuntu)
2) Gazebo 9
3) ROS bridge package
4) Gazebo-ROS package 
<p>

This has to be in 3 different terminals (Wait until each of the first 3 commands finish to execute the last one):
1) First terminal ```roscore```
2) Second terminal: ```rosrun gazebo_ros gazebo ./gazebo/envs/Teresa_Lightweight.world```
3) Third terminal: ```roslaunch rosbridge_server rosbridge_websocket.launch```



# Author and Advisor
[Daniel RODRIGUEZ](https://danielrs975.github.io), Student at Telecom SudParis, Institut Polytechnique de Paris

**Advisor:** Prof. Hossam Afifi (Telecom SudParis, Institut Polytechnique de Paris)

# Note about this project
This collaboration was made during my summer internship at Telecom SudParis. The project was the application of Reinforcement Learning in a Medical Robot to teach it the task of following an object. 
