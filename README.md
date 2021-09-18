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
      - [Verifying robot connection](#verifying-robot-connection)
      - [Verifying robot movement](#verifying-robot-movement)
      - [Training the robot ](#training-the-robot)
      - [Testing on the robot ](#testing-on-the-robot)
      - [Testing on the real Teresa ](#testing-on-the-real-teresa)
- [Authors and Advisor](#authors-and-advisor)
- [Note about this project](#note-about-this-project)

# Description
This directory contains the Teresa project which aims to allow the robot to follow a person autonomously thanks to the image rendering.<br /> 
![IMAGE ALT TEXT HERE]](./images/Teresa_Robot_Movement.gif)
[![Watch the video]](https://www.youtube.com/watch?v=Qo_Pitp4Zk8&ab_channel=SaadLahlali)
You can find a [video](https://www.youtube.com/watch?v=Qo_Pitp4Zk8&ab_channel=SaadLahlali) describing both the training and the testing with the Teresa robot [here](https://www.youtube.com/watch?v=Qo_Pitp4Zk8&ab_channel=SaadLahlali).<br />
Note : the ROS library allows us to generalize this project with ease to many other robots with similar movements.

## Reinforcement Learning Formulation of the Project
 ![reinforcement learning formulation](https://github.com/saad2050lahlali/Teresa_Robot/blob/master/images/rl_map.png)

## Tools used
One library it is used to connect with the ROS Server, ```roslibpy``` ([See docs here](https://roslibpy.readthedocs.io/en/latest/reference/index.html)) and also a ```ROSBridge``` library to connect remotely with the robot ([See docs here](http://wiki.ros.org/rosbridge_suite#:~:text=At%20its%20core%2C%20rosbridge%20is,Author%3A%20Maintained%20by%20Jonathan%20Mace)).

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
6) Copy the Teresa_Lightweight.world file in the gazebo_world folder to ./gazebo/envs/

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

#### Verifying robot connection
After runing the previous commands, you can verify that the robot can be connected to without problem with the following command:<br />
$ python3 ./verifications/robot_connect.py [HOST] [PORT]<br />
The arguments HOST and PORT are set by default respectively to ```localhost``` and ```9090``` which corresponds to local excecution of simulation.

#### Verifying robot movement
You can also verify that the robot movements manually with the following command:<br />
```$ python3 ./verifications/robot_move.py [HOST] [PORT]```

#### Training the robot 
After verifying that the robot is connected and the movements are the right ones, you can train the robot with the following command:  
```$ python3 ./train.py```<br />
The arguments HOST and PORT are set by default respectively to ```localhost``` and ```9090```.<br />
The model will be saved in the file ```saved_model.ckpt```. 

#### Testing on the simulation
After having trained the model, you can test the robot with the following command:<br />
```$ python3 ./test_simulation.py```

#### Testing on the real Teresa
You can also test on the real robot with the following command:<br />
```$ python3 ./test_teresa.py [HOST] [PORT] [HostCamera] [UserCamera] [PasswordCamera]```<br />
You have to specify the arguments. ```HOST``` and ```PORT``` correspond to the Robot and ```HostCamera```, ```UserCamera``` and ```PasswordCamera``` are necessary arguments to connect with the camera since we used in our project a external camera.


# Authors and Advisor
[Daniel RODRIGUEZ](https://danielrs975.github.io), Student at Telecom SudParis, Institut Polytechnique de Paris.<br />
[Quentin Addi](https://www.linkedin.com/in/quentin-addi-12482b194/), Student at Telecom SudParis, Institut Polytechnique de Paris.<br />
[Saad LAHLALI](https://www.linkedin.com/in/saad-lahlali/), Student at Telecom SudParis, Institut Polytechnique de Paris.<br />

**Advisor:** [Prof. Hossam Afifi](http://www-public.int-evry.fr/~afifi/cvusnew.html) (Telecom SudParis, Institut Polytechnique de Paris).

# Note about this project
This project was initiated by [Daniel RODRIGUEZ](https://github.com/danielrs975/robot_controller) during his summer internship afterward continued by Saad LAHLALI and Quentin Addi for a school project then Saad LAHLALI kept working on his free time.
