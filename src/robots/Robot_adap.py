"""
Robot Class
It contains all the common functionality
between robots
"""
import roslibpy
import time
from src.robots.actions.camera_train import take_picture_test
from src.robots.actions.camera_test import take_picture_train
from src.robots.actions.move_adap import execute_move

class Robot():
    """
    From here all the robots are generated. This robots are caracterized for
    having a camera and a set of movements.
    """

    def __init__(self, client, camera_name, move_topic, movements):
        """
        Constructor of Robot
        Args:
            client: Connection to ROS.
            camera_name: String. Name of the ROS topic that contains all the information related
            to the camera.
            move_topic: Dict. Contains the information about the ROS topic used to control
            the movement of the robot.
            movements: Array. Possible movements that the robot can do.
        """
        self.movements = movements
        self.camera_topic = roslibpy.Topic(client, '{}/image_raw/compressed'.format(camera_name), 'sensor_msgs/CompressedImage')
        self.move_topic = roslibpy.Topic(client, move_topic['topic_name'], move_topic['msg_type'])
        self.reset_service = roslibpy.Service(client, '/gazebo/reset_simulation', 'std_srvs/Empty')



    def move_robot_train(self, move):
        """
        Move the robot in the given direction
        Args:
            move: Integer. Index of the movement to execute. This number is restrained between 0
            and the number of possible movements.
        """
        execute_move(self.movements[move], self.move_topic)
        take_picture_train(self.camera_topic)

    def move_robot_test(self, move):
        """
        Move the robot in the given direction
        Args:
            move: Integer. Index of the movement to execute. This number is restrained between 0
            and the number of possible movements.
        """
        execute_move(self.movements[move], self.move_topic)
        take_picture_test(self.camera_topic)

    def remove_subscribers(self):
        """
        Remove the listeners to the topics of the robot
        """
        self.move_topic.unadvertise()
