import sys
from src.gym_envs.RobotEnv import RobotEnv # Training environment
import roslibpy # API of ROS
from src.robots.Teresa import Teresa # This is the representation of Teresa Robot


if __name__ == "__main__":

    HOST = 'localhost'
    PORT = 9090

    if len(sys.argv) > 0:
        HOST = int(sys.argv[1])
    if len(sys.argv) > 1:
        PORT = int(sys.argv[2])

    client = roslibpy.Ros(host=HOST, port=PORT)
    client.run()
    print("Is the client connected?")
    print(client.is_connected)
    client.terminate()