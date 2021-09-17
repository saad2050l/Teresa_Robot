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
    teresa_controller = Teresa(client)
    env = RobotEnv(teresa_controller, client)

    env.reset()
    finish = False

    while not finish:
        movement = input('Enter a movement (0 Right, 1 Left, 2 Backward, 3 Forward, exit): ')
        if movement == 'exit':
            finish = True
            continue
        movement = int(movement)
        state, reward, done, _ = env.step(movement)
        if done and reward:
            print("Centered")
            # env.reset()
        env.render()
    env.close()
    client.terminate()