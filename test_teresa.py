import sys
import roslibpy # Communication with the HMI
import time 
from src.gym_envs.RobotEnv_ import RobotEnv # Training environment
import numpy as np
import tensorflow as tf
import roslibpy # API of ROS
from src.robots.Teresa_adap import Teresa # This is the representation of Teresa Robot
from src.utils.training_tools import NB_STATES
from src.robots.actions.camera_adap import DlinkDCSCamera # class for the camera
import matplotlib.pyplot as plt
import gym
import cv2
import logging
import requests

def discount_correct_rewards(r, gamma=0.99):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        #if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    discounted_r -= discounted_r.mean()
    discounted_r /- discounted_r.std()
    return discounted_r

    def discount_and_normalize_rewards(episode_rewards):
        discounted_episode_rewards = np.zeros_like(episode_rewards)
        cumulative = 0.0
        #print("len episode rewards",episode_rewards)
        for i in reversed(range(len(episode_rewards))):
            cumulative = cumulative * gamma + episode_rewards[i]
            #print("dans boucle",episode_rewards[i],"cyl",cumulative)
            discounted_episode_rewards[i] = cumulative
        
        mean = np.mean(discounted_episode_rewards)
        std = np.std(discounted_episode_rewards)
        if std :
            discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)
        else:
            discounted_episode=[]
            discounted_episode_rewards[0] = np.array(mean)
            print("ATTTTTTTTTTTTTTTTTT")
        #print("dis",discounted_episode_rewards,"std",std)
        
        return discounted_episode_rewards

if __name__ == "__main__":

    HOST = int(sys.argv[1])
    PORT = int(sys.argv[2])
    host=str(sys.argv[2])
    user=str(sys.argv[3])
    password=str(sys.argv[4])

    #Connecting with the robot Teresa
    client = roslibpy.Ros(host=HOST, port=PORT)
    client.run()
    print("Is the client connected?")
    print(client.is_connected)

    #Connecting with the camera

    camera = DlinkDCSCamera(host = host, user = user, password = password)
    camera.set_day_night(2)
    print("The camera is connected.")


    #Setting-up the neural network 
    learning_rate = 0.01
    ima_size =  800
    action_size = 4
    new_graph = tf.Graph()

    initializer=tf.initializers.glorot_uniform()

    with tf.name_scope("inputs"):
        input_ = tf.placeholder(tf.float32, [None, ima_size, ima_size], name="input_")
        actions = tf.placeholder(tf.int32, [None, action_size], name="actions")
        discounted_episode_rewards_ = tf.placeholder(tf.float32, [None,], name="discounted_episode_rewards")
        
        # Add this placeholder for having this variable in tensorboard
        mean_reward_ = tf.placeholder(tf.float32 , name="mean_reward")

        with tf.name_scope("conv1"):
            conv1 = tf.layers.Conv2D(input_, 8, 8, subsample=(4,4), activation='relu')

        with tf.name_scope("conv2"):
            conv2 = tf.layers.Conv2D(conv1, 16, 8, subsample=(2,2), activation='relu')
        
        with tf.name_scope("conv3"):
            conv3 = tf.layers.Conv2D(conv2, 32, 4, subsample=(2,2), activation='relu')

        with tf.name_scope("flat"):
            flat = tf.layers.Flatten(conv3)
  
        with tf.name_scope("fc1"):
            fc1 = tf.layers.dense(flat, 512, activation='relu')

        with tf.name_scope("fc2"):
            fc2 = tf.layers.dense(fc1, action_size, activation='relu')

        with tf.name_scope("softmax"):
            action_distribution = tf.nn.softmax(fc2)

        with tf.name_scope("loss"):
            # tf.nn.softmax_cross_entropy_with_logits computes the cross entropy of the result after applying the softmax function
            # If you have single-class labels, where an object can only belong to one class, you might now consider using 
            # tf.nn.sparse_softmax_cross_entropy_with_logits so that you don't have to convert your labels to a dense one-hot array. 
            neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits = fc2, labels = actions)
            #loss = tf.nn.sparse_softmax_cross_entropy_with_logits (neg_log_prob * discounted_episode_rewards_)
            loss = tf.reduce_mean(neg_log_prob * discounted_episode_rewards_) 
            
        
        with tf.name_scope("train"):
            train_opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # Setup TensorBoard Writer


    ## Losses
    ## TRAINING Hyperparameters

    # tf.summary.scalar("Loss", loss)

    # ## Reward mean
    # tf.summary.scalar("Reward_mean", mean_reward_)

    max_episodes = 500

    gamma = 0.95 # Discount rate
    max_batch = NbStat*5
        
    episode_rewards_sum = 0

            # Launch the game
        #state = env.reset()
        #ne_state=np.identity(NbStat)[state:state+1]
        #env.render()
    episode_length=0

    saver = tf.train.Saver()

    client.run() # This run the main loop of ROS
    teresa_controller = Teresa(client) # Robot API
    env = RobotEnv(teresa_controller, client) # Training Environment

    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
    
            # Load the model
        print(saver.restore(sess, "saved_model.ckpt"))
        # if not saver.restore(sess, "pgpendul.ckpt"):
        #     print()
        total_rewards = 0
        for episode in range(50):
            state = env.reset()
            #ne_state=np.identity(NbStat)[state:state+1]
            step = 0
            done = False
            
            print("****************************************************")
            print("EPISODE ", episode)

        
            #while True:
            j = 0
            #The Q-Network
            while j < 500:
                j+=1
                state=int(state)
                ne_state=np.identity(NbStat)[state:state+1]
                # Choose action a, remember WE'RE NOT IN A DETERMINISTIC ENVIRONMENT, WE'RE OUTPUT PROBABILITIES.
                action_probability_distribution = sess.run(action_distribution, feed_dict={input_: ne_state.reshape([1,NbStat])})
                print(action_probability_distribution)
                action = np.random.choice(range(action_probability_distribution.shape[1]), p=action_probability_distribution.ravel())  # select action w.r.t the actions prob
                #action = np.argmax(action_probability_distribution)
                

                # new_state, reward, done, info = env.step(int(action),True)
                new_state, reward, done, info = env.step(int(action))
                env.render()

                print("state",state,"ne_state",new_state,"action",action) 
                total_rewards += reward
                if done:    
                    #rewards.append(total_rewards)
                    print ("Score", total_rewards)
                    break
                state = new_state
        env.close()
    print ("Score over time: " ,  total_rewards)
    client.terminate()