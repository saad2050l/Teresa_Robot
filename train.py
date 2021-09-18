import sys
from src.gym_envs.RobotEnv import RobotEnv # Training environment
import numpy as np
import tensorflow as tf
import roslibpy # API of ROS
from src.robots.Teresa import Teresa # This is the representation of Teresa Robot
from src.utils.training_tools import NB_STATES

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

    #Connection to local host
    HOST = 'localhost'
    PORT = 9090

    client = roslibpy.Ros(host=HOST, port=PORT)

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
    tf.summary.scalar("Loss", loss)

    ## Reward mean
    tf.summary.scalar("Reward_mean", mean_reward_)

    merged_summary_op = tf.summary.merge_all() #procedure d'affichage groupée dans tensorboard

    # Parameters for the training
    max_episodes = 100
    gamma = 0.95 # Discount rate
    #max_batch = NbStat*5
    max_batch = 10
    episodes_succeded = 0

    client.run() # This run the main loop of ROS
    teresa_controller = Teresa(client) # Robot API
    env = RobotEnv(teresa_controller, client) # Training Environment

    env.reset() # Restarting the environment to the initial state
    write_op = tf.summary.merge_all()
    allRewards = []
    total_rewards = 0
    maximumRewardRecorded = 0
    episode = 0
    episode_states, episode_actions, episode_rewards = [],[],[]

    saver = tf.train.Saver()
    NbStat = state_size

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        #writer = tf.summary.FileWriter("./tensorboard/pg/1",sess.graph)
        
        for episode in range(max_episodes):
            
            episode_rewards_sum = 0

            # Launch the game
            state = env.reset()
            print(NbStat)
            ne_state=np.identity(NbStat)[state:state+1]
            episode_length=0
            while True:
                episode_length+=1
                if episode_length > max_batch:
                    print ("tooooooooo long")
                    break      
                # Choose action a, remember WE'RE NOT IN A DETERMINISTIC ENVIRONMENT, WE'RE OUTPUT PROBABILITIES.
                #state=int(state)
                #print("state",state,NbStat)
                ne_state=np.identity(NbStat)[state:state+1]
                action_probability_distribution = sess.run(action_distribution, feed_dict={input_: ne_state.reshape([1,NbStat])})
                action = np.random.choice(range(action_probability_distribution.shape[1]), p=action_probability_distribution.ravel())  # select action w.r.t the actions prob

                # Perform a
                ext=False
                #print("in actor mstep",state,"real ibn self")
                new_state, reward, done, info = env.step(action)
                env.render()
                #print("after action state",new_state,NbStat)
                #print(" drz actor mstep",state,"real ibn self",new_state,"rew",reward,"done",done)
                # Store s, a, r
                episode_states.append(ne_state)
                            
                # For actions because we output only one (the index) we need 2 (1 is for the action taken)
                # We need [0., 1.] (if we take right) not just the index
                action_ = np.zeros((action_size), dtype=int)
                action_[action] = 1
                
                #print("action proba",action_probability_distribution,"st",state,"new",new_state)
                state = new_state
                episode_actions.append(action_)
                
                episode_rewards.append(reward)
                if done:
                    # Calculate sum reward
                    if reward == 1:
                        episodes_succeded += 1
                    
                    episode_rewards_sum = np.sum(episode_rewards)/episode_length  # HA addded the sum 
                    print(action_probability_distribution,"action proba",episode_rewards_sum,"length",episode_length,"rew",reward)
                    allRewards.append(episode_rewards_sum)
                    
                    total_rewards = np.sum(allRewards)
                    
                    # Mean reward
                    mean_reward = np.divide(total_rewards, episode+1)
                    
                    
                    maximumRewardRecorded = np.amax(allRewards)
                    
                    print("="*80)
                    print("Episode: ", episode,"length",episode_length)
                    print("Reward: ", episode_rewards_sum)
                    print("Mean Reward","val",mean_reward)
                    print("Max reward so far: ", maximumRewardRecorded)
                    
                    # Calculate discounted reward
                    discounted_episode_rewards = discount_and_normalize_rewards(episode_rewards)
                    
                    #print("disco",discounted_episode_rewards)               
                    # Feedforward, gradient and backpropagation
                    #loss_, _ = sess.run([loss, train_opt], feed_dict={input_: np.vstack(np.array(episode_states)),
                    #                                                 actions: np.vstack(np.array(episode_actions)),
                    #                                                 discounted_episode_rewards_: discounted_episode_rewards 
                    #                                                })
                    loss_, _ = sess.run([loss, train_opt], feed_dict={input_: np.vstack(np.array(episode_states)),
                                                                    actions: np.vstack(np.array(episode_actions)),
                                                                    discounted_episode_rewards_: discounted_episode_rewards
                                                                    })
                    
    
                                                                    
                    # Write TF Summaries
                    summary = sess.run(write_op, feed_dict={input_: np.vstack(np.array(episode_states)),actions: np.vstack(np.array(episode_actions)),discounted_episode_rewards_: discounted_episode_rewards,
                                                                        mean_reward_: mean_reward  })
                    
                
    #                 writer.add_summary(summary, episode)
    #                 writer.flush()
                    
                
                    
                    # Reset the transition stores
                    episode_states, episode_actions, episode_rewards = [],[],[]
                    
                    break
                
                
                
        # Save Model
        saver.save(sess, "saved_model.ckpt")
        env.close()
        client.terminate()
        print("Model saved")
        print("v"*80)
        print("Percentage of succeded episodes =  " + str((episodes_succeded / max_episodes)*100 +"%"))
