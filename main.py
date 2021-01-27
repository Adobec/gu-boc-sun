import gridworld_v1
import numpy as np
import time
import sys

max_episodes = 5000     # Maximum number of episodes
max_steps = 20          # Maximum number of steps in every episode


average_reward_criterion = 100
consecutive_iterations = 100
last_steps = np.zeros(consecutive_iterations)

env = gridworld_v1.GridWorld_DnS()

q_table = np.random.uniform(low=-1, high=1, size=(5 * 5, 4))

def action_decision(state, action, observation, reward, episode, epsilon_coefficient=0.0):
    next_state = observation
    epsilon = epsilon_coefficient * (0.99 ** episode)
    if epsilon <= np.random.uniform(0,1):
        next_action = np.argmax(q_table[next_state])
    else:
        next_action = np.random.choice([0, 1, 2, 3])
    
    # Update q_table
    alpha = 0.2
    gamma = 0.99
    q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (reward + gamma * q_table[next_state, next_action])

    return next_action, next_state

timer = time.time()
for episode in range(max_episodes): # In each episode,
    env.reset()                     # Reset the environment
    episode_reward = 0              # Reset the episode reward
    q_table_checkpoint = q_table    # Set a checkpoint for q_table

    for t in range(max_steps):  # At each step,
        time.sleep(0.05)
        env.render()
        state = env.state
        action = np.argmax(q_table[state])  # Implement action and state
        observation, reward, done, info = env.step(action)  # Get the result of this step
        action, state = action_decision(state, action, observation, reward, episode, 0.5)   # Decide what to do in next step
        episode_reward += reward

        if done:
            np.savetxt("q_table.txt", q_table, delimiter=",")
            print('Episode no.: %d, total steps: %d. Episode reward: %d, average reward: %f'%(episode, t+1, episode_reward, last_steps.mean()))
            last_steps = np.hstack((last_steps[1:],[reward]))
            if observation != 24:
                action, state = action_decision(state, action, observation, -200.0, episode, 0.5)
            break
    q_table = q_table_checkpoint

    # episode_reward = -100
    print('Episode no.: %d, total steps: %d. Episode reward: %d, average reward: %f'%(episode, t+1, reward, last_steps.mean()))
    last_steps = np.hstack((last_steps[1:],[reward]))

    if (last_steps.mean() >= average_reward_criterion):
        np.savetxt("q_table.txt", q_table, delimiter=",")
        print('Time used: %d s. After %d times of training, the model meets the criterion.'%(time.time() - timer, episode))

        env.close()
        sys.exit()

env.close()
sys.exit()