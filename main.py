import gym.spaces
import numpy as np
import matplotlib.pyplot as plt
from policy_gradient import Agent, process_observation

if __name__ == "__main__":
    EPISODES = 10000
    istrain = True
    env = gym.make("Boxing-v0")
    actions = env.action_space.n
    reward_list,  episode_list = [], []
    plt.ion()
    
    if istrain:
        agent = Agent(actions_size = actions)
        for i in range(EPISODES):
            total_reward = 0
            game_status = False
            last_observation = env.reset()
            observation, _, _, _ = env.step(env.action_space.sample())
            state = agent.initial_state(observation, last_observation)
            while not game_status:
                last_observation = observation
                action = agent.get_action(state)
                observation, reward, game_status, _ = env.step(action)
                env.render()
                agent.store_transition(state, action, reward)
                state = np.append(state[: ,:, 1:], process_observation(observation, last_observation), axis = 2)
                total_reward += reward
            agent.train_network()
            
            reward_list.append(total_reward)
            episode_list.append(i)
            plt.plot(episode_list, reward_list, '-r')
            plt.draw()
    else:
        agent = Agent(actions_size = actions, load_model = True)
        for i in range(EPISODES):
            total_reward = 0
            game_status = False
            last_observation = env.reset()
            observation, _, _, _ = env.step(env.action_space.sample())
            state = agent.initial_state(observation, last_observation)
            while not game_status:
                last_observation = observation
                action = agent.get_action(state)
                observation, reward, game_status, _ = env.step(action)
                env.render()
                state = process_observation(observation, last_observation)
                total_reward += reward
            
            reward_list.append(total_reward)
            episode_list.append(i)
            plt.plot(episode_list, reward_list, '-r')
            plt.draw()