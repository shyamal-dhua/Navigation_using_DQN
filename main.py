#Import necessary packages
from unityagents import UnityEnvironment
import numpy as np
import random
import argparse
import torch
from collections import deque
import matplotlib.pyplot as plt
from dqn_agent_navigation import Agent
import time, os, fnmatch, shutil

"""Training code -> train()
    
    Params
    ======
        env: an instance of the UnityEnvironment
        brain_name: default brain used in the environment [brains are responsible for deciding the actions of their associated agents]
        action_size (int): Number of possible actions (4)
        state_size (int): Number of features in any state of the environment (37)
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    
"""
def train(env, brain_name, action_size, state_size, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    max_mean = 0
    i_episode_solved = 0
    agent = Agent(state_size=state_size, action_size=action_size, seed=0)
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations[0] # get the current state
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps).item() #select an action
            env_info = env.step(action)[brain_name] # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            state = next_state # roll over the state to next time step
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        
        #Store the weights corresponding to the best mean score
        curr_mean = np.mean(scores_window)
        if ((curr_mean >= 13.0) and (curr_mean > max_mean)):
            max_mean = curr_mean
            i_episode_solved = i_episode
            t = time.localtime()
            timestamp = time.strftime('%b-%d-%Y_%H%M', t)
            #save best as well as intermediate checkpoints
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoints/checkpoint_' + timestamp+ '_' + str(round(max_mean,2)) + '.pth')
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoints/best_checkpoint.pth')
            #break
    print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode_solved - 100, max_mean))
    return scores

"""Testing code -> test()
    
    Params
    ======
        env: an instance of the UnityEnvironment
        brain_name: default brain used in the environment [brains are responsible for deciding the actions of their associated agents]
        checkpoint_path: File path for the saved model checkpoints
        action_size: Number of possible actions (4)
        state_size: Number of features in any state of the environment (37)
    
"""
def test(env, brain_name, checkpoint_path, action_size, state_size):
    agent = Agent(state_size=state_size, action_size=action_size, seed=0)
    # load the weights from file
    agent.qnetwork_local.load_state_dict(torch.load(checkpoint_path))
    
    # Run the evaluation for 1 episode
    for i in range(1):
        env_info = env.reset(train_mode=False)[brain_name]
        state = env_info.vector_observations[0]
        for j in range(200):
            action = agent.act(state).item()
            env_info = env.step(action)[brain_name]
            state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            if done:
                break 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch training script')
    
    parser.add_argument('--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('--n_episodes', default=2000, type=int, metavar='N', help='maximum number of training episodes')
    parser.add_argument('--max_t', default=1000, type=int, metavar='N', help='maximum number of timesteps per episode')
    parser.add_argument('--eps_start', default=1.0, type=float, metavar='N', help='starting value of epsilon')
    parser.add_argument('--eps_end', default=0.01, type=float, metavar='N', help='minimum value of epsilon')
    parser.add_argument('--eps_decay', default=0.995, type=float, metavar='N', help='multiplicative factor for epsilon decay')
    
    args = parser.parse_args()
    
    #Start the environment [change the file_name parameter to match the location of the Unity environment that you downloaded.]
    env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana.exe", seed=0)
    
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of actions
    action_size = brain.vector_action_space_size

    # examine the state space 
    state = env_info.vector_observations[0]
    state_size = len(state)

    if args.evaluate:
        print("\nRunning Test with the below parameters:\n")
        print("checkpoint path = ", args.evaluate)
        print("\n")
        test(env, brain_name, args.evaluate, action_size, state_size)
        env.close()
        exit(0)
    else:
        print("\nRunning Train with the below parameters:\n")
        print("n_episodes = ", args.n_episodes)
        print("max_t = ", args.max_t)
        print("eps_start = ", args.eps_start)
        print("eps_end = ", args.eps_end)
        print("eps_decay = ", args.eps_decay)
        print("\n")
        scores = train(env, brain_name, action_size, state_size, args.n_episodes, args.max_t, args.eps_start, args.eps_end, args.eps_decay)
        # plot the scores
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()
        fig.savefig("plots/training_plot.pdf", dpi=fig.dpi)
        env.close()
        exit(0)
