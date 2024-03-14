# Spring 2024, 535514 Reinforcement Learning
# HW1: Policy Iteration and Value iteration for MDPs

# pip install "cython<3"   

import numpy as np
import gym

def get_rewards_and_transitions_from_env(env):
    # Get state and action space sizes
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Intiailize matrices
    R = np.zeros((num_states, num_actions, num_states))
    P = np.zeros((num_states, num_actions, num_states))

    # Get rewards and transition probabilitites for all transitions from an OpenAI gym environment
    for s in range(num_states):
        for a in range(num_actions):
            for transition in env.P[s][a]:
                prob, s_, r, done = transition
                R[s, a, s_] = r
                P[s, a, s_] = prob
                
    return R, P

def value_iteration(env, gamma=0.9, max_iterations=10**6, eps=10**-3):
    """        
        Run value iteration (You probably need no more than 30 lines)
        
        Input Arguments
        ----------
            env: 
                the target environment
            gamma: float
                the discount factor for rewards
            max_iterations: int
                maximum number of iterations for value iteration
            eps: float
                for the termination criterion of value iteration 
        ----------
        
        Output
        ----------
            policy: np.array of size (500,)
        ----------
        
        TODOs
        ----------
            1. Initialize the value function V(s)
            2. Get transition probabilities and reward function from the gym env
            3. Iterate and improve V(s) using the Bellman optimality operator
            4. Derive the optimal policy using V(s)
        ----------
    """
    num_spaces = env.observation_space.n
    num_actions = env.action_space.n
    
    # Initialize with a random policy
    policy = np.array([env.action_space.sample() for _ in range(num_spaces)])
    
    ##### FINISH TODOS HERE #####
    #print('VI: ')
    R,P = get_rewards_and_transitions_from_env(env)
    V = np.zeros(num_spaces)
    for _ in range(max_iterations):
        #print(_)
        criterion = 0
        V_ = V.copy() # V_ stands for the last V_k, and V stands for V_k+1
        for s in range(num_spaces):
            max_v, best_a = -float('inf'), -1 # max_v: the maximum of Q_k(s,a), best_a: the best action to do given state s
            for a in range(num_actions):
                value = 0
                for s_ in range(num_spaces):
                    value += P[s, a, s_] * (R[s, a, s_] + gamma * V_[s_])
                if (value > max_v):
                    max_v, best_a = value, a
            V[s], policy[s] = max_v, best_a
            criterion = abs(np.sum(V) - np.sum(V_))
        if criterion < eps:
            break
    
    #############################
    
    # Return optimal policy    
    return policy

def policy_iteration(env, gamma=0.9, max_iterations=10**6, eps=10**-3):
    """ 
        Run policy iteration (You probably need no more than 30 lines)
        
        Input Arguments
        ----------
            env: 
                the target environment
            gamma: float
                the discount factor for rewards
            max_iterations: int
                maximum number of iterations for the policy evalaution in policy iteration
            eps: float
                for the termination criterion of policy evaluation 
        ----------  
        
        Output
        ----------
            policy: np.array of size (500,)
        ----------
        
        TODOs
        ----------
            1. Initialize with a random policy and initial value function
            2. Get transition probabilities and reward function from the gym env
            3. Iterate and improve the policy
        ----------
    """
    num_spaces = env.observation_space.n
    num_actions = env.action_space.n
    
    # Initialize with a random policy
    policy = np.array([env.action_space.sample() for _ in range(num_spaces)])
    
    ##### FINISH TODOS HERE #####
    # print('PI: ')
    R,P = get_rewards_and_transitions_from_env(env)
    policy_prev = np.zeros(num_spaces)
    V = np.zeros(num_spaces)
    # cnt = 0
    while not np.array_equal(policy, policy_prev):
        policy_prev = policy.copy()
        
        # IPE
        for _ in range(max_iterations):
            # print(_)
            criterion = 0
            V_ = V.copy() # V_ stands for the last V_k, and V stands for V_k+1
            for s in range(num_spaces):
                value = 0
                for s_ in range(num_spaces):
                    value += P[s, policy[s], s_] * (R[s, policy[s], s_] + gamma * V_[s_])
                V[s] = value
            criterion = abs(np.sum(V) - np.sum(V_))
            if criterion < eps:
                break

        # One-Step Policy Improvement
        for s in range(num_spaces):
            max_q, best_a = -float('inf'), -1
            for a in range(num_actions):
                value = 0
                for s_ in range(num_spaces):
                    value += P[s, a, s_] * (R[s, a, s_] + gamma * V[s_])
                if value > max_q:
                    max_q, best_a = value, a
            policy[s] = best_a
        # cnt += 1
        # print(cnt)
    #############################

    # Return optimal policy
    return policy

def print_policy(policy, mapping=None, shape=(0,)):
    print(np.array([mapping[action] for action in policy]).reshape(shape))


def run_pi_and_vi(env_name):
    """ 
        Enforce policy iteration and value iteration
    """    
    env = gym.make(env_name)
    print('== {} =='.format(env_name))
    print('# of actions:', env.action_space.n)
    print('# of states:', env.observation_space.n)
    print(env.desc)

    vi_policy = value_iteration(env)
    pi_policy = policy_iteration(env)

    return pi_policy, vi_policy


if __name__ == '__main__':
    # OpenAI gym environment: Taxi-v2 or Taxi-v3
    pi_policy, vi_policy = run_pi_and_vi('Taxi-v3')

    # For debugging
    action_map = {0: "S", 1: "N", 2: "E", 3: "W", 4: "P", 5: "D"}
    print_policy(pi_policy, action_map, shape=None)
    print_policy(vi_policy, action_map, shape=None)
    
    # Compare the policies obatined via policy iteration and value iteration
    diff = sum([abs(x-y) for x, y in zip(pi_policy.flatten(), vi_policy.flatten())])        
    print('Discrepancy:', diff)

# export MUJOCO_PY_MUJOCO_PATH=/Users/chueating/opt/anaconda3/envs/RL/lib/python3.9/site-packages
    
