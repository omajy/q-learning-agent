import gymnasium as gym
import numpy as np

# for the purpose of the project we will be training an agent to navigate a frozen lake to reach a reward without falling 
# the configuration that I have selected rewards the agent for making the goal, slightly penalises reward for staying in valid state, strongly penalises for dying (stepping in a hole)
env = gym.make(
    'FrozenLake-v1',
    desc=None,
    map_name="8x8",
    is_slippery=True,
    success_rate=1.0/3.0,
    reward_schedule=(10, -1, -0.1),
)

# define a q table of values the agent will learn, shaped rows (observation space) x columns (action space)
# the agent will learn the reward of taking an action a in a state s, through the Q Learning formula to define the policy Q(s,a)
states =  env.observation_space.n
actions = env.action_space.n
q_values = [[-0.1 for _ in range(actions)] for _ in range(states)]

episode_reward = 0
episode_terminated = False

episodes = 50000

# alpha is learning rate and balances the weight new information will have on updating the q values
alpha = 0.1
# gamma is the discount factor and balances immediate rewards with future rewards in updating q values
gamma = 0.99
# epsilon used in epsilon greedy policy, 1-epsilon is the probability the agent will choose an exploititative action (the max of the state in the q values) versus an explorative random
epsilon_start = 1.0
epsilon_finish = 0.01
epsilon_decay = 50000

def select_action(q_values, state, env, epsilon):
    # exploitative takes the action which maximises q value for the given state
    exploitative = q_values[state].index(max(q_values[state]))
    # explorative chooses a random action with equal likelihood 
    explorative = env.action_space.sample()

    if np.random.random() < epsilon:
        action = explorative
    else:
        action = exploitative

    return action

# simple training loop
for episode in range(episodes):

    # reset environment to capture the initial state
    state = env.reset()[0]
    episode_reward = 0
    epsilon = max(epsilon_finish, epsilon_start - episode * (epsilon_start - epsilon_finish) / epsilon_decay)

    while not episode_terminated: 

        action = select_action(q_values, state, env, epsilon)

        next_state, reward, terminated, truncated, information = env.step(action)

        q_values[state][action] += alpha * (reward + gamma * max(q_values[next_state]) - q_values[state][action])

        state = next_state

        episode_terminated = terminated or truncated

        episode_reward += reward

    # end of episode print the cumulative reward, reset the state to intial for next epoch
    print(f"Episode finished! Total reward: {episode_reward}") 
    episode_terminated = False

env.close()
print(q_values)