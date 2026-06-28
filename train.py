import gymnasium as gym
import numpy as np
import pickle

np.random.seed(42)

def run(episodes):
    # for the purpose of the project we will be training an agent to navigate a frozen lake to reach a reward without falling 
    # the configuration that I have selected rewards the agent for making the goal, slightly penalises reward for staying in valid state, strongly penalises for dying (stepping in a hole)
    env = gym.make(
        'FrozenLake-v1',
        desc=None,
        map_name="8x8",
        is_slippery=True,
        reward_schedule=(1, 0, 0),
    )

    # define a q table of values the agent will learn, shaped rows (observation space) x columns (action space)
    # the agent will learn the reward of taking an action a in a state s, through the Q Learning formula to define the policy Q(s,a)
    states =  env.observation_space.n
    actions = env.action_space.n
    q_values = np.zeros((states, actions))

    successes = []

    episode_rewards = np.zeros(episodes)

    episode_terminated = False

    data = {
        "q_values": q_values,
        "successes": successes
    }

    # alpha is learning rate and balances the weight new information will have on updating the q values
    alpha = 0.1
    # gamma is the discount factor and balances immediate rewards with future rewards in updating q values
    gamma = 0.99
    # epsilon used in epsilon greedy policy, 1-epsilon is the probability the agent will choose an exploititative action (the max of the state in the q values) versus an explorative random
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.0001

    # balance explorative with exploitative regarding epsilon value over episodes
    def select_action(q_values, state, env, epsilon):
        # exploitative takes the action which maximises q value for the given state
        exploitative = np.argmax(q_values[state,:])
        # explorative chooses a random action with equal likelihood 
        explorative = env.action_space.sample()
        # generate a random number between 0 and 1, if its smaller than epsilon choose random action otherwise exploit q values
        if np.random.random() < epsilon:
            action = explorative
        else:
            action = exploitative
        return action

    # agent trains for number of episodes passed in the run(episodes) parameter
    for episode in range(episodes):
        # reset environment to capture the initial state
        state = env.reset()[0]

        episode_reward = 0

        episode_terminated = False
    
        while not episode_terminated: 

            action = select_action(q_values, state, env, epsilon)

            next_state, reward, terminated, truncated, information = env.step(action)

            episode_reward += reward

            # prevents terminating behaviour from being implemented into the q table values
            reward = reward if terminated else reward + gamma * np.max(q_values[next_state, :])
            # encoded Bellman equation to update q table values
            q_values[state, action] += alpha * (reward - q_values[state, action])

            state = next_state

            episode_terminated = terminated or truncated

        episode_rewards[episode] = episode_reward

        successes.append(1 if episode_reward > 0 else 0)

        # epsilon iteratively decays every epsiode by the decay value and eventually will plateau at epsilon_min value to keep a base level of exploration 
        if episode >= 1000 and episode % 1000 == 0:
            moving_success_rate = sum(episode_rewards[episode-1000:episode])/1000
            print(f"Success rate from episode {episode-1000} to {episode}: {moving_success_rate:.1%}")

        epsilon = max(epsilon_min, epsilon-epsilon_decay)

    env.close()

    with open("train.pkl", "wb") as file:
        pickle.dump(data, file)

if __name__ == "__main__":
    run(20000)