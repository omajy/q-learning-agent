import gymnasium as gym
import numpy as np
import pickle

np.random.seed(42)

def run(episodes):
    with open("train.pkl", "rb") as file:
        data = pickle.load(file)
    
    q_values = data["q_values"]

    env = gym.make(
        'FrozenLake-v1',
        desc=None,
        map_name="8x8",
        is_slippery=True,
        reward_schedule=(1, 0, 0),
    )

    successes = []

    episode_rewards = np.zeros(episodes)

    episode_terminated = False

    for episode in range(episodes):
        # reset environment to capture the initial state
        state = env.reset()[0]

        episode_reward = 0

        episode_terminated = False
    
        while not episode_terminated: 

            action = np.argmax(q_values[state,:])

            next_state, reward, terminated, truncated, information = env.step(action)

            state = next_state

            episode_terminated = terminated or truncated

            episode_reward += reward

        episode_rewards[episode] = episode_reward

        successes.append(1 if episode_reward > 0 else 0)

    success_rate = sum(episode_rewards) / len(episode_rewards)
        
    env.close()
    print(f"Test Results over {episodes} episodes:")
    print(f"Win Rate: {success_rate:.1%}")

    with open("eval.pkl", "wb") as file:
        pickle.dump(successes, file)

if __name__ == "__main__":
    run(2000)