import pickle
import numpy as np
import matplotlib.pyplot as plt

def run():
    # source the training successes and episodes
    with open("train.pkl", "rb") as file:
        data = pickle.load(file)
        
    train_successes = data['successes']

    # source the evaluation successes and episodes 
    with open("eval.pkl", "rb") as file:
        eval_successes = pickle.load(file)

    eval_rate = np.mean(eval_successes)

    # convolve values in training successes to get the average performance across every 1000 episodes   
    window = 1000
    train_curve = np.convolve(
        train_successes,
        np.ones(window)/window,
        mode="valid"
    )

    # produce output figure comparing training performance vs evaluation performance
    train_x = np.arange(len(train_curve))
    plt.plot(train_x, train_curve, label="Training")

    plt.hlines(
        eval_rate,
        xmin=0,
        xmax=len(train_x),
        colors="red",
        linestyles="dashed",
        label="Evaluation"
    )

    plt.xlabel("Episodes")
    plt.ylabel(f"Success Rate (Convolution: {window})")
    plt.title("Frozen Lake Agent Performance")
    plt.legend(loc='lower right')
    plt.savefig('performance_graph.png')

    # now to watch our agent perform over 10 episodes and hope he passes 6 times 
    import gymnasium as gym

    env = gym.make(
        'FrozenLake-v1',
        desc=None,
        map_name="8x8",
        is_slippery=True,
        reward_schedule=(1, 0, 0),
        render_mode='human'
    )

    q_values = data['q_values']

    episode_terminated = False

    for episode in range(10):
            # reset environment to capture the initial state
            state = env.reset()[0]

            episode_terminated = False
        
            while not episode_terminated: 

                action = np.argmax(q_values[state,:])

                next_state, reward, terminated, truncated, information = env.step(action)

                state = next_state

                episode_terminated = terminated or truncated
            
    env.close()
     
if __name__ == "__main__":
     run()