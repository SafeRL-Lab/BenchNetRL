import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper

def make_env(gym_id="MiniGrid-MemoryS17Random-v0", seed=1):
    env = gym.make(
        gym_id,
        agent_view_size=3,
        tile_size=16,
        render_mode="rgb_array",
    )
    env = ImgObsWrapper(RGBImgPartialObsWrapper(env, tile_size=16))
    env = gym.wrappers.TimeLimit(env, max_episode_steps=96)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

def debug_minigrid_plot_12():
    # 1. Create environment
    env = make_env()

    # 2. Reset environment
    obs, info = env.reset()
    print("Observation shape:", obs.shape)
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)

    # We'll store up to X steps of (observation, action)
    steps_to_plot = 36
    observations = []
    actions = []

    # 3. Append the initial observation to the list
    observations.append(obs)
    actions.append(-1)  # no action led to the initial state, so store a placeholder

    # 4. Step through the environment, restricting actions to {0,1,2}
    for i in range(steps_to_plot):
        # Sample from {0,1,2} only
        action = np.random.choice([0,1,2])
        obs, reward, done, truncated, info = env.step(action)

        observations.append(obs)
        actions.append(action)

        print(f"Step={i}, Action={action}, Reward={reward}, Done={done}, Truncated={truncated}")

        if done or truncated:
            obs, info = env.reset()

    # 5. Plot resulting observations
    rows, cols = 6, 6
    fig, axs = plt.subplots(rows, cols, figsize=(12, 8))

    for idx in range(1, steps_to_plot + 1):
        ax = axs[(idx - 1) // cols, (idx - 1) % cols]
        obs_i = observations[idx]
        action_i = actions[idx]

        # Plot
        # obs_i is shape (height, width, 3) after wrappers => (28,28,3) typically
        ax.imshow(obs_i)
        ax.set_title(f"Step {idx}, Act={action_i}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    debug_minigrid_plot_12()
