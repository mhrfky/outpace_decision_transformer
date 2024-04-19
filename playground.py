import  gym

# Load the environment
env = gym.make('PointUMaze-v0')  # Replace with your actual environment name

# Try to get the bounds of the observation space if it's continuous
if isinstance(env.observation_space, gym.spaces.Box):
    low_bound = env.observation_space.low
    high_bound = env.observation_space.high
    print("Low bounds of the observation space:", low_bound)
    print("High bounds of the observation space:", high_bound)

    # If the environment represents a physical space, such as a maze,
    # these bounds may represent the size of that space.
else:
    print("The observation space is not continuous, and its size cannot be directly inferred.")
