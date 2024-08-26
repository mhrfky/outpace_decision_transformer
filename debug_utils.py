import numpy as np
import os
import time
import matplotlib.pyplot as plt

def plot_positions(positions):
    # Set x and y limits
    plt.xlim(-2, 10)
    plt.ylim(-2, 10)
    t = np.arange(len(positions))
    # Plot positions
    plt.scatter(positions[:, 0], positions[:, 1], c = t , c_map = 'viridis')
    
    # Add labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Plot of Positions')
    
    # Show the plot
    # Check if the 'debug' directory exists
    if not os.path.exists('debug'):
        os.makedirs('debug')

    # Find the latest index of existing debug plots
    existing_plots = [f for f in os.listdir('debug') if f.startswith('debug') and f.endswith('.png')]
    latest_index = 0
    for plot in existing_plots:
        index = int(plot.split('debug')[1].split('.png')[0])
        latest_index = max(latest_index, index)

    # Save the plot with the latest index
    plt.savefig(f'debug/debug{latest_index + 1}.png')
    plt.show()
    plt.close()

def plot_two_array_positions(positions, positions_2):
    # Set x and y limits
    plt.xlim(-2, 10)
    plt.ylim(-2, 10)
    t = np.arange(len(positions))
    t2 = np.arange(len(positions_2))
    # Plot positions
    plt.scatter(positions[:, 0], positions[:, 1], c = t , cmap = 'viridis')
    plt.scatter(positions_2[:, 0], positions_2[:, 1], c = t2 , marker ="*", cmap = 'viridis')

    
    # Add labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Plot of Positions')
    
    # Show the plot
    # Check if the 'debug' directory exists
    if not os.path.exists('debug'):
        os.makedirs('debug')

    # Find the latest index of existing debug plots
    existing_plots = [f for f in os.listdir('debug') if f.startswith('debug') and f.endswith('.png')]
    latest_index = 0
    for plot in existing_plots:
        index = int(plot.split('debug')[1].split('.png')[0])
        latest_index = max(latest_index, index)

    # Save the plot with the latest index
    plt.savefig(f'debug/debug{latest_index + 1}.png')
    plt.show()
    plt.close()

def time_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)  # Call the function
        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate the elapsed time
        
        # Ensure the directory exists
        os.makedirs('debug', exist_ok=True)
        
        # Open the log file and append the timing information
        with open('debug/timesteps.log', 'a') as f:
            f.write(f"Function '{func.__name__}'\ttook {elapsed_time:.4f} seconds to complete.\n")
        
        return result
    return wrapper