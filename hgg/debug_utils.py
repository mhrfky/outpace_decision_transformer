import os
import matplotlib.pyplot as plt
from functools import wraps

def visualize_output(update_func):
    def decorator(func):
        history = []

        @wraps(func)
        def wrapper(*args, **kwargs):
            if not os.path.exists("debug"):
                os.makedirs("debug")
            
            result = func(*args, **kwargs)
            history.append(result)
            
            update_func(history)
            
            file_path = os.path.join('debug', f"{func.__name__}.png")
            
            plt.savefig(file_path)
            plt.close()
            
            return result
        
        return wrapper
    return decorator

def visualize_length_of_output(update_func):
    def decorator(func):
        history = []

        @wraps(func)
        def wrapper(*args, **kwargs):
            if not os.path.exists("debug"):
                os.makedirs("debug")
            
            result = func(*args, **kwargs)
            length = len(result)
            history.append(length)
            
            update_func(history)
            
            file_path = os.path.join('debug', f"{func.__name__}.png")
            
            plt.savefig(file_path)
            plt.close()
            
            return result
        
        return wrapper
    return decorator



# Example update function
def update_plot(history):
    plt.figure()
    x = range(1, len(history) + 1)
    y = history
    plt.plot(x, y, marker='o')
    plt.xlabel('Run Number')
    plt.ylabel('Return Value')
    plt.title('Return Values Over Runs')
    plt.grid(True)

# Example usage
@visualize_output(update_plot)
def generate_data():
    # Your data generation code here
    import random
    return random.randint(0, 10)


