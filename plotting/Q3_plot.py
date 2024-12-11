import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load data from a NumPy file
numpy_data_1 = np.load('reward_data/reward_train_test_13_bus.npy')  # train and test 13
numpy_data_2 = np.load('reward_data/reward_train_13_bus_test_34_bus_og.npy') # train 13 test 34
numpy_data_3 = np.load('reward_data/reward_train_34_test_13_gnn_3e5.npy') # train 34 test 13
numpy_data_4 = np.load('reward_data/reward_train_test_34_bus.npy') #train and test 34


reward_sum_train_13_test_13 = np.sum(numpy_data_1,axis=1)
reward_sum_train_13_test_34 = np.sum(numpy_data_2,axis=1)
reward_sum_train_34_test_13 = np.sum(numpy_data_3,axis=1)
reward_sum_train_34_test_34 = np.sum(numpy_data_4,axis=1)

y1 = reward_sum_train_13_test_13
y2 = reward_sum_train_34_test_34
y3 = reward_sum_train_13_test_34
y4 = reward_sum_train_34_test_13


# Load data from a CSV file
csv_data = pd.read_csv('reward_data/13_bus/13_dqn_fe_graph_p_mlp_300000_1.csv')  # Replace with your actual file path
print(csv_data.head())
x_csv = csv_data['Step']  # Replace 'x' and 'y' with your CSV column names
y_csv = csv_data['Value']

#create gloabl font and bold
plt.rcParams.update({
    'font.size': 14,          # Increase font size for readability
    'font.weight': 'bold',    # Set font weight to bold
    'axes.titleweight': 'bold',  # Bold the subplot titles
    'axes.labelweight': 'bold',  # Bold the axis labels
})

# Create a figure with subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Plot the top-left subplot (from NumPy data)
axs[0, 0].scatter(np.arange(len(y1)), y1, color='blue', label='Cumulative Reward')
axs[0, 0].set_title('Train and Test on 13 Bus')
axs[0, 0].set_xlabel('Number of Load Profiles')
axs[0, 0].set_ylabel('Reward')
axs[0, 0].axhline(y=np.mean(reward_sum_train_13_test_13), color='r', linestyle='--', label=f'Mean: {np.mean(reward_sum_train_13_test_13):.2f}')
axs[0, 0].legend()
axs[0, 0].tick_params(axis='both', labelsize=12)  # Customize tick font size
axs[0, 0].set_ylim(-5.8,-5)

# Plot the top-right subplot (from NumPy data)
axs[1, 1].scatter(np.arange(len(y3)), y3, color='green', label='Cumulative Reward')
axs[1, 1].set_title('Train on 13 Bus and Test on 34 Bus')
axs[1, 1].set_xlabel('Number of Load Profiles')
axs[1, 1].set_ylabel('Reward')
axs[1, 1].axhline(y=np.mean(reward_sum_train_13_test_34), color='r', linestyle='--', label=f'Mean: {np.mean(reward_sum_train_13_test_34):.2f}')
axs[1, 1].legend()
axs[1, 1].tick_params(axis='both', labelsize=12)  # Customize tick font size
axs[1, 1].set_ylim(-10,-5)

# Plot the bottom -left subplot (from NumPy data)
axs[1, 0].scatter(np.arange(len(y2)), y2, color='green', label='Cumulative Reward')
axs[1, 0].set_title('Train and Test on 34 Bus')
axs[1, 0].set_xlabel('Number of Load Profiles')
axs[1, 0].set_ylabel('Reward')
axs[1, 0].axhline(y=np.mean(reward_sum_train_34_test_34), color='r', linestyle='--', label=f'Mean: {np.mean(reward_sum_train_34_test_34):.2f}')
axs[1, 0].legend()
axs[1, 0].tick_params(axis='both', labelsize=12)  # Customize tick font size
axs[1, 0].set_ylim(-6,-5)

# Plot the bottom-right subplot (from NumPy data)
axs[0, 1].scatter(np.arange(len(y4)), y4, color='blue', label='Cumulative Reward')
axs[0, 1].set_title('Train on 34 Bus and Test on 13 Bus')
axs[0, 1].set_xlabel('Number of Load Profiles')
axs[0, 1].set_ylabel('Reward')
axs[0, 1].axhline(y=np.mean(reward_sum_train_34_test_13), color='r', linestyle='--', label=f'Mean: {np.mean(reward_sum_train_34_test_13):.2f}')
axs[0, 1].legend()
axs[0, 1].tick_params(axis='both', labelsize=12)  # Customize tick font size
axs[0, 1].set_ylim(-7.6,-5.5)


# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])

fig.suptitle('Left: Train & Test on same architecture | Right: Train on small & Test on big architecuture & vice versa', fontsize=20, fontweight='bold')

# Show the plot
plt.show()
