import os
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
    
SAVEFIG = True
#SAVEFIG = False
algo = 'ppo'
#policy = 'graph'
policy = 'mlp'
#env_name = "13Bus"
env_name = "13Bus"
std_dev_scale = 5
#combined_results = read_eval_results(base_directory)
#print(combined_results)

base_dir = f"/home/hugo/experiment/AI_Power/powergym_plus/powergym_plus/logs/{algo}_{policy}/"  # Change this to the root directory containing the folders
index_range = range(1,6,1)

def collect_data(base_dir, index_range, env_name):
    all_data = []
    for loop1 in index_range:
        folder = f"{base_dir}{env_name}_{loop1}/"
        file_path = os.path.join(folder, "eval_results.txt")
        if os.path.isfile(file_path):
            try:
                # df = pd.read_csv(file_path)
                data = np.loadtxt(file_path, delimiter = ',', skiprows = 1)
                all_data.append(data)
                print(f'current datasize: {data.shape}')
            except Exception as e:
                print(f"Failed to read {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")

    # calculate average rewards and std rewards across sampling seeds
    num_row, num_col = all_data[0].shape
    average_seed = np.empty(all_data[0].shape)
    average_seed[:,0] = all_data[0][:,0]

    for loop1 in range(0, num_row):
        cur_total1 = 0
        cur_total2 = 0
        for data in all_data:
            cur_total1 = cur_total1 + data[loop1, 1]
            cur_total2 = cur_total2 + data[loop1, 2]
        average_seed[loop1, 1] = cur_total1 / len(all_data)
        average_seed[loop1, 2] = cur_total2 / len(all_data)
    return average_seed

average_seed13 = collect_data(base_dir, index_range, env_name)
#print(average_seed)
env_name = "34Bus"
average_seed34 = collect_data(base_dir, index_range, env_name)
env_name = "123Bus"
average_seed123 = collect_data(base_dir, index_range, env_name)

plt.rcParams["font.family"] = "Times New Roman"
fig = plt.figure(num = 1, figsize = (6, 3), dpi = 300)
ax1 = plt.subplot(111)
# x = np.array(range(1, len(average_seed[0]) + 1))
p11, = ax1.plot(average_seed13[:, 0], average_seed13[:, 1], linewidth = 0.7, color='#003f5c')
ax1.fill_between(average_seed13[:, 0], average_seed13[:, 1] - std_dev_scale * average_seed13[:, 2], average_seed13[:, 1] + std_dev_scale * average_seed13[:, 2], alpha = 0.5, label = 'std dev', color = '#ff7c00')
p12, = ax1.plot(average_seed34[:, 0], average_seed34[:, 1], linewidth = 0.7, color='#2f4b7c')
ax1.fill_between(average_seed34[:, 0], average_seed34[:, 1] - std_dev_scale * average_seed34[:, 2], average_seed34[:, 1] + std_dev_scale * average_seed34[:, 2], alpha = 0.5, label = 'std dev', color = '#7ad6c1')
p13, = ax1.plot(average_seed123[:, 0], average_seed123[:, 1], linewidth = 0.7, color='#d62728')
ax1.fill_between(average_seed123[:, 0], average_seed123[:, 1] - std_dev_scale * average_seed123[:, 2], average_seed123[:, 1] + std_dev_scale * average_seed123[:, 2], alpha = 0.5, label = 'std dev', color = '#f7a1a1')

ax1.legend([p11, p12, p13], ['13Bus', '34Bus', '123Bus'], fontsize = 12, ncol = 3, loc='upper center')
ax1.set_ylim((-120, 20))
ax1.tick_params(axis='both', labelsize=14)
plt.ylabel('Cumulative Rewards', fontsize = 14)
plt.xlabel('# of Episodes', fontsize = 14)

plt.tight_layout()

fig_name = f'{algo}_{policy}_training'
if SAVEFIG == True:
    #plt.savefig('north_gp.eps', dpi = 200, format = 'eps')
    plt.savefig(f'{fig_name}.svg', dpi = 300, format = 'svg')
plt.show()

 

