import os
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
    
SAVEFIG = False
algo = 'ppo'
# policy = 'graph'
policy = 'mlp'
#env_name = "13Bus"
env_name = "34Bus"
#combined_results = read_eval_results(base_directory)
#print(combined_results)

base_dir = f"/home/hugo/experiment/AI_Power/powergym_plus/powergym_plus/logs/{algo}_{policy}/"  # Change this to the root directory containing the folders
index_range = range(1,2,1)

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

print(average_seed)

plt.rcParams["font.family"] = "Times New Roman"
fig = plt.figure(num = 1, figsize = (6, 3), dpi = 300)
ax1 = plt.subplot(111)
# x = np.array(range(1, len(average_seed[0]) + 1))
p11 = ax1.plot(average_seed[:, 0], average_seed[:, 1], linewidth = 0.5, color='#003f5c')
ax1.fill_between(average_seed[:, 0], average_seed[:, 1] - average_seed[:, 2], average_seed[:, 1] + average_seed[:, 2], alpha = 0.3, label = 'std dev', color = '#ff7c00')


plt.tight_layout()

if SAVEFIG == True:
    #plt.savefig('north_gp.eps', dpi = 200, format = 'eps')
    plt.savefig('se_time.svg', dpi = 300, format = 'svg')
plt.show()

 

