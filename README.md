# What is PowerGym +
PowerGym+ is a gym environment for Volt-Var control in power distribution systems. The Volt-Var control aims to minimize voltage violations, control losses, and power losses under physical networked constraints and device constraints. The networked constraints are maintained by the power distribution system simulator, OpenDSS, while device constraints typically involve integer constraints on actions.

# How is it different than Siemens/PowerGym
1. Compatibility: It is fully compatible with Gymnasium, the current standard for reinforcement learning gym environments.
2. Graph-Based Observations: Observations can be wrapped as a graph, enabling advanced processing.
3. Inductive Learning Support: Designed to support inductive learning.



## Requirements
------------
For the complete installation
```
pip install -r requirements.txt
```

## Run options
### Action spaces
There are two action option: `action_type = Reduced or Original`

1. The Reduced action space for 13 bus is [2,2,33,33]. 
    Where the first two are for capacitor, 
    the next 33 for voltage regulator and 
    the last 33 is for the battery
2. The orignal action space for 13 bus is [2,2,33,33,33,33] 
    Where the first two are for capacitor, 
    the next three 33 are for voltage regulator and 
    the last 33 is for battery

The range of action space is given below:
|**Action Space** | |
| ------------- | ------------- |
| **Variable**| **Range**|
| Capacitor status     | {0, 1} |
| Regulator tap number | {0, ..., 32} |
| Discharge power (disc.) | {0, ..., 32} |
| Discharge power (cont.) | [-1, 1]  |

In the reduced action space we assume that the same tap_num is applied across all the voltage regulators. This allow us to simplify the aciton space.
### Observation spaces
There are two observation option: `obs_type = graph or non-graph`
Below is a description of non-graph observation. {} denotes a finite set and [] denote a continuous interval.
|**Observation Space** | |
| ------------- | ------------- |
| **Variable**| **Range**|
| Bus voltage     | [0.8, 1.2] |
| Capacitor status     | {0, 1} |
| Regulator tap number | {0, ..., 32} |
| State-of-charge (soc) | [0, 1] |
| Discharge power  | [-1, 1]  |

For a graph observation we calculate the node and the number of edges based on the architecture given. Each node has a node feature. In our case each node has 3 feature. Here is an example of how the 13 bus graph observation looks like. 

graph_observation
{'node_features': tensor([[1.0000, 1.0000, 1.0000],
        [1.0000, 1.0000, 1.0000],
        [1.0062, 1.0062, 1.0062],
        [0.9803, 0.9959, 0.9784],
        [0.9686, 0.9874, 0.9692],
        [0.9615, 0.9962, 0.9589],
        [0.9924, 0.9789, 0.0000],
        [0.9916, 0.9779, 0.0000],
        [0.9615, 0.9962, 0.9589],
        [0.9574, 0.9967, 0.9570],
        [0.9591, 0.0000, 0.0000],
        [0.9584, 0.0000, 0.0000],
        [0.9751, 0.9962, 0.9722],
        [0.9817, 0.9968, 0.9797],
        [0.9615, 0.9962, 0.9589],
        [0.9606, 0.9590, 0.0000]]), 
'edge_index': tensor([[ 0,  2,  2,  2,  3, 13, 13, 12,  5, 13, 13,  7,  9,  5, 10, 11,  5],
        [ 1,  1,  1,  1,  4,  2, 12,  5, 14,  3,  6,  6,  8, 15, 15, 15,  8]])}

### Bateries
There are two kinds of batteries. 
1. Discrete battery has discretized choices on the discharge power (e.g., choose from {0,...,32})
2. continuous battery chooses the normalized discharge power from the interval [-1,1]. 
The user should specify the battery's kind upon calling the environment.

### Rewards
The reward function is a combination of three losses: 
1. voltage violation, 
2. control error, and 
3. power loss 

The control error is further decomposed into capacitor's & regulator's switching cost and battery's discharge loss & soc loss. The weights among these losses depends on the circuit system and is listed in the Appendix of our paper.


The implemented circuit systems are summerized as follows.
| **System**| **# Caps**| **# Regs**| **# Bats**|
| ------------- | ------------- |------- |------- |
| 13Bus     | 2 | 3 | 1 |
| 34Bus | 4 | 6 | 2 |
| 123Bus | 4 | 7 | 4 |
| 8500Node | 10 | 12 | 10 |


# Training

### DQN
To run a DQN run the following code
```
python training\dqn.py --policy --env_name --total_training_step --buffer_size --learning_starts --action_type --obs_type --load_profile_id
```

|**Argument Parser** | |
| ------------- | ------------- |
| **Argument**| **Default**| **Range**|
| policy     | MLP | MLP, GNN_FE, GNN_Policy |
| env_name     | 13Bus | 13Bus, 34Bus, 123Bus, 8500-Node |
| total_training_step    |1e5 | [1 - Any larger number] |
| buffer_size     | 1e4 | [1 - Any larger number] Note: Make sure it is less than training steps |
| learning_starts    | 1e3 |[1 - Any larger number] Note: Make sure it is less than training steps|
| action_type     | Reduced | Reduced, Original |
| obs_type     | graph | graph, non_graph |
| load_profile_id     | 0 |13bus: [0,72], 34bus: [0,15], 123Bus: [0,15] |


Example with `GNN_FE` and `34_Bus`

```
python training\dqn.py --policy GNN_FE --env_name 34Bus
```

The models are saved directly in `saved_models`. 
The name of the model is of the format `name = f"{args.env_name}_{args.policy}_ts_{args.total_training_step}_bs_{args.buffer_size}_ls_{args.learning_starts}"`
The tensorboad file is saved in `data\dqn_tensorboard`

To view the reward in tensorboard after executing a script `tensorborad --logdir Path_to_dqn_tensorboard`
For example my path is `tensorboard --logdir C:\Users\Romesh\Desktop\volt_var\volt_var\data\dqn_tensorboard`

# Testing
```
python .\testing\test_dqn.py --env_name 34Bus --policy GNN_FE --test all_load_profile --saved_training_file 34_dqn_fe_graph_p_mlp_300000.zip
```

All the other parser arguments for testing are similar to training except `--test` and `saved_training_file`. 
We give you two options `all_load_profiles` or `single_load_profile`. If you select `all_load_profiles` based on the type of `env_name` it will automatically select all the loadprofile for that particular instance of the environment. But if you only want to test for single load profile you can select `single_load_profile`. The `saved_training_file` is **required**. We provide you with one saved model to test the agent. But please add the file name of the saved model. You will find this **name** in the `saved_models` directory.

# Understanding Policy.

Raw Observations:
    The environment provides raw observations, which typically include node features and edge indices for graph-structured data.

Feature Extraction and Q-Value Computation:

    Feature Extraction:
        The CombinedGCN model starts by processing these raw observations through the Graph Convolutional Network (GCN) layers.
        The GCN layers (conv1 and conv2) transform node features based on the graph structure defined by edge_index.
        This step aggregates and extracts meaningful features from the graph data.

    Q-Value Computation:
        After feature extraction, the output from the GCN layers is pooled into a single vector representing the entire graph.
        This pooled feature vector is then processed through fully connected layers (fc and output_layer).
        The final layer computes the Q-values for each possible action, which are used to evaluate the quality of taking each action given the current state.

Output:
    The output of the CombinedGCN model is the Q-values for each action.
    These Q-values are used by the DQN algorithm to decide which action to take and to update the Q-values based on the rewards received.

This integrated approach ensures that both feature extraction and Q-value computation are handled within a single network, streamlining the architecture and making it more efficient for DQN.