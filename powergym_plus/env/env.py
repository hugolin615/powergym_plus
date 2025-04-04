# Copyright 2021 Siemens Corporation
# SPDX-License-Identifier: MIT

import os
import gymnasium as gym
import numpy as np
from powergym_plus.env.circuit import Circuits
from powergym_plus.env.loadprofile import LoadProfile
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import torch as th
from torch_geometric.data import Data
from gymnasium.spaces import Graph, Box, Discrete

#this is our new imports
from powergym_plus.env.action import ActionSpace
# from env.reward import MyReward
# from env.obs import Observation
#### helper functions ####

#### environment class ####
class volt_var(gym.Env):
    """Enviroment to train RL agentS
    
    Attributes:
        obs (dict): Observation/state of system
        dss_folder_path (str): Path to folder containing DSS file
        dss_file (str): DSS simulation filename
        source_bus (str): the bus (with coordinates in BusCoords.csv) closest to the source
        node_size (int): the size of node in plots
        shift (int): the shift amount of labels in plots
        show_node_labels (bool): show node labels in plots
        scale (float): scale of the load profile
        wrap_observation (bool): whether to flatten obs into array at the outputs of reset & step
        observe_load (bool): whether to include the nodal loads in the observation
        
        load_profile (obj): Class for load profile management
        num_profiles (int): number of distinct profiles generated by load_profile
        horizon (int): Maximum steps in a episode
        circuit (obj): Circuit object linking to DSS simulation
        all_bus_names (list): All bus names in system
        cap_names (list): List of capacitor bus
        reg_names (list): List of regulator bus
        bat_names (list): List of battery bus
        cap_num (int): number of capacitors
        reg_num (int): number of regulators
        bat_num (int): number of batteries
        reg_act_num (int): Number of reg control actions
        bat_act_num (int): Number of bat control actions
        topology (graph): NxGraph of power system
        reward_func (obj): Class of reward fucntions
        t (int): Timestep for environment state
        ActionSpace (obj): Action space class. Use for sampling random actions
        action_space (gym.spaces): the base action space from class ActionSpace
        observation_space (gym.spaces): observation space of environment.
        
    Defined at self.step(), self.reset():
        all_load_profiles (dict): 2D array of load profile for all bus and time
    
    Defined at self.step() and used at self.plot_graph()
        self.str_action: the action string to be printed at self.plot_graph()
        
    Defined at self.build_graph():
        edges (dict): Dict of edges connecting nodes in circuit
        lines (dict): Dict of edges with components in circuit
        transformers (dict): Dictionary of transformers in system
    """
    # def __init__(self, folder_path, profile_id ,action_type, obs_type, info, dss_act=False):
    def __init__(self, folder_path, info, dss_act=False):
        super().__init__()
        self.obs = dict()
        """
        Example: 
        self.dss_folder_path = '/home/romesh-prasad/ONR/volt_var/volt_var/systems/13Bus' 
        self.dss_file = 'IEEE13Nodeckt_daily.dss'
        self.source_bus = 'source_bus'
        self.node_size = 500
        self.shift = 10
        self.show_node_labels = True
        self.scale = 1.0
        self.wrap_observation = True
        self.observe_load = False
        """
        self.dss_folder_path = os.path.join(folder_path, info['system_name'])
        print(f"inside_volt_var:{self.dss_folder_path}") 
        self.dss_file = info['dss_file'] 
        self.source_bus = info['source_bus']
        self.node_size = info['node_size']
        self.shift = info['shift']
        self.show_node_labels = info['show_node_labels']
        self.scale = info['scale'] if 'scale' in info else 1.0
        self.wrap_observation = True
        self.observe_load = True
        #self.action_type = action_type
        #self.obs_type = obs_type
        #self.profile_id = profile_id

        # generate load profile files
        self.load_profile = LoadProfile(\
                 info['max_episode_steps'],
                 self.dss_folder_path,
                 self.dss_file,
                 worker_idx = info['worker_idx'] if 'worker_idx' in info else None)

        self.num_profiles = self.load_profile.gen_loadprofile(scale=self.scale) #73 for 13 bus
        # choose a dummy load profile for the initialization of the circuit
        # self.load_profile.choose_loadprofile(self.profile_id)
        self.load_profile.choose_loadprofile(0)
        print('number of profiles: ', self.num_profiles)
        
        # problem horizon is the length of load profile
        self.horizon = info['max_episode_steps']
        self.reg_act_num = info['reg_act_num']
        self.bat_act_num = info['bat_act_num']
        assert self.horizon>=1, 'invalid horizon'
        assert self.reg_act_num>=2 and self.bat_act_num>=2, 'invalid act nums'
        
        self.circuit = Circuits(os.path.join(self.dss_folder_path, self.dss_file),
                                RB_act_num=(self.reg_act_num, self.bat_act_num),
                                dss_act=dss_act)
        self.all_bus_names = self.circuit.dss.ActiveCircuit.AllBusNames
        self.cap_names = list(self.circuit.capacitors.keys())
        self.reg_names = list(self.circuit.regulators.keys())
        self.bat_names = list(self.circuit.batteries.keys())
        self.cap_num = len(self.cap_names)
        self.reg_num = len(self.reg_names)
        self.bat_num = len(self.bat_names)
        assert self.cap_num>=0 and self.reg_num>=0 and self.bat_num>=0 and \
               self.cap_num + self.reg_num + self.bat_num>=1,'invalid CRB_num'
        
        #self.topology = self.build_graph()
        self.t = 0

        #self.reward_func = MyReward(volt_var, info, self.obs, self.t, self.horizon)
        self.reward_func = self.MyReward(self, info)
        
        
        # create action space and observation space
        self.ActionSpace = ActionSpace( (self.cap_num, self.reg_num, self.bat_num),
                                        (self.reg_act_num, self.bat_act_num) )
        print(f"self.cap_num:{self.cap_num}, self.reg_num:{self.reg_num}, self.bat_num:{self.bat_num}, self.reg_act_num: {self.reg_act_num}, self.bat_act_num:{self.bat_act_num}")
        self.action_space = self.ActionSpace.space
        
        # HL: ignore the wrap_observation at this moment, using the dict types for experiments first
        #      need to determine appropriate way of wrapping observation
        # self.reset()
        self.reset_obs_space()
        
        #print(f"This is in init, {observation}, {_}")
        # nodes, edges, lines, transformer = self.build_edges()
        #obs = Observation((self.cap_num, self.reg_num, self.bat_num, self.reg_act_num, self.bat_act_num), self.wrap_observation, self.observe_load, observation, self.obs)
        # if self.obs_type == "non_graph":
        #    self.observation_space = obs.processflatdata()
        #elif self.obs_type == "graph":
        #    self.observation_space = obs.processgraphdata(nodes, edges)
    
    def step(self, action):
        """Steps through one step of enviroment and calls DSS solver after one control action
        
        Args:
            action [array]: Integer array of actions for capacitors, regulators and batteries
        
        Returns:
            self.wrap_obs(self.obs), reward, done, all_reward, all_diff
            next wrapped observation (array), reward (float), Done (bool), all rewards (dict), all state errors (dict)
        """
        action_idx = 0
        self.str_action = '' # the action string to be printed at self.plot_graph()
        
        # if self.action_type == "Reduced":
        #    act = self.ActionSpace
        #    action = act.decode_action(dis_action)
        #    print(f"Dis_action: {dis_action} and multi_dis action:{action}")
        #elif self.action_type == "Original":
        #    action = dis_action        

        ### capacitor control
        # print(f"This is the selected action: {action}")
        if self.cap_num>0:
            #print(f'action: {action}, action_idx: {action_idx}, cap_num: {self.cap_num}')
            statuses = action[action_idx:action_idx+self.cap_num] #use index to get the right action for cap
            capdiff = self.circuit.set_all_capacitor_statuses(statuses) #array of size 2
            cap_statuses = {cap:status for cap, status in \
                            zip(self.circuit.capacitors.keys(), statuses)}
            action_idx += self.cap_num
            self.str_action += 'Cap Status:'+str(statuses)
        else: capdiff, cap_statuses = [], dict()

        if self.reg_num>0:
            tapnums = action[action_idx:action_idx+self.reg_num]
            #if self.action_type == "Original":
            #    tapnums = action[action_idx:action_idx+self.reg_num]
            #elif self.action_type == "Reduced":
            #    tapnums = [action[action_idx]] * self.reg_num
            #    print(f"This is tapnums: {tapnums}")
            regdiff = self.circuit.set_all_regulator_tappings(tapnums)
            reg_statuses = {reg:self.circuit.regulators[reg].tap \
                            for reg in self.reg_names}
            action_idx += self.reg_num
            self.str_action += 'Reg Tap Status:'+str(tapnums)
        else: regdiff, reg_statuses = [], dict()

        ### battery control
        if self.bat_num>0:
            #states = [action[-1]]*self.bat_num
            states = action[action_idx:]
            self.circuit.set_all_batteries_before_solve(states)
            self.str_action += 'Bat Status:'+str(states)
        
        self.circuit.dss.ActiveCircuit.Solution.Solve()

        ### update battery kWh. record soc_err and discharge_err
        if self.bat_num>0:
            soc_errs, dis_errs = self.circuit.set_all_batteries_after_solve()
            bat_statuses = {name:[bat.soc, -1*bat.actual_power()/bat.max_kw] for name, bat in self.circuit.batteries.items()}
        else: soc_errs, dis_errs, bat_statuses = [], [], dict()
        #I have only got the action from action space and checked with previous action for now
        ### update time step
        self.t += 1 
 
        ### Update obs ###
        bus_voltages = dict()
        for bus_name in self.all_bus_names:
            bus_voltages[bus_name] = self.circuit.bus_voltage(bus_name) #get bus voltage based on the actions
            bus_voltages[bus_name] = [bus_voltages[bus_name][i] for i in range(len(bus_voltages[bus_name])) if i%2==0] #only takes even index
        
        self.obs['bus_voltages'] = bus_voltages
        self.obs['cap_statuses'] = cap_statuses
        self.obs['reg_statuses'] = reg_statuses
        self.obs['bat_statuses'] = bat_statuses
        self.obs['power_loss'] = - self.circuit.total_loss()[0]/self.circuit.total_power()[0]
        self.obs['time'] = self.t
        if self.observe_load:
            self.obs['load_profile_t'] = self.all_load_profiles.iloc[self.t%self.horizon].to_dict()

        done = (self.t >= self.horizon)
    
        # Determine if the episode is truncated
        truncated = False  # Update this based on your specific condition for truncation

        reward, info = self.reward_func.composite_reward(capdiff, regdiff,\
                                                         soc_errs, dis_errs)
        # avoid dividing by zero
        info.update( {'av_cap_err': sum(capdiff)/(self.cap_num+1e-10),
                      'av_reg_err': sum(regdiff)/(self.reg_num+1e-10),
                      'av_dis_err': sum(dis_errs)/(self.bat_num+1e-10),
                      'av_soc_err': sum(soc_errs)/(self.bat_num+1e-10),
                      'av_soc': sum([soc for soc, _ in bat_statuses.values()])/ \
                                   (self.bat_num+1e-10)  })
        
        #print(f"observation_prior_wrap_in_step: {self.obs}")
        if self.wrap_observation:
            return self.wrap_obs(self.obs), reward, done, truncated, info
            #if self.obs_type == "non_graph":
            #    observation = self.flat_wrap_obs(self.obs)
            #elif self.obs_type == "graph":
            #    observation = self.graph_wrap_obs(self.obs)
        else:
            #observation = self.obs
            return self.obs, reward, done, truncated, info
        
        #print(f"observation_after_wrap_in_step: {observation}")
        
        #return observation, reward, done, truncated, info

    def reset_obs_space(self, wrap_observation=True, observe_load=False):
        '''
        reset the observation space based on the option of wrapping and load.
        
        instead of setting directly from the attribute (e.g., Env.wrap_observation)
        it is suggested to set wrap_observation and observe_load through this function
        
        '''
        self.wrap_observation = wrap_observation
        self.observe_load = observe_load
        
        self.reset(load_profile_idx=0)
        #self.reset()
        #nnode = len(self.obs['bus_voltages'])
        nnode = len(np.hstack( list(self.obs['bus_voltages'].values()) ))
        if observe_load: nload = len(self.obs['load_profile_t'])
        
        if self.wrap_observation:
            low, high = [0.8]*nnode, [1.2]*nnode  # add voltage bound
            low, high = low+[0]*self.cap_num, high+[1]*self.cap_num # add cap bound
            low, high = low+[0]*self.reg_num, high+[self.reg_act_num]*self.reg_num # add reg bound
            low, high = low+[0,-1]*self.bat_num, high+[1,1]*self.bat_num # add bat bound
            if observe_load: low, high = low+[0.0]*nload, high+[1.0]*nload # add load bound
            low, high = np.array(low, dtype=np.float32), np.array(high, dtype=np.float32)
            self.observation_space = gym.spaces.Box(low, high) 
        else:
            bat_dict = {bat: gym.spaces.Box(np.array([0,-1]), np.array([1,1]), dtype=np.float32) 
                        for bat in self.obs['bat_statuses'].keys()}
            obs_dict = {
                'bus_voltages': gym.spaces.Box(0.8, 1.2, shape=(nnode,)),
                'cap_statuses': gym.spaces.MultiDiscrete([2]*self.cap_num),
                'reg_statuses': gym.spaces.MultiDiscrete([self.reg_act_num]*self.cap_num),
                'bat_statuses': gym.spaces.Dict(bat_dict)
            }
            if observe_load: obs_dict['load_profile_t'] = gym.spaces.Box(0.0, 1.0, shape=(nload,))
            self.observation_space = gym.spaces.Dict(obs_dict)

    #def reset(self, seed = None): # put seed = None so that it is compatible with openAI gymnasium env
    def reset(self, seed = None, load_profile_idx = 0): # put seed = None so that it is compatible with openAI gymnasium env
        """
        Reset state of enviroment for new episode
        
        Args:
            load_profile_idx (int, optional): ID number for load profile
        
        Returns:
            numpy array: wrapped observation
        """
        super().reset(seed=seed)
        ###reset time
        self.t = 0
 
        ### choose load profile
        self.load_profile.choose_loadprofile(load_profile_idx)
        #self.load_profile.choose_loadprofile(0)
        self.all_load_profiles = self.load_profile.get_loadprofile(load_profile_idx)
        # self.all_load_profiles = self.load_profile.get_loadprofile(0)
        # print('what is all_load_profiles: ', self.all_load_profiles)

        #print(f"This is the profile: {load_profile_idx}")
        
        ### re-compile dss and reset batteries
        self.circuit.reset()

        ### node voltages
        bus_voltages = dict()
        for bus_name in self.all_bus_names:
            bus_voltages[bus_name] = self.circuit.bus_voltage(bus_name)
            bus_voltages[bus_name] = [bus_voltages[bus_name][i] for i in range(len(bus_voltages[bus_name])) if i%2==0]
        self.obs['bus_voltages'] = bus_voltages

        ### status of capacitor
        cap_statuses = {name:cap.status for name, cap in self.circuit.capacitors.items()}
        self.obs['cap_statuses'] = cap_statuses
        
        ### status of regulator
        reg_statuses = {name:reg.tap for name, reg in self.circuit.regulators.items()}
        self.obs['reg_statuses'] = reg_statuses

        ### status of battery
        bat_statuses = {name:[bat.soc, -1*bat.actual_power()/bat.max_kw] for name, bat in self.circuit.batteries.items()}
        self.obs['bat_statuses'] = bat_statuses

        ### total power loss
        self.obs['power_loss'] = -self.circuit.total_loss()[0]/self.circuit.total_power()[0]
        
        ### time step tracker
        self.obs['time'] = self.t

        ### load for current timestep
        if self.observe_load:
            self.obs['load_profile_t'] = self.all_load_profiles.iloc[self.t].to_dict()

        ### Edge weight
        #self.obs['Y_matrix'] = self.circuit.edge_weight

        #print(f"observation_prior_to_wrap_in_reset: {self.obs}")
        
        # optional: extra info returned by env
        info = {}

        if self.wrap_observation:
            #if self.obs_type == "non_graph":
            #    observation = self.flat_wrap_obs(self.obs)
            #elif self.obs_type == "graph":
            #    observation = self.graph_wrap_obs(self.obs)
            return self.wrap_obs(self.obs), info
        else:
            #observation = self.obs
            return self.obs, info

        #print(f"observation_after_wrap_in_reset: {observation}")

        # return observation, {} #reset requires tuple of observation and info or etc.

    def build_edges(self):
        
        """Constructs a NetworkX graph for downstream use
        
        Returns:
            Graph: Network graph
        """
        self.lines = dict()
        self.circuit.dss.ActiveCircuit.Lines.First
        while(True):
            bus1 = self.circuit.dss.ActiveCircuit.Lines.Bus1.split('.', 1)[0].lower() #gets the first bus name
            bus2 = self.circuit.dss.ActiveCircuit.Lines.Bus2.split('.', 1)[0].lower() #gets the second bus name
            line_name = self.circuit.dss.ActiveCircuit.Lines.Name.lower() #gets the line number
            self.lines[line_name] = (bus1, bus2) 
            if self.circuit.dss.ActiveCircuit.Lines.Next==0:
                break

        transformer_names = self.circuit.dss.ActiveCircuit.Transformers.AllNames
        self.transformers = dict()
        for transformer_name in transformer_names:
            self.circuit.dss.ActiveCircuit.SetActiveElement('Transformer.' + transformer_name)
            buses = self.circuit.dss.ActiveCircuit.ActiveElement.BusNames
            #assert len(buses) == 2, 'Transformer {} has more than two terminals'.format(transformer_name)
            bus1 = buses[0].split('.', 1)[0].lower()
            bus2 = buses[1].split('.', 1)[0].lower()
            self.transformers[transformer_name] = (bus1, bus2)

        self.edges = [frozenset(edge) for _, edge in self.transformers.items()] + [frozenset(edge) for _, edge in self.lines.items()]
        if len(self.edges) != len(set(self.edges)):
            print('There are ' + str(len(self.edges)) + ' edges and ' + str(len(set(self.edges))) + ' unique edges. Overlapping transformer edges')

        self.circuit.topology.add_edges_from(self.edges)
        print(f"len_node: {len(self.circuit.topology.nodes)}")
        # print(f"nodes: {self.circuit.topology.nodes}")
        print(f"len_edges: {len(self.circuit.topology.edges)}")
        # print(f"edges: {self.circuit.topology.edges}")

        self.adj_mat = nx.adjacency_matrix(self.circuit.topology)
        return self.circuit.topology.nodes, self.circuit.topology.edges, self.lines, self.transformers
    
    def _process_edge_index(self, edge_index):
        if isinstance(edge_index, np.ndarray):
            edge_index = th.from_numpy(edge_index).long()
        else:
            edge_index = edge_index.long()

        if edge_index.dim() == 1:
            edge_index = edge_index.unsqueeze(0)
        
        if edge_index.size(0) == 1:
            edge_index = edge_index.repeat(2, 1)
        elif edge_index.size(0) > 2:
            edge_index = edge_index.t()
        
        assert edge_index.size(0) == 2, f"Edge index should have shape [2, num_edges], got {edge_index.shape}"
        
        return edge_index.contiguous()

    def wrap_obs(self, obs):
        """ Wrap the observation dictionary (i.e., self.obs) to a numpy array
        
        Attribute:
            obs: the observation distionary generated at self.reset() and self.step()
        
        Return:
            a numpy array of observation.
        
        """
        key_obs = ['bus_voltages', 'cap_statuses', 'reg_statuses', 'bat_statuses']
        if self.observe_load: key_obs.append('load_profile_t')

        mod_obs = []
        for var_dict in key_obs:
            # node voltage is a dict of dict, we only take minimum phase node voltage
            #if var_dict == 'bus_voltages': 
            #    for values in obs[var_dict].values():
            #        mod_obs.append(min(values))
            if var_dict in \
                ['bus_voltages','cap_statuses','reg_statuses', 'bat_statuses', 'load_profile_t']:
                mod_obs = mod_obs + list(obs[var_dict].values())
            elif var_dict == 'power_loss':
                mod_obs.append(obs['power_loss'])
        return np.hstack(mod_obs)

    def graph_wrap_obs(self, obs):
        """ Wrap the observation dictionary (i.e., self.obs) to a gym graph space
        and add the node features.
        Currently, the node features are: bus_voltages (3), cap_status(1), VR_status(3), battery_status(2).
        So, each node will have 9 features.
        Attribute:
            obs: the observation distionary generated at self.reset() and self.step()
        
        Return:
            a gym graph space of observation
        
        """
        #__________________building nodes and features___________________________#

        bus_voltages = list(obs['bus_voltages'].values())
        #print(f"bus_voltages: {bus_voltages}")

        # Find the maximum length of the voltage sequences
        max_length = max(len(v) for v in bus_voltages)

        # Pad shorter sequences with zeros to make all sequences the same length
        padded_voltages = [v + [0.0] * (max_length - len(v)) for v in bus_voltages]
        #print(f"padded:{padded_voltages}")

        # Convert the list of padded voltage lists to a PyTorch tensor
        node_features = th.tensor(padded_voltages, dtype = th.float32)
        #print(f"node_features:{node_features}")


        #__________________stuff_I_added___________________________#

        # max_nodes = 1000

        # num_actual_nodes = node_features.shape[0]

        # if num_actual_nodes < max_nodes:
        #     padding_size = max_nodes - num_actual_nodes
        #     padding_tensor = th.zeros((padding_size, node_features.shape[1]), dtype=node_features.dtype)
        #     node_features = th.cat([node_features, padding_tensor], dim=0)

        #__________________building edges___________________________#

        all_buses = list(obs['bus_voltages'].keys())
        #print(f"Unique bus names: {all_buses}")

        #index to the buses
        bus_to_index = {bus: idx for idx, bus in enumerate(all_buses)}
        #print(f"indexing: {bus_to_index}")

        topology, edges, lines, transformer = self.build_edges()

        #create mapping
        edge = []
        for (bus1, bus2) in edges:
            # print(f"bus 1: {bus1}, bus 2: {bus2}")
            # print(f"index_1: {bus_to_index[bus1]}, index_2: {bus_to_index[bus2]}")
            edge.append((bus_to_index[bus1], bus_to_index[bus2]))
        #print(f"mapping: {edge}")

        #edge_index = th.tensor(edge, dtype=th.long).t().contiguous()
        #print(f"edge converted: {edge_index}")


        edge_index = th.tensor(edge, dtype=th.long).t()


        #__________________stuff_I_added___________________________#
        # # Pad edge index if necessary
        # edge_index_np = edge_index.numpy()  # Convert to numpy array for padding
        # num_edges = edge_index_np.shape[1]
        
        # if num_edges < max_nodes:
        #     pad_width = ((0, 0), (0, max_nodes - num_edges))
        #     padded_edge_index = np.pad(edge_index_np, pad_width=pad_width, mode='constant', constant_values=0)
        #     edge_index = th.tensor(padded_edge_index, dtype=th.long)

        #convert into PyG data
        data = Data(x=node_features, edge_index=edge_index)
        #print(f"Data: {data}")

        edge_index = self._process_edge_index(edge_index)

        observation = {"node_features": node_features, "edge_index" : edge_index}

        # print("Node features shape:", node_features.shape)
        # print("Node features dtype:", node_features)
        # print("Edge index shape:", edge_index.shape)
        # print("Edge index dtype:", edge_index)
              
        return observation
    
    def flat_wrap_obs(self, obs):
        """ Wrap the observation dictionary (i.e., self.obs) to a numpy array
        
        Attribute:
            obs: the observation distionary generated at self.reset() and self.step()
        
        Return:
            a numpy array of observation.
        
        """
        key_obs = ['bus_voltages', 'cap_statuses', 'reg_statuses', 'bat_statuses']
        if self.observe_load: key_obs.append('load_profile_t')

        mod_obs = []
        for var_dict in key_obs:
            if var_dict in \
                ['bus_voltages','cap_statuses','reg_statuses', 'bat_statuses', 'load_profile_t']:
                mod_obs = mod_obs + list(obs[var_dict].values())
            elif var_dict == 'power_loss':
                mod_obs.append(obs['power_loss'])
        return np.hstack(mod_obs)

    def seed(self, seed):
        self.ActionSpace.seed(seed)

    def random_action(self):
        """Samples random action
        
        Returns:
            Array: Random control actions
        """
        return self.ActionSpace.sample()

    class MyReward:
        """Reward definition class
        
        Attributes:
            env (obj): Inherits all attributes of environment 
        """
        def __init__(self, env, info):
            self.env = env
            self.power_w = info['power_w']
            self.cap_w = info['cap_w']
            self.reg_w = info['reg_w']
            self.soc_w = info['soc_w']
            self.dis_w = info['dis_w']

        def powerloss_reward(self):
            # Penalty for power loss of entire system at one time step
            #loss = self.env.circuit.total_loss()[0] # a postivie float
            #gen = self.env.circuit.total_power()[0] # a negative float
            ratio = max(0.0, min(1.0, self.env.obs['power_loss']))
            return -ratio * self.power_w

        def ctrl_reward(self, capdiff, regdiff, soc_err, discharge_err):
            # penalty of actions
            ## capdiff: abs(current_cap_state - new_cap_state)
            ## regdiff: abs(current_reg_tap_num - new_reg_tap_num)
            ## soc_err: abs(soc - initial_soc)
            ## discharge_err: max(0, kw) / max_kw
            ### discharge_err > 0 means discharging
            cost =  self.cap_w * sum(capdiff) + \
                    self.reg_w * sum(regdiff) + \
                    (0.0 if self.env.t != self.env.horizon else self.soc_w * sum(soc_err)) + \
                    self.dis_w * sum(discharge_err)
            return -cost

        def voltage_reward(self, record_node = False):
            # Penalty for node voltage being out of [0.95, 1.05] range
            violated_nodes = []
            total_violation = 0
            for name, voltages in self.env.obs['bus_voltages'].items():
                max_penalty = min(0, 1.05 - max(voltages)) #penalty is negative if above max
                min_penalty = min(0, min(voltages) - 0.95) #penalty is negative if below min
                total_violation += (max_penalty + min_penalty)
                if record_node and (max_penalty != 0 or min_penalty != 0):
                    violated_nodes.append(name)
            return total_violation, violated_nodes
        
        def composite_reward(self, cd, rd, soc, dis, full=True, record_node=False):
            # the main reward function
            p = self.powerloss_reward()
            v, vio_nodes = self.voltage_reward(record_node)
            t = self.ctrl_reward(cd, rd, soc, dis)
            summ = p + v + t
            
            info = dict() if not record_node else {'violated_nodes': vio_nodes}
            if full: info.update( {'power_loss_ratio':-p/self.power_w, 
                                    'vol_reward':v, 'ctrl_reward':t} )
            
            return summ, info
    def get_num_profiles(self):
        return self.num_profiles

    def get_ep_len(self):
        return self.horizon
