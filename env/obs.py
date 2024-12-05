import gymnasium as gym
import numpy as np

class Observation:
    def __init__(self,CRB_num, wrap_observation, observe_load, observation, obs) -> None:
        self.cap_num, self.reg_num, self.bat_num, self.reg_act_num, self.bat_act_num = CRB_num
        self.wrap_observation = wrap_observation
        self.observation = observation #this is what we get from reset
        self.observe_load = observe_load
        self.obs = obs
        #print(f"Inside Observation: {self.observation}, {self.obs}")
                
    def processflatdata(self):        
        '''
        reset the observation space based on the option of wrapping and load.
        
        instead of setting directly from the attribute (e.g., Env.wrap_observation)
        it is suggested to set wrap_observation and observe_load through this function
        
        '''
        nnode = len(np.hstack( list(self.obs['bus_voltages'].values()) ))
        if self.observe_load: nload = len(self.obs['load_profile_t'])
        
        if self.wrap_observation:
            low, high = [0.8]*nnode, [1.2]*nnode  # add voltage bound
            low, high = low+[0]*self.cap_num, high+[1]*self.cap_num # add cap bound
            low, high = low+[0]*self.reg_num, high+[self.reg_act_num]*self.reg_num # add reg bound
            low, high = low+[0,-1]*self.bat_num, high+[1,1]*self.bat_num # add bat bound
            if self.observe_load: low, high = low+[0.0]*nload, high+[1.0]*nload # add load bound
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
            if self.observe_load: obs_dict['load_profile_t'] = gym.spaces.Box(0.0, 1.0, shape=(nload,))
            self.observation_space = gym.spaces.Dict(obs_dict)
        
        return self.observation_space
            
    def processgraphdata(self, nodes, edges):
        
        '''
        reset the observation space based on the option of wrapping and load.

        instead of setting directly from the attribute (e.g., Env.wrap_observation)
        it is suggested to set wrap_observation and observe_load through this function
        '''
        node = len(nodes)
        edge = len(edges)
        self.observation_space = gym.spaces.Dict({"node_features": gym.spaces.Box(low=0.8, high=1.2, shape=(node, 3), dtype=np.float32), 
                                                  "edge_index": gym.spaces.Box(low=0, high=10000, shape=(2, edge), dtype=np.int64)})
        
        #Use this for experiments
        # nodes = 1000        
        # self.observation_space = gym.spaces.Dict({"node_features": gym.spaces.Box(low=0.8, high=1.2, shape = (nodes,3),dtype=np.float32), 
        #                                  "edge_index": gym.spaces.Box(low=0, high = nodes, shape = (2,nodes), dtype=np.int64)}) 
        
        return self.observation_space
     
                