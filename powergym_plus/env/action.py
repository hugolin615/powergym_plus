import gymnasium as gym
import numpy as np

class ActionSpace:
    '''Action Space Wrapper for Capacitors, Regulators, and Batteries
   

    Attributes:
        cap_num, reg_num, bat_num (int): number of capacitors, regulators, and batteries.
        reg_act_num, bat_act_num: number of actions for regulators and batteries.
        space (gym.spaces): the space object from gym

    Note:
        There are two options: Original and Reduced
        "original" : Action space includes num_cap*2,num_reg*33,num_bat*33
            Inside "original" there are two options, you can specify with action_space: discrete and multidiscrete
        Reduced : Action space include num_cap*2, 1_reg*33, 1_bat*33 
        The reduced action space allows earlier convergence
    '''
    def __init__(self, CRB_num, RB_act_num):
        self.cap_num, self.reg_num, self.bat_num = CRB_num
        self.reg_act_num, self.bat_act_num = RB_act_num

        if self.bat_act_num < float('inf'):
            cap_actions = [2]* self.cap_num
            reg_actions = [self.reg_act_num]*self.reg_num
            bat_actions = [self.bat_act_num]*self.bat_num
            self.dimension = cap_actions + reg_actions + bat_actions
            self.space = gym.spaces.MultiDiscrete(self.dimension)
            print(f"This is action {self.space}")                
        
        else:
            self.space = gym.spaces.Tuple((\
               gym.spaces.MultiDiscrete([2]*self.cap_num + [self.reg_act_num]*self.reg_num),\
               gym.spaces.Box(low=-1, high=1, shape=(self.bat_num,)) ))
    
    #  TODO: put these functions with wrap_action
    def encode_action(self,multi_discrete_action):
        """
        Encodes a MultiDiscrete action to a single discrete action index.
        """
        index = 0
        base = 1
        for i in reversed(range(len(self.dimension))):
            index += multi_discrete_action[i] * base
            base *= self.dimension[i]
            #print(f"for i {i} the index: {index}, base: {base}")
        return index
    
    #  TODO: put these functions with wrap_action
    def decode_action(self,discrete_action):
        """
        Decodes a single discrete action index to a MultiDiscrete action.
        """
        multi_discrete_action = []
        for dim in reversed(self.dimension):
            multi_discrete_action.append(discrete_action % dim)
            discrete_action //= dim
            #print(f"for dim : {dim}, multi_discrete_action: {multi_discrete_action} is and discrete_action : {discrete_action}")
        return list(reversed(multi_discrete_action))
    
    def seed(self, seed):
        self.space.seed(seed)
     
    def sample(self):
        ss = self.space.sample()
        if self.bat_act_num == np.inf:
            return np.concatenate(ss)
        return ss
        
    def dim(self):
        if self.bat_act_num == np.inf:
            return self.space[0].shape[0] + self.space[1].shape[0]
        return self.space.shape[0]

    def CRB_num(self):
        return self.cap_num, self.reg_num, self.bat_num

    def RB_act_num(self):
        return self.reg_act_num, self.bat_act_num
    
