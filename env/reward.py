class MyReward:
    """Reward definition class
    
    Attributes:
        env (obj): Inherits all attributes of environment 
    """
    def __init__(self, env, info, obs, t, horizon):
        self.env = env
        self.obs = obs
        self.power_w = info['power_w']
        self.cap_w = info['cap_w']
        self.reg_w = info['reg_w']
        self.soc_w = info['soc_w']
        self.dis_w = info['dis_w']
        self.time = t
        self.horizon = horizon

    def powerloss_reward(self):
        # Penalty for power loss of entire system at one time step
        #loss = self.env.circuit.total_loss()[0] # a postivie float
        #gen = self.env.circuit.total_power()[0] # a negative float
        ratio = max(0.0, min(1.0, self.obs['power_loss']))
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
                (0.0 if self.time != self.horizon else self.soc_w * sum(soc_err)) + \
                self.dis_w * sum(discharge_err)
        return -cost

    def voltage_reward(self, record_node = False):
        # Penalty for node voltage being out of [0.95, 1.05] range
        violated_nodes = []
        total_violation = 0
        for name, voltages in self.obs['bus_voltages'].items():
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