import numpy as np
from torch_geometric.data import Data
import torch as th
import networkx as nx
from core_open_dss.circuit import Circuits
from core_open_dss.loadprofile import LoadProfile


class Wrap_obs:
    def __init__(self, obs, circuits) -> None:
        self.obs = obs
        self.circuit = circuits
        pass

    def wrapflatdata(self):
        """ 
        Wrap the observation dictionary (i.e., self.obs) to a numpy array
        
        Attribute:
            obs: the observation distionary generated at self.reset() and self.step()
        
        Return:
            a numpy array of observation.
        
        """
        key_obs = ['bus_voltages', 'cap_statuses', 'reg_statuses', 'bat_statuses']
        if self.observe_load: key_obs.append('load_profile_t')

        mod_obs = []
        # voltages = obs['bus_voltages'].values()
        # for v in voltages:
        #     mod_obs = mod_obs + v
        # return np.hstack(mod_obs)

        for var_dict in key_obs:
            # node voltage is a dict of dict, we only take minimum phase node voltage
            if var_dict in ['bus_voltages']:
                mod_obs = mod_obs + list(self.obs[var_dict].values())
            # elif var_dict == 'power_loss':
            #     mod_obs.append(obs['power_loss'])
        return np.hstack(mod_obs)
    
    def wrapgraphdata(self):
        """ 
        Wrap the observation dictionary (i.e., self.obs) to a gym graph space
        and add the node features.
        Currently, the node features are: bus_voltages (3), cap_status(1), VR_status(3), battery_status(2).
        So, each node will have 9 features.
        Attribute:
            obs: the observation distionary generated at self.reset() and self.step()
        
        Return:
            a gym graph space of observation
        
        """
        #__________________building nodes and features___________________________#

        bus_voltages = list(self.obs['bus_voltages'].values())
        #print(f"bus_voltages: {bus_voltages}")

        # Find the maximum length of the voltage sequences
        max_length = max(len(v) for v in bus_voltages)

        # Pad shorter sequences with zeros to make all sequences the same length
        padded_voltages = [v + [0.0] * (max_length - len(v)) for v in bus_voltages]
        #print(f"padded:{padded_voltages}")

        # Convert the list of padded voltage lists to a PyTorch tensor
        node_features = th.tensor(padded_voltages, dtype = th.float32)
        #print(f"node_features:{node_features}")

        #__________________building edges___________________________#

        all_buses = list(self.obs['bus_voltages'].keys())
        #print(f"Unique bus names: {all_buses}")

        #index to the buses
        bus_to_index = {bus: idx for idx, bus in enumerate(all_buses)}
        #print(f"indexing: {bus_to_index}")

        topology, edges, lines, transformer = self.build_edges()

        #create mapping
        edge = []
        for (bus1, bus2) in edges:
            edge.append((bus_to_index[bus1], bus_to_index[bus2]))
        #print(f"mapping: {edge}")

        #edge_index = th.tensor(edge, dtype=th.long).t().contiguous()
        #print(f"edge converted: {edge_index}")

        edge_index = th.tensor(edge, dtype=th.long).t()

        #convert into PyG data
        data = Data(x=node_features, edge_index=edge_index)
        #print(f"Data: {data}")

        edge_index = self._process_edge_index(edge_index)

        observation = {"node_features": node_features, "edge_index" : edge_index}

        print("Node features shape:", node_features.shape)
        print("Edge index shape:", edge_index.shape)
        print("Node features dtype:", node_features.dtype)
        print("Edge index dtype:", edge_index.dtype)
              
        return observation
    
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

        self.adj_mat = nx.adjacency_matrix(self.circuit.topology)
        return self.circuit.topology, self.edges, self.lines, self.transformers
    
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

        