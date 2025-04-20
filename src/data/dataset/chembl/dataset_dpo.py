import pickle
from tqdm import tqdm
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
        

class DPODataset(Dataset):
    def __init__(self, dataset, affinity_info, pair_dict):
        self.dataset = dataset
        
        with open(affinity_info, 'rb') as f:
            self.affinity_info = pickle.load(f)

        with open(pair_dict, 'rb') as f:
            self.pair_dict = pickle.load(f)
        
        self.pair_pool = [item for v in self.pair_dict.values() for item in v]
            
        self.length = len(self.pair_pool)
        
        print("Convert uuid to index for dataset...")
        self.uuid_to_idx = {item['uuid']: idx for idx, item in tqdm(enumerate(self.dataset), total=len(self.dataset))}


    def __len__(self):
        return self.length
  
    def __getitem__(self, idx):
        
        winner_identifier = self.pair_pool[idx][0]
        loser_identifer = self.pair_pool[idx][1]
        
        winner_idx = self.uuid_to_idx[winner_identifier]
        loser_idx = self.uuid_to_idx[loser_identifer]
        
        winner_sample = self.dataset[winner_idx]
        loser_sample = self.dataset[loser_idx]
        
        assert winner_sample['uuid'] == winner_identifier, f"Expected winner {winner_identifier}, but got {winner_sample['uuid']}"
        assert loser_sample['uuid'] == loser_identifer, f"Expected loser {loser_identifer}, but got {loser_sample['uuid']}"
        
        winner_affinity = self.affinity_info[winner_identifier]['pAffinity']
        loser_affinity = self.affinity_info[loser_identifer]['pAffinity']
        
        assert winner_affinity > loser_affinity, f"Winner affinity {winner_affinity} should be greater than loser affinity {loser_affinity}"
        
        paired_data = HeteroData()

        # Add winner sample data
        for node_type in winner_sample.node_types:
            for key in winner_sample[node_type]:
                paired_data[f"w_{node_type}"][key] = winner_sample[node_type][key]
                
        paired_data["w_uuid"] = winner_sample["uuid"]
        paired_data["w_ligand", "to", "w_ligand"].edge_index = winner_sample["ligand", "ligand"].edge_index
        paired_data["w_affinity"] = winner_affinity
        
        # Add loser sample data
        for node_type in loser_sample.node_types:
            for key in loser_sample[node_type]:
                paired_data[f"l_{node_type}"][key] = loser_sample[node_type][key]

        paired_data["l_uuid"] = loser_sample["uuid"]
        paired_data["l_ligand", "to", "l_ligand"].edge_index = loser_sample["ligand", "ligand"].edge_index
        paired_data["l_affinity"] = loser_affinity

        return paired_data