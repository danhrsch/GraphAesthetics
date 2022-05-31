from torch.utils.data.dataset import DFIterDataPipe
import os.path as osp
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset, download_url


class Grapholulu(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['annotated_links.csv', 'colours.csv']

    @property
    def processed_file_names(self):
        return ['Grapholulu.pt']

    def download(self):
        pass

    def process(self):
        idx = 0
        for raw_path in self.raw_paths:
            
            df= pd.read_csv(self.raw_paths[0], names=['source', 'target', 'link']).drop('link', axis=1)
            df.index+=1
            link = pd.read_csv(self.raw_paths[0], names=['source', 'target', 'link'])['link']
            link.index+=1

            cl=pd.read_csv(self.raw_paths[1]).drop('ImgName', axis=1)
            cl.index+=1
            
            node_feats= torch.tensor(cl.to_numpy(), dtype=torch.float)
            
            
            edge_feats = torch.tensor(link.to_numpy(), dtype=torch.float)
            
            edge_index = torch.tensor(df.to_numpy(), dtype=torch.float)
            # Get labels info
            label = 0
            idx += 1
            data = Data(x=node_feats, edge_index=edge_index,  y=edge_feats)
            
            torch.save((data), self.processed_paths[0])

