import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset

class Grapholulu(InMemoryDataset):
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
            node_ids = df.index
            link = pd.read_csv(self.raw_paths[0], names=['source', 'target', 'link'])#['link']
            link.index+=1

            cl=pd.read_csv(self.raw_paths[1])
            cl.index+=1
            di=cl['ImgName'].to_dict()
            node2index = {v: k for k, v in di.items()}
            df=df.replace({"source": node2index}).replace({"target": node2index})
            cl=cl.drop('ImgName', axis=1)
            
            node_feats= torch.tensor(cl.to_numpy(), dtype=torch.float)
            
            
            edge_feats = torch.tensor(link['link'].to_numpy(), dtype=torch.float)
            
            edges = link.loc[link['source'].isin(node_ids)]
            edge_index = torch.tensor(df.to_numpy(), dtype=torch.float).T
            
            

            data = Data(x=node_feats, edge_index=edge_index,  y=edge_feats, node2index=node2index)
            
            torch.save((data), self.processed_paths[0])

