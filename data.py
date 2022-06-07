import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
import pandas as pd

def get_data(path):
            df= pd.read_csv(path+'annotated_links.csv', names=['source', 'target', 'link']).drop('link', axis=1)
            #df.index+=1
            node_ids = df.index
            link = pd.read_csv(path+'annotated_links.csv', names=['source', 'target', 'link'])#['link']
            link.index+=1

            cl=pd.read_csv(path+'colours.csv')
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
            return data

if __name__ == '__main__':
    
    path = "/GitHub/GraphAesthetics-PreProcessing/investigating-aesthetics_metadata/"
    data = get_data(path)
    transform = RandomLinkSplit(is_undirected=True)
    train_data, val_data, test_data = transform(data)
    print(train_data, val_data, test_data)
    