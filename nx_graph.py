import pandas as pd
import networkx as nx


def get_nx_Graph(node_path, edge_path):

    # load nodes details
    with open(node_path) as f:
        img_nodes = f.read().splitlines() 
    
    # load edges (or links)
    with open(edge_path) as f:
            img_links = f.read().splitlines()  
    len(img_nodes), len(img_links)
    
    node_list_1 = []
    node_list_2 = []
    
    for i in img_links:
      if i.split(',')[2]=='1' or i.split(',')[2]=='0.5':
        node_list_1.append(i.split(',')[0])
        node_list_2.append(i.split(',')[1])
        
    
    img_df = pd.DataFrame({'node_1': node_list_1, 'node_2': node_list_2, 'Link': 1})
    
    G = nx.from_pandas_edgelist(img_df, "node_1", "node_2", create_using=nx.Graph())
    return G
