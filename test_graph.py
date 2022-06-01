from deepsnap.graph import Graph
from deepsnap.dataset import GraphDataset

from nx_graph import get_nx_Graph
from dataset import Grapholulu

def nx2deepsnap(nxGraph):
    return Graph(nxGraph)

def pyg2deepsnap(PyGdataset):
    return GraphDataset.pyg_to_graphs(PyGdataset)[0]


def show_nx_vs_pyg_graph(path):
    g1=get_nx_Graph(path + 'raw/colours.csv', path + 'raw/annotated_links.csv')
    g2=Grapholulu(path)
    
    print(nx2deepsnap(g1))
    print(pyg2deepsnap(g2))

