import argparse
import networkx as nx
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--adjs',
                    nargs='*',
                    default=[],
                    type=str)
parser.add_argument('--target',
                    default=[],
                    type=str)
args=parser.parse_args()

edges=pd.DataFrame()
for filename in args.adjs:
    edge=pd.read_csv(filename, sep="\t", names=['source', 'edgetype', 'target'])
    edges=pd.concat([edges, edge])
G = nx.from_pandas_edgelist(edges, source='source', target='target', edge_attr=True, create_using=nx.DiGraph())

num_edge_maxnetwork=0
for c in nx.weakly_connected_components(G):
    subgraph = G.subgraph(c)
    if(num_edge_maxnetwork < len(subgraph.edges)):
        maxnetwork=subgraph
        num_edge_maxnetwork = len(subgraph.edges)
adjs=nx.to_pandas_edgelist(maxnetwork, source='source', target='target')
adjs[['source', 'edgetype', 'target']].to_csv('./data/adjs.graph.tsv', index=False, header=False, sep='\t')

nodes_adj=set(list(adjs['target'])+list(adjs['source']))
target=pd.read_csv(args.target, sep="\t", names=['source', 'edgetype', 'target'])
target_on_adj=target[target['source'].isin(nodes_adj) & target['target'].isin(nodes_adj)]
target_on_adj[['source', 'edgetype', 'target']].to_csv('./data/targets.graph.tsv', index=False, header=False, sep='\t')
