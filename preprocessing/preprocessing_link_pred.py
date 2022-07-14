import numpy as np
import csv
import os, sys
import json
import joblib
import argparse

def load_graph(filenames,labels):
    edges=set()
    nodes=set()

    for filename in filenames:
        print("[LOAD]",filename)
        temp_base, ext = os.path.splitext(filename)
        base, data_type = os.path.splitext(temp_base)
        if data_type==".graph" or ext==".sif":
            for line in open(filename):
                arr=line.strip().split("\t")
                if len(arr)==2:
                    if arr[0]!=arr[1]:
                        edges.add((arr[0],"",arr[1]))
                    else:
                        print("[skip self-loop]",arr[0])
                    nodes.add(arr[0])
                    nodes.add(arr[1])
                elif len(arr)==3:
                    if arr[1] not in labels:
                        labels[arr[1]]=len(labels)
                    if arr[0]!=arr[2]:
                        #edges.add((arr[0],arr[2]))
                        edges.add((arr[0],arr[1],arr[2]))
                    else:
                        print("[skip self-loop]",arr[0])
                    nodes.add(arr[0])
                    nodes.add(arr[2])
                else:
                    print("[ERROR] unknown format")
    return edges,nodes,labels

def sample_neg_list(nodes0,nodes1,train_target_edges,n):
    neg_label_list=[]
    i_list=np.random.choice(nodes0,n)
    j_list=np.random.choice(nodes1,n)
    s=set(train_target_edges)
    for i,j in zip(i_list,j_list):
        if (i,0,j) not in s:
            neg_label_list.append((i,0,j))
    #r_list=np.random.choice(np.arange(2,len(labels)),n)
    #for i,j,r in zip(i_list,j_list,r_list):
    #	if (i,r,j) not in train_target_edges:
    #		neg_label_list.append((i,r,j))
    return neg_label_list

def build_label_list(target_nodes,train_target_edges):
    label_list=[]
    pi=0
    ni=0
    neg_sample=100
    neg_label_list=[None]
    nodes0=[idx for idx,type in enumerate(node_type) if node_type[idx]==0]
    nodes1=[idx for idx,type in enumerate(node_type) if node_type[idx]==1]
    for i in range(len(train_target_edges)):
        if i%len(neg_label_list)==0:
            neg_label_list=sample_neg_list(nodes0,nodes1,train_target_edges,neg_sample)
            ni=0
        if pi==len(train_target_edges):
            np.random.shuffle(train_target_edges)
            pi=0
        pos=train_target_edges[pi]
        neg=neg_label_list[ni]
        label_list.append(pos+neg)
        ni+=1
        pi+=1
    return label_list

def build_adjs(base_edges,self_edges,node_num):
    adj_idx=[]
    adj_val=[]

    edges_forward=set([(e[0],e[2]) for e in base_edges])
    edges_reverse=set([(e[2],e[0]) for e in base_edges])
    edges_self=set([(e[0],e[2])for e in self_edges])
    for e in sorted(edges_forward|edges_reverse|edges_self):
        assert len(e)==2, "length mismatch"
        adj_idx.append([e[0],e[1]])
        adj_val.append(1)
    adjs=[(np.array(adj_idx),np.array(adj_val),np.array((node_num,node_num)))]
    return adjs

def build_multi_adjs(base_edges,self_edges,node_num,labels):
    all_edges=[[] for l in range(len(labels))]
    edges_forward= set([(e[0],e[1],e[2]) for e in base_edges])
    edges_reverse=set([(e[2],e[1],e[0]) for e in base_edges])
    edges_self=set([(e[0],1,e[2])for e in self_edges])
    for e in sorted(edges_forward|edges_reverse|edges_self):
        assert len(e)==3, "length mismatch"
        all_edges[e[1]].append([e[0],e[2]])
    adjs=[]
    for es in all_edges:
        if len(es)>0:
            adj_val=[]
            adj_idx=[]
            for e in sorted(es):
                adj_idx.append([e[0],e[1]])
                adj_val.append(1)
            adj=(np.array(adj_idx),np.array(adj_val),np.array((node_num,node_num)))
            adjs.append(adj)
    return [adjs]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
        nargs='*',
        default=[],
        type=str)
    parser.add_argument('--target',
        nargs='*',
        default=[],
        type=str)
    parser.add_argument('--test_rate', type=float,
        default=0.2,
        help='save jbl file')
    parser.add_argument('--test_label_rate', type=float,
        default=0.1,
        help='save jbl file')
    parser.add_argument('--output_jbl', type=str,
        default="dataset.jbl",
        help='save jbl file [default: dataset.jbl]')
    parser.add_argument('--output_csv', type=str,
        default="dataset_node.csv",
        help='save csv file [default: dataset_node.csv]')
    parser.add_argument('--neg_sample', type=int,
        default=1000,
        help='sample')
    parser.add_argument('--multi_edges',
        action="store_true",
        help='multi-label')

    args=parser.parse_args()
    for t in args.target:
        if  t in args.input:
            args.input.remove(t)
            print("[INFO]",t," is removed from inputs")
    labels={"negative":0,"self":1}
    base_edges,base_nodes,labels=load_graph(args.input,labels)
    target_edges,target_nodes,labels=load_graph(args.target,labels)

    all_edges=base_edges|target_edges
    all_nodes=base_nodes|target_nodes

    print("#non-target edges:",len(base_edges))
    print("#target edges:",len(target_edges))
    print("#all edges:",len(all_edges))
    print("===")
    print("#non-target nodes:",len(base_nodes))
    print("#target nodes:",len(target_nodes))
    print("#all nodes:",len(all_nodes))
    print("===")
    all_nodes_list=sorted(list(all_nodes))
    node_num=len(all_nodes_list)
    all_nodes_mapping={el:i for i,el in enumerate(all_nodes_list)}
    #node_num=len(all_nodes)

    base_edges=[(all_nodes_mapping[e[0]],labels[e[1]],all_nodes_mapping[e[2]]) for e in base_edges]
    target_edges=[(all_nodes_mapping[e[0]],labels[e[1]],all_nodes_mapping[e[2]]) for e in target_edges]
    all_edges=[(all_nodes_mapping[e[0]],labels[e[1]],all_nodes_mapping[e[2]]) for e in all_edges]

    self_edges=[(i,labels["self"],i) for i in range(len(all_nodes))]
    base_nodes=[all_nodes_mapping[e] for e in base_nodes]
    target_nodes=[all_nodes_mapping[e] for e in target_nodes]

    np.random.shuffle(target_edges)
    ##
    ## target/base/self => train/label
    ##
    """
    test_num=int(target_edge_num*args.test_rate)
    train_edges=target_edges[:target_edge_num-test_num]+base_edges+self_edges
    label_edges=target_edges[target_edge_num-test_num:]
    """
    target_edge_num=len(target_edges)
    test_num=int(target_edge_num*args.test_rate)
    train_target_edges=target_edges[:target_edge_num-test_num]
    test_target_edges=target_edges[target_edge_num-test_num:]

    print("#train target edges:",len(train_target_edges))
    print("#test target edges:",len(test_target_edges))
    
    target_type = sorted(labels.items(), key = lambda x: x[1], reverse=True)[0][0]
    if (target_type == 'drug_gene'):
        node_type=[0 if node.startswith('DB') else 1 if node.isnumeric() else -1 for node in all_nodes_list]
    elif (target_type == 'drug_disease'):
        node_type=[0 if node.startswith('DB') else 1 if node.startswith('C') else -1 for node in all_nodes_list]
    elif (target_type == 'disease_gene'):
        node_type=[0 if node.startswith('C') else 1 if node.isnumeric() else -1 for node in all_nodes_list]

    label_list=build_label_list(target_type,train_target_edges)
    test_label_list=build_label_list(target_type,test_target_edges)
    if args.multi_edges:
        adjs=build_multi_adjs(base_edges,self_edges,node_num,labels)
    else:
        adjs=build_adjs(base_edges,self_edges,node_num)
    node_ids=np.array([list(range(node_num))])
    graph_names=["one"]
    max_node_num = node_num

    obj={
        "adj":adjs,
        "node":np.expand_dims(np.stack([list(range(len(all_nodes_list))),node_type]).T,0),
        "node_num":max_node_num,
        "label_list":np.array([label_list]),
        "test_label_list":np.array([test_label_list]),
        "max_node_num":max_node_num}

    print(obj)
    print("[SAVE]",args.output_jbl)
    joblib.dump(obj, args.output_jbl)
    fp=open(args.output_csv,"w")
    print("[SAVE]",args.output_csv)
    for node in all_nodes_list:
        fp.write(node)
        fp.write("\n")



