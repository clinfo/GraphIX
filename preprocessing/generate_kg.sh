#!/bin/sh
python preprocessing/get_max_network.py --adjs ./data/disease_disease.treenum.graph.tsv ./data/disease_gene.leaf.graph.tsv ./data/drug_gene.graph.tsv ./data/network.graph.tsv --target ./data/drug_disease.treenum.graph.tsv

python preprocessing/preprocessing_link_pred.py --input ./data/adjs.graph.tsv --target ./data/targets.graph.tsv --multi_edges --test_rate 0
