# XaiDR: XAI model for Drug Repoisiotning
# Overview
This repository is the python implementation of paper 'XaiDR: A graph-based drug repositioning prediction with underlying proteins interpretable'.<br>
XaiDR is interpretable supervised learning framework for drug repositioning. It can present important genes that have high contribution to disease-drug association prediction.

# Requirements
XaiDR is testet to work with python 3.6. The required dependencies are:<br>
```
networkx
numpy
pandas
tensorflow>=1.12
joblib
scipy
scikit-learn>=0.21
```
# Usage
In the top directory, run the code as following:<br>
## Generate a knowledge graph
```
sh preprocessing/generate_kg.sh
```
## Perform 5-fold cross validation
python gcn.py train_cv --config config.json
## Calculate contribution to the novel edge
As an example, calculate the contribution of surrounding 1-hop nodes to Intestinal neplasms(node number: 15958)-Salicylic acid(node number: 20380) novel edge mentioned in the paper.<br>
```
python gcn.py --config config.json visualize --visualize_type edge_score --visualize_node0 20380 --visualize_node1 15958 --graph_distance 1
```