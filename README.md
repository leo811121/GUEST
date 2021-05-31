# GUEST
![image](https://github.com/leo811121/GUEST/blob/master/pictures/model.PNG)
- **Intro** Recently, research works of fake news detection are devoted to
studying explainable neural network frameworks to fulfill the task.
However, most frameworks are focus on generating interpretable
evidences from language viewpoint while other information like
user profile or information about news structure and time are able
to offer firmer evidences. Hence, we propose a new framework
Graph of User Encoder and Semantic Attention Network (GUEST)
to explore discover explainable evidences through social media
contexts, user profiles, structure and time information of social media
conversation. Experiments from two public datasets indicates
that GUEST simultaneously outperforms state-of-art models and
provides interpretable evidences from semantics and user characteristics.

More details and reseatch about GUEST can be seen in [here](https://github.com/leo811121/GUEST/blob/master/GUEST.pdf).

# Result of Performance Comparison


| Dataset   |  Metrics  | SVM | CNN | DeClarE | RvNN-BU | RvNN-TD | BaysienDL | AIFN | GUEST |
| ----------|---------- |-----|-----|---------|---------|---------|-----------|------|-------|
| RumorEval | Accuracy  |0.500|0.450|  0.750  |  0.700  |  0.750  |   0.706   | 0.750| **0.900** |
|           | Precision |0.720|0.710|  0.807  |  0.647  |  0.807  |   0.715   | 0.807| **0.930** |
|           |  Recall   |0.580|0.540|  0.791  |  0.750  |  0.791  |   0.734   | 0.791| **0.880** |
|           |    F1     |0.670|0.370|  0.749  |  0.700  |  0.749  |   0.580   | 0.749| **0.890** |
|-----------|-----------|-----|-----|---------|---------|---------|-----------|------|-------|
| Pheme     | Accuracy  |0.834|0.835|  0.870  |  0.785  |  0.806  |   0.863   | 0.763| **0.914** |
|           | Precision |0.837|0.767|  0.809  |  0.647  |  0.753  |   0.890   | 0.656| 0.881 |
|           |  Recall   |0.818|0.814|  0.779  |  0.588  |  0.791  |   0.683   | 0.629| **0.874** |
|           |    F1     |0.825|0.785|  0.793  |  0.672  |  0.749  |   0.726   | 0.683| **0.887** |


# Code explanation  

[model](https://github.com/leo811121/GUEST/tree/master/model) : we introduce our fake news detection model Graph
of User Encoder and Semantic Attention Network (GUEST), which contains four components. Graph Attention
Network (GAT) is for generating sentence representation. Fi-GNN implements self-attention mechanism and GGNN to generate user feature embedding. 
Co-attention fusion enables interaction between each sentence representation and corresponding
user feature embedding. Temporal and structure embedding layer applies biLSTM to capture temporal and structure features of a Twitter conversation.

- **Graph Attention Network (GAT)** -> [guest.py](https://github.com/leo811121/GUEST/blob/master/model/guest.py)
- **Fi-GNN** -> [fi_gnn.py](https://github.com/leo811121/GUEST/blob/master/model/fi_gnn.py)
- **Temporal and structure embedding layer** -> [bilstm.py](https://github.com/leo811121/GUEST/blob/master/model/bilstm.py)

[data](https://github.com/leo811121/GUEST/tree/master/data) : This directory includes RumorEval datset and [data-preprocessing](https://github.com/leo811121/GUEST/blob/master/data/data_preprocessing.py)
file (Pheme dataset is too large to put on Github).

[embeddings](https://github.com/leo811121/GUEST/tree/master/embeddings) : implementing BERT pretrained embedding on claims and comments

[BertDataset.py](https://github.com/leo811121/GUEST/blob/master/BertDataset.py) : It implements custom pytorch dataset to solve data-imbalanced problem in RumorEval during training.

[training.py](https://github.com/leo811121/GUEST/blob/master/training.py) : This file runs model training and evaluation.

# Run the code
```
!python /GUEST/training.py
```


