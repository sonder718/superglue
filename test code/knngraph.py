import torch
from torch_cluster import knn_graph
import networkx as nx
import matplotlib.pyplot as plt
import dgl
x1 = torch.tensor([[0.0, 0.0, 1.0],
                   [1.0, 0.5, 0.5],
                   [0.5, 0.2, 0.2],
                   [0.3, 0.2, 0.4]])
x2 = torch.tensor([[0.0, 1.0, 1.0],
                   [0.3, 0.3, 0.3],
                   [0.4, 0.4, 1.0],
                   [0.3, 0.8, 0.2]])
x = torch.stack([x1, x2], dim=0)
knn_g =knn_graph(x, 2)  # Each node has two predecessors
knn_g.edges()
#######################################################
# x = torch.tensor([[10., 0.], [10., 1.], [10., 2.], [10., 3.],[-10., 0.], [-10., 1.], [-10., 2.], [-10., 3.]])
# # batch = torch.tensor([0, 0, 0, 0])
# edge_index = knn_graph(x, k=3, loop=False)
# print(edge_index)
# edge_index=edge_index.t().numpy()
# print(edge_index)
# #只绘制1号节点的邻接关系
# G1 = nx.Graph()
# # G1.add_edges_from(edge_index[0:3])#3近邻则为3条边
# nx.draw(G1,with_labels=True)
# plt.show()

####################################################
# G = nx.Graph()
# G.add_edges_from(edge_index)

    
# nx.draw(G,with_labels=True)
# # nx.draw(G, pos=nx.spring_layout(G), node_size=300, font_size=10)
# #绘制节点标签
# # nx.draw_networkx_labels(G, pos=nx.spring_layout(G), labels=nx.get_node_attributes(G, 'label'))
# plt.show()