from glineage import *#GridGraphLineage, ParametrizedGraphLineage, CompleteGraphLineage, PathGraphLineage, CycleGraphLineage
from skel import *
from kron import *

from sklearn.datasets import load_digits

from torch.nn import Linear
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import torch
from torch.nn.utils.parametrizations import orthogonal
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_col
import networkx as nx

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from skimage.transform import resize

from time import time

def combine_colors(glin, color_list):
    raws = [glin.getX(i) for i in range(len(glin.xl))]

    #print([item.shape for item in raws])
    raws = [item + 1e-3 for item in raws]
    sums = [(.3 + .3*torch.clamp(item.sum(1,keepdim=True), 0, 1)) for item in raws]
    #print(sums)
    raws = [item / item.sum(1,keepdim=True) for item in raws]

    color_mat = torch.tensor(np.stack([mpl_col.to_rgb(item) for item in color_list]))

    result = [torch.matmul(item, color_mat) for item in raws]
    result = [torch.cat([rgb, a], 1) for rgb, a in zip(result, sums)]
    return result

def glin_visualizer(glin,
                    pos=None,
                    add_self_loops=True,
                    plot_dim = 2,
                    draw_self_loops=False,
                    node_colors='black',
                    SEP=5,
                    same_edge_color = (0.0, 0.0, 0.0, 1.0),
                    delta_edge_color = (0.0, 0.0, 0.0, 1.0)
    ):
    glist = [item.cpu().detach().to_dense().numpy() for item in glin.gl]
    if add_self_loops:
        glist = [item + np.eye(item.shape[0]) for item in glist]

    slist = [item.cpu().detach().to_dense().numpy() for item in glin.pl]
    k = len(glist)
    sizes = [item.shape[0] for item in glist]
    N = sum(sizes)
    blocks = [[np.zeros((sizes[i], sizes[j])) for j in range(k)] for i in range(k)]
    for i in range(k):
        blocks[i][i] = glist[i]
    for i in range(k-1):
        blocks[i][i+1] = slist[i].T
        blocks[i+1][i] = slist[i]

    print([[item.shape for item in sublist] for sublist in blocks])

    entire = np.block(blocks)
    entire_graph = nx.from_numpy_array(entire)

    graphs = [nx.from_numpy_array(item) for item in glist]
    if pos is None:
        embeds = [nx.spring_layout(gr, dim=plot_dim) for gr in graphs]
        final_pos = [np.stack(list(item.values())) for item in embeds]
    else:
        final_pos = pos
    labels = [i for i in range(k) for j in range(len(graphs[i].nodes())) ]
    #print(labels)
    #print(final_pos)

    spacing = .01
    #final_pos[0][:,0] += 1.0
    for i in range(1, len(final_pos)):
        final_pos[i][:, 0] += SEP + final_pos[i-1][:, 0].max() + abs(final_pos[i-1][:, 0].max())*spacing

    #final_pos[0][:,0] -= 2.0

    all_node_pos_array = np.concatenate(final_pos)
    all_node_pos = {i:all_node_pos_array[i] for i in entire_graph.nodes()}
    #all_node_pos = nx.spring_layout(entire_graph)
    #print(entire_graph.nodes())
    if node_colors != 'black':
        node_colors = [tuple(item) for item in np.concatenate(node_colors)]

    if not draw_self_loops:
        entire_graph.remove_edges_from(nx.selfloop_edges(entire_graph))
    same_L_edges = [(u,v) for (u,v) in entire_graph.edges() if labels[u] == labels[v]]
    delta_L_edges = [(u,v) for (u,v) in entire_graph.edges() if abs(labels[u] - labels[v])==1]

    #print(len(same_L_edges),len(delta_L_edges))
    #quit()

    fig,ax = plt.subplots()

    NODE_SIZE=150.0
    nx.draw_networkx_nodes(entire_graph, pos=all_node_pos, node_color=node_colors, edgecolors='black', linewidths=.5, node_size=NODE_SIZE, ax=ax)
    nx.draw_networkx_edges(entire_graph, edgelist = same_L_edges, pos=all_node_pos, width=1.25, ax=ax, edge_color=same_edge_color)
    nx.draw_networkx_edges(entire_graph, edgelist = delta_L_edges, style=":", pos=all_node_pos, width=1.25, ax=ax, edge_color=delta_edge_color)
    #nx.draw_networkx_nodes(entire_graph, pos=all_node_pos, node_color='black', node_size=.25*NODE_SIZE)

    if all_node_pos_array.std(0)[1] < 0.1:
        ax.set_ylim([-1., 1.])
    return fig,ax

def convert_level_points(X, dim=2):
    #return X
    if X.shape[0] < dim:
        return np.zeros((1,dim))
    scl = StandardScaler()
    pca = PCA(dim)
    return pca.fit_transform(scl.fit_transform(X))

def hierarchical_layout(glin, poslist):
    pmats = glin.pl
    pmats = [item.to_dense().T for item in pmats]
    pmats = [item / item.sum(1, keepdim=True) for item in pmats]
    pmats.reverse()
    new_pos = [torch.tensor(poslist[-1]).float()]
    for i in range(len(pmats)):
        pm = pmats.pop(0)
        new_pos.insert(0, torch.matmul(pm.float(), new_pos[0]))
    return [item.detach().numpy() for item in new_pos]
