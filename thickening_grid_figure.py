from glineage import *
from skel import *
from kron import *

from sklearn.datasets import load_digits

from torch.nn import Linear
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import torch
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from skimage.transform import resize

from time import time

from visualize_graph_lineage import *

def thickening_op(gl, pl):
    new_gl = []

def torch_glin_to_scipy_glin(t_gl, t_pl):
    s_gl = [torch_sparse_to_scipy_sparse(gmat) for gmat in t_gl]
    s_pl = [torch_sparse_to_scipy_sparse(pmat) for pmat in t_pl]
    return s_gl, s_pl

def scipy_glin_to_torch_glin(s_gl, s_pl):
    t_gl = [scipy_spr_to_torch_sparse(gmat) for gmat in s_gl]
    t_pl = [scipy_spr_to_torch_sparse(pmat) for pmat in s_pl]
    return t_gl, t_pl



if __name__ == '__main__':
    K = 2

    gsizes = [1,2,4,8]

    test = GridGraphLineage([(gsizes[i],gsizes[i]) for i in range(len(gsizes))])
    for i in range(len(gsizes)):
        if i == 0:
            test.setX(torch.tensor([[0.0,0.0]]), i)
        else:
            t = (np.mgrid[:2**i, :2**i]) / 2**i
            X = t[0].reshape(2**i * 2**i, )
            Y = t[1].reshape(2**i * 2**i, )
            coords = np.stack([X,Y]).T
            coords -= .5
            test.setX(torch.tensor(coords).float(),i)
    tograph = test

    #tograph = tograph.dilated_skelprod(tograph, test, 1.0, 1.0)
    #quit()

    #for item in tograph.gl:
    #    plt.matshow(item.detach().cpu().to_dense().numpy())
    #    plt.show()
    #quit()
    for i in range(K):
        dim=tograph.getX(0).shape[1]
        #print(dim)
        pos = [convert_level_points(tograph.getX(i).numpy(), dim=2) for i in range(len(tograph.gl))]
        #pos = [np.dot(item, np.array([[np.cos(3.0), np.sin(3.0)],[-np.sin(3.0), np.cos(3.0)]])) for item in pos]
        pos = hierarchical_layout(tograph, pos)
        final_pos = pos
        SEP=3
        spacing=0.2
        for i in range(1, len(final_pos)):
            final_pos[i][:, 0] += SEP + final_pos[i-1][:, 0].max() + abs(final_pos[i-1][:, 0].max())*spacing
        print(len(pos),[item.shape for item in pos])
        #print(len(tograph.gl),[item.shape for item in tograph.gl])
        #for j in range(len(gsizes)):
        #    #print(torch.tensor(pos[j]).float().shape)
        #    tograph.setX( j/len(gsizes)+torch.cat([torch.tensor(pos[j]).float(), -j/len(gsizes) + torch.linspace(-1,1,tograph.xl[j].shape[0]).reshape(-1,1)],1),j)
        #    #tograph.setX( j + torch.cat([torch.tensor(pos[j]).float(),(j/len(gsizes))*torch.ones((tograph.xl[j].shape[0],1))],1),j)
        for j in range(len(gsizes)):
            tograph.setX(torch.tensor(final_pos[j]).float(),j)
        tograph = tograph.thicken()
        #max = torch.zeros((1,3))
        #for j in range(len(gsizes)):
        #    tograph.setX(
        #        max + j*2*torch.eye(3)[0] + torch.cat(
        #            [torch.tensor(np.concatenate(final_pos[:j+1],0)).float(),
        #             torch.linspace(-2,2,tograph.xl[j].shape[0]).reshape(-1,1)],
        #            1
        #        ),
        #        j
        #    )
        #    max = tograph.xl[j].max(0)[0] + 100000
    #print(pos)
    #quit()

    pos = [convert_level_points(tograph.getX(i).numpy(), dim=2) for i in range(len(tograph.gl))]
    pos = hierarchical_layout(tograph, pos)

    test_colors = np.linspace(0, 1, len(gsizes))
    for i in range(len(gsizes)):
        test.setX(torch.ones(gsizes[i]**2).reshape(-1,1).double(),i)

    tograph2 = test
    for i in range(K):
        tograph2 = tograph2.thicken()
        for j in range(len(tograph2.xl)):
            #print(tograph2.xl[j].shape, torch.ones((tograph2.xl[j].shape[0],1)).shape )
            tograph2.setX(

                torch.cat([tograph2.xl[j],test_colors[j]*torch.ones(tograph2.xl[j].shape[0]).reshape(-1,1).double()],1),
                j
            )

    for i in range(len(tograph2.xl)):
        tograph2.xl[i] = tograph2.xl[i] / tograph2.xl[i].max(0, keepdim=True)[0]

    #print(tograph2.xl[-1])

    colors = combine_colors(tograph2, plt.rcParams['axes.prop_cycle'].by_key()['color'][:K+1])
    colors = [item.numpy() for item in colors]

    #for i in range(len(colors)):
    #    print(colors[i])#(colors[-1].shape, tograph.gl[-1].shape)

    plt.rcParams["figure.figsize"] = (10,5)

    fig, ax = glin_visualizer(tograph, pos = pos, node_colors = colors)

    ax.axis('off')

    fig.tight_layout()
    fig.savefig("thickened_grids_v2_2x.png", transparent=True, dpi=300, pad_inches=0)
