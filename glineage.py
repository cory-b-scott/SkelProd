import torch
import networkx as nx
import numpy as np

from math import floor,ceil

from skel import *
from kron import *

class GraphLineage():

    def __init__(self, glist, plist):
        self.gl = glist
        self.xl = [None for i in range(len(self.gl))]
        self.yl = [None for i in range(len(self.gl))]
        self.pl = plist

    def kron_multi(self, gp, pp, hp, sp, prod=torch_skeletal_cross_product_blocks):
        #print([item[-1].shape for item in [gp, pp, xp, hp, sp, yp]])
        skel_blocks = prod(gp, hp[::-1], pp, sp[::-1])

        """for item in skel_blocks:
            for subitem in item:
                print(subitem.shape, end= " ")
            print()
        """
        skprod = torch_block_matrix_with_padding(skel_blocks)

        #xprod = [torch_sparse_kron(torch.cat(xp[i],0).coalesce(),
        #                           torch.cat(yp[::-1][i],0).coalesce()).coalesce() for i in range(len(xp))]
        #xprod /= xprod.values().max()
        #print(xprod.shape)

        spblcks = []

        #print(len(pp))

        for i in range(len(pp)+1):
            row = []
            for j in range(len(pp)):
                #row.append(0)
                """if i == j:
                    row.append(
                        torch_sparse_kron(torch_sparse_eye(gp[j].shape[-1],gp[j].shape[-1]), sp[::-1][j]).coalesce()
                    )
                elif i-j == 1:
                    row.append(
                        torch_sparse_kron(pp[j], torch_sparse_eye(hp[::-1][j].shape[-1],hp[::-1][j].shape[-1])).coalesce()
                    )
                else:"""
                if True:
                    row.append(
                        torch.zeros(
                            (
                                0,#gp[i].shape[0]*hp[::-1][i].shape[0],
                                0#gp[j].shape[0]*hp[-(j-1)].shape[0]
                            )).to_sparse().coalesce()
                    )
                #print(gp[i].shape[0]*hp[::-1][i].shape[0], gp[j].shape[0]*hp[-(j-1)].shape[0],row[-1].shape, end = " ")
            #print()

            spblcks.append(row)
        #print(len(spblcks), len(spblcks[0]))
        for i in range(len(pp)):
            spblcks[i][i] = torch_sparse_kron(
                torch_sparse_eye(gp[i].shape[-1],gp[i].shape[-1]),
                sp[::-1][i]
            )
            spblcks[i+1][i] = torch_sparse_kron(
                pp[i],
                torch_sparse_eye(hp[:-1][::-1][i].shape[-1],hp[:-1][::-1][i].shape[-1])
            )


        """for item in spblcks:
            for subitem in item:
                print(subitem.shape, end= " ")
            print()
        """
        if len(spblcks) > 1:
            spprod = torch_block_matrix_with_padding(spblcks).coalesce()
            #print(spprod.shape)
        else:
            spprod = torch.zeros((1,1)).to_sparse().coalesce()
        return skprod, spprod

    def __pow__(self, exp):
        if exp == 0:
            return self
        elif exp == 1:
            return self
        else:
            return self * (self ** (exp - 1))



    def __mul__(self, other):
        return self.skelprod(other, prod=torch_skeletal_cross_product_blocks)

    def __add__(self, other):
        return self.skelprod(other, prod=torch_skeletal_box_product_blocks)

    def __mod__(self, other):
        return self.skelprod(other, prod=torch_skeletal_box_cross_product_blocks)

    def dilated_skelprod(self, other, rho1, rho2, prod=torch_skeletal_cross_product_blocks):
        gl, pl = self.gl, self.pl
        gl = [item + torch_sparse_eye(*item.shape).to(item.device) for item in gl]
        hl, sl = other.gl, other.pl
        hl = [item + torch_sparse_eye(*item.shape).to(item.device) for item in hl]

        hlp = hl[::-1]
        slp = sl[::-1]

        new_gl = []
        new_pl = []
        new_xl = []

        i = 0

        last_k1 = None
        last_k2 = None

        inc1 = 0
        inc2 = 0
        while i < min(len(sl), len(pl))-1:
            gl_small = []
            pl_small = []
            hl_small = []
            sl_small = []

            xl_small = []
            yl_small = []


            k1_list = np.arange(0, i+rho1, rho1)
            k2_list = np.arange(0, i+rho2, rho2)
            lv = min(len(k1_list), len(k2_list))

            k1_list = np.floor(np.linspace(0, i+rho1, lv+1)[:-1]).astype('int')
            k2_list = np.ceil(np.linspace(0, i+rho2, lv+1)[:-1]).astype('int')

            #print(rho1*np.linspace(0,(i)/rho1,lv), np.floor(rho1*np.linspace(0,(i)/rho1,lv)), k1_list, k2_list)

            #print(k1_list, k2_list, k1_list + k2_list[::-1])

            sel = np.where(np.abs(k1_list + k2_list[::-1] - i) < 1)
            k1_list = k1_list[sel]
            k2_list = k2_list[sel]
            print(">>>")
            print(last_k1, last_k2)
            print(k1_list, k2_list)

            gl_small = [gl[s] for s in k1_list]
            pl_small = [pl[s] for s in k1_list]

            for s in range(len(gl_small)-1):
                if gl_small[s].shape == gl_small[s+1].shape:
                    pl_small[s] = torch_sparse_eye(*gl_small[s+1].shape)

            hl_small = [hl[s] for s in k2_list]
            sl_small = [sl[s] for s in k2_list]

            for s in range(len(hl_small)-1):
                if hl_small[s].shape == hl_small[s+1].shape:
                    sl_small[s] = torch_sparse_eye(*hl_small[s+1].shape)

            if other.xl[0] is not None and self.xl[0] is not None:
                xl_small = [self.xl[k1] for k1 in k1_list]
                yl_small = [other.xl[k2] for k2 in k2_list[::-1]]

                #yl_small.reverse()

            gg, pp = self.kron_multi(gl_small, pl_small[:-1], hl_small, sl_small[:-1], prod=prod)

            #hl_small.reverse()
            #sl_small.reverse()

            if last_k1 is not None:
                spblocks = [[None for l in range(len(last_k1))] for m in range(len(k1_list))]
                for m in range(len(k1_list)):
                    for l in range(len(last_k1)):
                        #print(last_k1[l], k1_list[m])
                        if (k1_list[m] - last_k1[l]) + (k2_list[::-1][m] - last_k2[::-1][l]) <= 1:
                            if k1_list[m] - last_k1[l] == 1:
                                arga = pl[last_k1[l]]
                            elif k1_list[m] == last_k1[l]:
                                arga = torch_sparse_eye(*gl[last_k1[l]].shape)
                            else:
                                arga = torch.zeros(
                                    gl[k1_list[m]].shape[0],
                                    gl[last_k1[l]].shape[0]
                                ).to_sparse()

                            if k2_list[::-1][m] - last_k2[::-1][l] == 1:
                                argb = sl[last_k2[::-1][l]]
                            elif k2_list[::-1][m] == last_k2[::-1][l]:
                                argb = torch_sparse_eye(*hl[last_k2[::-1][l]].shape)
                            else:
                                argb = torch.zeros(
                                    hl[k2_list[::-1][m]].shape[0],
                                    hl[last_k2[::-1][l]].shape[0]
                                ).to_sparse()
                        else:
                            arga = torch.zeros(
                                gl[k1_list[m]].shape[0],
                                gl[last_k1[l]].shape[0]
                            ).to_sparse()
                            argb = torch.zeros(
                                hl[k2_list[::-1][m]].shape[0],
                                hl[last_k2[::-1][l]].shape[0]
                            ).to_sparse()
                        spblocks[m][l] = torch_sparse_kron(arga, argb)
                        #spblocks[m][l] = (arga, argb)

                print(gg.shape)
                #print("@@@@@")
                #for sublist in spblocks:
                    #print([item for item in sublist])
                #    print([item.shape for item in sublist])
                    #print([(item[0].shape,item[1].shape) for item in sublist])

                pp = torch_block_matrix_with_padding(spblocks).coalesce()

            else:
                pp = torch_sparse_eye(1,1)

            print(pp.shape, gg.shape)

            new_gl.append(gg)
            new_pl.append(pp)

            if len(xl_small) > 0:
                newxs = []
                #yl_small = [item.contiguous() for item in yl_small]
                for k in range(len(xl_small)):

                    print(xl_small[k].shape, "<->", yl_small[k].shape)
                    xa = torch.kron(
                        xl_small[k],
                        torch.ones( (yl_small[k].shape[0], 1) )
                    )
                    xb = torch.kron(
                        torch.ones( (xl_small[k].shape[0], 1) ) ,
                        yl_small[k]
                    )
                    print(k, xa.shape, xb.shape)
                    newxs.append(torch.cat([xa, xb], -1))
                new_xl.append(torch.cat(newxs,0))
                print(new_xl[-1].shape)

            i+=min(rho1,rho2)
            last_k1 = k1_list
            last_k2 = k2_list

        print("###",[item.shape for item in new_pl])
        print("###",[item.shape for item in new_gl])
        print("###",[item.shape for item in new_xl])

        newgl = GraphLineage(new_gl, new_pl[1:])

        if len(new_xl) > 0:
            for i in range(len(newgl.gl)):
                newgl.setX(new_xl[i], i)
        return newgl

    def skelprod(self, other, prod=torch_skeletal_cross_product_blocks):
        gl, pl = self.gl, self.pl
        gl = [item + torch_sparse_eye(*item.shape).to(item.device) for item in gl]
        hl, sl = other.gl, other.pl
        hl = [item + torch_sparse_eye(*item.shape).to(item.device) for item in hl]

        new_gl = []
        new_pl = []

        for i in range(1,len(gl)+1):
            gl_small = gl[:i]
            pl_small = pl[:i-1]

            hl_small = hl[:i]
            sl_small = sl[:i-1]

            gg, pp = self.kron_multi(gl_small, pl_small, hl_small, sl_small, prod=prod)
            new_gl.append(gg)
            new_pl.append(pp)

        newgl = GraphLineage(new_gl, new_pl[1:])
        if self.xl[0] is not None and other.xl[0] is not None:
            for i in range(len(newgl.gl)):
                #print(newgl.gl[i].shape)
                newxs = []
                for j in range(i+1):
                    xa = torch.kron(
                        self.xl[j],
                        torch.ones( (other.xl[i-j].shape[0], 1) )
                    )
                    xb = torch.kron(
                        torch.ones( (self.xl[j].shape[0], 1) ) ,
                        other.xl[i-j]
                    )

                    newxs.append(torch.cat([xa, xb], -1))
                newgl.setX(torch.cat(newxs,0), i)
            #x = torch.cat()
            #print([item.shape for item in newgl.gl])
        return newgl

    def get_graph(self, n=-1):
        return self.gl[n]

    def __str__(self):
        return "Graph Lineage with:\n\t-gmats "+\
               str([item.shape for item in self.gl]) + "\n\t-pmats " + \
               str([item.shape for item in self.pl]) + "\n\t-xmats " + \
               str([(item.shape if item is not None else None) for item in self.xl]) + "\n\t-ymats " + \
               str([(item.shape if item is not None else None) for item in self.yl])


    def setX(self, X, n):
        self.xl[n] = X

    def getX(self,n=-1):
        return self.xl[n]

    def setY(self, Y, n):
        self.yl[n] = Y

    def getY(self,n=-1):
        return self.yl[n]

    def save_to_npz(self, filename):

        Xarrays = {"x_%d"%i:self.getX(i) for i in range(len(self.xl))}
        Yarrays = {"x_%d"%i:self.getY(i) for i in range(len(self.yl))}
        Garrays = {"g_%d"%i:self.gl[i].to_dense().numpy() for i in range(len(self.gl))}
        Parrays = {"p_%d"%i:self.pl[i].to_dense().numpy() for i in range(len(self.pl))}

        Garrays.update(Xarrays)
        Garrays.update(Yarrays)
        Garrays.update(Parrays)

        np.savez_compressed(filename, **Garrays)

    def as_graded_graph(self, returnX=False, blocks=True):
        gsizes = [item.shape[0] for item in self.gl]

        Gblocks = [[torch_empty_sparse(gsizes[j], gsizes[i]) for i in range(len(gsizes))] for j in range(len(gsizes))]

        for i in range(len(self.gl)):
            Gblocks[i][i] = self.gl[i]
        for i in range(len(self.gl)-1):
            Gblocks[i][i+1] = self.pl[i].T.coalesce()
            Gblocks[i+1][i] = self.pl[i].coalesce()

        if blocks:
            G = Gblocks
        else:
            G = torch_block_matrix_with_padding(Gblocks)

        if returnX:
            return G, torch.cat(self.xl)
        else:
            return G

    def reverse(self):
        self.gl.reverse()
        self.xl.reverse()
        self.pl = [item.T.coalesce() for item in self.pl]
        self.pl.reverse()

    def thicken(self):
        new_gl = []
        new_pl = []
        new_xl = []

        gsizes = [item.shape for item in self.gl]

        for L in range(1,len(self.gl)+1):
            gblocks = [[
                torch.sparse_coo_tensor([[],[]], [], [
                    gsizes[i][0],
                    gsizes[j][1]
                ],device=self.gl[0].device).coalesce()
              for j in range(L)
            ] for i in range(L)]
            #print(gblocks)
            for i in range(L):
                gblocks[i][i] = self.gl[i]
            for i in range(L-1):
                gblocks[i][i+1] = self.pl[i].T.coalesce()
                gblocks[i+1][i] = self.pl[i]
            gg = torch_block_matrix_with_padding(gblocks)
            new_gl.append(gg)

        for L in range(1,len(self.gl)):
            pblocks = [[
                torch.sparse_coo_tensor([[],[]], [], [
                    gsizes[i][0],
                    gsizes[j][1]
                ],device=self.gl[0].device).coalesce()
              for j in range(L)
            ] for i in range(L+1)]

            for i in range(L):
                pblocks[i][i] = torch_sparse_eye(*self.gl[i].shape)

            for i in range(L):
                pblocks[i+1][i] = self.pl[i]#.T.coalesce()

            #print("^^^", [(self.gl[i].shape) for i in range(L)])
            #pblocks = [torch_sparse_eye(*self.gl[i].shape) for i in range(L)]
            #pblocks[-1] = self.pl[L-1]
            #print("$$$", [item.shape for item in pblocks])
            #new_pl.append(torch_sparse_matrix_concat(pblocks,1))
            pp = torch_block_matrix_with_padding(pblocks)
            #import matplotlib.pyplot as plt
            #plt.matshow(pp.detach().to_dense().numpy())
            #plt.show()
            new_pl.append(pp)

        #print("%%%",[item.shape for item in new_gl])
        #print("%%%",[item.shape for item in new_pl])
        new_lin = GraphLineage(new_gl, new_pl)
        for i in range(1,len(self.gl)+1):
            new_lin.setX(torch.cat( self.xl[:i], 0 ), i-1 )
            #print(new_lin.xl[i-1].shape)
        return new_lin

def glineage_from_npz(filename):
    npz_file = np.load(filename)

    Xarrs = []
    Parrs = []
    Garrs = []

    for k,v in npz_file.items():
        if "p_" in k:
            Parrs.append(v)
        if "x_" in k:
            Xarrs.append(v)
        if "g_" in k:
            Garrs.append(v)

    Gt = [np_arr_to_torch_sparse(item) for item in sorted(Garrs, key = lambda x: x.shape[0])]
    Xt = [torch.tensor(item) for item in sorted(Xarrs, key = lambda x: x.shape[0])]
    Pt = [np_arr_to_torch_sparse(item) for item in sorted(Parrs, key = lambda x: x.shape[0])]

    gg = GraphLineage(Gt, Pt)
    for i in range(len(Xt)):
        gg.setX(Xt[i],i)
    return gg

class RandomGraphLineage(GraphLineage):

    def __init__(self, sizes, pval=2):
        gl = [
            nxgraph_to_torch_sparse(nx.erdos_renyi_graph(ii,pval/ii)) for ii in sizes
        ]

        pl = [
            torch.tensor(np.random.randint(low=0,high=2,size=(sizes[i], sizes[i-1]))).to_sparse().coalesce() for i in range(1,len(sizes))
        ]

        xl = [
            torch.rand((ii,1)) for ii in sizes
        ]

        GraphLineage.__init__(self,gl,pl)
        self.xl = xl

class CompleteGraphLineage(GraphLineage):

    def __init__(self, sizes):
        gl = [
            nxgraph_to_torch_sparse(nx.complete_graph(ii)) for ii in sizes
        ]

        pl = [
            torch.tensor(np.ones((sizes[i], sizes[i-1]))).to_sparse().coalesce() for i in range(1,len(sizes))
        ]

        xl = [
            torch.rand((ii,1)).to_sparse().coalesce() for ii in sizes
        ]

        super().__init__(gl,pl)

class NullGraphLineage(GraphLineage):

    def __init__(self, n):
        gl = [torch.tensor([[1]]).to_sparse_coo() for i in range(n)]
        pl = [torch.tensor([[1]]).to_sparse_coo() for i in range(n)]
        super().__init__(gl, pl)

class PathGraphLineage(GraphLineage):

    def __init__(self, sizes):
        gl = [
            nxgraph_to_torch_sparse(nx.path_graph(ii)).float() for ii in sizes
        ]

        pl = [
            (torch.tensor(np.kron(np.eye(sizes[i-1]),np.ones((int(sizes[i]/sizes[i-1]),1))))).float().to_sparse_coo() for i in range(1,len(sizes))
        ]

        super().__init__(gl,pl)


class CycleGraphLineage(GraphLineage):

    def __init__(self, sizes):
        gl = [
            nxgraph_to_torch_sparse(nx.cycle_graph(ii)) for ii in sizes
        ]

        pl = [
            (torch.tensor(np.kron(np.eye(sizes[i-1]),np.ones((int(sizes[i]/sizes[i-1]),1))))).to_sparse() for i in range(1,len(sizes))
        ]

        super().__init__(gl,pl)

class GridGraphLineage(GraphLineage):

    def __init__(self, sizes):
        gl = [
            nxgraph_to_torch_sparse(nx.grid_graph(sizes[i])) for i in range(len(sizes))
        ]

        pl = [
            (
                torch.tensor(
                    np.kron(
                        np.kron(
                            np.eye(sizes[i-1][0]),np.ones((int(sizes[i][0]/sizes[i-1][0]),1))
                        ),
                        np.kron(
                            np.eye(sizes[i-1][1]),np.ones((int(sizes[i][1]/sizes[i-1][1]),1))
                        ),
                    )
                )

            ).to_sparse() for i in range(1,len(sizes))
        ]


        super().__init__(gl,pl)

class LearnedGraphLineage(GraphLineage):

    def __init__(self, old_lineage, learn_G=False, learn_P=False):
        self.LP = learn_P
        self.LG = learn_G


        GraphLineage.__init__(self,gl,pl)
        self.xl = old_lineage.xl
        self.rebuild()

class GridPoolingLineage(GraphLineage):

    def __init__(self, sizes, randomize=False):
        self.g_orig_list = [
            nxgraph_to_torch_sparse(nx.grid_graph(sizes[i])) for i in range(len(sizes))
        ]

        self.g_param_list = torch.nn.ParameterList([torch.nn.Parameter(1.0*torch.ones(item.values().shape)) for item in self.g_orig_list])
        self.g_idx_list = [item.indices() for item in self.g_orig_list]

        gl = [torch.sparse_coo_tensor(idxs, vals, size=mat.shape).coalesce() for (idxs, vals, mat) in zip(self.g_idx_list, self.g_param_list, self.g_orig_list)]

        self.sizes = sizes

        if randomize:
            poolA = [
                .5*torch.rand((int(sizes[i][0]/sizes[i-1][0]),1)).to_sparse_coo()
                for i in range(1, len(sizes))
            ]
            poolB = [
                .5*torch.rand((int(sizes[i][1]/sizes[i-1][1]),1)).to_sparse_coo()
                for i in range(1, len(sizes))
            ]
        else:
            poolA = [
                .5*torch.ones((int(sizes[i][0]/sizes[i-1][0]),1)).to_sparse_coo()
                for i in range(1, len(sizes))
            ]
            poolB = [
                .5*torch.ones((int(sizes[i][1]/sizes[i-1][1]),1)).to_sparse_coo()
                for i in range(1, len(sizes))
            ]

        self.poolA_idxs = [item.indices() for item in poolA]
        self.poolB_idxs = [item.indices() for item in poolB]

        self.poolA_shapes = [item.shape for item in poolA]
        self.poolB_shapes = [item.shape for item in poolB]

        self.poolA_values = torch.nn.ParameterList([torch.nn.Parameter(item.values().float()) for item in poolA])
        self.poolB_values = torch.nn.ParameterList([torch.nn.Parameter(item.values().float()) for item in poolA])

        poolA_sparse = [torch.sparse_coo_tensor(idxs, vals, size=shape).coalesce()
            for (idxs, vals, shape) in zip(self.poolA_idxs, self.poolA_values, self.poolA_shapes)
        ]

        poolB_sparse = [torch.sparse_coo_tensor(idxs, vals, size=shape).coalesce()
            for (idxs, vals, shape) in zip(self.poolB_idxs, self.poolB_values, self.poolB_shapes)
        ]

        self.pl = [
            torch_sparse_kron(
                torch_sparse_kron(
                    torch_sparse_eye(size[0],size[0]),pA
                ),
                torch_sparse_kron(
                    torch_sparse_eye(size[1],size[1]),pB
                )
            )
            for size,pA,pB in zip(self.sizes, poolA_sparse, poolB_sparse)
        ]

        #print([item.shape for item in self.pl])

        super().__init__(gl,self.pl)

    def rebuild(self):
        poolA_sparse = [torch.sparse_coo_tensor(idxs, vals, size=shape, device=vals.device).coalesce()
            for (idxs, vals, shape) in zip(self.poolA_idxs, self.poolA_values, self.poolA_shapes)
        ]

        poolB_sparse = [torch.sparse_coo_tensor(idxs, vals, size=shape, device=vals.device).coalesce()
            for (idxs, vals, shape) in zip(self.poolB_idxs, self.poolB_values, self.poolB_shapes)
        ]



        self.pl = [
            torch_sparse_kron(
                torch_sparse_kron(
                    torch_sparse_eye(size[0],size[0],dev=pA.device),pA
                ),
                torch_sparse_kron(
                    torch_sparse_eye(size[1],size[1],dev=pA.device),pB
                )
            )
            for size,pA,pB in zip(self.sizes, poolA_sparse, poolB_sparse)
        ]

        self.gl = [torch.sparse_coo_tensor(idxs, vals, size=mat.shape, device=vals.device).coalesce() for (idxs, vals, mat) in zip(self.g_idx_list, self.g_param_list, self.g_orig_list)]
        self.gl = [.5*(item + item.T).coalesce() for item in self.gl]

        #print([item.layout for item in self.pl])
        #print([item.layout for item in poolB_sparse])
        #print([item.layout for item in poolA_sparse])

    def to(self,device):
        self.poolA_values = [item.to(device) for item in self.poolA_values]
        self.poolA_idxs = [item.to(device) for item in self.poolA_idxs]
        self.poolB_values = [item.to(device) for item in self.poolB_values]
        self.poolB_idxs = [item.to(device) for item in self.poolB_idxs]
        self.g_param_list = [item.to(device) for item in self.g_param_list]
        self.g_idx_list = [item.to(device) for item in self.g_idx_list]

    def parameters(self):
        return torch.nn.ParameterList([*self.poolA_values, *self.poolB_values, *self.g_param_list ])

class ClusteredGraphLineage(GraphLineage):

    def __init__(self, old_lineage, randomize=False, opt_G = False, opt_P=False, norm_G=True, norm_P=True):

        self.opt_G = opt_G
        self.opt_P = opt_P
        self.norm_P = norm_P
        self.norm_G = norm_G

        if randomize:
            if opt_G:
                self.gl_orig = torch.nn.ParameterList([torch.nn.Parameter(1.0-2*torch.rand(old_lineage.gl[-1].shape).float())])
            else:
                self.gl_orig = [old_lineage.gl[-1].to_dense().float()]
            if opt_P:
                self.pl_orig = torch.nn.ParameterList([torch.nn.Parameter(1.0-2*torch.rand(item.shape).float()) for item in old_lineage.pl])
            else:
                self.pl_orig = [item.to_dense().float() for item in old_lineage.pl]
        else:
            if opt_G:
                self.gl_orig = torch.nn.ParameterList([torch.nn.Parameter(old_lineage.gl[-1].to_dense().float())])
            else:
                self.gl_orig = [old_lineage.gl[-1].to_dense().float()]
            if opt_P:
                self.pl_orig = torch.nn.ParameterList([torch.nn.Parameter(item.to_dense().float()) for item in old_lineage.pl])
            else:
                self.pl_orig = [item.to_dense().float() for item in old_lineage.pl]


        self.gl = [(torch.nn.functional.sigmoid(.5*(item + item.T)) if norm_G else item) for item in self.gl_orig]
        self.pl = [(item / item.sum(0,keepdim=True) if norm_P else item) for item in self.pl_orig]

        GraphLineage.__init__(self,self.gl,self.pl)
        self.xl = torch.nn.ParameterList([
            torch.nn.Parameter(torch.rand((item.shape[0],1))) for item in self.gl_orig
        ])
        self.rebuild()

    def rebuild(self):
        if self.norm_G:
            self.gl = [torch.nn.functional.sigmoid(.5*(item + item.T)) for item in self.gl_orig]
        else:
            self.gl = [.5*(item + item.T) for item in self.gl_orig]
        if self.norm_P:
            self.pl = [item / item.sum(0,keepdim=True) for item in self.pl_orig]
        else:
            self.pl = [item for item in self.pl_orig]

        #print(self.gl[0].shape, [item.shape for item in self.pl])
        #quit()
        for i in range(len(self.pl)-1,-1,-1):
            newP = self.pl[i]
            newG = self.gl[0]
            newG = torch.matmul(newP.T, torch.matmul(newG, newP))
            if self.norm_G:
                newG = torch.nn.functional.sigmoid(newG)
            newG = newG - torch.diag(torch.diag(newG))
            self.gl.insert(0, newG)

    def parameters(self):
        if self.opt_G and self.opt_P:
            return torch.nn.ParameterList([*self.gl_orig, *self.pl_orig])
        elif not self.opt_P:
            return self.gl_orig
        elif not self.opt_G:
            return self.pl_orig
        else:
            return torch.nn.ParameterList([])

class DenseParametrizedGraphLineage(GraphLineage):

    def __init__(self, old_lineage, randomize=False, opt_G = False, opt_P=False, norm_G=True, norm_P=True):

        self.opt_G = opt_G
        self.opt_P = opt_P
        self.norm_P = norm_P
        self.norm_G = norm_G

        if randomize:
            if opt_G:
                self.gl_orig = torch.nn.ParameterList([torch.nn.Parameter(1.0-2*torch.rand(item.shape).float()) for item in old_lineage.gl])
            else:
                self.gl_orig = [item.to_dense().float() for item in old_lineage.gl]
            if opt_P:
                self.pl_orig = torch.nn.ParameterList([torch.nn.Parameter(1.0-2*torch.rand(item.shape).float()) for item in old_lineage.pl])
            else:
                self.pl_orig = [item.to_dense().float() for item in old_lineage.pl]
        else:
            if opt_G:
                self.gl_orig = torch.nn.ParameterList([torch.nn.Parameter(item.to_dense().float()) for item in old_lineage.gl])
            else:
                self.gl_orig = [item.to_dense().float() for item in old_lineage.gl]
            if opt_P:
                self.pl_orig = torch.nn.ParameterList([torch.nn.Parameter(item.to_dense().float()) for item in old_lineage.pl])
            else:
                self.pl_orig = [item.to_dense().float() for item in old_lineage.pl]


        self.gl = [(torch.nn.functional.sigmoid(.5*(item + item.T)) if norm_G else item) for item in self.gl_orig]
        self.pl = [(item / item.sum(0,keepdim=True) if norm_P else item) for item in self.pl_orig]

        GraphLineage.__init__(self,self.gl,self.pl)
        self.xl = torch.nn.ParameterList([
            torch.nn.Parameter(torch.rand((item.shape[0],1))) for item in self.gl_orig
        ])
        self.rebuild()

    def rebuild(self):
        if self.norm_G:
            self.gl = [torch.nn.functional.sigmoid(.5*(item + item.T)) for item in self.gl_orig]
        else:
            self.gl = [.5*(item + item.T) for item in self.gl_orig]
        if self.norm_P:
            self.pl = [item / item.sum(0,keepdim=True) for item in self.pl_orig]
        else:
            self.pl = [item for item in self.pl_orig]


    def parameters(self):
        if self.opt_G and self.opt_P:
            return torch.nn.ParameterList([*self.gl_orig, *self.pl_orig])
        elif not self.opt_P:
            return self.gl_orig
        elif not self.opt_G:
            return self.pl_orig
        else:
            return torch.nn.ParameterList([])

class ParametrizedGraphLineage(GraphLineage):

    def __init__(self, old_lineage, randomize=False, norm_G=False, norm_P=False):
        self.norm_G = norm_G
        self.norm_P = norm_P

        if randomize:
            self.g_orig_list = [torch.sparse_coo_tensor(item.indices(), 1.5 - 3*torch.rand(item.values().shape), size=item.shape).float().coalesce() for item in old_lineage.gl]
            self.p_orig_list = [torch.sparse_coo_tensor(item.indices(), 1.5 - 3*torch.rand(item.values().shape), size=item.shape).float().coalesce() for item in old_lineage.pl]
        else:
            self.g_orig_list = [torch.sparse_coo_tensor(item.indices(),item.values(), size=item.shape).float().coalesce() for item in old_lineage.gl]
            self.p_orig_list = [torch.sparse_coo_tensor(item.indices(),item.values(), size=item.shape).float().coalesce() for item in old_lineage.pl]


        self.g_param_list = torch.nn.ParameterList([torch.nn.Parameter(item.values()) for item in self.g_orig_list])
        self.g_idx_list = [item.indices() for item in self.g_orig_list]

        gl = [torch.sparse_coo_tensor(idxs, vals, size=mat.shape).coalesce() for (idxs, vals, mat) in zip(self.g_idx_list, self.g_param_list, self.g_orig_list)]


        self.p_param_list = torch.nn.ParameterList([torch.nn.Parameter(item.values()) for item in self.p_orig_list])
        self.p_idx_list = [item.indices() for item in self.p_orig_list]

        pl = [torch.sparse_coo_tensor(idxs, vals, size=mat.shape).coalesce() for (idxs, vals, mat) in zip(self.p_idx_list, self.p_param_list, self.p_orig_list)]

        xl = torch.nn.ParameterList([
            torch.nn.Parameter(torch.rand((item.shape[0],1))) for item in self.g_orig_list
        ])
        GraphLineage.__init__(self,gl,pl)
        self.xl = xl
        self.rebuild()

    def rebuild(self):
        self.gl = [torch.sparse_coo_tensor(idxs, vals, size=mat.shape, device=vals.device).coalesce() for (idxs, vals, mat) in zip(self.g_idx_list, self.g_param_list, self.g_orig_list)]
        self.gl = [.5*(item + item.T).coalesce() for item in self.gl]

        if self.norm_G:
            for j in range(3):
                self.gl = [torch.sparse.softmax(item,0) for item in self.gl]
                self.gl = [torch.sparse.softmax(item,1) for item in self.gl]

        self.pl = [torch.sparse_coo_tensor(idxs, vals, size=mat.shape, device=vals.device).coalesce() for (idxs, vals, mat) in zip(self.p_idx_list, self.p_param_list, self.p_orig_list)]

        if self.norm_P:
            self.pl = [torch.sparse.softmax(item,1) for item in self.pl]
        #print(self.gl[0].dtype)

    def to(self, device):
        self.g_param_list = [item.to(device) for item in self.g_param_list]
        self.g_idx_list = [item.to(device) for item in self.g_idx_list]
        self.p_param_list = [item.to(device) for item in self.p_param_list]
        self.p_idx_list = [item.to(device) for item in self.p_idx_list]

    def parameters(self):
        return torch.nn.ParameterList([*self.g_param_list, *self.p_param_list])
