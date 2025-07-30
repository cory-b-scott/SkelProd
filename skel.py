import torch
import numpy as np
import networkx as nx
import scipy as sp

def torch_mixed_sparse_dense_kron(A,B):
    return torch_sparse_kron(A.tosparse(),B.tosparse())


def torch_sparse_kron(A, B):
    #if (not A.is_sparse) or (not B.is_sparse):
    #    return torch_mixed_sparse_dense_kron(A,B)
    #if not A.is_sparse:
    #    A = A.to_sparse_coo()
    #if not B.is_sparse:
    #    B = B.to_sparse_coo()

    A_idx = A.indices()
    B_idx = B.indices()
    #print(torch.kron(A_idx, torch.ones_like(B_idx)) + torch.kron(torch.ones_like(A_idx), B_idx))
    A_idx_mod = torch.kron(
        torch.stack(
            [A_idx[0] * B.shape[0],
             A_idx[1] * B.shape[1]
            ]
        ),
        torch.ones_like(B_idx[0])
    )
    i = (A_idx_mod + torch.kron(torch.ones_like(A_idx[0]),B_idx))
    v = torch.kron(A.values(), B.values())
    return torch.sparse_coo_tensor(i, v, [A.shape[0]*B.shape[0], A.shape[1]*B.shape[1]],device=A.device).coalesce()

def torch_sparse_kronsum(A,B):
    #print(A,B)
    return (
        torch_sparse_kron(
            A,
            torch_sparse_eye(*B.shape,dev=B.device)
        ) + \
        torch_sparse_kron(
            torch_sparse_eye(*A.shape,dev=A.device),
            B
        )).coalesce()

def torch_sparse_eye(n1,n2,dev='cpu'):
    n = min(n1,n2)
    i = torch.stack([torch.arange(n),
                      torch.arange(n)])
    v = torch.ones(n)
    return torch.sparse_coo_tensor(i, v, [n1,n2],device=dev).coalesce()

def torch_empty_sparse(n1, n2, dev='cpu'):
     return torch.sparse_coo_tensor([[],[]], [], [n1,n2],device=dev).coalesce()

def torch_sparse_transpose(A):
    A_idx = A.indices()
    return torch.sparse_coo_tensor(torch.flip(A_idx,[0]), A.values(), A.shape[::-1],device=A.device).coalesce()

def torch_block_matrix_with_padding(blocks):
    shapes = [[item.shape for item in rowBlocks] for rowBlocks in blocks]
    shapeArr = np.array(shapes)
    rsizes = np.max(shapeArr[:,:,0], 1)
    csizes = np.max(shapeArr[:,:,1], 0)
    #for i in range(len(rsizes)):
    #    for j in range(len(csizes)):
    #        print(rsizes[i], csizes[j],end = " ")
    #    print()
    #print(rsizes,csizes)
    #for zz in shapes:
    #    print([item for item in zz])
    mod_blocks = [
        [
            blocks[i][j] if blocks[i][j].shape[0] > 0 else torch_empty_sparse(rsizes[i],csizes[j], dev=blocks[i][j].device)
            for j in range(len(blocks[i]))
        ]
        for i in range(len(blocks))
    ]

    return torch_block_matrix(mod_blocks)

def torch_block_matrix(blocks):
    return torch_sparse_matrix_concat(
        [torch_sparse_matrix_concat(
            row,
            1
        ).coalesce() for row in blocks],
        0
    )

def torch_sparse_matrix_concat(submats, dim):
    #print([item.shape for item in submats])
    other_dim = (dim+1)%2
    shapes = [0] + [item.shape[dim] for item in submats]
    if not all([item.shape[other_dim] == submats[0].shape[other_dim] for item in submats]):
        raise ValueError("Incorrect matrix shapes")
    offsets = torch.cumsum(torch.tensor(shapes),0)
    final_shape = [0,0]
    final_shape[dim] = offsets[-1]
    final_shape[other_dim] = submats[0].shape[other_dim]
    idxs = [item.indices() for item in submats]
    all_offsets = torch.zeros((2,offsets.shape[0]),device=submats[0].device)
    all_offsets[dim] = offsets
    mod_idxs = [(item.T+off).T for item, off in zip(idxs, all_offsets[:,:-1].T)]
    concat_idxs = torch.cat(mod_idxs,1)
    concat_values = torch.cat([item.values() for item in submats],0)
    return torch.sparse_coo_tensor(concat_idxs, concat_values, final_shape).coalesce()

def torch_thickening_op_blocks(glist, slist):
    k = len(glist)
    blocks = [[
        torch.sparse_coo_tensor([[],[]], [], [
            gsizes[i][0],
            gsizes[j][1]
        ],device=glist[0].device).coalesce()
      for j in range(k)
    ] for i in range(k)]

def torch_skeletal_box_product_blocks(glist, hlist, slist, plist):
    blocks = []
    gsizes = [item.shape for item in glist]
    hsizes = [item.shape for item in hlist]
    for i in range(len(glist)):
        brow = []
        for j in range(len(hlist)):
            if i == j:
                brow.append(torch_sparse_kronsum(glist[i], hlist[j]))
            else:
                brow.append(
                    torch.sparse_coo_tensor([[],[]], [], [
                        gsizes[i][0]*hsizes[i][0],
                        gsizes[j][1]*hsizes[j][1]
                    ],device=glist[0].device).coalesce()
                )
        blocks.append(brow)
    return blocks

def torch_skeletal_cross_product_blocks(glist, hlist, slist, plist):
    blocks = []
    gsizes = [item.shape for item in glist]
    hsizes = [item.shape for item in hlist]
    for i in range(len(glist)):
        brow = []
        for j in range(len(hlist)):
            if i == j:
                brow.append(torch_sparse_kron(glist[i], hlist[j]))
            elif i-j == -1:
                brow.append(torch_sparse_kron(
                    torch_sparse_transpose(slist[i]),plist[j-1]
                ))
            elif i-j == 1:
                brow.append(torch_sparse_kron(
                    slist[i-1],torch_sparse_transpose(plist[j].to_sparse_coo())
                ))
            else:
                brow.append(
                    torch.sparse_coo_tensor([[],[]], [], [
                        gsizes[i][0]*hsizes[i][0],
                        gsizes[j][1]*hsizes[j][1]
                    ],device=glist[0].device).coalesce()
                )
        blocks.append(brow)
    return blocks

def torch_skeletal_box_cross_product_blocks(glist, hlist, slist, plist):
    blocks = []
    gsizes = [item.shape for item in glist]
    hsizes = [item.shape for item in hlist]
    for i in range(len(glist)):
        brow = []
        for j in range(len(hlist)):
            if i == j:
                brow.append((torch_sparse_kronsum(glist[i], hlist[j])+torch_sparse_kron(glist[i], hlist[j])).coalesce())
            elif i-j == -1:
                brow.append(torch_sparse_kron(
                    torch_sparse_transpose(slist[i]),plist[j-1]
                ))
            elif i-j == 1:
                brow.append(torch_sparse_kron(
                    slist[i-1],torch_sparse_transpose(plist[j].to_sparse_coo())
                ))
            else:
                brow.append(
                    torch.sparse_coo_tensor([[],[]], [], [
                        gsizes[i][0]*hsizes[i][0],
                        gsizes[j][1]*hsizes[j][1]
                    ],device=glist[0].device).coalesce()
                )
        blocks.append(brow)
    return blocks

def torch_skeletal_wrong_box_product_blocks(glist, hlist, slist, plist):
    blocks = []
    gsizes = [item.shape for item in glist]
    hsizes = [item.shape for item in hlist]
    for i in range(len(glist)):
        brow = []
        for j in range(len(hlist)):
            if i == j:
                brow.append((torch_sparse_kronsum(glist[i], hlist[j])).coalesce())
            elif i-j == -1:
                brow.append(torch_sparse_kron(
                    torch_sparse_transpose(slist[i]),plist[j-1]
                ))
            elif i-j == 1:
                brow.append(torch_sparse_kron(
                    slist[i-1],torch_sparse_transpose(plist[j].to_sparse_coo())
                ))
            else:
                brow.append(
                    torch.sparse_coo_tensor([[],[]], [], [
                        gsizes[i][0]*hsizes[i][0],
                        gsizes[j][1]*hsizes[j][1]
                    ],device=glist[0].device).coalesce()
                )
        blocks.append(brow)
    return blocks

def torch_sparse_activation(A, act):
    return torch.sparse_coo_tensor(A.indices(), act(A.values()), A.shape,device=A.device).coalesce()

def scipy_spr_to_torch_sparse(scipy_spr):
    return torch.sparse_coo_tensor(
        torch.tensor(np.array([scipy_spr.row,scipy_spr.col])),
        torch.tensor(scipy_spr.data),
        scipy_spr.shape
    ).coalesce()

def nxgraph_to_torch_sparse(nxgr):
    Asparse = nx.to_scipy_sparse_array(nxgr, format='coo')
    return scipy_spr_to_torch_sparse(
        Asparse
    )

def np_arr_to_torch_sparse(nparr):
    return scipy_spr_to_torch_sparse(
        sp.sparse.coo_array(nparr)
    )

def torch_sparse_to_scipy_sparse(t_coo):
    idx = t_coo.indices().detach().cpu().numpy()
    val = t_coo.values().detach().cpu().numpy()
    return sp.sparse.coo_array((val,idx),shape=t_coo.shape,dtype=val.dtype)
