import torch

def torch_dense_cat(blocks):
    return torch.cat(
        [torch.cat(
            row,
            1
        ) for row in blocks],
        0
    )

def torch_kronsum(A,B):
    return torch.kron(A,torch.eye(B.shape[0])) + torch.kron(torch.eye(A.shape[0]), B)

def torch_skeletal_box_product_blocks_dense(glist, hlist, slist, plist):
    #print([item.is_contiguous() for item in glist])
    #print([item.is_contiguous() for item in hlist])
    #print([item.is_contiguous() for item in slist])
    #print([item.is_contiguous() for item in plist])
    blocks = []
    gsizes = [item.shape for item in glist]
    hsizes = [item.shape for item in hlist]
    for i in range(len(glist)):
        brow = []
        for j in range(len(hlist)):
            if i == j:
                brow.append(torch_kronsum(glist[i], hlist[j]))
            elif i-j == -1:
                brow.append(torch.kron(
                    slist[i].T.contiguous(),plist[j-1]
                ))
            elif i-j == 1:
                brow.append(torch.kron(
                    slist[i-1],plist[j].T.contiguous()
                ))
            else:
                brow.append(
                    torch.zeros((gsizes[i][0]*hsizes[i][0], gsizes[j][1]*hsizes[j][1]))
                )
        blocks.append(brow)
    return blocks
