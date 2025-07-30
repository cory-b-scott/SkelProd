import torch

def recursive_split(T, dims, ii=0):
    if ii > len(dims)-1:
        return T
    else:
        em = [recursive_split(Tsmall, dims, ii=ii+1) for Tsmall in torch.split(T, dims[ii], ii)]
        #print(em)
        return torch.stack(em)
        
        
def kron_approx(A, dim1, dim2):
    splits = recursive_split(A, dim2, 0)
    test = splits.reshape(torch.prod(torch.tensor(dim1)), torch.prod(torch.tensor(dim2)))
    U, S, V = torch.pca_lowrank(test, q=1, center=False)
    #print(S)
    return U.reshape(dim1).contiguous(), S[0], V.reshape(dim2).contiguous()

def kron_with_cat(A,B):
    return torch.cat(
        [torch.kron(A, torch.ones(B.shape[0],1)),
        torch.kron(torch.ones(A.shape[0],1), B)],
        -1
    )

def kron_factor_with_split(X, dim1, dim2):
    A,B = torch.split(X, [dim1[-1], dim2[-1]],-1)
    A1, S1, B1 = kron_approx(A, dim1, (dim2[0], 1))
    A2, S2, B2 = kron_approx(B, (dim1[0],1), dim2)
    #print(A1.shape, B1.shape, A2.shape, B2.shape)
    return (A1 * S1 * B1[0,0]), (B2 * S2 * A2[0,0])

