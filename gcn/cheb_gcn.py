import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from scipy.sparse.linalg import eigs


class cheb_conv(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''
    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv, self).__init__()
        self.K = K
        self.cheb_polynomials=cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])
        for i in self.Theta:
            #print(i)
            nn.init.kaiming_uniform_(i)
            #print(i)
        #nn.init.xavier_uniform_(self.Theta)
        # for i in self.Theta:
        #     print(i)
        #     init.normal_(i.data)



    def forward(self, x):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        batch_size, num_of_vertices, in_channels = x.shape

        # adj=self.adj_conn
        # # k阶拉普拉斯矩阵
        # Ls = []
        # L1 = adj.repeat(batch_size,1,1).cuda()
        # L0 = torch.eye(num_of_vertices).repeat(batch_size, 1, 1).cuda()
        # Ls.append(L0)
        # Ls.append(L1)
        # for k in range(2, self.K):
        #     L2 = 2 * torch.matmul(adj, L1) - L0
        #     L0, L1 = L1, L2
        #     Ls.append(L2)
        #
        # Lap = torch.stack(Ls, 1)  # [B, K,nNode, nNode]



        output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

        for k in range(self.K):

            #T_k = Lap[:,k]  # (b,N,N)
            T_k=self.cheb_polynomials[k]
            #print(T_k.shape)
            #print(T_k)
            theta_k = self.Theta[k]  # (in_channel, out_channel)
            #print()
            #print(x)
            rhs = x.permute(0, 2, 1).matmul(T_k).permute(0, 2, 1) #x(b,n,c)->(b,c,n)*T(b,n,n)->(b,c,n)->(b,n,c)

            #print(rhs)
            #print(theta_k)

            output = output + rhs.matmul(theta_k)
            #print(output)
            #print(outputs)
        return F.relu(output)


def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real  #最大特征值的实数 1个特征值

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials
#
# a=3
# x = torch.zeros(a * a, a * a)
# for i in range(0, a):
#     for j in range(0, a):
#         index1 = i * a + j
#         for k in range(0, a):
#             for q in range(0, a):
#                 index2 = k * a + q
#                 if index1 == index2:
#                      x[index1, index2] = 1
#                 else:
#                      x[index1, index2] = (i - k) ** 2 + (j - q) ** 2
# print(x)
#
# x = torch.sqrt(x)
# x = 1 / x
# print(x)
#
# a_min = torch.full((a * a, 1), 0.49)
# print(a_min)
# ge = torch.ge(x, a_min)
# # 设置zero变量，方便后面的where操作
# zero = torch.zeros_like(x)
# result = torch.where(ge, x, zero)
# print(result)
# #归一化
# result_sum=result.sum(-1)
# print(result_sum.unsqueeze(-1))
# result=(result/result_sum.unsqueeze(-1)).numpy()
# print(result)
# s=scaled_Laplacian(result)
# print(x)
# print(s)
# print(s.sum(-1))
# #print()
