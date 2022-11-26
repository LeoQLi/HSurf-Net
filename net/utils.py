import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_points


def knn_group(x:torch.FloatTensor, idx:torch.LongTensor):
    """
    :param  x:      (B, N, F)
    :param  idx:    (B, M, k)
    :return (B, M, k, F)
    """
    B, N, F = tuple(x.size())
    _, M, k = tuple(idx.size())

    x = x.unsqueeze(1).expand(B, M, N, F)
    idx = idx.unsqueeze(3).expand(B, M, k, F)

    return torch.gather(x, dim=2, index=idx)


def get_knn_idx(pos, query, k, offset=0):
    """
    :param  pos:     (B, N, F)
    :param  query:   (B, M, F)
    :return knn_idx: (B, M, k)
    """
    _, knn_idx, _ = knn_points(pos, query, K=k+offset, return_nn=False)
    return knn_idx[:, :, offset:]


class LinearLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, with_bn=0, activation=None):
        super().__init__()
        assert with_bn in [0, 1, 2]
        self.with_bn = with_bn > 0 and activation is not None

        self.linear = nn.Linear(in_features, out_features)

        if self.with_bn:
            if with_bn == 2:
                self.bn = nn.BatchNorm2d(out_features)
            else:
                self.bn = nn.BatchNorm1d(out_features)

        if activation is None:
            self.activation = nn.Identity()
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(alpha=1.0)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.1)
        else:
            raise ValueError()

    def forward(self, x):
        """
        x: (*, C)
        y: (*, C)
        """
        y = self.linear(x)
        if self.with_bn:
            if x.dim() == 2:    # (B, C)
                y = self.activation(self.bn(y))
            elif x.dim() == 3:  # (B, N, C)
                y = self.activation(self.bn(y.transpose(1, 2))).transpose(1, 2)
            elif x.dim() == 4:  # (B, H, W, C)
                y = self.activation(self.bn(y.permute(0, 3, 1, 2))).permute(0, 2, 3, 1)
        else:
            y = self.activation(y)
        return y


class Conv1D(nn.Module):
    def __init__(self, input_dim, output_dim, with_bn=True, with_relu=True):
        super(Conv1D, self).__init__()
        self.with_bn = with_bn
        self.with_relu = with_relu
        self.conv = nn.Conv1d(input_dim, output_dim, 1)
        if with_bn:
            self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        """
        x: (B, C, N)
        """
        if self.with_bn:
            x = self.bn(self.conv(x))
        else:
            x = self.conv(x)

        if self.with_relu:
            x = F.relu(x)
        return x


class FC(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FC, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        """
        x: (B, C)
        """
        x = F.relu(self.bn(self.fc(x)))
        return x


class GlobalShiftLayer(nn.Module):
    def __init__(self, output_scale, input_dim, output_dim, with_last=False, with_fc=True):
        super(GlobalShiftLayer, self).__init__()
        self.output_scale = output_scale
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.with_last = with_last
        self.with_fc = with_fc

        self.conv_in = Conv1D(input_dim, input_dim*2, with_bn=True, with_relu=with_fc)

        if with_fc:
            self.fc = FC(input_dim*2, input_dim//2)
            if with_last:
                self.conv_out = Conv1D(input_dim + input_dim//2 + input_dim//4, output_dim, with_bn=True)
            else:
                self.conv_out = Conv1D(input_dim + input_dim//2, output_dim, with_bn=True)
        else:
            if with_last:
                self.conv_out = Conv1D(input_dim + input_dim*2 + input_dim, output_dim, with_bn=True)
            else:
                self.conv_out = Conv1D(input_dim + input_dim*2, output_dim, with_bn=True)

    def forward(self, x, x_last=None):
        """
        x: (B, C, N)
        x_last: (B, C)
        """
        BS = x.size()[0]

        ### Global information
        y = self.conv_in(x)
        x_global = torch.max(y, dim=2, keepdim=False)[0]    # (B, C*2)
        if self.with_fc:
            x_global = self.fc(x_global)                    # (B, C/2)

        ### Feature fusion for shifting
        if self.with_last:
            x = torch.cat([x_global.view(BS, -1, 1).repeat(1, 1, self.output_scale),
                            x_last.view(BS, -1, 1).repeat(1, 1, self.output_scale),
                            x[:, :, :self.output_scale],
                ], dim=1)
        else:
            x = torch.cat([x_global.view(BS, -1, 1).repeat(1, 1, self.output_scale),
                            x[:, :, :self.output_scale],
                ], dim=1)

        x = self.conv_out(x)
        return x, x_global



