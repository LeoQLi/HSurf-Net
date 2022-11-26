import torch
import torch.nn as nn
from .utils import knn_group, get_knn_idx

from .utils import LinearLayer as FCLayer
BN1d = 1
BN2d = 2


class Aggregator(nn.Module):
    def __init__(self, oper):
        super().__init__()
        assert oper in ('mean', 'sum', 'max')
        self.oper = oper

    def forward(self, x, dim=2):
        if self.oper == 'mean':
            return x.mean(dim=dim, keepdim=False)
        elif self.oper == 'sum':
            return x.sum(dim=dim, keepdim=False)
        elif self.oper == 'max':
            ret, _ = x.max(dim=dim, keepdim=False)
            return ret


class GraphConv(nn.Module):
    def __init__(self, in_channels, num_fc_layers, growth_rate, knn=16, aggr='max', with_bn=BN2d, activation='relu', relative_feat_only=False):
        super().__init__()
        self.in_channels = in_channels
        self.knn = knn
        assert num_fc_layers > 2
        self.num_fc_layers = num_fc_layers
        self.growth_rate = growth_rate
        self.relative_feat_only = relative_feat_only

        if relative_feat_only:
            self.layer_first = FCLayer(in_channels+3, growth_rate, with_bn=with_bn, activation=activation)
        else:
            self.layer_first = FCLayer(in_channels*3, growth_rate, with_bn=with_bn, activation=activation)

        self.layers_mid = nn.ModuleList()
        for i in range(1, num_fc_layers-1):
            self.layers_mid.append(FCLayer(in_channels + i * growth_rate, growth_rate, with_bn=with_bn, activation=activation))

        self.layer_last = FCLayer(in_channels + (num_fc_layers - 1) * growth_rate, growth_rate, with_bn=False, activation=None)

        self.aggr = Aggregator(aggr)

    @property
    def out_channels(self):
        return self.in_channels + self.num_fc_layers * self.growth_rate

    def get_edge_feature(self, x, pos, knn_idx):
        """
        :param        x: (B, N, c)
        :param  knn_idx: (B, N, K)
        :return edge_feat: (B, N, K, C)
        """
        knn_feat = knn_group(x, knn_idx)   # (B, N, K, c)
        x_tiled = x.unsqueeze(-2).expand_as(knn_feat)
        if self.relative_feat_only:
            knn_pos = knn_group(pos, knn_idx)
            pos_tiled = pos.unsqueeze(-2)
            edge_feat = torch.cat([knn_pos - pos_tiled, knn_feat - x_tiled], dim=3)
        else:
            edge_feat = torch.cat([x_tiled, knn_feat, knn_feat - x_tiled], dim=3)
        return edge_feat

    def forward(self, x, pos, knn_idx=None):
        """
        :param  x: (B, N, x)
              pos: (B, N, y)
        :return y: (B, N, z)
          knn_idx: (B, N, K)
        """
        if knn_idx is None:
            knn_idx = get_knn_idx(pos, pos, k=self.knn, offset=1)
        edge_feat = self.get_edge_feature(x, pos, knn_idx=knn_idx) # (B, N, K, C)

        ### First Layer
        y = torch.cat([
            self.layer_first(edge_feat),              # (B, N, K, c)
            x.unsqueeze(-2).repeat(1, 1, self.knn, 1) # (B, N, K, d)
        ], dim=-1)                                    # (B, N, K, c+d)

        ### Intermediate Layers
        for layer in self.layers_mid:
            y = torch.cat([
                layer(y),                             # (B, N, K, c)
                y                                     # (B, N, K, c+d)
            ], dim=-1)                                # (B, N, K, c+d+...)

        ### Last Layer
        y = torch.cat([
            self.layer_last(y),                       # (B, N, K, c)
            y                                         # (B, N, K, d+(L-1)*c)
        ], dim=-1)                                    # (B, N, K, d+L*c)

        ### Pooling Layer
        y = self.aggr(y, dim=-2)

        return y, knn_idx


class EncodeNet(nn.Module):
    def __init__(self,
        num_convs=4,
        in_channels=3,
        conv_channels=24,
        num_fc_layers=3,
        growth_rate=12,
        knn=16,
        aggr='max',
        activation='relu',
    ):
        super().__init__()
        self.num_convs = num_convs
        self.in_channels = in_channels

        self.trans = nn.ModuleList()
        self.convs = nn.ModuleList()
        for i in range(num_convs):
            tran = FCLayer(in_channels, conv_channels, with_bn=BN1d, activation=activation)
            conv = GraphConv(
                in_channels=conv_channels,
                num_fc_layers=num_fc_layers,
                growth_rate=growth_rate,
                knn=knn,
                aggr=aggr,
                activation=activation,
                relative_feat_only=(i == 0),
            )
            self.trans.append(tran)
            self.convs.append(conv)
            in_channels = conv.out_channels

    @property
    def out_channels(self):
        return self.convs[-1].out_channels

    def forward(self, x, knn_idx=None):
        """
        :param  x: (B, N, 3+c)
          knn_idx: (B, N, K)
        :return y: (B, N, C), C = conv_channels+num_fc_layers*growth_rate
        """
        pos = x[:,:,:3]
        for i in range(self.num_convs):
            x = self.trans[i](x)
            x, knn_idx = self.convs[i](x, pos=pos, knn_idx=knn_idx)
        return x