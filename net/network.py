import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_points, knn_gather

from .encode import EncodeNet
from .decode import DecodeNet
from .utils import GlobalShiftLayer, Conv1D


class PointEncoder(nn.Module):
    def __init__(self, num_out=1, encode_knn=16, code_dim=128):
        super(PointEncoder, self).__init__()
        self.num_out = num_out

        self.encodeNet = EncodeNet(num_convs=4,
                                    in_channels=3,
                                    conv_channels=24,
                                    knn=encode_knn)
        dim_1 = self.encodeNet.out_channels

        self.conv_1 = Conv1D(dim_1, 128)

        dim_2 = 512
        self.shift_1 = GlobalShiftLayer(self.num_out*2, 128, 256, with_fc=True)
        self.shift_2 = GlobalShiftLayer(self.num_out, 256, dim_2, with_last=True, with_fc=True)
        self.shift_3 = GlobalShiftLayer(self.num_out, dim_2, dim_2, with_fc=True)

        self.conv_2 = Conv1D(dim_2, 256)
        self.conv_3 = Conv1D(256, code_dim)

    def forward(self, pos, knn_idx):
        """
            pos: (B, N, 3)
            knn_idx: (B, N, K)
        """
        ### Local Aggregation
        y = self.encodeNet(pos, knn_idx=knn_idx).transpose(2, 1)   # (B, C, N)
        y = self.conv_1(y)

        ### Multi-scale Global Shift
        y, global_1 = self.shift_1(y)                              # (B, C, n*2), (B, C)
        y, global_2 = self.shift_2(y, global_1)                    # (B, C, n), (B, C)
        y, global_3 = self.shift_3(y)

        y = self.conv_3(self.conv_2(y))                            # (B, C, n)
        return y


class Network(nn.Module):
    def __init__(self, num_in=1, num_knn=16, decode_knn=16):
        super(Network, self).__init__()
        self.num_in = num_in
        self.num_out = num_in // 4
        self.num_knn = num_knn
        self.decode_knn = decode_knn
        assert decode_knn >= num_knn
        code_dim = 128

        self.pointEncoder = PointEncoder(num_out=self.num_out, encode_knn=self.num_knn, code_dim=code_dim)

        pos_dim = 64
        self.out_dim = 128
        self.featDecoder = DecodeNet(in_dim=code_dim,
                                    pos_dim=pos_dim + 3,
                                    out_dim=self.out_dim,
                                    hidden_size=128,
                                    num_blocks=3)

        self.mlp_pos = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, pos_dim),
        )

        self.conv_1 = Conv1D(128, 128)
        self.conv_2 = Conv1D(128, 128)
        self.conv_w = nn.Conv1d(128, 1, 1)
        self.mlp_n  = nn.Linear(128, 3)
        # self.mlp_nn = nn.Linear(128, 3)

    def forward(self, pos, mode_test=False):
        """
            pos: (B, N, 3)
        """
        _, knn_idx, _ = knn_points(pos, pos, K=self.decode_knn+1, return_nn=False)  # (B, N, K+1)

        ### Space Transformation
        y = self.pointEncoder(pos, knn_idx=knn_idx[:,:,1:self.num_knn+1])           # (B, C, n)
        B, Cy, _ = y.size()

        ### Relative Position Encoding
        pos_sub = pos[:, :self.num_out, :]                                          # (B, n, 3)
        knn_idx = knn_idx[:, :self.num_out, :self.decode_knn]                       # (B, n, K)

        nn_pc = knn_gather(pos, knn_idx)                                            # (B, n, K, 3)
        nn_pc = nn_pc - pos_sub.unsqueeze(2)                                        # (B, n, K, 3)

        nn_feat = self.mlp_pos(nn_pc)                                               # (B, n, K, C)
        nn_feat = torch.cat([nn_pc, nn_feat], dim=-1)                               # (B, n, K, 3+C)

        ### Surface Fitting
        Cp = nn_feat.size()[-1]
        feat = self.featDecoder(x=nn_feat.view(B*self.num_out, self.decode_knn, Cp),    # (B*n, K, C)
                                c=y.transpose(2, 1).reshape(B*self.num_out, Cy),        # (B*n, C)
                            )
        feat = feat.reshape(B, self.num_out, self.out_dim, self.decode_knn)             # (B, n, C, K)
        feat = feat.max(dim=3, keepdim=False)[0]                                        # (B, n, C)

        ### Output Module
        feat = self.conv_1(feat.transpose(2, 1))                                        # (B, C, n)
        weights = 0.01 + torch.sigmoid(self.conv_w(feat))                               # (B, 1, n)
        normal = self.mlp_n(self.conv_2(feat * weights).max(dim=2, keepdim=False)[0])   # (B, 3)

        neighbor_normal = None
        # neighbor_normal = self.mlp_nn(feat.transpose(2, 1))                           # (B, n, 3)

        normal = F.normalize(normal, p=2, dim=-1)
        if neighbor_normal is not None:
            neighbor_normal = F.normalize(neighbor_normal, p=2, dim=-1)

        if mode_test:
            return normal
        return normal, weights, neighbor_normal


    def get_loss(self, q_target, q_pred, pred_weights=None, normal_loss_type='sin', pcl_in=None):
        """
            q_target: (B, 3)
            q_pred: (B, 3)
            pred_weights: (B, 1, N)
            pcl_in: (B, N, 3)
        """
        def cos_angle(v1, v2):
            return torch.bmm(v1.unsqueeze(1), v2.unsqueeze(2)).view(-1) / torch.clamp(v1.norm(2, 1) * v2.norm(2, 1), min=0.000001)

        weight_loss = torch.zeros(1, device=q_pred.device, dtype=q_pred.dtype)

        ### query point normal
        o_pred = q_pred
        o_target = q_target

        if normal_loss_type == 'mse_loss':
            normal_loss = 0.5 * F.mse_loss(o_pred, o_target)
        elif normal_loss_type == 'ms_euclidean':
            normal_loss = 0.1 * torch.min((o_pred-o_target).pow(2).sum(1), (o_pred+o_target).pow(2).sum(1)).mean()
        elif normal_loss_type == 'ms_oneminuscos':
            cos_ang = cos_angle(o_pred, o_target)
            normal_loss = 1.0 * (1-torch.abs(cos_ang)).pow(2).mean()
        elif normal_loss_type == 'sin':
            normal_loss = 0.1 * torch.norm(torch.cross(o_pred, o_target, dim=-1), p=2, dim=1).mean()
        else:
            raise ValueError('Unsupported loss type: %s' % (normal_loss_type))

        ### compute the true weight by fitting distance
        pcl_in = pcl_in[:, :self.num_out, :]
        pred_weights = pred_weights.squeeze()
        if pred_weights is not None:
            thres_d = 0.05 * 0.05
            normal_dis = torch.bmm(o_target.unsqueeze(1), pcl_in.transpose(2, 1)).pow(2).squeeze()    # (B, N)
            sigma = torch.mean(normal_dis, dim=1) * 0.3 + 1e-5                                        # (B,)
            threshold_matrix = torch.ones_like(sigma) * thres_d                                       # (B,)
            sigma = torch.where(sigma < thres_d, threshold_matrix, sigma)                             # (B,)  all sigma >= thres_d
            true_weight = torch.exp(-1 * torch.div(normal_dis, sigma.unsqueeze(-1)))                  # (B, N)

            weight_loss = (true_weight - pred_weights).pow(2).mean()

        loss = normal_loss + weight_loss

        return loss, (normal_loss, weight_loss)


