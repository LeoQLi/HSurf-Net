import torch
import torch.nn as nn


class ResnetBlock(nn.Module):
    def __init__(self, c_dim, size_in, size_h=None, size_out=None):
        """
        size_in (int): input dimension
        size_h (int): hidden dimension
        size_out (int): output dimension
        """
        super().__init__()
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in

        self.mlp_1 = nn.Sequential(
            nn.BatchNorm1d(size_in),
            nn.ReLU(),
            nn.Conv1d(size_in, size_h, 1)
        )
        self.mlp_2 = nn.Sequential(
            nn.BatchNorm1d(size_h),
            nn.ReLU(),
            nn.Conv1d(size_h, size_out, 1)
        )

        self.fc_c = nn.Conv1d(c_dim, size_out, 1)

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False)

    def forward(self, x, c):
        dx = self.mlp_1(x)
        dx = self.mlp_2(dx)

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        out = x_s + dx + self.fc_c(c)
        return out


class DecodeNet(nn.Module):
    def __init__(self, in_dim, pos_dim, out_dim, hidden_size, num_blocks):
        """
        in_dim: Dimension of context vectors
        pos_dim: Point dimension
        out_dim: Output dimension
        hidden_size: Hidden state dimension
        """
        super().__init__()

        c_dim = in_dim + pos_dim
        self.conv_p = nn.Conv1d(c_dim, hidden_size, 1)
        self.blocks = nn.ModuleList([
            ResnetBlock(c_dim, hidden_size) for _ in range(num_blocks)
        ])
        self.mlp_out = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Conv1d(hidden_size, out_dim, 1)
        )

    def forward(self, x, c):
        """
        x: (B, N, C)
        c: (B, in_dim), latent code
        """
        x = x.transpose(2, 1)         # (B, C, N)
        num_points = x.size(-1)
        c = c.unsqueeze(2).expand(-1, -1, num_points)  # (B, C, N)

        xc = torch.cat([x, c], dim=1)
        net = self.conv_p(xc)

        for block in self.blocks:
            net = block(net, xc)

        out = self.mlp_out(net)       # (B, out_dim, N)
        return out

