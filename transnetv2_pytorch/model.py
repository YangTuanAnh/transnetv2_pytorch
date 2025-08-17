import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class TransNetV2(nn.Module):
    def __init__(
        self,
        F=16, L=3, S=2, D=1024,
        use_many_hot_targets=True,
        use_frame_similarity=True,
        use_color_histograms=True,
        use_mean_pooling=False,
        dropout_rate=0.5,
        use_convex_comb_reg=False,
        use_resnet_features=False,
        use_resnet_like_top=False,
        frame_similarity_on_last_layer=False,
    ):
        super().__init__()

        if (use_resnet_features or use_resnet_like_top or
                use_convex_comb_reg or frame_similarity_on_last_layer):
            raise NotImplemented("Some options not implemented in Pytorch version of Transnet!")

        self.SDDCNN = nn.ModuleList(
            [StackedDDCNNV2(in_filters=3, n_blocks=S, filters=F)] +
            [StackedDDCNNV2(in_filters=(F * 2 ** (i - 1)) * 4, n_blocks=S, filters=F * 2 ** i)
             for i in range(1, L)]
        )

        self.frame_sim_layer = (
            FrameSimilarity(sum([(F * 2 ** i) * 4 for i in range(L)]),
                            lookup_window=101, output_dim=128, similarity_dim=128)
            if use_frame_similarity else None
        )
        self.color_hist_layer = (
            ColorHistograms(lookup_window=101, output_dim=128)
            if use_color_histograms else None
        )

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate is not None else None

        output_dim = (F * 2 ** (L - 1)) * 4 * 3 * 6  # 3x6 spatial grid
        if use_frame_similarity:
            output_dim += 128
        if use_color_histograms:
            output_dim += 128

        self.fc1 = nn.Linear(output_dim, D)
        self.cls_layer1 = nn.Linear(D, 1)
        self.cls_layer2 = nn.Linear(D, 1) if use_many_hot_targets else None
        self.use_mean_pooling = use_mean_pooling

    def forward(self, inputs):
        assert (
            isinstance(inputs, torch.Tensor)
            and list(inputs.shape[2:]) == [27, 48, 3]
            and inputs.dtype == torch.uint8
        ), "incorrect input type and/or shape"

        x = inputs.permute(0, 4, 1, 2, 3).float().div_(255.)

        block_features = []
        for block in self.SDDCNN:
            x = block(x)
            if self.frame_sim_layer is not None:
                block_features.append(x)

        if self.use_mean_pooling:
            x = x.mean(dim=[3, 4]).permute(0, 2, 1)
        else:
            x = x.permute(0, 2, 3, 4, 1).reshape(x.size(0), x.size(1), -1)

        if self.frame_sim_layer is not None:
            x = torch.cat([self.frame_sim_layer(block_features), x], dim=2)

        if self.color_hist_layer is not None:
            x = torch.cat([self.color_hist_layer(inputs), x], dim=2)

        x = F.relu(self.fc1(x))

        if self.dropout is not None:
            x = self.dropout(x)

        one_hot = self.cls_layer1(x)
        if self.cls_layer2 is not None:
            return one_hot, {"many_hot": self.cls_layer2(x)}
        return one_hot


class StackedDDCNNV2(nn.Module):
    def __init__(self, in_filters, n_blocks, filters, pool_type="avg",
                 shortcut=True, stochastic_depth_drop_prob=0.0):
        super().__init__()
        assert pool_type in ("avg", "max")

        self.shortcut = shortcut
        self.stochastic_depth_drop_prob = stochastic_depth_drop_prob

        self.DDCNN = nn.ModuleList([
            DilatedDCNNV2(
                in_filters if i == 1 else filters * 4,
                filters,
                activation=F.relu if i != n_blocks else None
            ) for i in range(1, n_blocks + 1)
        ])
        self.pool = (
            nn.AvgPool3d(kernel_size=(1, 2, 2))
            if pool_type == "avg" else
            nn.MaxPool3d(kernel_size=(1, 2, 2))
        )

    def forward(self, inputs):
        x = inputs
        shortcut = None
        for block in self.DDCNN:
            x = block(x)
            if shortcut is None:
                shortcut = x

        x = F.relu(x)

        if self.shortcut:
            if self.stochastic_depth_drop_prob and self.training:
                if random.random() < self.stochastic_depth_drop_prob:
                    x = shortcut
                else:
                    x += shortcut
            else:
                x += shortcut

        return self.pool(x)


class DilatedDCNNV2(nn.Module):
    def __init__(self, in_filters, filters, batch_norm=True, activation=None):
        super().__init__()
        self.activation = activation

        self.Conv3D_1 = Conv3DConfigurable(in_filters, filters, dilation_rate=1,
                                           use_bias=not batch_norm)
        self.Conv3D_2 = Conv3DConfigurable(in_filters, filters, dilation_rate=2,
                                           use_bias=not batch_norm)
        self.Conv3D_4 = Conv3DConfigurable(in_filters, filters, dilation_rate=4,
                                           use_bias=not batch_norm)
        self.Conv3D_8 = Conv3DConfigurable(in_filters, filters, dilation_rate=8,
                                           use_bias=not batch_norm)

        self.bn = nn.BatchNorm3d(filters * 4, eps=1e-3) if batch_norm else None

    def forward(self, inputs):
        x = torch.cat([
            self.Conv3D_1(inputs),
            self.Conv3D_2(inputs),
            self.Conv3D_4(inputs),
            self.Conv3D_8(inputs)
        ], dim=1)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Conv3DConfigurable(nn.Module):
    def __init__(self, in_filters, filters, dilation_rate,
                 separable=True, use_bias=True):
        super().__init__()

        if separable:
            self.layers = nn.ModuleList([
                nn.Conv3d(in_filters, 2 * filters, kernel_size=(1, 3, 3),
                          padding=(0, 1, 1), bias=False),
                nn.Conv3d(2 * filters, filters, kernel_size=(3, 1, 1),
                          dilation=(dilation_rate, 1, 1),
                          padding=(dilation_rate, 0, 0),
                          bias=use_bias)
            ])
        else:
            self.layers = nn.ModuleList([
                nn.Conv3d(in_filters, filters, kernel_size=3,
                          dilation=(dilation_rate, 1, 1),
                          padding=(dilation_rate, 1, 1),
                          bias=use_bias)
            ])

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


class FrameSimilarity(nn.Module):
    def __init__(self, in_filters, similarity_dim=128,
                 lookup_window=101, output_dim=128, use_bias=False):
        super().__init__()
        self.projection = nn.Linear(in_filters, similarity_dim, bias=use_bias)
        self.fc = nn.Linear(lookup_window, output_dim)
        self.lookup_window = lookup_window
        assert lookup_window % 2 == 1

    def forward(self, inputs):
        # inputs is a list of tensors: [B,C,T,H,W]
        # combine blocks → [B, totalC, T]
        x = torch.stack(inputs, dim=1).mean(dim=[3, 4])  # [B, blocks, T]
        x = x.view(x.size(0), -1, x.size(2))

        x = F.normalize(self.projection(torch.transpose(x, 1, 2)), p=2, dim=2)

        b, t, _ = x.shape
        sims = torch.bmm(x, x.transpose(1, 2))
        sims_pad = F.pad(sims, [(self.lookup_window - 1) // 2] * 2)

        idx = torch.arange(self.lookup_window, device=x.device)
        idx = idx[None, None, :].expand(b, t, -1)
        center = torch.arange(t, device=x.device)[None, :, None]
        center = center.expand(b, -1, self.lookup_window)

        sims = sims_pad[:, center + idx]

        return F.relu(self.fc(sims))


class ColorHistograms(nn.Module):
    def __init__(self, lookup_window=101, output_dim=None):
        super().__init__()
        self.fc = nn.Linear(lookup_window, output_dim) if output_dim else None
        self.lookup_window = lookup_window
        assert lookup_window % 2 == 1

    @staticmethod
    def compute_color_histograms(frames):
        # frames uint8 → [B,T,H,W,C]
        b, t, h, w, _ = frames.shape
        flat = frames.view(b * t, h * w, 3)
        R, G, B = flat[..., 0] >> 5, flat[..., 1] >> 5, flat[..., 2] >> 5
        bins = (R << 6) + (G << 3) + B
        bins += (torch.arange(b * t, device=frames.device) << 9)[:, None]
        hist = torch.zeros(b * t * 512, dtype=torch.float32, device=frames.device)
        hist.scatter_add_(0, bins.view(-1), torch.ones_like(bins, dtype=torch.float32).view(-1))
        hist = hist.view(b, t, 512)
        return F.normalize(hist, p=2, dim=2)

    def forward(self, inputs):
        x = self.compute_color_histograms(inputs)
        b, t, _ = x.shape
        sims = torch.bmm(x, x.transpose(1, 2))
        sims_pad = F.pad(sims, [(self.lookup_window - 1) // 2] * 2)
        idx = torch.arange(self.lookup_window, device=x.device)
        idx = idx[None, None, :].expand(b, t, -1)
        center = torch.arange(t, device=x.device)[None, :, None]
        center = center.expand(b, -1, self.lookup_window)
        sims = sims_pad[:, center + idx]
        return F.relu(self.fc(sims)) if self.fc else sims
