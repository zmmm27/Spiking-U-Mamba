import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Union, Type, List, Tuple
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from torch.cuda.amp import autocast
from mamba_ssm import Mamba
from spikingjelly.clock_driven import neuron, functional


def get_matching_pool_op(conv_op, pool_type='conv'):
    if pool_type == 'conv':
        return None
    if conv_op == nn.Conv2d:
        return nn.MaxPool2d
    elif conv_op == nn.Conv3d:
        return nn.MaxPool3d
    else:
        raise ValueError(f"Unsupported conv_op: {conv_op}")


def maybe_convert_scalar_to_list(conv_op, scalar):
    if conv_op == nn.Conv2d:
        return [scalar, scalar] if isinstance(scalar, int) else scalar
    elif conv_op == nn.Conv3d:
        return [scalar, scalar, scalar] if isinstance(scalar, int) else scalar
    else:
        return scalar


def convert_conv_op_to_dim(conv_op):
    if conv_op == nn.Conv2d:
        return 2
    elif conv_op == nn.Conv3d:
        return 3
    else:
        return 2


class BasicBlockD(nn.Module):

    def __init__(
            self,
            conv_op,
            input_channels,
            output_channels,
            kernel_size,
            stride,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            nonlin,
            nonlin_kwargs
    ):
        super().__init__()
        self.conv = conv_op(
            input_channels, output_channels, kernel_size,
            stride=stride, padding=kernel_size // 2, bias=conv_bias
        )
        self.norm = norm_op(output_channels, **norm_op_kwargs)
        self.nonlin = nonlin(**nonlin_kwargs)

    def forward(self, x):
        return self.nonlin(self.norm(self.conv(x)))


class UpsampleLayer(nn.Module):
    def __init__(
            self,
            conv_op,
            input_channels,
            output_channels,
            pool_op_kernel_size,
            mode='nearest'
    ):
        super().__init__()
        self.conv = conv_op(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        x = self.conv(x)
        return x


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        return out


class BasicResBlock(nn.Module):
    def __init__(
            self,
            conv_op,
            input_channels,
            output_channels,
            norm_op,
            norm_op_kwargs,
            kernel_size=3,
            padding=1,
            stride=1,
            use_1x1conv=False,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True}
    ):
        super().__init__()

        self.conv1 = conv_op(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = norm_op(output_channels, **norm_op_kwargs)
        self.act1 = nonlin(**nonlin_kwargs)

        self.conv2 = conv_op(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = norm_op(output_channels, **norm_op_kwargs)
        self.act2 = nonlin(**nonlin_kwargs)

        if use_1x1conv:
            self.conv3 = conv_op(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.act2(y)


class UNetResEncoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 return_skips: bool = False,
                 stem_channels: int = None,
                 pool_type: str = 'conv',
                 ):
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages

        assert len(kernel_sizes) == n_stages
        assert len(n_blocks_per_stage) == n_stages
        assert len(features_per_stage) == n_stages
        assert len(strides) == n_stages

        self.conv_pad_sizes = []
        for krnl in kernel_sizes:
            self.conv_pad_sizes.append([i // 2 for i in (krnl if isinstance(krnl, (list, tuple)) else [krnl, krnl])])

        stem_channels = features_per_stage[0] if stem_channels is None else stem_channels

        # Stem
        stem_blocks = [
            BasicResBlock(
                conv_op=conv_op,
                input_channels=input_channels,
                output_channels=stem_channels,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                kernel_size=kernel_sizes[0] if isinstance(kernel_sizes[0], int) else kernel_sizes[0][0],
                padding=self.conv_pad_sizes[0][0] if isinstance(self.conv_pad_sizes[0], list) else self.conv_pad_sizes[
                    0],
                stride=1,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                use_1x1conv=True
            )
        ]

        for _ in range(n_blocks_per_stage[0] - 1):
            stem_blocks.append(
                BasicBlockD(
                    conv_op=conv_op,
                    input_channels=stem_channels,
                    output_channels=stem_channels,
                    kernel_size=kernel_sizes[0] if isinstance(kernel_sizes[0], int) else kernel_sizes[0][0],
                    stride=1,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                )
            )

        self.stem = nn.Sequential(*stem_blocks)

        input_channels = stem_channels

        # Build stages
        stages = []
        for s in range(n_stages):
            kernel = kernel_sizes[s] if isinstance(kernel_sizes[s], int) else kernel_sizes[s][0]
            pad = self.conv_pad_sizes[s][0] if isinstance(self.conv_pad_sizes[s], list) else self.conv_pad_sizes[s]
            stride = strides[s] if isinstance(strides[s], int) else strides[s][0]

            stage_blocks = [
                BasicResBlock(
                    conv_op=conv_op,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    input_channels=input_channels,
                    output_channels=features_per_stage[s],
                    kernel_size=kernel,
                    padding=pad,
                    stride=stride,
                    use_1x1conv=True,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs
                )
            ]

            for _ in range(n_blocks_per_stage[s] - 1):
                stage_blocks.append(
                    BasicBlockD(
                        conv_op=conv_op,
                        input_channels=features_per_stage[s],
                        output_channels=features_per_stage[s],
                        kernel_size=kernel,
                        stride=1,
                        conv_bias=conv_bias,
                        norm_op=norm_op,
                        norm_op_kwargs=norm_op_kwargs,
                        nonlin=nonlin,
                        nonlin_kwargs=nonlin_kwargs,
                    )
                )

            stages.append(nn.Sequential(*stage_blocks))
            input_channels = features_per_stage[s]

        self.stages = nn.ModuleList(stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        x = self.stem(x)
        ret = []
        for s in self.stages:
            x = s(x)
            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]


class UNetResDecoder(nn.Module):
    def __init__(self,
                 encoder,
                 num_classes,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)

        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1

        stages = []
        upsample_layers = []
        seg_layers = []

        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_upsampling = encoder.strides[-s]

            kernel = encoder.kernel_sizes[-(s + 1)]
            if isinstance(kernel, (list, tuple)):
                kernel = kernel[0]
            pad = encoder.conv_pad_sizes[-(s + 1)]
            if isinstance(pad, (list, tuple)):
                pad = pad[0]

            upsample_layers.append(
                UpsampleLayer(
                    conv_op=encoder.conv_op,
                    input_channels=input_features_below,
                    output_channels=input_features_skip,
                    pool_op_kernel_size=stride_for_upsampling[0] if isinstance(stride_for_upsampling, (
                        list, tuple)) else stride_for_upsampling,
                    mode='nearest'
                )
            )

            stage_blocks = [
                BasicResBlock(
                    conv_op=encoder.conv_op,
                    norm_op=encoder.norm_op,
                    norm_op_kwargs=encoder.norm_op_kwargs,
                    nonlin=encoder.nonlin,
                    nonlin_kwargs=encoder.nonlin_kwargs,
                    input_channels=2 * input_features_skip if s < n_stages_encoder - 1 else input_features_skip,
                    output_channels=input_features_skip,
                    kernel_size=kernel,
                    padding=pad,
                    stride=1,
                    use_1x1conv=True
                )
            ]

            for _ in range(n_conv_per_stage[s - 1] - 1):
                stage_blocks.append(
                    BasicBlockD(
                        conv_op=encoder.conv_op,
                        input_channels=input_features_skip,
                        output_channels=input_features_skip,
                        kernel_size=kernel,
                        stride=1,
                        conv_bias=encoder.conv_bias,
                        norm_op=encoder.norm_op,
                        norm_op_kwargs=encoder.norm_op_kwargs,
                        nonlin=encoder.nonlin,
                        nonlin_kwargs=encoder.nonlin_kwargs,
                    )
                )

            stages.append(nn.Sequential(*stage_blocks))
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.upsample_layers = nn.ModuleList(upsample_layers)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []

        for s in range(len(self.stages)):
            x = self.upsample_layers[s](lres_input)
            if s < (len(self.stages) - 1):
                x = torch.cat((x, skips[-(s + 2)]), 1)
            x = self.stages[s](x)

            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))

            lres_input = x

        seg_outputs = seg_outputs[::-1]
        return seg_outputs if self.deep_supervision else seg_outputs[0]


class MultiStepLIFEncoder(nn.Module):
    def __init__(self, in_channels, embed_dim, time_steps=4):
        super().__init__()
        self.time_steps = time_steps

        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.LeakyReLU(inplace=True)
        )

        self.conv_out = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim)
        )


        self.log_tau = nn.Parameter(torch.log(torch.tensor(2.0)))
        self.v_threshold = nn.Parameter(torch.tensor(1.0))


        self.lifs = nn.ModuleList([
            neuron.MultiStepLIFNode(tau=2.0, v_threshold=1.0, detach_reset=True)
            for _ in range(3)
        ])


        self.time_attention = nn.Parameter(torch.ones(time_steps) / time_steps)

    def forward(self, x):
        tau = torch.exp(self.log_tau)
        B, C, H, W = x.shape


        for lif in self.lifs:
            if hasattr(lif, 'reset'):
                lif.reset()


        x_encoded = self.conv_in(x)


        x_time = x_encoded.unsqueeze(0).repeat(self.time_steps, 1, 1, 1, 1)


        time_modulation = torch.linspace(0.8, 1.2, self.time_steps, device=x.device)
        time_modulation = time_modulation.view(self.time_steps, 1, 1, 1, 1)
        x_time = x_time * time_modulation


        all_spikes = []
        for lif in self.lifs:
            spike_seq = lif(x_time)  # [T, B, embed_dim, H, W]
            all_spikes.append(spike_seq)


        spike_seq = torch.stack(all_spikes).mean(dim=0)


        time_weights = F.softmax(self.time_attention, dim=0)
        time_weights = time_weights.view(self.time_steps, 1, 1, 1, 1)


        spike_aggregated = (spike_seq * time_weights).sum(dim=0)


        out = self.conv_out(spike_aggregated)

        return out



class ChannelAttention(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)


class BidirectionalFeatureFusion(nn.Module):

    def __init__(self, dim, reduction=16):
        super().__init__()


        self.gate_mamba = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.Sigmoid()
        )
        self.gate_snn = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.Sigmoid()
        )


        self.mamba_to_snn = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(inplace=True)
        )
        self.snn_to_mamba = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(inplace=True)
        )


        self.channel_attn_mamba = ChannelAttention(dim, reduction)
        self.channel_attn_snn = ChannelAttention(dim, reduction)


        self.spatial_attn = SpatialAttention()


        self.fusion_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim)
        )


        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, mamba_feat, snn_feat):

        mamba_to_snn = self.mamba_to_snn(mamba_feat)
        snn_to_mamba = self.snn_to_mamba(snn_feat)


        mamba_gate = self.gate_mamba(mamba_feat)
        snn_gate = self.gate_snn(snn_feat)


        snn_enhanced = snn_feat + mamba_to_snn * mamba_gate
        mamba_enhanced = mamba_feat + snn_to_mamba * snn_gate


        mamba_enhanced = self.channel_attn_mamba(self.spatial_attn(mamba_enhanced))
        snn_enhanced = self.channel_attn_snn(self.spatial_attn(snn_enhanced))


        fused = torch.cat([mamba_enhanced, snn_enhanced], dim=1)
        fused = self.fusion_conv(fused)


        out = fused + self.gamma * (mamba_feat + snn_feat) / 2

        return out



class SpikingUMamba(nn.Module):


    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 stem_channels: int = None,
                 fusion_type: str = 'bidirectional'
                 ):
        super().__init__()

        n_blocks_per_stage = n_conv_per_stage
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)


        for s in range(math.ceil(n_stages / 2), n_stages):
            n_blocks_per_stage[s] = 1

        for s in range(math.ceil((n_stages - 1) / 2 + 0.5), n_stages - 1):
            n_conv_per_stage_decoder[s] = 1

        assert len(n_blocks_per_stage) == n_stages
        assert len(n_conv_per_stage_decoder) == (n_stages - 1)

        self.encoder = UNetResEncoder(
            input_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_blocks_per_stage,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            nonlin,
            nonlin_kwargs,
            return_skips=True,
            stem_channels=stem_channels
        )

        self.mamba_layer = MambaLayer(dim=features_per_stage[-1])


        bottleneck_dim = features_per_stage[-1]
        self.snn_enhancer = MultiStepLIFEncoder(
            in_channels=bottleneck_dim,
            embed_dim=bottleneck_dim,
            time_steps=4
        )


        self.fusion = BidirectionalFeatureFusion(bottleneck_dim, reduction=16)
        self.use_gates = False


        self.post_fusion = nn.Sequential(
            nn.Conv2d(bottleneck_dim, bottleneck_dim, 3, padding=1),
            nn.BatchNorm2d(bottleneck_dim),
            nn.LeakyReLU(inplace=True),
            ChannelAttention(bottleneck_dim),
            nn.Conv2d(bottleneck_dim, bottleneck_dim, 3, padding=1),
            nn.BatchNorm2d(bottleneck_dim)
        )

        self.decoder = UNetResDecoder(
            self.encoder,
            num_classes,
            n_conv_per_stage_decoder,
            deep_supervision
        )

    def forward(self, x, return_features=False):
        skips = self.encoder(x)
        bottleneck = skips[-1]


        mamba_features = self.mamba_layer(bottleneck)


        snn_features = self.snn_enhancer(bottleneck)


        if self.use_gates:
            fused_features, gates = self.fusion(mamba_features, snn_features)
        else:
            fused_features = self.fusion(mamba_features, snn_features)


        enhanced = self.post_fusion(fused_features)


        enhanced = enhanced + bottleneck

        skips[-1] = enhanced


        seg_output = self.decoder(skips)

        if return_features:
            return seg_output, mamba_features, snn_features, enhanced

        return seg_output



def create_spiking_umamba(
        input_channels: int = 1,
        num_classes: int = 4,
        image_size: tuple = (224, 224),
        deep_supervision: bool = True,
        base_features: int = 32,
        max_features: int = 320,
        fusion_type: str = 'bidirectional'
):


    min_dim = min(image_size)
    n_stages = min(5, int(np.log2(min_dim)) - 2)


    features_per_stage = [min(base_features * 2 ** i, max_features) for i in range(n_stages)]


    model = SpikingUMamba(
        input_channels=input_channels,
        n_stages=n_stages,
        features_per_stage=features_per_stage,
        conv_op=nn.Conv2d,
        kernel_sizes=[3] * n_stages,
        strides=[1] + [2] * (n_stages - 1),
        n_conv_per_stage=[2] * n_stages,
        num_classes=num_classes,
        n_conv_per_stage_decoder=[2] * (n_stages - 1),
        conv_bias=True,
        norm_op=nn.InstanceNorm2d,
        norm_op_kwargs={'eps': 1e-5, 'affine': True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={'inplace': True},
        deep_supervision=deep_supervision,
        stem_channels=base_features,
        fusion_type=fusion_type
    )

    return model



def create_umamba_bot_2d(
        input_channels: int = 1,
        num_classes: int = 4,
        image_size: tuple = (224, 224),
        deep_supervision: bool = True,
        base_features: int = 32,
        max_features: int = 320
):

    return create_spiking_umamba(
        input_channels=input_channels,
        num_classes=num_classes,
        image_size=image_size,
        deep_supervision=deep_supervision,
        base_features=base_features,
        max_features=max_features,
        fusion_type='bidirectional'
    )