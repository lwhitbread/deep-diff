import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions.normal import Normal
import functools
import inspect
from typing import List, Tuple, Callable

sys.path.append("../")
from src.modules import *

InitType = Callable[
    [
        nn.Module, 
        nn.init.xavier_uniform_, 
        torch.nn.init.xavier_normal_, 
        torch.nn.init.kaiming_uniform_, 
        torch.nn.init.kaiming_normal_, 
        torch.nn.init.zeros_, 
        dict
    ], 
    None,
]

def store_config_args(  
        func: Callable,
    ):
    """
    Class-method decorator that saves every argument provided to the
    function as a dictionary in 'self.config'.
    """

    attrs, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations = inspect.getfullargspec(func)

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        self.config = {}

        # first save the default values
        if defaults:
            for attr, val in zip(
                reversed(attrs), 
                reversed(defaults)
            ):
                self.config[attr] = val

        # next handle positional args
        for attr, val in zip(
            attrs[1:], 
            args
        ):
            self.config[attr] = val

        # lastly handle keyword args
        if kwargs:
            for attr, val in kwargs.items():
                self.config[attr] = val

        return func(self, *args, **kwargs)
    
    return wrapper

class ConditionalFeaturesGenerator(nn.Module):
    
    @store_config_args    
    def __init__(
            self, 
            gen_features: int,
            nb_out_feats: int, 
            nb_dims: int = 2,
            nb_streams: int = 1,
            img_dims: Tuple[int, ...] = (192, 192),
            nb_linear_layers: int = 1,
            nb_blocks: int = 6, 
            transpose_kernel_size: int = 2, 
            stride: int = 2, 
            activation: str = "GELU", 
            normalisation: str = None, 
            dropout: float = 0.0, 
            padding_mode: str = "zeros", 
            bias: bool = True,
    ) -> None:
        super().__init__()
        
        assert nb_dims == len(img_dims), \
            "nb_dims must be equal to the length of warp_spatial_dims"
        assert nb_dims in [2, 3], \
            "nb_dims must be 2 or 3"
        assert nb_streams in [1, 2], \
            "nb_streams must be 1 or 2"
        assert transpose_kernel_size % 2 == 0, \
            "transpose_kernel_size must be even"
        assert transpose_kernel_size == stride, \
            "transpose_kernel_size must be equal to stride"
        for dim in img_dims:
            assert dim % (2 ** nb_blocks) == 0, \
                "warp_spatial_dims must be divisible by 2**nb_blocks"

        self.gen_features = gen_features
        self.nb_out_feats = nb_out_feats
        self.nb_streams = nb_streams
        self.nb_blocks = nb_blocks

        self.kernel_size = transpose_kernel_size
        self.activation = getattr(nn, activation)()
        self.dropout = dropout # not used
        self.padding_mode = padding_mode 
        self.bias = bias

        self.init_conv_feats = nb_out_feats * (2 ** (nb_blocks - 1))
        self.spatial_init = []
        
        self.prior_z = None
        self.prior_x = None
        self.count = 0
        
        for dim in img_dims:
            _spatial = dim / (2 ** nb_blocks)
            assert _spatial.is_integer(), \
                "img_dims must be divisible by 2**nb_blocks"
            self.spatial_init.append(int(_spatial))
        
        self.linear_out_feats = np.prod(self.spatial_init) * self.init_conv_feats

        linear_layers_mean = nn.ModuleList()
        linear_layers_std = nn.ModuleList()
        
        for layer in range(nb_linear_layers):
            
            if layer == 0:
            
                linear_layers_mean.append(
                    nn.Linear(
                        gen_features, 
                        self.linear_out_feats // (nb_linear_layers - layer), 
                        bias = self.bias
                    )
                )
                linear_layers_mean.append(self.activation)
                linear_layers_std.append(
                    nn.Linear(
                        gen_features, 
                        self.linear_out_feats // (nb_linear_layers - layer), 
                        bias = self.bias
                    )
                )
                linear_layers_std.append(self.activation)
            
            else:
            
                linear_layers_mean.append(
                    nn.Linear(
                        self.linear_out_feats // (nb_linear_layers - layer + 1), 
                        self.linear_out_feats // (nb_linear_layers - layer), 
                        bias = self.bias
                    )
                )
                linear_layers_mean.append(self.activation)
                linear_layers_std.append(
                    nn.Linear(
                        self.linear_out_feats // (nb_linear_layers - layer + 1), 
                        self.linear_out_feats // (nb_linear_layers - layer), 
                        bias = self.bias
                    )
                )
                linear_layers_std.append(self.activation)
            
            self.linear_mean = nn.Sequential(*linear_layers_mean)
            self.linear_std = nn.Sequential(*linear_layers_std)
        
        self.streams_wrapper = nn.ModuleList()

        for _ in range(nb_streams):

            nb_feats = self.init_conv_feats
            transpose_conv_layers = nn.ModuleList()

            for block in range(nb_blocks):
                transpose_conv_layers.append(
                    ConvBlock(
                        nb_dims, 
                        nb_feats, 
                        nb_feats // 2 if nb_feats // 2 > nb_out_feats else nb_out_feats, 
                        kernel_size = transpose_kernel_size, 
                        stride = stride, 
                        padding = 0,
                        activation = activation, 
                        normalisation = normalisation, 
                        bias = bias,
                        transpose = True,
                    )
                )
                if block != nb_blocks - 1:
                    nb_feats = nb_feats // 2
                if block == nb_blocks - 1:
                    assert nb_feats == nb_out_feats, \
                        "nb_out_feats must be equal to the number of features in the last block"
            
            self.streams_wrapper.append(nn.Sequential(*transpose_conv_layers))
    
    def forward(
            self, 
            x: torch.Tensor, 
            training: bool = True,
    ) -> torch.Tensor: 

        batch_size = x.shape[0]
        mean = self.linear_mean(x)
        mean = mean.view(
            batch_size, 
            self.init_conv_feats, 
            *self.spatial_init
        )

        std = self.linear_std(x)
        std = std.view(
            batch_size, 
            self.init_conv_feats, 
            *self.spatial_init
        )

        if training:
            eps = torch.randn_like(std)
            z = eps.mul(std).add_(mean)
        else:
            z = mean
                      
        if self.nb_streams == 1:
            return [self.streams_wrapper[0](z)]
        elif self.nb_streams == 2:
            return [self.streams_wrapper[0](z), self.streams_wrapper[1](z)]
            # out = []
            # stream_0 = torch.cuda.Stream()
            # stream_1 = torch.cuda.Stream()
            # torch.cuda.synchronize()
            # with torch.cuda.stream(stream_0):
            #     out.append(self.streams_wrapper[0](z))
            # with torch.cuda.stream(stream_1):
            #     out.append(self.streams_wrapper[1](z))
            # torch.cuda.synchronize()
            # return out

class UNet(nn.Module):
    
    """
    Adapted from https://github.com/voxelmorph/voxelmorph.

        in_shape: tuple
            the shape of the spatial input to the network e.g., (64, 64, 64)
        in_features: int
            the number of input features e.g., 1 for a single channel image
        nb_dims: int
            the number of spatial dimensions of the input
        nb_features: list[list, list]
            the number of features per layer for the encoder and decoder respectively
        nb_levels: int
            the number of levels in the network, used if nb_features is not specified
        nb_convs_per_level: int
            the number of convolutions per level
        feature_multiple: int
            the multiple of features to use for each level
        max_pool: int
            the size of the max pooling kernel
        activation: str
            the activation function to use (i.e., GELU), can use any of the activations in torch.nn 
        normalisation: str
            the normalisation to use (i.e., Instance), can use any of the normalisations in torch.nn
        bias: bool
            whether to use bias in the convolutional layers
        dropout: float
            the dropout probability to use (Not implemented yet)
        upsample_mode: str
            the mode to use for upsampling (i.e., nearest)
    """

    @store_config_args
    def __init__(
            self,
            in_shape: Tuple[int, ...],
            in_features: int,
            nb_dims: int,
            nb_features: List[int] = None,
            nb_levels: int = None,
            nb_convs_per_level: int = 1,
            feature_multiple: int = 1,
            max_pool: int = 2,
            activation: str = "GELU",
            normalisation: str = "Instance",
            bias: bool = True,
            upsample_mode: str = "nearest",
            half_resolution: bool = False,
    ) -> None:
        super().__init__()
        
        assert nb_dims in [2, 3], \
            "nb_dims must be 2 or 3"
        assert nb_dims == len(in_shape), \
            "nb_dims must be equal to the length of in_shape"
        assert activation in ["GELU", "ReLU", "LeakyReLU", "ELU", "SELU", "Tanh", "Sigmoid"], \
            "activation must be one of GELU, ReLU, LeakyReLU, ELU, SELU, Tanh, Sigmoid"
        assert normalisation in ["Instance", "Batch", "Layer", "Group", "None"], \
            "normalisation must be one of Instance, Batch, Layer, Group, None"
        assert upsample_mode in ["nearest", "linear", "bilinear", "trilinear", "area"], \
            "upsample_mode must be one of nearest, linear, bilinear, trilinear, area"
        
        if nb_dims == 2:
            upsample_mode = "bilinear" if upsample_mode == "linear" else upsample_mode
        elif nb_dims == 3:
            upsample_mode = "trilinear" if upsample_mode == "linear" else upsample_mode

        self.half_res = half_resolution

        if nb_features is None:
            nb_features = self.default_features()
        
        if isinstance(nb_features, int):
            assert nb_levels is not None, \
                "nb_levels must be specified if nb_features is an integer"
            
            features = np.round(nb_features * feature_multiple ** np.arange(nb_levels)).astype(int)
            
            nb_features = [
                np.repeat(features[:-1], nb_convs_per_level),
                np.repeat(features[::-1], nb_convs_per_level)
            ]
        
        encoder_feats = nb_features[0]
        nb_decoder_convs = len(encoder_feats)
        additional_convs = nb_features[1][nb_decoder_convs:]
        decoder_feats = nb_features[1][:nb_decoder_convs]
        self.nb_levels = nb_decoder_convs // nb_convs_per_level + 1  

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels
        MaxPool = getattr(nn, f"MaxPool{nb_dims}d")
        self.max_pool = nn.ModuleList([MaxPool(kernel_size = k) for k in max_pool])
        self.up_sample = nn.ModuleList(
            [
                nn.Upsample(
                    scale_factor = k, 
                    mode = upsample_mode,
                    align_corners = None if upsample_mode == "nearest" else True,
                ) for k in max_pool
            ]
        )
    
        #encoder
        nb_feats_in = in_features
        nb_encoder_feats = [nb_feats_in]
        self.encoder = nn.ModuleList()
        for idx_0 in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for idx_1 in range(nb_convs_per_level):
                nb_feats_out = encoder_feats[idx_0 * nb_convs_per_level + idx_1]
                convs.append(
                    ConvBlock(
                        nb_dims, 
                        nb_feats_in, 
                        nb_feats_out, 
                        kernel_size = 3, 
                        stride = 1, 
                        padding = 1,
                        activation = activation, 
                        normalisation = normalisation, 
                        bias = bias,
                        transpose = False,
                    )
                )
                nb_feats_in = nb_feats_out
            
            nb_encoder_feats.append(nb_feats_in)
            self.encoder.append(convs)

        # decoder
        reversed_nb_encoder_feats = nb_encoder_feats[::-1]
        self.decoder = nn.ModuleList()
        for idx_0 in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for idx_1 in range(nb_convs_per_level):
                nb_feats_out = decoder_feats[idx_0 * nb_convs_per_level + idx_1]
                convs.append(
                    ConvBlock(
                        nb_dims,
                        nb_feats_in,
                        nb_feats_out,
                        kernel_size = 3,
                        stride = 1,
                        padding = 1,
                        activation = activation,
                        normalisation = normalisation,
                        bias = bias,
                        transpose = False
                    )
                )
                nb_feats_in = nb_feats_out
            
            self.decoder.append(convs)
            if not half_resolution or idx_0 < self.nb_levels - 2:
                nb_feats_in = nb_feats_in + reversed_nb_encoder_feats[idx_0]
    
        # additional convs
        self.additional_convs = nn.ModuleList()
        for idx_0 in range(len(additional_convs)):
            nb_feats_out = additional_convs[idx_0]
            self.additional_convs.append(
                ConvBlock(
                    nb_dims,
                    nb_feats_in,
                    nb_feats_out,
                    kernel_size = 3,
                    stride = 1,
                    padding = 1,
                    activation = activation,
                    normalisation = normalisation,
                    bias = bias,
                    transpose = False
                )
            )
            nb_feats_in = nb_feats_out
        
        self.final_nb_feats = nb_feats_in

    
    def forward(
            self, 
            x: torch.Tensor,
    ) -> torch.Tensor:

        history = [x]

        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            history.append(x)
            x = self.max_pool[level](x)
        
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < self.nb_levels - 2:
                x = self.up_sample[level](x)
                x = torch.cat([x, history.pop()], dim = 1)

        for conv in self.additional_convs:
            x = conv(x)
        
        return x
    
    @staticmethod
    def default_features(
    ) -> List[List[int]]:
        nb_features = [
            [16, 32, 32, 32],             # encoder
            [32, 32, 32, 32, 32, 16, 16]  # decoder
        ]
        return nb_features

class RegNet(nn.Module):
    """ 
    Registration network
    Adapted from https://github.com/voxelmorph/voxelmorph
    """

    @store_config_args
    def __init__(
            self,
            in_shape: Tuple[int, ...],
            nb_dims: int,
            nb_unet_features: List[List[int]] = None,
            nb_unet_levels: int = None,
            nb_unet_convs_per_level: int = 1,
            unet_feature_multiple: int = 1,
            unet_max_pool: int = 2,
            activation: str = "GELU",
            normalisation: str = "Instance",
            bias: bool = True,
            unet_upsample_mode: str = "linear",
            unet_half_resolution: bool = False,
            int_steps: int = 7,
            int_downsize: int = 1,
            int_smoothing_args: dict = dict(use_smoothing = False, kernel_size = 3, sigma = 1.0),
            bidirectional: bool = False,
            source_feats: int = 1,
            target_feats: int = 1,
            use_tanh_vecint_act: bool = False,
            # use_cortical_mask: bool = False,
            nb_field_layers: int = 3,
    ) -> None: 
        super().__init__()

        assert nb_dims in [2, 3], \
            "nb_dims must be 2 or 3"
        
        assert nb_dims == len(in_shape), \
            "nb_dims must be equal to the length of in_shape"
        
        assert activation in ["GELU", "ReLU", "LeakyReLU", "ELU", "SELU", "Tanh", "Sigmoid"], \
            "activation must be one of GELU, ReLU, LeakyReLU, ELU, SELU, Tanh, Sigmoid"
        
        assert normalisation in ["Instance", "Batch", "Layer", "Group", "None"], \
            "normalisation must be one of Instance, Batch, Layer, Group, None"
        
        assert unet_upsample_mode in ["nearest", "linear", "bilinear", "trilinear", "area"], \
            "upsample_mode must be one of nearest, linear, bilinear, trilinear, area"
        
        assert isinstance(int_smoothing_args, dict), \
            "int_smoothing_args must be a dict"

        # self.use_cortical_mask = use_cortical_mask
        
        self.unet = UNet(
            in_shape = in_shape,
            in_features = source_feats + target_feats,
            nb_dims = nb_dims,
            nb_features = nb_unet_features,
            nb_levels = nb_unet_levels,
            nb_convs_per_level = nb_unet_convs_per_level,
            feature_multiple = unet_feature_multiple,
            max_pool = unet_max_pool,
            activation = activation,
            normalisation = normalisation,
            bias = bias,
            upsample_mode = unet_upsample_mode,
            half_resolution = unet_half_resolution,
        )
        
        self.flow_field = CreateField(
            dec_feats = self.unet.final_nb_feats, 
            nb_dims = nb_dims,
            nb_layers = nb_field_layers,
        )

        if not unet_half_resolution and int_steps > 0 and int_downsize > 1:
            self.downsize = ResizeTransform(
                vel_resize = int_downsize, 
                nb_dims = nb_dims,
                out_shape = None,
            )
        else:
            self.downsize = None
        
        if int_steps > 0 and int_downsize > 1:
            self.upsize = ResizeTransform(
                vel_resize = 1 / int_downsize, 
                nb_dims = nb_dims,
                out_shape = in_shape,
            )
        else:
            self.upsize = None
        
        self.bidirectional = bidirectional
        self.int_smoothing_args = int_smoothing_args

        downsize_shape = tuple([int(s / int_downsize) for s in in_shape])
        
        self.integrator = IntegrationLayer(
            in_shape = downsize_shape,
            nb_steps = int_steps,
            kernel_size = int_smoothing_args["kernel_size"],
            sigma = int_smoothing_args["sigma"],
            use_tanh_act = use_tanh_vecint_act,
        ) if int_steps > 0 else None

        self.spatial_transformer = SpatialTransformer(in_shape)

        # if self.use_cortical_mask:
        #     self.spatial_transformer_mask = SpatialTransformer(in_shape, "nearest")
    
    def forward(
            self, 
            source: torch.Tensor, 
            target: torch.Tensor,
            # cortical_mask: torch.Tensor = None,
    ) -> tuple:

        x = torch.cat([source, target], dim = 1)
        x = self.unet(x)
        flow = self.flow_field(x)

        pos_flow = flow
        if self.downsize is not None:
            pos_flow = self.downsize(pos_flow)
        
        neg_flow = -pos_flow if self.bidirectional else None

        if self.integrator is not None:
            pos_flow = self.integrator(
                pos_flow,
                smoothing = self.int_smoothing_args["use_smoothing"],
            )
            neg_flow = self.integrator(
                neg_flow,
                smoothing = self.int_smoothing_args["use_smoothing"],
            ) if self.bidirectional else None

        if self.upsize is not None:
            pos_flow = self.upsize(pos_flow)
            neg_flow = self.upsize(neg_flow) if self.bidirectional else None
        
        y_source = self.spatial_transformer(
            source, 
            pos_flow
        )
        y_target = self.spatial_transformer(
            target, 
            neg_flow
        ) if self.bidirectional else None

        # if self.use_cortical_mask:
        #     assert cortical_mask is not None, \
        #         "cortical_mask must be provided if use_cortical_mask is True"
        #     cortical_mask = self.spatial_transformer_mask(
        #         cortical_mask,
        #         pos_flow,
        #     )

        return (
            y_source, 
            y_target, 
            pos_flow,
            neg_flow,
            None, # used previously when experimenting with a cortical mask
            flow,
        ) if self.bidirectional else (
            y_source, 
            pos_flow,
            None, # used previously when experimenting with a cortical mask
            flow,
        )

class IntensityUNet(nn.Module):
    """ 
    IntensityUNet is a U-Net based intensity normalisation network.
    """

    @store_config_args
    def __init__(
            self,
            in_shape: Tuple[int, ...],
            nb_dims: int,
            nb_unet_features: List[List[int]] = None,
            nb_unet_levels: int = None,
            nb_unet_convs_per_level: int = 1,
            unet_feature_multiple: int = 1,
            unet_max_pool: int = 2,
            activation: str = "GELU",
            normalisation: str = "Instance",
            bias: bool = True,
            unet_upsample_mode: str = "linear",
            unet_half_resolution: bool = False,
            int_downsize: int = 2,
            input_feats: int = None,
            use_flow: bool = False,
            intensity_act = "exp",
            intensity_field_act = "Sigmoid",
            intensity_field_act_mult = 3.,
            nb_field_layers: int = 3,
    ) -> None: 
        super().__init__()

        assert nb_dims in [2, 3], \
            "nb_dims must be 2 or 3"
        assert nb_dims == len(in_shape), \
            "nb_dims must be equal to the length of in_shape"
        if unet_upsample_mode == "linear":
            unet_upsample_mode = "bilinear" if nb_dims == 2 else "trilinear"

        assert activation in ["GELU", "ReLU", "LeakyReLU", "ELU", "SELU", "Tanh", "Sigmoid"], \
            "activation must be one of GELU, ReLU, LeakyReLU, ELU, SELU, Tanh, Sigmoid"
        assert normalisation in ["Instance", "Batch", "Layer", "Group", "None"], \
            "normalisation must be one of Instance, Batch, Layer, Group, None"
        assert unet_upsample_mode in ["nearest", "linear", "bilinear", "trilinear", "area"], \
            "upsample_mode must be one of nearest, linear, bilinear, trilinear, area"

        if input_feats is None:
            input_feats = 2
        if use_flow:
            input_feats = input_feats + nb_dims
        
        self.unet = UNet(
            in_shape = in_shape,
            in_features = input_feats,
            nb_dims = nb_dims,
            nb_features = nb_unet_features,
            nb_levels = nb_unet_levels,
            nb_convs_per_level = nb_unet_convs_per_level,
            feature_multiple = unet_feature_multiple,
            max_pool = unet_max_pool,
            activation = activation,
            normalisation = normalisation,
            bias = bias,
            upsample_mode = unet_upsample_mode,
            half_resolution = unet_half_resolution,
        )
        
        self.intensity_field = CreateField(
            dec_feats = self.unet.final_nb_feats, 
            nb_dims = nb_dims,
            intensity_modulation = True,
            intensity_field_act = intensity_field_act,
            intensity_field_act_mult = intensity_field_act_mult,
            nb_layers = nb_field_layers,
        )

        self.intensity = IntensityModulationBlock(
            in_shape = in_shape,
            nb_dims = nb_dims,            
            intensity_act = intensity_act,
            int_downsize = int_downsize,
        )
    
    def freeze_weights(
        self,
        freeze: bool,
    ) -> None:
        for param in self.unet.parameters():
            param.requires_grad = not freeze
        for param in self.intensity_field.parameters():
            param.requires_grad = not freeze

        if freeze:
            print("Freezing weights of IntensityUNet")
    
    def forward(
            self, 
            source: torch.Tensor, 
            target: torch.Tensor,
            field_multiplier: float = 1.,
            displacement: torch.Tensor = None,
    ) -> tuple:
        
        if displacement is None:
            x = torch.cat([source, target], dim = 1)
        else:
            x = torch.cat([source, target, displacement], dim = 1)
        int_field = self.unet(x)
        int_field = self.intensity_field(int_field, field_multiplier) # change to int_field after debugging
        x, int_modulation = self.intensity(source, int_field, field_multiplier)

        return x, int_field, int_modulation


class Warp(nn.Module):

    @store_config_args
    def __init__(
            self, 
            nb_dims: int = 2,
            gen_features: int = 1,
            gen_features_linear_layers: int = 1,
            gen_features_blocks: int = 6,
            nb_gen_features_streams: int = 1,
            img_dims: Tuple[int, ...] = (192, 192),
            int_steps: int = 7,
            interp_mode: str = "linear",
            nb_unet_features_reg: List[List[int]] = [
                [16, 32, 32, 32], # encoder
                [32, 32, 32, 32, 32, 16, 16]  # decoder
            ],
            unet_upsample_mode: str = "linear",
            nb_unet_features_int: List[List[int]] = [
                [16, 32, 32, 32], # encoder
                [32, 32, 32, 32, 32, 16, 16]  # decoder
            ],
            int_smoothing_args_st_1: dict = dict(
                use_smoothing = False, 
                kernel_size = 3, 
                sigma = 1.0
            ),
            int_smoothing_args_st_2: dict = dict(
                use_smoothing = False,
                kernel_size = 3,
                sigma = 1.0,
            ),
            int_smoothing_args_mean_trackers: dict = dict(
                use_smoothing = False,
                kernel_size = 3,
                sigma = 1.0,
            ),
            intensity_field_act = "Sigmoid",
            intensity_field_act_mult = 3.,
            intensity_act = "exp",
            downsize_factor_intensity_st_1: float = 2,
            downsize_factor_intensity_st_2: float = 2,
            downsize_factor_vec_st_1: float = 2,
            downsize_factor_vec_st_2: float = 1,
            device: str = "cpu",
            use_tanh_vecint_act: bool = False,
            zero_mean_cons: bool = False,
            adj_warps: bool = False,
            lambda_rate: float = 0.9,
            min_max_args: dict = None,
            # use_cortical_mask: bool = False,
            nb_field_layers: int = 3,
            pred_arbitrary_cond_temps: bool = False,
    ) -> None:
        super().__init__()
        
        self.nb_dims = nb_dims
        self.gen_features = gen_features
        self.nb_gen_features_streams = nb_gen_features_streams
        self.gen_features_linear_layers = gen_features_linear_layers
        self.gen_features_blocks = gen_features_blocks
        self.img_dims = img_dims
        self.int_steps = int_steps
        self.interp_mode = interp_mode
        self.nb_unet_features_reg = nb_unet_features_reg
        self.nb_unet_features_int = nb_unet_features_int
        self.int_smoothing_args_st_1 = int_smoothing_args_st_1
        self.int_smoothing_args_st_2 = int_smoothing_args_st_2
        self.int_smoothing_args_mean_trackers = int_smoothing_args_mean_trackers
        self.intensity_act = intensity_act
        self.device = device
        self.zero_mean_cons = zero_mean_cons
        self.adj_warps = adj_warps
        # self.use_cortical_mask = use_cortical_mask
        
        self.gen_features = ConditionalFeaturesGenerator(
            gen_features = self.gen_features, 
            nb_out_feats = 8,
            nb_streams = self.nb_gen_features_streams,
            nb_dims = self.nb_dims,
            img_dims = self.img_dims,
            nb_linear_layers = self.gen_features_linear_layers, 
            nb_blocks = self.gen_features_blocks, 
            normalisation = "Instance",
        )

        self.gen_cond_temp_field = CreateField(
            dec_feats = self.gen_features.nb_out_feats, 
            nb_dims = self.nb_dims,
            nb_layers = nb_field_layers,
        )
        
        if self.int_steps > 0 and downsize_factor_vec_st_1 > 1:
            self.vec_downsize = ResizeTransform(
                vel_resize = downsize_factor_vec_st_1, 
                nb_dims = self.nb_dims,
                out_shape = None,
            )
            self.vec_upsize = ResizeTransform(
                vel_resize = 1 / downsize_factor_vec_st_1, 
                nb_dims = self.nb_dims,
                out_shape = self.img_dims,
            )
            self.integrator_in_shape = tuple([int(s / downsize_factor_vec_st_1) for s in self.img_dims])
        else:
            self.vec_downsize = None
            self.vec_upsize = None
            self.integrator_in_shape = self.img_dims

        self.integrate_flow = IntegrationLayer(
            self.integrator_in_shape, 
            self.int_steps, 
            kernel_size = self.int_smoothing_args_st_1["kernel_size"], 
            sigma = self.int_smoothing_args_st_1["sigma"],
            use_tanh_act = use_tanh_vecint_act,
        )

        if self.int_smoothing_args_mean_trackers is not None:
            self.integrate_mean_tracker_flow = IntegrationLayer(
                self.integrator_in_shape, 
                self.int_steps, 
                kernel_size = self.int_smoothing_args_mean_trackers["kernel_size"], 
                sigma = self.int_smoothing_args_mean_trackers["sigma"],
                use_tanh_act = use_tanh_vecint_act,
            )
        else:
            self.integrate_mean_tracker_flow = None

        self.warp = SpatialTransformer(
            self.img_dims, 
            self.interp_mode,
        )

        # if self.use_cortical_mask:
        #     self.warp_cortical_mask = SpatialTransformer(
        #         self.img_dims, 
        #         "nearest",
        #     )

        self.cond_temp_intensity_field = CreateField(
            dec_feats = self.gen_features.nb_out_feats,
            nb_dims = len(self.img_dims),
            intensity_modulation = True,
            intensity_field_act = intensity_field_act,
            intensity_field_act_mult = intensity_field_act_mult,
            nb_layers = nb_field_layers,
        )

        self.cond_temp_intensity = IntensityModulationBlock(
            in_shape = self.img_dims,
            nb_dims = self.nb_dims,
            intensity_act = self.intensity_act,
            int_downsize = downsize_factor_intensity_st_1,
        )
                  
        self.reg = RegNet(
            self.img_dims, 
            self.nb_dims, 
            nb_unet_features = self.nb_unet_features_reg, 
            int_steps = self.int_steps, 
            int_smoothing_args = self.int_smoothing_args_st_2,
            int_downsize = downsize_factor_vec_st_2,
            use_tanh_vecint_act = use_tanh_vecint_act,
            # use_cortical_mask = self.use_cortical_mask,
            bidirectional = True,
            nb_field_layers = nb_field_layers,
            unet_upsample_mode = unet_upsample_mode,
        )

        self.intensity_net_phase_2 = IntensityUNet(
            self.img_dims, 
            self.nb_dims, 
            nb_unet_features = self.nb_unet_features_int,
            use_flow = False, 
            intensity_act = self.intensity_act,
            int_downsize = downsize_factor_intensity_st_2,
            intensity_field_act = intensity_field_act,
            intensity_field_act_mult = intensity_field_act_mult,
            nb_field_layers = nb_field_layers,
            unet_upsample_mode = unet_upsample_mode,
        )

        if self.zero_mean_cons:
            self.mean_trackers = AllMeansTracker(
                img_dims = self.img_dims,
                nb_dims = self.nb_dims,
                nb_mean_fields = min_max_args["nb_means"],
                device = self.device,
                lambda_rate = lambda_rate,
            )

        self.summary_mode_flag = False

        self.prior_field = None
        self.prior_params = None

        self.pred_arbitrary_cond_temps = pred_arbitrary_cond_temps

        self.initialise_parameters()

    @staticmethod
    def weight_init(
            module: nn.Module, 
            method: InitType, 
            **kwargs,
    ) -> None:
        
        if isinstance(
            module, 
            (
                nn.Conv3d, 
                nn.Conv2d, 
                nn.ConvTranspose3d, 
                nn.ConvTranspose2d, 
                nn.Linear,
            )
        ):
            method(module.weight, **kwargs)

    @staticmethod
    def bias_init(
            module: nn.Module, 
            method: InitType, 
            **kwargs
    ) -> None:
        
        if isinstance(
            module, 
            (
                nn.Conv3d, 
                nn.Conv2d, 
                nn.ConvTranspose3d, 
                nn.ConvTranspose2d, 
                nn.Linear,
            )
        ):
            if module.bias is not None:
                method(module.bias, **kwargs)
    
    def initialise_parameters(
            self,
            method_weights: InitType = nn.init.kaiming_uniform_,
            method_bias: InitType = nn.init.zeros_,
            kwargs_weights = {},
            kwargs_bias = {},
    ) -> None:
        
        for module in self.modules():
            self.weight_init(
                module, 
                method_weights, 
                **kwargs_weights,
            )
            self.bias_init(
                module, 
                method_bias, 
                **kwargs_bias,
            )
        
        # Initialise flow weights
        self.gen_cond_temp_field.conv_layers[-1].conv.weight = nn.Parameter(
            Normal(0, 1e-5).sample(
                self.gen_cond_temp_field.conv_layers[-1].conv.weight.shape
            )
        )
        self.gen_cond_temp_field.conv_layers[-1].conv.bias = nn.Parameter(
            torch.zeros(
                self.gen_cond_temp_field.conv_layers[-1].conv.bias.shape
            )
        )

        self.cond_temp_intensity_field.conv_layers[-1].conv.weight = nn.Parameter(
            Normal(0, 1e-5).sample(
                self.cond_temp_intensity_field.conv_layers[-1].conv.weight.shape
            )
        )

        self.cond_temp_intensity_field.conv_layers[-1].conv.bias = nn.Parameter(
            torch.zeros(
                self.cond_temp_intensity_field.conv_layers[-1].conv.bias.shape
            )
        )
        
        self.reg.flow_field.conv_layers[-1].conv.weight = nn.Parameter(
            Normal(0, 1e-5).sample(
                self.reg.flow_field.conv_layers[-1].conv.weight.shape
            )
        )
        self.reg.flow_field.conv_layers[-1].conv.bias = nn.Parameter(
            torch.zeros(
                self.reg.flow_field.conv_layers[-1].conv.bias.shape
            )
        )

        self.intensity_net_phase_2.intensity_field.conv_layers[-1].conv.weight = nn.Parameter(
            Normal(0, 1e-5).sample(
                self.intensity_net_phase_2.intensity_field.conv_layers[-1].conv.weight.shape
            )
        )

        self.intensity_net_phase_2.intensity_field.conv_layers[-1].conv.bias = nn.Parameter(
            torch.zeros(
                self.intensity_net_phase_2.intensity_field.conv_layers[-1].conv.bias.shape
            )
        )

    def freeze_intensity_weights(
        self,
        freeze_st_1: bool,
        freeze_st_2: bool,
    ) -> None:
        
        self.cond_temp_intensity_field.freeze_weights(freeze_st_1)
        self.intensity_net_phase_2.freeze_weights(freeze_st_2)
    
    def summary_mode(
        self,
        summary_mode: bool = False,
    ) -> None:
        
        self.summary_mode_flag: bool = summary_mode

    def forward(
            self, 
            params: torch.Tensor, 
            template: torch.Tensor = None,
            target: torch.Tensor = None,
            training: bool = True,
            intensity_field_multiplier_st_1: float = 1.0,
            intensity_field_multiplier_st_2: float = 1.0,
            # epoch: int = None,
            means_idx_0: torch.Tensor = None,
            prop_means_idx_0: torch.Tensor = None,
            # cortical_mask: torch.Tensor = None,
    ) -> list:
        
        # generate features     
        gen_features = self.gen_features(
            params,
            training = training,
        )

        flow_st_1 = self.gen_cond_temp_field(
            gen_features[0],
        )

        # if self.mean_outside_loop or not self.zero_mean_cons:

        cond_temp_warp = flow_st_1
        neg_cond_temp_warp = -cond_temp_warp

        if self.vec_downsize is not None:
            cond_temp_warp = self.vec_downsize(
                cond_temp_warp,
            )

            neg_cond_temp_warp = self.vec_downsize(
                neg_cond_temp_warp,
            )
        
        cond_temp_warp = self.integrate_flow(
            cond_temp_warp, 
            smoothing = self.int_smoothing_args_st_1["use_smoothing"],
        )

        neg_cond_temp_warp = self.integrate_flow(
            neg_cond_temp_warp, 
            smoothing = self.int_smoothing_args_st_1["use_smoothing"],
        )

        if self.vec_upsize is not None:
            cond_temp_warp = self.vec_upsize(
                cond_temp_warp,
            )

            neg_cond_temp_warp = self.vec_upsize(
                neg_cond_temp_warp,
            )

        _cond_temp = self.warp(
            template, 
            cond_temp_warp,
        )

        # if self.use_cortical_mask:
        #     assert cortical_mask is not None, \
        #         "Cortical mask is None, but use_cortical_mask is True"
            
        #     cortical_mask = self.warp_cortical_mask(
        #         cortical_mask, 
        #         cond_temp_warp,
        #     ) 

        if self.zero_mean_cons:
            
            _mean_fields_idx_0 = self.mean_trackers.get_mean_field(means_idx_0)
            _mean_fields_idx_1 = self.mean_trackers.get_mean_field(means_idx_0 + 1)
            
            _mean_fields_idx_0 = torch.einsum("i,ijklm->ijklm", prop_means_idx_0, _mean_fields_idx_0)
            _mean_fields_idx_1 = torch.einsum("i,ijklm->ijklm", 1 - prop_means_idx_0, _mean_fields_idx_1)
            
            _mean_fields = _mean_fields_idx_0 + _mean_fields_idx_1
            cond_temp_warp_unbiased = flow_st_1 + _mean_fields
            # cond_temp_warp_unbiased = flow_st_1 - _mean_fields

            if self.vec_downsize is not None:
                cond_temp_warp_unbiased = self.vec_downsize(
                    cond_temp_warp_unbiased,
                )
            
            if self.integrate_mean_tracker_flow is not None:
                cond_temp_warp_unbiased = self.integrate_mean_tracker_flow(
                    cond_temp_warp_unbiased, 
                    smoothing = self.int_smoothing_args_mean_trackers["use_smoothing"],
                )

            else:
                cond_temp_warp_unbiased = self.integrate_flow(
                    cond_temp_warp_unbiased, 
                    smoothing = self.int_smoothing_args_st_1["use_smoothing"],
                )

            if self.vec_upsize is not None:
                cond_temp_warp_unbiased = self.vec_upsize(
                    cond_temp_warp_unbiased,
                )

            if self.adj_warps:
                # if self.mean_outside_loop:
                _cond_temp_unbiased = self.warp(
                    template, 
                    cond_temp_warp_unbiased,
                )

        # if self.use_cortical_mask:
        #     if not self.zero_mean_cons:
        #         pass
        #     else:   
        #         assert cortical_mask is not None, \
        #             "Cortical mask is None, but use_cortical_mask is True"
                
        #         cortical_mask = self.warp_cortical_mask(
        #             cortical_mask, 
        #             cond_temp_warp,
        #         ) 

        intensity_field_st_1 = self.cond_temp_intensity_field(
            gen_features[0] if self.nb_gen_features_streams == 1 else gen_features[1],
            intensity_field_multiplier_st_1,
        )
        
        cond_temp, intensity_modulation_st_1 = self.cond_temp_intensity(
           _cond_temp,
           intensity_field_st_1,
           intensity_field_multiplier_st_1, 
        )

        if self.pred_arbitrary_cond_temps:

            return [
                cond_temp,
                cond_temp_warp,
                intensity_field_st_1,
                _cond_temp,
                _cond_temp_unbiased,
                neg_cond_temp_warp,
                flow_st_1,
                _mean_fields,
                None, # used previously when experimenting with a cortical mask
                intensity_modulation_st_1,
            ]

        _x, y, pos_flow, neg_flow, _, flow_st_2 = self.reg(
            cond_temp,
            target,
            # cortical_mask = cortical_mask if self.use_cortical_mask else None,
        )

        x, intensity_field_st_2, intensity_modulation_st_2 = self.intensity_net_phase_2(
            _x,
            target,
            intensity_field_multiplier_st_2,
        )

        if self.zero_mean_cons and training:
            self.mean_trackers.forward(
                flow_st_2.clone().detach(), 
                means_idx_0, 
                prop_means_idx_0,
            )
        
        if not self.zero_mean_cons:
            return [
                x,
                cond_temp_warp,
                pos_flow,
                intensity_field_st_1,
                intensity_field_st_2,
                _cond_temp,
                _x,
                cond_temp,
                neg_cond_temp_warp,
                flow_st_1,
                None, # used previously when experimenting with a cortical mask
                intensity_modulation_st_1,
                intensity_modulation_st_2,
            ]
        else:
            return [
                x,
                cond_temp_warp,
                pos_flow,
                intensity_field_st_1,
                intensity_field_st_2,
                _cond_temp,
                _x,
                cond_temp,
                _cond_temp_unbiased,
                neg_cond_temp_warp,
                flow_st_1,
                _mean_fields,
                None,  # used previously when experimenting with a cortical mask 
                intensity_modulation_st_1,
                intensity_modulation_st_2,
            ]

# TODO: build inverse warp map model to train with warps from st 1, and trained against st 1 flow
# TODO: Model is to ingest st 1 cond warps and predict st 1 flow, using trained warps and flow from st 1
# TODO: Model is to be trained with st 1 flow as target
# TODO: Model will then be frozen, and will move st 2 warps, tracked in mean trackers back to st 1 velocity field space using inverse warp map
# TODO: We can then differentially use smoothing in st 1 and st 2
# TODO: Model is to be trained with the same smoothing hyperparams as st 1, and then frozen
        
            # nb_dims: int = 2,
            # gen_features: int = 1,
            # gen_features_linear_layers: int = 1,
            # gen_features_blocks: int = 6,
            # nb_gen_features_streams: int = 1,
            # img_dims: Tuple[int, ...] = (192, 192),
            # int_steps: int = 7,
            # interp_mode: str = "linear",

            # #gen_intensity_blocks: int = 6,
            # nb_unet_features_reg: List[List[int]] = [
            #     [16, 32, 32, 32], # encoder
            #     [32, 32, 32, 32, 32, 16, 16]  # decoder
            # ],
            # nb_unet_features_int: List[List[int]] = [
            #     [16, 32, 32, 32], # encoder
            #     [32, 32, 32, 32, 32, 16, 16]  # decoder
            # ],
            # int_smoothing_args_st_1: dict = dict(
            #     use_smoothing = False, 
            #     kernel_size = 3, 
            #     sigma = 1.0
            # ),
            # int_smoothing_args_st_2: dict = dict(
            #     use_smoothing = False,
            #     kernel_size = 3,
            #     sigma = 1.0,
            # ),
            # int_smoothing_args_mean_trackers: dict = dict(
            #     use_smoothing = False,
            #     kernel_size = 3,
            #     sigma = 1.0,
            # ),
            # intensity_field_act = "Sigmoid",
            # intensity_field_act_mult = 3.,
            # intensity_act = "exp",
            # downsize_factor_intensity_st_1: float = 2,
            # downsize_factor_intensity_st_2: float = 2,
            # downsize_factor_vec_st_1: float = 2,
            # downsize_factor_vec_st_2: float = 1,
            # device: str = "cpu",
            # use_tanh_vecint_act: bool = False,
            # zero_mean_cons: bool = False,
            # adj_warps: bool = False,
            # prior_discount_factor: float = 0.2,
            # mean_tracker_max_weight: float = 10.,
            # use_cum_tracker: bool = False,
            # use_lambda_update_mean_tracker: bool = False,
            # lambda_rate: float = 0.9,
            # mean_outside_loop: bool = False,
            # min_max_args: dict = None,
            # use_cortical_mask: bool = False,
            # nb_field_layers: int = 3,
            # pred_arbitrary_cond_temps: bool = False,

# class InverseWarpMap(nn.Module):

#     @store_config_args
#     def __init__(
#             self,
#             img_dims: Tuple[int, ...],
#             nb_dims: int,
#             int_steps: int = 7,
#             interp_mode: str = "linear",
#             nb_unet_features: List[List[int]] = [
#                 [16, 32, 32, 32], # encoder
#                 [32, 32, 32, 32, 32, 16, 16]  # decoder
#             ],
#             int_smoothing_args: dict = dict(
#                 use_smoothing = False,
#                 kernel_size = 3,
#                 sigma = 1.0,
#             ),
#             use_tanh_vecint_act: bool = False,
#             downsize_factor: float = 1.05,
#             device: str = "cpu",
#             nb_field_layers: int = 2,
#     ) -> None:
#         super().__init__()

#         self.img_dims = img_dims
#         self.nb_dims = nb_dims
#         self.int_steps = int_steps
#         self.interp_mode = interp_mode
#         self.nb_unet_features = nb_unet_features
#         self.int_smoothing_args = int_smoothing_args
#         self.use_tanh_vecint_act = use_tanh_vecint_act
#         self.downsize_factor = downsize_factor
#         self.device = device
        

# # class UNet(nn.Module):
    
# #     """
# #     Adapted from https://github.com/voxelmorph/voxelmorph.

# #         in_shape: tuple
# #             the shape of the spatial input to the network e.g., (64, 64, 64)
# #         in_features: int
# #             the number of input features e.g., 1 for a single channel image
# #         nb_dims: int
# #             the number of spatial dimensions of the input
# #         nb_features: list[list, list]
# #             the number of features per layer for the encoder and decoder respectively
# #         nb_levels: int
# #             the number of levels in the network, used if nb_features is not specified
# #         nb_convs_per_level: int
# #             the number of convolutions per level
# #         feature_multiple: int
# #             the multiple of features to use for each level
# #         max_pool: int
# #             the size of the max pooling kernel
# #         activation: str
# #             the activation function to use (i.e., GELU), can use any of the activations in torch.nn 
# #         normalisation: str
# #             the normalisation to use (i.e., Instance), can use any of the normalisations in torch.nn
# #         bias: bool
# #             whether to use bias in the convolutional layers
# #         dropout: float
# #             the dropout probability to use (Not implemented yet)
# #         upsample_mode: str
# #             the mode to use for upsampling (i.e., nearest)
# #     """

# #     @store_config_args
# #     def __init__(
# #             self,
# #             in_shape: Tuple[int, ...],
# #             in_features: int,
# #             nb_dims: int,
# #             nb_features: List[int] = None,
# #             nb_levels: int = None,
# #             nb_convs_per_level: int = 1,
# #             feature_multiple: int = 1,
# #             max_pool: int = 2,
# #             activation: str = "GELU",
# #             normalisation: str = "Instance",
# #             bias: bool = True,
# #             upsample_mode: str = "nearest",
# #             half_resolution: bool = False,


#         self.inverse_warp_map = UNet(
#             in_shape = img_dims,
#             in_features = 3, # 3 features for i,j,k for 3 dim warp fields
#             nb_dims = nb_dims,
#             nb_features = nb_unet_features,
#             nb_levels = 4,
#             nb_convs_per_level = 2,
#             feature_multiple = 1,
#             max_pool = 2,
#             activation = "GELU",
#             normalisation = "Instance",
#             bias = True,
#             upsample_mode = interp_mode,
#             half_resolution = False,
#         )

#         self.create_field = CreateField(
#             dec_feats = self.inverse_warp_map.final_nb_feats, 
#             nb_dims = nb_dims,
#             nb_layers = nb_field_layers,
#         )

#         if self.int_steps > 0 and downsize_factor > 1:
#             self.vec_downsize = ResizeTransform(
#                 vel_resize = downsize_factor, 
#                 nb_dims = self.nb_dims,
#                 out_shape = None,
#             )
#             self.vec_upsize = ResizeTransform(
#                 vel_resize = 1 / downsize_factor, 
#                 nb_dims = self.nb_dims,
#                 out_shape = self.img_dims,
#             )
#             self.integrator_in_shape = tuple([int(s / downsize_factor) for s in self.img_dims])

#         else:
#             self.vec_downsize = None
#             self.vec_upsize = None
#             self.integrator_in_shape = self.img_dims
        
#         self.integrate_flow = IntegrationLayer(
#             self.integrator_in_shape, 
#             self.int_steps, 
#             kernel_size = self.int_smoothing_args["kernel_size"], 
#             sigma = self.int_smoothing_args["sigma"],
#             use_tanh_act = use_tanh_vecint_act,
#         )

#         self.warp = SpatialTransformer(
#             self.img_dims, 
#             self.interp_mode,
#         )

#         self.initialise_parameters()

#     @staticmethod
#     def weight_init(
#             module: nn.Module, 
#             method: InitType, 
#             **kwargs,
#     ) -> None:
        
#         if isinstance(
#             module, 
#             (
#                 nn.Conv3d, 
#                 nn.Conv2d, 
#                 nn.ConvTranspose3d, 
#                 nn.ConvTranspose2d, 
#                 nn.Linear,
#             )
#         ):
#             method(module.weight, **kwargs)

#     @staticmethod
#     def bias_init(
#             module: nn.Module, 
#             method: InitType, 
#             **kwargs
#     ) -> None:
        
#         if isinstance(
#             module, 
#             (
#                 nn.Conv3d, 
#                 nn.Conv2d, 
#                 nn.ConvTranspose3d, 
#                 nn.ConvTranspose2d, 
#                 nn.Linear,
#             )
#         ):
#             if module.bias is not None:
#                 method(module.bias, **kwargs)
    
#     def initialise_parameters(
#             self,
#             method_weights: InitType = nn.init.kaiming_uniform_,
#             method_bias: InitType = nn.init.zeros_,
#             kwargs_weights = {},
#             kwargs_bias = {},
#     ) -> None:
        
#         for module in self.modules():
#             self.weight_init(
#                 module, 
#                 method_weights, 
#                 **kwargs_weights,
#             )
#             self.bias_init(
#                 module, 
#                 method_bias, 
#                 **kwargs_bias,
#             )
        
#         # Initialise flow weights
#         self.create_field.conv_layers[-1].conv.weight = nn.Parameter(
#             Normal(0, 1e-5).sample(
#                 self.create_field.conv_layers[-1].conv.weight.shape
#             )
#         )
#         self.create_field.conv_layers[-1].conv.bias = nn.Parameter(
#             torch.zeros(
#                 self.create_field.conv_layers[-1].conv.bias.shape
#             )
#         )
    
#     def forward(
#             self, 
#             warp: torch.Tensor, 
#             # flow: torch.Tensor,
#     ) -> torch.Tensor:
        
#         # predict flow from deformation field
#         flow = self.inverse_warp_map(warp)
#         flow = self.create_field(flow)

#         # for evaluation, integrate the predicted flow to get the predicted warp
#         # to compare with the ground truth warp
#         if self.int_steps > 0 and self.downsize_factor > 1:
#             pred_warp = self.vec_downsize(flow)
#             pred_warp = self.integrate_flow(pred_warp)
#             pred_warp = self.vec_upsize(pred_warp)
#         else:
#             pred_warp = self.integrate_flow(flow)

#         return [
#             flow,
#             pred_warp,
#         ]
