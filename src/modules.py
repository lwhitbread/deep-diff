import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions.normal import Normal
import numbers
from typing import List, Tuple, Union, Iterable

CompositeIntType = Union[int, List[int], Tuple[int, ...]]
CompositeFloatType = Union[float, List[float], Tuple[float, ...]]
IntIterable = Union[int, Iterable[int]]
 
class CreateField(nn.Module):
    """Create Displacement Field Layer"""

    def __init__(
            self,
            dec_feats: int,
            nb_dims: int = None,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1,
            dilation: int = 1,
            nb_layers: int = 1,
            intensity_modulation: bool = False,
            intensity_field_act: str = "Sigmoid",
            intensity_field_act_mult: float = 3,
        ) -> None:
        super().__init__()

        assert type(intensity_modulation) == bool, "intensity_modulation must be a boolean"
        self.intensity_modulation = intensity_modulation

        assert nb_dims in [2, 3], "ndims must be 2 or 3" 
        if nb_dims == 3:
            from torch.nn import Conv3d as Conv
        else:
            from torch.nn import Conv2d as Conv

        conv_layers = nn.ModuleList()
        
        for idx in range(nb_layers):
            if idx == nb_layers - 1:
                conv_layers.append(
                    ConvBlock(
                        nb_dims,
                        dec_feats,
                        1 if intensity_modulation else nb_dims,
                        kernel_size = kernel_size,
                        stride = stride,
                        padding = 0 if kernel_size == 1 else padding,
                        dilation = dilation,
                        groups = 1,
                        bias = True,
                        padding_mode = "zeros",
                        activation = "Identity",
                        transpose = False,
                        normalisation = None,
                    )
                )
            else:
                conv_layers.append(
                    ConvBlock(
                        nb_dims,
                        dec_feats,
                        dec_feats,
                        kernel_size = 3,
                        stride = stride,
                        padding = padding,
                        dilation = dilation,
                        groups = 1,
                        bias = True,
                        padding_mode = "zeros",
                        activation = "GELU",
                        transpose = False,
                        normalisation = "Instance",
                    )
                )
        
        self.conv_layers = conv_layers
        self.conv_layers[-1].conv.weight = nn.Parameter(Normal(0, 1e-5).sample(self.conv_layers[-1].conv.weight.shape))
        self.conv_layers[-1].conv.bias = nn.Parameter(torch.zeros(self.conv_layers[-1].conv.bias.shape))

        if intensity_modulation:
            self.activation = getattr(nn, intensity_field_act)()
            self.post_activation_mult = intensity_field_act_mult

    def freeze_weights(
        self,
        freeze: bool,
    ) -> None:
        assert self.intensity_modulation, "intensity_modulation must be True to freeze weights"
        
        for module in self.conv_layers:
            for param in module.parameters():
                param.requires_grad = not freeze
        
        if freeze:
            print("Freezing weights of CreateField layer for intensity modulation.")
    
    def forward(
            self, 
            x: torch.Tensor,
            field_multiplier: float = 1., # only used for intensity modulation
        ) -> torch.Tensor:
        
        if self.intensity_modulation:
            for module in self.conv_layers:
                x = module(x)
            return self.post_activation_mult * (self.activation(x) - 0.5) * field_multiplier
        else:
            for module in self.conv_layers:
                x = module(x)
            
            return x

class IntensityModulationBlock(nn.Module):
    """
    Intensity modulation block to modulate the intensity of the input image.
    """

    def __init__(
            self,
            in_shape: Tuple[int, ...],
            nb_dims: int = 2,
            intensity_act: str = "exp",
            norm_percentile: float = 0.999,
            int_downsize: int = 2,
        ) -> None:
        super().__init__()

        assert intensity_act in ["Sigmoid", "exp"], "act_type must be either Sigmoid or exp"

        if intensity_act == "Sigmoid":
            self.activation = getattr(nn, intensity_act)()
            self.post_activation_mult = 2
        elif intensity_act == "exp":
            self.activation = getattr(torch, intensity_act)
            self.post_activation_mult = 1

        self.norm_percentile = norm_percentile

        if int_downsize > 1:
            mode = "bilinear" if nb_dims == 2 else "trilinear"
            new_shape = [int(s / int_downsize) for s in in_shape]
            self.downsize = nn.Upsample(
                size = new_shape, 
                mode = mode,
                align_corners = True,
            )
            self.upsize = nn.Upsample(
                size = in_shape,
                mode = mode,
                align_corners = True,
            )
        else:
            self.downsize = None
            self.upsize = None
        
    def forward(
            self, 
            x: torch.Tensor,
            intensity_modulation: torch.Tensor,
            field_multiplier: float = 1.,
        ) -> torch.Tensor:
        
        assert x.shape == intensity_modulation.shape, \
            "both tensors passed as arguments must have the same size.\n" + \
            f"tensor 1: {x.shape}, tensor 2: {intensity_modulation.shape}"
                
        if self.downsize is not None:
            intensity_modulation = self.downsize(intensity_modulation)
        
        intensity_modulation = self.activation(intensity_modulation) * self.post_activation_mult
        
        if field_multiplier == 0:
            assert torch.allclose(intensity_modulation, torch.ones_like(intensity_modulation), atol = 1e-8), \
                "intensity_modulation must be 1 when field_multiplier is 0"
        
        if self.upsize is not None:
            intensity_modulation = self.upsize(intensity_modulation)
        
        x = x * intensity_modulation
        
        if field_multiplier == 0:
            return x, intensity_modulation

        x = torch.clip(x, 0, 1.1)
        
        return x, intensity_modulation

class SpatialTransformer(nn.Module):
    
    def __init__(
            self,
            size: Tuple[int, ...],
            mode: str = "linear",
    ) -> None:
        super().__init__()

        self.mode = mode
        self.size = size
        # Establish a sampling grid   
        vectors = [torch.arange(0, s) for s in self.size]
        grids = torch.meshgrid(vectors, indexing = "ij")
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0).type(torch.FloatTensor)
        self.register_buffer("grid", grid)

        if self.mode == "linear":
            if len(self.size) == 2:
                self.mode = "bilinear"
            elif len(self.size) == 3:
                self.mode = "bilinear"
    
    def forward(
            self, 
            x: torch.Tensor, 
            flow: torch.Tensor,
    ) -> torch.Tensor:

        assert flow.shape[2:] == x.shape[2:] == self.size, \
            "flow and x must have the same size as the grid"
        
        # get the new locations
        new_locs = self.grid + flow
        

        for idx in range(len(self.size)):
            new_locs[:, idx, ...] = 2 * (new_locs[:, idx, ...] / (self.size[idx] - 1) - 0.5)

        if len(self.size) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        
        elif len(self.size) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
    
        return F.grid_sample(x, new_locs, mode = self.mode, align_corners = True)

class GaussianSmoothing(nn.Module):
    """
    Adrian Sahlman:
    https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/7
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed separately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel. If it is less than 0, then it will learn sigma
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(
            self, 
            channels: CompositeIntType, 
            kernel_size: CompositeIntType = 5, 
            sigma: CompositeFloatType = 2, 
            dim: int = 2,
    ) -> None:
        super().__init__()

        self.og_sigma = sigma
        
        kernel_dic = {3:1,5:2}
        self.pad = kernel_dic[kernel_size]

        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(
                    size, 
                    dtype = torch.float32
                ) for size in kernel_size
            ]
            , indexing = "ij"
        )

        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel = kernel * (1 / (std * np.sqrt(2 * np.pi)) * \
                    torch.exp((-((mgrid - mean) / std) ** 2) / 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
    
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        # print(kernel)
        
        if self.og_sigma < 0:
            # --- Learnable Sigma---------------:
            sigma = 2
            self.learnable = 1 # is the network learnable or static?

            if dim == 1:
                self.conv = nn.Conv1d(
                    in_channels = dim,
                    out_channels = dim,
                    kernel_size = kernel_size,
                    padding = self.pad,
                )
            
            elif dim == 2:
                self.conv = nn.Conv2d(
                    in_channels = dim,
                    out_channels = dim,
                    kernel_size = kernel_size,
                    padding = self.pad,
                )
            
            elif dim == 3:
                self.conv = nn.Conv3d(
                    in_channels = dim,
                    out_channels = dim,
                    kernel_size = kernel_size,
                    padding = self.pad,
                )
            
            else:
                raise RuntimeError('Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim))

            
            kernel_to_cat = [kernel for _ in range(channels)]
            kernel_to_cat = tuple(kernel_to_cat)
            
            # Initialise with normal dist
            self.conv.weight = nn.Parameter(torch.cat(kernel_to_cat,dim=1))
            self.conv.bias = nn.Parameter(torch.zeros(self.conv.bias.shape))

            print("Smoothing Layer: Learnable Sigma, initialised sigma = {self.og_sigma}")
            print(self.conv.weight.shape)
            print(self.conv.weight)


        else:
            # --- Static network---------------:
            self.learnable = 0

            self.register_buffer('weight', kernel)

            self.groups = channels

            if dim == 1:
                self.conv = F.conv1d
            elif dim == 2:
                self.conv = F.conv2d
            elif dim == 3:
                self.conv = F.conv3d
            else:
                raise RuntimeError('Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim))
            
            print(f"Smoothing Layer: Static Sigma = {self.og_sigma}")
            print(self.weight.shape)
            print(self.weight)


    def forward(
            self, 
            input: torch.Tensor,
        ) -> torch.Tensor:
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        # if static or trainable:
        if self.learnable == 1:
            return self.conv(input)
        else:
            return self.conv(input, weight=self.weight, groups=self.groups, padding=self.pad)


class IntegrationLayer(nn.Module):
    """
    Integrates a vector field via scaling and squaring. 
    See: https://github.com/voxelmorph/voxelmorph.
    """

    def __init__(
            self, 
            in_shape: Tuple[int, ...],
            nb_steps: int,
            kernel_size: CompositeIntType = 3,
            sigma: float = 2.,
            use_tanh_act: bool = True,
        ) -> None:
        super().__init__()

        assert nb_steps > 0, \
            "nsteps should be > 0, found: %d" % nb_steps
        
        self.in_shape = in_shape
        self.nsteps = nb_steps
        self.scale = 2. ** -self.nsteps
        self.transformer = SpatialTransformer(in_shape)
        self.use_tanh_act = use_tanh_act
        if use_tanh_act:
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()

        self.nb_dims = len(in_shape)
        self.smoothing = GaussianSmoothing(
            channels = self.nb_dims, 
            kernel_size = kernel_size, 
            sigma = sigma, 
            dim = self.nb_dims,
        )

    def forward(
            self, 
            vec: torch.Tensor, 
            smoothing = True,
        ) -> torch.Tensor:
        
        # see if works
        if self.use_tanh_act:
            if self.nb_dims == 3:
                vec_1 = self.activation(vec[:,0,:,:,:]) * (1/self.in_shape[0])
                vec_2 = self.activation(vec[:,1,:,:,:]) * (1/self.in_shape[1])
                vec_3 = self.activation(vec[:,2,:,:,:]) * (1/self.in_shape[2])
                vec = torch.stack((vec_1, vec_2, vec_3), dim=1)
            elif self.nb_dims == 2:
                vec_1 = self.activation(vec[:,0,:,:]) * (1/self.in_shape[0])
                vec_2 = self.activation(vec[:,1,:,:]) * (1/self.in_shape[1])
                vec = torch.stack((vec_1, vec_2), dim=1)
            
        else:
            vec = self.activation(vec * self.scale)
        
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)  
            if smoothing:
                vec = self.smoothing(vec)

        return vec


class ConvBlock(nn.Module):
    """
    Convolutional block consisting of a convolutional
    layer, normalization and activation.
    """

    def __init__(
            self,
            nb_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: CompositeIntType = 3,
            stride: CompositeIntType = 1,
            padding: CompositeIntType = 1,
            dilation: CompositeIntType = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = "zeros",
            activation: str = "GELU",
            transpose: bool = False,
            normalisation: str = None,
        ) -> None:
        super().__init__()

        if transpose:
            Conv = getattr(nn, 'ConvTranspose%dd' % nb_dims)
        else:
            Conv = getattr(nn, 'Conv%dd' % nb_dims)
        
        # Get convolutional layer
        self.conv = Conv(
            in_channels = in_channels, 
            out_channels = out_channels, 
            kernel_size = kernel_size, 
            stride = stride, 
            padding = padding, 
            dilation = dilation, 
            groups = groups, 
            bias = bias, 
            padding_mode = padding_mode, 
        )
        
        # get normalisation if passed
        if normalisation:
            if normalisation == "Instance" or normalisation == "Batch":
                Norm = getattr(nn, "%sNorm%dd" % (normalisation, nb_dims))
            elif normalisation == "Group" or normalisation == "Layer":
                Norm = getattr(nn, "%sNorm" % (normalisation))
            else:
                raise ValueError("normalisation must be one of Instance, Batch, Group or Layer")   
            
            self.normalisation = Norm(out_channels)
        else:
            self.normalisation = None
        
        # Get activation layer
        try:
            self.activation = getattr(nn, activation)()
        except:
            raise AttributeError("Activation function not found")

    def forward(
            self, 
            x
        ) -> torch.Tensor:
        
        x = self.conv(x)
        if self.normalisation is not None:
            x = self.normalisation(x)  
        return self.activation(x)

class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(
            self, 
            vel_resize: float = 1.0, 
            nb_dims: int = 3,
            out_shape: Tuple[int, ...] = None,
        ) -> None: 
        super().__init__()
        
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if nb_dims == 2:
            self.mode = 'bi' + self.mode
        elif nb_dims == 3:
            self.mode = 'tri' + self.mode
        self.out_shape = out_shape

    def forward(
            self, 
            x: torch.Tensor,
        ) -> torch.Tensor:
        
        if self.factor < 1:
            # resize first to save memory
            x = F.interpolate(
                x, 
                align_corners = True, 
                scale_factor = self.factor if self.out_shape is None else None, 
                size = self.out_shape,
                mode = self.mode,
            )
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = F.interpolate(
                x, 
                align_corners = True, 
                scale_factor = self.factor if self.out_shape is None else None, 
                size = self.out_shape,
                mode = self.mode,
            )

        # don't do anything if resize is 1
        return x

class MeanTracker(nn.Module):
    """A class to track mean vector fields."""

    def __init__(
            self, 
            img_dims: Tuple[int, ...],
            nb_dims: int,
            mean_field_idx: int,
            lambda_rate: float = 0.9,
            *args, 
            **kwargs,
        ) -> None:
        super().__init__()

        assert len(img_dims) == nb_dims
        assert nb_dims in [2, 3], "image dims must be 2 or 3"
        
        self.img_dims = img_dims
        self.nb_dims = nb_dims
        self.mean_field_idx = mean_field_idx
        self.lambda_rate = lambda_rate
        
        mean_field = torch.zeros(
            nb_dims, 
            *img_dims,
            requires_grad = False,
        )

        self.register_buffer(
            "mean_field",
            mean_field,
        )
        
        self.weight_accumulator = 0.
        self.applied_weight_accumulator = 0.
        self.mean_field_mag = 0.

    # use the forward to update the mean field
    def forward(
            self, 
            x: torch.Tensor, # the vector field  
            prop_contribution: float, # the proportion of the vector field to add to the mean field
        ) -> None:
        
        assert prop_contribution >= 0 and prop_contribution <= 1, \
            "prop_contribution must be greater than 0 and less than or equal to 1"

        self.mean_field = (1 - prop_contribution * (1 - self.lambda_rate)) * self.mean_field + prop_contribution * (1 - self.lambda_rate) * x

    def get_mean_field(self) -> torch.Tensor:
        return self.mean_field
    
    def move_to(self, device) -> None:
        self.mean_field = self.mean_field.to(device)
        self.to(device)

class AllMeansTracker(nn.Module):
    """A class to track agglomerate of mean vector fields.""" 

    def __init__(
            self, 
            img_dims: Tuple[int, ...],
            nb_dims: int,
            nb_mean_fields: int,
            device: torch.device = torch.device(0 if torch.cuda.is_available() else "cpu"),
            lambda_rate: float = 0.9,
            *args, 
            **kwargs,
        ) -> None:
        super().__init__()

        assert len(img_dims) == nb_dims
        assert nb_dims in [2, 3], "image dims must be 2 or 3"
        
        self.img_dims = img_dims
        self.nb_dims = nb_dims
        
        mean_trackers = []
        for idx in range(nb_mean_fields):
            mean_trackers.append(
                MeanTracker(
                    img_dims = img_dims,
                    nb_dims = nb_dims,
                    mean_field_idx = idx,
                    lambda_rate = lambda_rate,
                )
            )
        
        self.mean_trackers = nn.ModuleList(mean_trackers)
        self.device = device
        
        for tracker in self.mean_trackers:
            tracker.move_to(self.device)

    # use the forward to update the mean fields
    def forward(
            self,
            x: torch.Tensor, # warp field for full batch
            means_idx_0: torch.Tensor, # means_idx_0 for full batch
            prop_means_idx_0: torch.Tensor, # prop_means_idx_0 for full batch
    ) -> None: 

        assert isinstance(means_idx_0, Iterable), "means_idx_0 must be iterable"
        assert isinstance(prop_means_idx_0, Iterable), "prop_means_idx_0 must be iterable"
        
        for idx in range(len(means_idx_0)):
            # get the mean field idx for each sample in the batch
            _means_idx_0 = int(means_idx_0[idx])
            _prop_means_idx_0 = prop_means_idx_0[idx]
            _means_idx_1 = _means_idx_0 + 1
            _prop_means_idx_1 = 1 - _prop_means_idx_0
            
            self.mean_trackers[_means_idx_0].forward(
                x = x[idx],
                prop_contribution = _prop_means_idx_0,
            )
            
            self.mean_trackers[_means_idx_1].forward(
                x = x[idx],
                prop_contribution = _prop_means_idx_1,
            )
            
    def get_mean_field(
            self,
            mean_field_idx: IntIterable,
        ) -> torch.Tensor:
        
        if isinstance(mean_field_idx, Iterable):
            mean_fields = []
            for idx_0 in mean_field_idx:
                for idx_1, tracker in enumerate(self.mean_trackers):
                    if idx_0 == idx_1:
                        # mean_fields.append(tracker.to(self.device).get_mean_field())
                        mean_fields.append(tracker.get_mean_field())
                # mean_fields.append(self.mean_trackers[idx].to(self.device).get_mean_field())
            return torch.stack(mean_fields, dim = 0)
        else:
            for idx, tracker in enumerate(self.mean_trackers):
                if mean_field_idx == idx:
                    # return tracker.to(self.device).get_mean_field()
                    return tracker.get_mean_field()



#############################################################################
#############################################################################
## UNUSED CODE
#############################################################################
#############################################################################

## TBD WITH MJ: HAVE NOT GOT TANH ACTIVATION TO WORK YET
## To be considered and adjusted - we want to use tanh activation to limit the displacement field
def DiffeomorphicActivate(flow_field,size):
    """ Activation Function
    Args:
        flow_field ([tensor array]): A n-dimension array containing the flow field in each direction
        size ([list]): [description]: The maximum size of the field, to limit the size of the initial displacement
    Returns:
        flow_field [tensor array]: Flow field after the activation function has been applied.
    """

    # Assert ndims is 2D or 3D
    assert flow_field.size()[1] in [2,3]
    assert len(size) in [2,3]
    

    if len(size) == 3:
        flow_1= torch.tanh(flow_field[:,0,:,:,:])*(1/size[0]) 
        flow_2 = torch.tanh(flow_field[:,1,:,:,:])*(1/size[1])
        flow_3= torch.tanh(flow_field[:,2,:,:,:])*(1/size[2])
        flow_field =torch.stack((flow_1,flow_2,flow_3), dim=1)
    elif len(size) == 2:
        flow_1= torch.tanh(flow_field[:,0,:,:])*(1/size[0])
        flow_2 = torch.tanh(flow_field[:,1,:,:])*(1/size[1])
        flow_field =torch.stack((flow_1,flow_2), dim=1)

    return flow_field


