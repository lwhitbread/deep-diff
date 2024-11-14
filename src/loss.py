import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import kornia.filters as filters
from typing import Union, List

Scalar = Union[int, float]
ncc_win = List[int]

class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """
    def __init__(
            self, 
            device: str, 
            win: ncc_win = None
    ) -> None:
        self.device = device
        self.win = win

    def loss(self, y_true, y_pred):
        with torch.autocast(
            enabled=False, 
            dtype=torch.float32, 
            device_type="cuda",
        ):
        # Assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
            y_true = y_true.float()
            y_pred = y_pred.float()
            
            ndims = y_true.dim() - 2
            assert 1 <= ndims <= 3, f"volumes should be 1 to 3 dimensions. found: {ndims}"

            # Set window size
            win = [9] * ndims if self.win is None else self.win

            # Create filter for summing in neighborhoods
            sum_filt = torch.ones([1, 1, *win], dtype=y_true.dtype, device=self.device)

            pad_no = math.floor(win[0] / 2)
            stride = (1,) * ndims
            padding = (pad_no,) * ndims

            # Use F.conv1d, F.conv2d, or F.conv3d based on the number of dimensions
            conv_fn = getattr(F, f'conv{ndims}d')

            # Compute squares and products needed for NCC
            I2, J2, IJ = y_true * y_true, y_pred * y_pred, y_true * y_pred

            I_sum = conv_fn(y_true, sum_filt, stride=stride, padding=padding)
            J_sum = conv_fn(y_pred, sum_filt, stride=stride, padding=padding)
            I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
            J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
            IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

            # Calculate means
            u_I, u_J = I_sum / sum_filt.numel(), J_sum / sum_filt.numel()

            # Calculate variances and covariance
            cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * sum_filt.numel()
            I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * sum_filt.numel()
            J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * sum_filt.numel()

            # Compute NCC
            cc = cross * cross / (I_var * J_var + 1e-5)
            out = 1 - torch.mean(cc)

        return out
    
class MixedNCCMSE:
    """
    Mixed loss of NCC and MSE
    """
    def __init__(
            self, 
            device: str, 
            ncc_weight: float = 0.005,
            win: ncc_win = None,
    ) -> None:
        self.device = device
        self.ncc_weight = ncc_weight
        self.ncc = NCC(device=device, win=win)
        self.mse = MSE()

    def loss(self, y_true, y_pred):
        ncc_loss = self.ncc.loss(y_true, y_pred)
        mse_loss = self.mse.loss(y_true, y_pred)
        return self.ncc_weight * ncc_loss + (1 - self.ncc_weight) * mse_loss

class MSE:
    """
    Mean squared error loss.
    """

    def loss(
        self, 
        y_true: torch.Tensor, 
        y_pred: torch.Tensor, 
        region_weights: torch.Tensor = None
    ) -> torch.Tensor:
        if region_weights is not None:
            return torch.mean(region_weights * (y_true - y_pred) ** 2)
        else:
            return torch.mean((y_true - y_pred) ** 2)

class MAE:
    """
    Mean absolute error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean(torch.abs(y_true - y_pred))


class SparsePenalty:
    """
    Implements a penalty using a thresholded version of the input
    """
    def __init__(
            self, 
            threshold: float = 0.1,
            threshold_type: str = "soft",
            sigmoid_centre_y: float = 0.5,
            nb_dims: int = 2, # for soft thresholding
            device: str = "cuda",
            use_spatial_grad: bool = True,
            spatial_grad_mult: float = 1.,
            downsample_factor: int = 1,
            img_dims: tuple = None,
        ) -> None:
        
        self.threshold = threshold
        self.threshold_type = threshold_type
        self.sigmoid_centre_y = sigmoid_centre_y
        self.nb_dims = nb_dims
        self.device = device
        if use_spatial_grad is None:
            self.use_spatial_grad = True
        self.use_spatial_grad = use_spatial_grad
        self.spatial_grad_mult = spatial_grad_mult
        self.downsample_factor = downsample_factor

        if downsample_factor > 1:
            assert img_dims is not None, "if downsample factor is greater than 1, img_dims must be provided"
            assert len(img_dims) == nb_dims, "img_dims must have the same number of dimensions as nb_dims"
            mode = "bilinear" if nb_dims == 2 else "trilinear"
            new_shape = [int(s / downsample_factor) for s in img_dims]
            self.downsize = nn.Upsample(
                size = new_shape, 
                mode = mode,
                align_corners = True,
            )
            self.upsize = nn.Upsample(
                size = img_dims,
                mode = mode,
                align_corners = True,
            )
        else:
            self.downsize = None
            self.upsize = None

    def loss(
         self, 
         image: torch.Tensor, # used for spatial gradient
         y_pred: torch.Tensor, # the intensity modulation field
         region_weights: torch.Tensor = None
         ) -> torch.Tensor:

        if self.downsize is not None:
            image = self.downsize(image)
            y_pred = self.downsize(y_pred)
            if region_weights is not None:
                region_weights = self.downsize(region_weights)

        if self.threshold_type == "soft":
            if self.use_spatial_grad:
                # update to order 1f
                if self.nb_dims == 2:
                    spatial_grad_adj = filters.spatial_gradient(
                        image, 
                        mode = "diff", 
                        order = 1 # TODO: play around with order 1 or 2
                    )
                
                elif self.nb_dims == 3:
                    spatial_grad_adj = filters.spatial_gradient3d(
                        image, 
                        mode = "diff", 
                        order = 1 # TODO: play around with order 1 or 2
                    )
                
                # can we gaussian blur this?
                spatial_grad_adj = torch.mean(
                    torch.square(
                        spatial_grad_adj
                    ) * self.spatial_grad_mult,
                    dim = 2,
                    keepdim = False,
                ) + 1.
  
        # threshold the input  
        if self.threshold_type == "soft":
            y_pred = torch.sigmoid(y_pred / self.threshold) 
            y_pred = torch.square(y_pred - self.sigmoid_centre_y) 

            # apply the spatial gradient
            if self.use_spatial_grad:
                if region_weights is not None:
                    y_pred = y_pred * spatial_grad_adj * region_weights
                    if self.downsize is not None:
                        y_pred = self.upsize(y_pred)
                    return torch.mean(y_pred)
                else:
                    y_pred = y_pred * spatial_grad_adj
                    if self.downsize is not None:
                        y_pred = self.upsize(y_pred)
                    return torch.mean(y_pred)
            else:
                if region_weights is not None:
                    y_pred = y_pred * region_weights
                    if self.downsize is not None:
                        y_pred = self.upsize(y_pred)
                    return torch.mean(y_pred)
                else:
                    if self.downsize is not None:
                        y_pred = self.upsize(y_pred)
                    return torch.mean(y_pred)

        elif self.threshold_type == "hard":
            y_pred = torch.where(y_pred > self.threshold, torch.ones_like(y_pred), torch.zeros_like(y_pred))
            if self.downsize is not None:
                y_pred = self.upsize(y_pred)
            return torch.mean(y_pred)
        else:
            raise ValueError(f"threshold type {self.threshold_type} not recognized")

class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice

class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, 
                 penalty: str = "l1", 
                 loss_mult: Scalar = None, 
                 nb_dims: int = 2, 
                 order: int = 1,
                 use_grad_filter: bool = True,
                 ) -> None:

        self.penalty = penalty
        self.loss_mult = loss_mult
        self.nb_dims = nb_dims
        assert self.nb_dims in [2, 3], f"should be 2 or 3 dimensions. found: {self.ndims}"
        self.order = order
        self.use_grad_filter = use_grad_filter

    def loss(self, 
             _: None, 
             y_pred: torch.Tensor,
             region_weights: torch.Tensor = None,
    ) -> torch.Tensor:
             
        if self.use_grad_filter:
            if self.nb_dims == 2:
                spatial_grad = filters.spatial_gradient(
                    y_pred, 
                    mode = "diff", 
                    order = self.order
                )
            elif self.nb_dims == 3:
                spatial_grad = filters.spatial_gradient3d(
                    y_pred, 
                    mode = "diff", 
                    order = self.order
                )
                
            if region_weights is not None:
                region_weights = region_weights.unsqueeze(1)
                spatial_grad = spatial_grad * region_weights

            if self.penalty == "l1":
                if self.loss_mult is not None:
                    return torch.mean(torch.abs(spatial_grad)) * self.loss_mult
                else:
                    return torch.mean(torch.abs(spatial_grad))
            elif self.penalty == "l2":
                if self.loss_mult is not None:
                    return torch.mean(spatial_grad ** 2) * self.loss_mult
                else:
                    return torch.mean(spatial_grad ** 2)

        else:
            dx, dy = y_pred, y_pred
            if self.nb_dims == 3:
                dz = y_pred
            
            for n in range(self.order):

                if self.nb_dims == 3:
                    dy = torch.abs(dy[:, :, 1:, :, :] - dy[:, :, :-1, :, :])
                    dx = torch.abs(dx[:, :, :, 1:, :] - dx[:, :, :, :-1, :])
                    dz = torch.abs(dz[:, :, :, :, 1:] - dz[:, :, :, :, :-1])
                
                elif self.nb_dims == 2:
                    dy = torch.abs(dy[:, :, 1:, :] - dy[:, :, :-1, :])
                    dx = torch.abs(dx[:, :, :, 1:] - dx[:, :, :, :-1])

            if self.penalty == 'l2':
                dy = dy * dy
                dx = dx * dx
                if self.nb_dims == 3:
                    dz = dz * dz
            
            if self.nb_dims == 3:
                d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
            else:
                d = torch.mean(dx) + torch.mean(dy)

            grad = d / self.nb_dims

            if self.loss_mult is not None:
                grad = grad * self.loss_mult
            return grad

def Get_Ja(flow):
    '''
    Calculate the Jacobian value at each point of the displacement map having
    size of b*h*w*d*3 and in the cubic volume of [-1, 1]^3
    '''
    D_y = (flow[:, 1:, :-1, :-1, :] - flow[:, :-1, :-1, :-1, :])
    D_x = (flow[:, :-1, 1:, :-1, :] - flow[:, :-1, :-1, :-1, :])
    D_z = (flow[:, :-1, :-1, 1:, :] - flow[:, :-1, :-1, :-1, :])
    D1 = (D_x[..., 0] + 1) * ((D_y[..., 1] + 1) * (D_z[..., 2] + 1) - D_z[..., 1] * D_y[..., 2])
    D2 = (D_x[..., 1]) * (D_y[..., 0] * (D_z[..., 2] + 1) - D_y[..., 2] * D_x[..., 0])
    D3 = (D_x[..., 2]) * (D_y[..., 0] * D_z[..., 1] - (D_y[..., 1] + 1) * D_z[..., 0])
    return D1 - D2 + D3


def NJ_loss(y_pred):
    '''
    Penalizing locations where Jacobian has negative determinants
    '''
    Neg_Jac = 0.5 * (torch.abs(Get_Ja(y_pred)) - Get_Ja(y_pred))
    return torch.sum(Neg_Jac)

class VectorLoss:
    def __init__(
        self,
        magnitude_weight: float = 0.5,
        magnitude_penalty: str = "l2",
        cosine_weight: float = 0.5,
    ) -> None:
        
        self.magnitude_weight = magnitude_weight
        self.cosine_weight = cosine_weight

        if magnitude_penalty == "l1":
            self.magnitude_penalty = "l1"
        elif magnitude_penalty == "l2":
            self.magnitude_penalty = "l2"
        else:
            raise ValueError(f"magnitude type {magnitude_penalty} not recognized")

        super(VectorLoss, self).__init__()

    def loss(
        self, 
        y_pred: torch.Tensor, 
        y_true: torch.Tensor,
    ) -> List[torch.Tensor]:
        # magnitude loss
        if self.magnitude_penalty == "l1":
            magnitude_loss = torch.mean(torch.abs(y_pred - y_true))
        elif self.magnitude_penalty == "l2":
            magnitude_loss = torch.mean(torch.square(y_pred - y_true))

        # cosine loss
        cosine_loss = torch.mean(1 - F.cosine_similarity(y_pred, y_true, dim=1))

        combined_loss = self.magnitude_weight * magnitude_loss + self.cosine_weight * cosine_loss
        return [combined_loss, magnitude_loss, cosine_loss]
