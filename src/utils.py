import torch
import torch.nn.functional as F


def generate_gaussian_kernel(kernel_size, sigma):
    # Generate a 1D Gaussian kernel
    x = torch.linspace(-1, 1, kernel_size)
    x = torch.exp(-x ** 2 / (2 * sigma ** 2))
    x /= x.sum()

    # Create a 3D Gaussian kernel using the outer product
    gaussian_kernel = x.view(1, -1) * x.view(-1, 1) * x.view(-1, 1, 1)
    return gaussian_kernel

def apply_gaussian_blur(
        input_image: torch.Tensor, 
        kernel_size: int, 
        sigma: float, 
        blur_mask: bool):
    
    if blur_mask:
        nb_dims = len(input_image.shape)
        assert nb_dims < 6, "Only 3D tensors are supported"

        # Generate the Gaussian kernel
        gaussian_kernel = generate_gaussian_kernel(kernel_size, sigma)
        
        if nb_dims < 5:
            for _ in range(nb_dims, 5):
                gaussian_kernel = gaussian_kernel.unsqueeze(0)
                input_image = input_image.unsqueeze(0)

        # Pad the input image 
        padding = kernel_size // 2
        padding_tup = tuple([padding] * 6)
        padded_image = F.pad(input_image, padding_tup)

        # Apply blur
        blurred_image = F.conv3d(
            input = padded_image, 
            weight = gaussian_kernel, 
            stride = 1,
            padding = 0,
            )

        # remove leading dimensions
        for _ in range(nb_dims, len(blurred_image.shape)):
            blurred_image = blurred_image.squeeze(0)
        
        return blurred_image
    
    else:
        return input_image