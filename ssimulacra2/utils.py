"""
Utility functions for SSIMULACRA2 implementation.
"""

import numpy as np
from PIL import Image


def prepare_images(ref_img, dist_img):
    """
    Prepare images for SSIMULACRA2 calculation.

    Args:
        ref_img: PIL Image object of the reference image
        dist_img: PIL Image object of the distorted image

    Returns:
        tuple: (reference image, distorted image) as PIL Image objects
            with matching sizes and in RGB mode
    """
    # Convert to RGB if not already
    if ref_img.mode != "RGB":
        ref_img = ref_img.convert("RGB")
    if dist_img.mode != "RGB":
        dist_img = dist_img.convert("RGB")

    # Get dimensions
    ref_width, ref_height = ref_img.size
    dist_width, dist_height = dist_img.size

    # Check if sizes match
    if ref_width != dist_width or ref_height != dist_height:
        # Resize distorted image to match reference
        dist_img = dist_img.resize((ref_width, ref_height), Image.BICUBIC)

    return ref_img, dist_img


def pad_to_size(image, target_size):
    """
    Pad an image to the target size.

    Args:
        image: NumPy array representing an image
        target_size: Tuple (height, width) for the target size

    Returns:
        NumPy array: Padded image
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size

    # Calculate padding
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)

    # Calculate padding on each side
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # Pad array
    if len(image.shape) == 3:  # RGB image
        padded = np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="reflect",
        )
    else:  # Grayscale image
        padded = np.pad(
            image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="reflect"
        )

    return padded


def get_gaussian_kernel(radius, sigma=1.5):
    """
    Generate a 2D Gaussian kernel.

    Args:
        radius: Radius of the kernel (kernel size will be 2*radius+1)
        sigma: Standard deviation of the Gaussian

    Returns:
        NumPy array: 2D Gaussian kernel
    """
    size = 2 * radius + 1
    x, y = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    kernel = np.exp(-((x**2 + y**2) / (2.0 * sigma**2))) * normal

    # Normalize the kernel
    return kernel / np.sum(kernel)


def rgb_to_yuv(rgb_img):
    """
    Convert RGB image to YUV color space.

    Args:
        rgb_img: NumPy array with RGB values (0-255)

    Returns:
        NumPy array: YUV image
    """
    # RGB to YUV conversion matrix
    conversion_matrix = np.array(
        [
            [0.299, 0.587, 0.114],
            [-0.14713, -0.28886, 0.436],
            [0.615, -0.51499, -0.10001],
        ]
    )

    # Normalize RGB to [0, 1]
    rgb_norm = rgb_img.astype(np.float32) / 255.0

    # Reshape for matrix multiplication
    rgb_reshaped = rgb_norm.reshape(-1, 3)

    # Apply conversion
    yuv_reshaped = np.dot(rgb_reshaped, conversion_matrix.T)

    # Reshape back to image dimensions
    yuv_img = yuv_reshaped.reshape(rgb_img.shape)

    return yuv_img


def create_pyramid(image, num_scales, subsample_factors):
    """
    Create a Gaussian pyramid for the image.

    Args:
        image: NumPy array representing an image
        num_scales: Number of scales in the pyramid
        subsample_factors: List of subsampling factors for each scale

    Returns:
        list: List of images at different scales
    """
    pyramid = [image]  # First level is the original image

    for i in range(1, num_scales):
        # Apply Gaussian blur before downsampling
        sigma = 0.5 * subsample_factors[i]
        kernel_size = int(2 * np.ceil(3 * sigma) + 1)
        kernel = get_gaussian_kernel(kernel_size // 2, sigma)

        # Apply convolution for each channel
        if len(image.shape) == 3:  # RGB image
            filtered = np.zeros_like(pyramid[-1])
            for c in range(image.shape[2]):
                filtered[:, :, c] = ndimage.convolve(pyramid[-1][:, :, c], kernel)
        else:  # Grayscale image
            filtered = ndimage.convolve(pyramid[-1], kernel)

        # Subsample
        downsampled = filtered[:: subsample_factors[i], :: subsample_factors[i]]
        pyramid.append(downsampled)

    return pyramid
