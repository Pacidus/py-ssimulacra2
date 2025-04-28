"""
Core implementation of the SSIMULACRA2 image quality metric.
Based on the original C++ implementation by Jon Sneyers, Cloudinary.
"""

import numpy as np
from scipy import ndimage
from PIL import Image
import cv2

# Constants for SSIMULACRA2
kC2 = 0.0009
kNumScales = 6


class MsssimScale:
    """Represents scores at a specific scale."""

    def __init__(self):
        self.avg_ssim = np.zeros(3 * 2)  # 3 channels x 2 norms
        self.avg_edgediff = np.zeros(
            3 * 4
        )  # 3 channels x 2 types of edge differences x 2 norms


class Msssim:
    """Stores scores across multiple scales and computes the final score."""

    def __init__(self):
        self.scales = []

    def score(self):
        """
        Compute the final SSIMULACRA2 score using the tuned weights.
        Returns a value from 0 to 100, where 100 is perfect quality.
        """
        # These weights were obtained by optimizing against multiple image quality datasets
        weights = [
            0.0,
            0.0007376606707406586,
            0.0,
            0.0,
            0.0007793481682867309,
            0.0,
            0.0,
            0.0004371155730107379,
            0.0,
            1.1041726426657346,
            0.00066284834129271,
            0.00015231632783718752,
            0.0,
            0.0016406437456599754,
            0.0,
            1.8422455520539298,
            11.441172603757666,
            0.0,
            0.0007989109436015163,
            0.000176816438078653,
            0.0,
            1.8787594979546387,
            10.94906990605142,
            0.0,
            0.0007289346991508072,
            0.9677937080626833,
            0.0,
            0.00014003424285435884,
            0.9981766977854967,
            0.00031949755934435053,
            0.0004550992113792063,
            0.0,
            0.0,
            0.0013648766163243398,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            7.466890328078848,
            0.0,
            17.445833984131262,
            0.0006235601634041466,
            0.0,
            0.0,
            6.683678146179332,
            0.00037724407979611296,
            1.027889937768264,
            225.20515300849274,
            0.0,
            0.0,
            19.213238186143016,
            0.0011401524586618361,
            0.001237755635509985,
            176.39317598450694,
            0.0,
            0.0,
            24.43300999870476,
            0.28520802612117757,
            0.0004485436923833408,
            0.0,
            0.0,
            0.0,
            34.77906344483772,
            44.835625328877896,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0008680556573291698,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0005313191874358747,
            0.0,
            0.00016533814161379112,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0004179171803251336,
            0.0017290828234722833,
            0.0,
            0.0020827005846636437,
            0.0,
            0.0,
            8.826982764996862,
            23.19243343998926,
            0.0,
            95.1080498811086,
            0.9863978034400682,
            0.9834382792465353,
            0.0012286405048278493,
            171.2667255897307,
            0.9807858872435379,
            0.0,
            0.0,
            0.0,
            0.0005130064588990679,
            0.0,
            0.00010854057858411537,
        ]

        ssim = 0.0
        i = 0

        # Sum all weighted scores across channels, scales, and norms
        for c in range(3):  # Three channels: X, Y, B
            for scale in range(len(self.scales)):
                for n in range(2):  # Two norms: L1 and L4
                    # SSIM component
                    ssim += weights[i] * abs(self.scales[scale].avg_ssim[c * 2 + n])
                    i += 1

                    # Ringing component
                    ssim += weights[i] * abs(self.scales[scale].avg_edgediff[c * 4 + n])
                    i += 1

                    # Blurring component
                    ssim += weights[i] * abs(
                        self.scales[scale].avg_edgediff[c * 4 + n + 2]
                    )
                    i += 1

        # Apply non-linear mapping to match perceptual scale
        ssim = ssim * 0.9562382616834844
        ssim = (
            2.326765642916932 * ssim
            - 0.020884521182843837 * ssim * ssim
            + 6.248496625763138e-05 * ssim * ssim * ssim
        )

        if ssim > 0:
            ssim = 100.0 - 10.0 * pow(ssim, 0.6276336467831387)
        else:
            ssim = 100.0

        return ssim


def rgb_to_xyb(img_rgb):
    """
    Convert RGB to XYB color space, approximating the JPEG XL transformation.

    Args:
        img_rgb: RGB image with values in range [0, 1]

    Returns:
        XYB image
    """
    # Linear RGB to XYB transformation (approximation of JPEG XL's transformation)
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]

    # Y (luminance)
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b

    # X and B (color difference channels)
    x = 0.5 * (r - g)
    by = 0.5 * (b - y)

    # Combine channels
    img_xyb = np.stack([x, y, by], axis=-1)

    return img_xyb


def make_positive_xyb(img_xyb):
    """
    Normalize XYB values to a positive range [0,1] as in the C++ implementation.

    Args:
        img_xyb: XYB image

    Returns:
        Normalized XYB image with values in a positive range
    """
    # Make a copy to avoid modifying the original
    img = img_xyb.copy()

    # Apply the same transformations as in the C++ MakePositiveXYB function
    img[:, :, 2] = (img[:, :, 2] - img[:, :, 1]) + 0.55  # B-Y and shift
    img[:, :, 0] = img[:, :, 0] * 14.0 + 0.42  # Scale and shift X
    img[:, :, 1] += 0.01  # Shift Y

    return img


def downsample(img, factor):
    """
    Downsample an image by a given factor.

    Args:
        img: Input image as numpy array
        factor: Downsampling factor

    Returns:
        Downsampled image
    """
    if factor == 1:
        return img

    # Use CV2 resize with area interpolation for best quality downsampling
    h, w = img.shape[:2]
    new_h, new_w = h // factor, w // factor

    result = np.zeros((new_h, new_w, img.shape[2]), dtype=img.dtype)

    for c in range(img.shape[2]):
        result[:, :, c] = cv2.resize(
            img[:, :, c], (new_w, new_h), interpolation=cv2.INTER_AREA
        )

    return result


def gaussian_blur(img, sigma=1.5):
    """
    Apply Gaussian blur to an image.

    Args:
        img: Input image as numpy array
        sigma: Standard deviation for Gaussian kernel

    Returns:
        Blurred image
    """
    result = np.zeros_like(img)

    # Apply Gaussian blur to each channel
    for c in range(img.shape[2]):
        result[:, :, c] = ndimage.gaussian_filter(img[:, :, c], sigma=sigma)

    return result


def to_the_4th(x):
    """Raise x to the 4th power."""
    x_sq = x * x
    return x_sq * x_sq


def calculate_ssim_map(m1, m2, s11, s22, s12, plane_averages):
    """
    Calculate SSIM map and norms.

    Args:
        m1, m2: Mean images
        s11, s22: Variance images
        s12: Covariance image
        plane_averages: Output array for results
    """
    h, w = m1.shape[1:3]
    one_per_pixels = 1.0 / (h * w)

    for c in range(3):  # For each channel
        sum1 = [0.0, 0.0]  # For L1 and L4 norms

        # Calculate modified SSIM index for each pixel
        for y in range(h):
            for x in range(w):
                mu1 = m1[c, y, x]
                mu2 = m2[c, y, x]
                mu11 = mu1 * mu1
                mu22 = mu2 * mu2
                mu12 = mu1 * mu2

                # Modified SSIM formula (without double gamma correction)
                num_m = 1.0 - (mu1 - mu2) * (mu1 - mu2)
                num_s = 2 * (s12[c, y, x] - mu12) + kC2
                denom_s = (s11[c, y, x] - mu11) + (s22[c, y, x] - mu22) + kC2

                # Error is 1 - SSIM
                d = 1.0 - (num_m * num_s / denom_s)
                d = max(d, 0.0)

                # L1 norm (mean)
                sum1[0] += d
                # L4 norm
                sum1[1] += to_the_4th(d)

        # Store results
        plane_averages[c * 2] = one_per_pixels * sum1[0]
        plane_averages[c * 2 + 1] = np.sqrt(np.sqrt(one_per_pixels * sum1[1]))


def calculate_edge_diff_map(img1, mu1, img2, mu2, plane_averages):
    """
    Calculate edge difference maps for ringing and blurring detection.

    Args:
        img1, img2: Original and distorted images
        mu1, mu2: Blurred versions of img1 and img2
        plane_averages: Output array for results
    """
    h, w = img1.shape[1:3]
    one_per_pixels = 1.0 / (h * w)

    for c in range(3):  # For each channel
        sum1 = [0.0, 0.0, 0.0, 0.0]  # For ringing (L1, L4) and blurring (L1, L4)

        for y in range(h):
            for x in range(w):
                # Edge strength ratio: how much stronger are edges in distorted vs original
                d1 = (1.0 + abs(img2[c, y, x] - mu2[c, y, x])) / (
                    1.0 + abs(img1[c, y, x] - mu1[c, y, x])
                ) - 1.0

                # d1 > 0: distorted has an edge where original is smooth (ringing, etc.)
                artifact = max(d1, 0.0)
                sum1[0] += artifact
                sum1[1] += to_the_4th(artifact)

                # d1 < 0: original has an edge where distorted is smooth (blurring, etc.)
                detail_lost = max(-d1, 0.0)
                sum1[2] += detail_lost
                sum1[3] += to_the_4th(detail_lost)

        # Store results
        plane_averages[c * 4] = one_per_pixels * sum1[0]  # Ringing L1
        plane_averages[c * 4 + 1] = np.sqrt(
            np.sqrt(one_per_pixels * sum1[1])
        )  # Ringing L4
        plane_averages[c * 4 + 2] = one_per_pixels * sum1[2]  # Blurring L1
        plane_averages[c * 4 + 3] = np.sqrt(
            np.sqrt(one_per_pixels * sum1[3])
        )  # Blurring L4


def alpha_blend(img_array, alpha_array, bg=0.5):
    """
    Apply alpha blending with a background color.

    Args:
        img_array: RGB image array
        alpha_array: Alpha channel array
        bg: Background intensity (0-1)

    Returns:
        Blended RGB image
    """
    return (
        alpha_array[:, :, np.newaxis] * img_array
        + (1 - alpha_array[:, :, np.newaxis]) * bg
    )


def calculate_ssimulacra2(reference_img, distorted_img):
    """
    Calculate the SSIMULACRA2 score between reference and distorted images.

    Args:
        reference_img: PIL Image object of the reference image
        distorted_img: PIL Image object of the distorted image

    Returns:
        float: SSIMULACRA2 score (0-100, higher is better)
    """
    # Check if images are identical
    ref_array = np.array(reference_img)
    dist_array = np.array(distorted_img)

    if np.array_equal(ref_array, dist_array):
        return 100.0

    # Ensure images are in RGB
    ref_rgb = reference_img.convert("RGB")
    dist_rgb = distorted_img.convert("RGB")

    # Convert PIL images to numpy arrays in range [0, 1]
    ref_array = np.array(ref_rgb).astype(np.float32) / 255.0
    dist_array = np.array(dist_rgb).astype(np.float32) / 255.0

    # Handle alpha if present
    has_alpha = False
    alpha = None

    if reference_img.mode == "RGBA":
        has_alpha = True
        alpha = np.array(reference_img.split()[3]).astype(np.float32) / 255.0

    # Convert RGB to linear RGB (assuming images are in sRGB)
    def srgb_to_linear(x):
        return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

    ref_linear = srgb_to_linear(ref_array)
    dist_linear = srgb_to_linear(dist_array)

    # If there's alpha, we need to calculate scores for dark and light backgrounds
    if has_alpha:
        # Calculate for dark background (bg=0.1)
        ref_dark = alpha_blend(ref_linear, alpha, 0.1)
        dist_dark = alpha_blend(dist_linear, alpha, 0.1)

        # Calculate for light background (bg=0.9)
        ref_light = alpha_blend(ref_linear, alpha, 0.9)
        dist_light = alpha_blend(dist_linear, alpha, 0.9)

        # Calculate scores for both backgrounds
        score_dark = compute_ssimulacra2_core(ref_dark, dist_dark)
        score_light = compute_ssimulacra2_core(ref_light, dist_light)

        # Return the worse of the two scores
        return min(score_dark, score_light)
    else:
        # No alpha, just calculate the score directly
        return compute_ssimulacra2_core(ref_linear, dist_linear)


def compute_ssimulacra2_core(ref_linear, dist_linear):
    """
    Core computation of SSIMULACRA2 score for linear RGB images.

    Args:
        ref_linear: Linear RGB reference image as numpy array
        dist_linear: Linear RGB distorted image as numpy array

    Returns:
        float: SSIMULACRA2 score (0-100, higher is better)
    """
    # Convert from linear RGB to XYB
    ref_xyb = rgb_to_xyb(ref_linear)
    dist_xyb = rgb_to_xyb(dist_linear)

    # Normalize to positive range
    ref_xyb = make_positive_xyb(ref_xyb)
    dist_xyb = make_positive_xyb(dist_xyb)

    # Transpose to channel-first for easier processing
    ref_xyb = np.transpose(ref_xyb, (2, 0, 1))
    dist_xyb = np.transpose(dist_xyb, (2, 0, 1))

    # Create result structure
    msssim = Msssim()

    # Process each scale
    for scale in range(kNumScales):
        if ref_xyb.shape[1] < 8 or ref_xyb.shape[2] < 8:
            break

        # Create scale result
        scale_result = MsssimScale()

        # Square and product images for SSIM calculation
        ref_sq = ref_xyb * ref_xyb
        dist_sq = dist_xyb * dist_xyb
        ref_dist = ref_xyb * dist_xyb

        # Apply Gaussian blur
        mu1 = gaussian_blur(ref_xyb)
        mu2 = gaussian_blur(dist_xyb)

        sigma1_sq = gaussian_blur(ref_sq) - mu1 * mu1
        sigma2_sq = gaussian_blur(dist_sq) - mu2 * mu2
        sigma12 = gaussian_blur(ref_dist) - mu1 * mu2

        # Calculate SSIM map
        calculate_ssim_map(
            mu1, mu2, sigma1_sq, sigma2_sq, sigma12, scale_result.avg_ssim
        )

        # Calculate edge difference maps
        calculate_edge_diff_map(ref_xyb, mu1, dist_xyb, mu2, scale_result.avg_edgediff)

        # Add scale result
        msssim.scales.append(scale_result)

        # Downsample for next scale (except the last iteration)
        if scale < kNumScales - 1:
            # Transpose back to height, width, channels for downsampling
            ref_xyb_hwc = np.transpose(ref_xyb, (1, 2, 0))
            dist_xyb_hwc = np.transpose(dist_xyb, (1, 2, 0))

            # Convert back to RGB for downsampling (better quality)
            # This is a simplified inverse of our XYB conversion
            ref_linear_hwc = np.zeros_like(ref_xyb_hwc)
            dist_linear_hwc = np.zeros_like(dist_xyb_hwc)

            # Restore original scale/shift
            ref_x = (ref_xyb_hwc[:, :, 0] - 0.42) / 14.0
            ref_y = ref_xyb_hwc[:, :, 1] - 0.01
            ref_b = ref_xyb_hwc[:, :, 2] - 0.55 + ref_y

            dist_x = (dist_xyb_hwc[:, :, 0] - 0.42) / 14.0
            dist_y = dist_xyb_hwc[:, :, 1] - 0.01
            dist_b = dist_xyb_hwc[:, :, 2] - 0.55 + dist_y

            # Approximate conversion from XYB to RGB
            ref_linear_hwc[:, :, 0] = ref_y + ref_x  # R = Y + X
            ref_linear_hwc[:, :, 1] = ref_y - ref_x  # G = Y - X
            ref_linear_hwc[:, :, 2] = ref_b  # B

            dist_linear_hwc[:, :, 0] = dist_y + dist_x
            dist_linear_hwc[:, :, 1] = dist_y - dist_x
            dist_linear_hwc[:, :, 2] = dist_b

            # Clip to valid range
            ref_linear_hwc = np.clip(ref_linear_hwc, 0, 1)
            dist_linear_hwc = np.clip(dist_linear_hwc, 0, 1)

            # Downsample in linear RGB
            ref_linear_hwc = downsample(ref_linear_hwc, 2)
            dist_linear_hwc = downsample(dist_linear_hwc, 2)

            # Convert back to XYB
            ref_xyb_downsampled = rgb_to_xyb(ref_linear_hwc)
            dist_xyb_downsampled = rgb_to_xyb(dist_linear_hwc)

            # Apply normalization again
            ref_xyb_downsampled = make_positive_xyb(ref_xyb_downsampled)
            dist_xyb_downsampled = make_positive_xyb(dist_xyb_downsampled)

            # Transpose back to channel-first
            ref_xyb = np.transpose(ref_xyb_downsampled, (2, 0, 1))
            dist_xyb = np.transpose(dist_xyb_downsampled, (2, 0, 1))

    # Calculate and return final score
    return msssim.score()
