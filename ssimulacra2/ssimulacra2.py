"""SSIMULACRA2 implementation in Python."""

import numpy as np
from PIL import Image
from scipy import ndimage

# Constants from the original implementation
kC2 = 0.0009
kNumScales = 6

# Parameters for opsin absorbance - exact same values as in C++
# From src/lib/jxl/opsin_params.h
kM02 = 0.078
kM00 = 0.30
kM01 = 1.0 - kM02 - kM00  # Computed exactly as in C++

kM12 = 0.078
kM10 = 0.23
kM11 = 1.0 - kM12 - kM10  # Computed exactly as in C++

kM20 = 0.24342268924547819
kM21 = 0.20476744424496821
kM22 = 1.0 - kM20 - kM21  # Computed exactly as in C++

kB0 = 0.0037930732552754493
kB1 = kB0
kB2 = kB0

# Create the matrix and bias arrays exactly as in C++
kOpsinAbsorbanceMatrix = np.array(
    [kM00, kM01, kM02, kM10, kM11, kM12, kM20, kM21, kM22]
)

kOpsinAbsorbanceBias = np.array([kB0, kB1, kB2])


def srgb_to_linear(img):
    """Convert sRGB to linear RGB."""
    # Handle the two-part curve
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    mask = rgb <= 0.04045
    rgb[mask] = rgb[mask] / 12.92
    rgb[~mask] = ((rgb[~mask] + 0.055) / 1.055) ** 2.4
    return rgb


def linear_rgb_to_xyb(linear):
    """Convert linear RGB to XYB color space."""
    # Apply opsin absorbance (matrix multiplication)
    r, g, b = linear[:, :, 0], linear[:, :, 1], linear[:, :, 2]

    # Reshape the matrix for easier operations
    matrix = kOpsinAbsorbanceMatrix.reshape(3, 3)

    # Apply matrix multiplication
    mixed0 = (
        matrix[0, 0] * r + matrix[0, 1] * g + matrix[0, 2] * b + kOpsinAbsorbanceBias[0]
    )
    mixed1 = (
        matrix[1, 0] * r + matrix[1, 1] * g + matrix[1, 2] * b + kOpsinAbsorbanceBias[1]
    )
    mixed2 = (
        matrix[2, 0] * r + matrix[2, 1] * g + matrix[2, 2] * b + kOpsinAbsorbanceBias[2]
    )

    # Zero if negative
    mixed0 = np.maximum(mixed0, 0)
    mixed1 = np.maximum(mixed1, 0)
    mixed2 = np.maximum(mixed2, 0)

    # Apply cube root and subtract bias
    mixed0 = np.cbrt(mixed0) - np.cbrt(kOpsinAbsorbanceBias[0])
    mixed1 = np.cbrt(mixed1) - np.cbrt(kOpsinAbsorbanceBias[1])
    mixed2 = np.cbrt(mixed2) - np.cbrt(kOpsinAbsorbanceBias[2])

    # Store in XYB format
    x = 0.5 * (mixed0 - mixed1)
    y = 0.5 * (mixed0 + mixed1)
    b_y = mixed2

    return np.stack([x, y, b_y], axis=2)


def make_positive_xyb(xyb):
    """Adjust XYB values to positive range as in SSIMULACRA2."""
    result = xyb.copy()

    # MakePositiveXYB function from SSIMULACRA2
    result[:, :, 2] = (result[:, :, 2] - result[:, :, 1]) + 0.55  # B-Y
    result[:, :, 0] = result[:, :, 0] * 14.0 + 0.42  # X scaling
    result[:, :, 1] += 0.01  # Y offset

    return result


def downsample(img, fx, fy):
    """Downsample image by a factor along each dimension."""
    h, w, c = img.shape
    out_h, out_w = (h + fy - 1) // fy, (w + fx - 1) // fx
    result = np.zeros((out_h, out_w, c), dtype=np.float32)

    # This matches the C++ implementation's averaging approach
    normalize = 1.0 / (fx * fy)

    for ch in range(c):
        for oy in range(out_h):
            for ox in range(out_w):
                sum_val = 0.0
                for iy in range(fy):
                    for ix in range(fx):
                        x = min(ox * fx + ix, w - 1)
                        y = min(oy * fy + iy, h - 1)
                        sum_val += img[y, x, ch]
                result[oy, ox, ch] = sum_val * normalize

    return result


def multiply_images(a, b):
    """Element-wise multiplication of two images."""
    return a * b


def blur_image(img, sigma=1.5):
    """Apply Gaussian blur to an image.

    This uses scipy's Gaussian filter with mode='mirror' to
    match the mirroring behavior in the C++ implementation.
    """
    # Apply gaussian blur to each channel separately to ensure precise handling
    result = np.zeros_like(img)

    for c in range(img.shape[2]):
        # Use 'mirror' mode to match C++ boundary handling
        # and truncate at 4*sigma to match typical Gaussian filter behavior
        result[:, :, c] = ndimage.gaussian_filter(
            img[:, :, c], sigma=sigma, mode="mirror", truncate=4.0
        )

    return result


def ssim_map(m1, m2, s11, s22, s12):
    """Compute the SSIM map between two images."""
    h, w, c = m1.shape
    plane_averages = np.zeros(c * 2)

    # For each channel
    for ch in range(c):
        sum1 = np.zeros(2)

        # Process each pixel
        for y in range(h):
            for x in range(w):
                mu1 = m1[y, x, ch]
                mu2 = m2[y, x, ch]
                mu11 = mu1 * mu1
                mu22 = mu2 * mu2
                mu12 = mu1 * mu2

                # Modified SSIM formula without the spurious gamma correction term
                num_m = 1.0 - (mu1 - mu2) * (mu1 - mu2)
                num_s = 2.0 * (s12[y, x, ch] - mu12) + kC2
                denom_s = (s11[y, x, ch] - mu11) + (s22[y, x, ch] - mu22) + kC2

                # Error score (1 - SSIM')
                d = 1.0 - (num_m * num_s / denom_s)
                d = max(d, 0.0)

                # L1 and L4 norms
                sum1[0] += d
                sum1[1] += d**4

        # Store averages
        one_per_pixels = 1.0 / (h * w)
        plane_averages[ch * 2] = one_per_pixels * sum1[0]
        plane_averages[ch * 2 + 1] = (one_per_pixels * sum1[1]) ** 0.25

    return plane_averages


def edge_diff_map(img1, mu1, img2, mu2):
    """Compute edge difference maps."""
    h, w, c = img1.shape
    plane_averages = np.zeros(c * 4)

    # For each channel
    for ch in range(c):
        sum1 = np.zeros(4)

        # Process each pixel
        for y in range(h):
            for x in range(w):
                # Compute ratio of edge strengths
                d1 = (1.0 + abs(img2[y, x, ch] - mu2[y, x, ch])) / (
                    1.0 + abs(img1[y, x, ch] - mu1[y, x, ch])
                ) - 1.0

                # d1 > 0: distorted has edges where original is smooth (ringing/blockiness)
                artifact = max(d1, 0.0)
                sum1[0] += artifact
                sum1[1] += artifact**4

                # d1 < 0: original has edges where distorted is smooth (blurring)
                detail_lost = max(-d1, 0.0)
                sum1[2] += detail_lost
                sum1[3] += detail_lost**4

        # Store averages
        one_per_pixels = 1.0 / (h * w)
        plane_averages[ch * 4] = one_per_pixels * sum1[0]
        plane_averages[ch * 4 + 1] = (one_per_pixels * sum1[1]) ** 0.25
        plane_averages[ch * 4 + 2] = one_per_pixels * sum1[2]
        plane_averages[ch * 4 + 3] = (one_per_pixels * sum1[3]) ** 0.25

    return plane_averages


def alpha_blend(img, alpha, bg=0.5):
    """Alpha blend image with background color."""
    if alpha is None:
        return img

    result = img.copy()
    for c in range(3):
        result[:, :, c] = alpha * img[:, :, c] + (1.0 - alpha) * bg

    return result


class MsssimScale:
    """Data structure for one scale's metrics."""

    def __init__(self):
        self.avg_ssim = np.zeros(3 * 2)  # 3 channels, 2 norms
        self.avg_edgediff = np.zeros(
            3 * 4
        )  # 3 channels, 4 values (ringing and blurring, each with 2 norms)


class Msssim:
    """Multi-scale structural similarity metric."""

    def __init__(self):
        self.scales = []

    def score(self):
        """Compute the final SSIMULACRA2 score."""
        # These weights were obtained from the original C++ implementation
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

        # Apply weights to each component
        for c in range(3):
            for scale in range(len(self.scales)):
                for n in range(2):
                    ssim += weights[i] * abs(self.scales[scale].avg_ssim[c * 2 + n])
                    i += 1
                    ssim += weights[i] * abs(self.scales[scale].avg_edgediff[c * 4 + n])
                    i += 1
                    ssim += weights[i] * abs(
                        self.scales[scale].avg_edgediff[c * 4 + n + 2]
                    )
                    i += 1

        # Transform to final score
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


def compute_ssimulacra2(orig_path, dist_path, bg=0.5):
    """Compute SSIMULACRA2 score between original and distorted images."""
    # Load images
    orig_img = np.array(Image.open(orig_path).convert("RGB"), dtype=np.float32)
    dist_img = np.array(Image.open(dist_path).convert("RGB"), dtype=np.float32)

    # Check if alpha channel exists (assuming 4-channel means RGBA)
    orig_alpha = None
    dist_alpha = None
    if len(orig_img.shape) > 2 and orig_img.shape[2] == 4:
        orig_alpha = orig_img[:, :, 3] / 255.0
        orig_img = orig_img[:, :, :3]
    if len(dist_img.shape) > 2 and dist_img.shape[2] == 4:
        dist_alpha = dist_img[:, :, 3] / 255.0
        dist_img = dist_img[:, :, :3]

    # Alpha blending if needed
    if orig_alpha is not None:
        orig_img = alpha_blend(orig_img, orig_alpha, bg)
    if dist_alpha is not None:
        dist_img = alpha_blend(dist_img, dist_alpha, bg)

    # Convert sRGB to linear RGB
    orig_linear = srgb_to_linear(orig_img)
    dist_linear = srgb_to_linear(dist_img)

    # Convert to XYB
    orig_xyb = linear_rgb_to_xyb(orig_linear)
    dist_xyb = linear_rgb_to_xyb(dist_linear)

    # Make XYB values positive
    orig_xyb = make_positive_xyb(orig_xyb)
    dist_xyb = make_positive_xyb(dist_xyb)

    # Initialize multi-scale metric
    msssim = Msssim()

    # Process at multiple scales
    img1 = orig_xyb
    img2 = dist_xyb

    for scale in range(kNumScales):
        # Check if image is too small to process
        if img1.shape[0] < 8 or img1.shape[1] < 8:
            break

        # Create multiplied images for variance calculations
        mul = multiply_images(img1, img1)
        sigma1_sq = blur_image(mul)

        mul = multiply_images(img2, img2)
        sigma2_sq = blur_image(mul)

        mul = multiply_images(img1, img2)
        sigma12 = blur_image(mul)

        # Compute blurred means
        mu1 = blur_image(img1)
        mu2 = blur_image(img2)

        # Create scale data structure
        scale_data = MsssimScale()

        # Compute SSIM map
        scale_data.avg_ssim = ssim_map(mu1, mu2, sigma1_sq, sigma2_sq, sigma12)

        # Compute edge difference maps
        scale_data.avg_edgediff = edge_diff_map(img1, mu1, img2, mu2)

        # Add to scales
        msssim.scales.append(scale_data)

        # Downsample for next scale (if not the last scale)
        if scale < kNumScales - 1:
            # Downsample in linear RGB space for better quality
            orig_linear = downsample(orig_linear, 2, 2)
            dist_linear = downsample(dist_linear, 2, 2)

            # Convert back to XYB
            img1 = linear_rgb_to_xyb(orig_linear)
            img2 = linear_rgb_to_xyb(dist_linear)

            # Make XYB values positive
            img1 = make_positive_xyb(img1)
            img2 = make_positive_xyb(img2)

    # Compute final score
    return msssim.score()


def compute_ssimulacra2_with_alpha(orig_path, dist_path):
    """For images with alpha, compute SSIMULACRA2 with dark/light backgrounds."""
    # Test if images have alpha channel
    has_alpha = False

    try:
        img = Image.open(orig_path)
        has_alpha = img.mode == "RGBA"
    except:
        pass

    if not has_alpha:
        return compute_ssimulacra2(orig_path, dist_path)

    # For images with alpha, compute with dark and light backgrounds
    score_dark = compute_ssimulacra2(orig_path, dist_path, bg=0.1)
    score_light = compute_ssimulacra2(orig_path, dist_path, bg=0.9)

    # Return the worse (lower) of the two scores
    return min(score_dark, score_light)
