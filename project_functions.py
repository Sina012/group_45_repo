###################################  Functions for the IAPR project  ####################################
##################################  Group 45 (Romane, Thomas & Sina)  ####################################


##################################  IMPORT NECESSARY LIBRARIES  ####################################

import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Sequence
from pathlib import Path
from collections import deque
import cv2
import pandas as pd

from scipy import ndimage as ndi
from scipy.ndimage import binary_fill_holes
from skimage import color, filters, morphology, measure, exposure
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern

############################  1) PLOTTING & LOADING DATA FUNCTIONS  ##############################

def load_and_show_image(path_to_img):
    assert os.path.exists(path_to_img), "ERROR: Image not found"
    img = np.array(Image.open(path_to_img))

    # Display image
    plt.imshow(img)
    plt.axis('off')
    return img


def load_labels_csv(data_path):
    """
    Load the training-dataset label file.

    Parameters
    ----------
    data_path : str
        Path to the CSV file that contains:
        ┌──────────────┬───────────────┬───┬───────────────┐
        │ image_name   │ choco_type_01 │ … │ choco_type_13 │
        └──────────────┴───────────────┴───┴───────────────┘
        (one header row + one data row per training image)

    Returns
    -------
    filenames : np.ndarray, shape (N,)
        1D array with the JPEG/PNG filenames for the N training images
        (exact strings as they appear in the first column).

    counts    : np.ndarray, shape (N, 13)
        Integer matrix where each row holds the 13 chocolate type counts
        for the corresponding image.
    """

    df = pd.read_csv(data_path)
    filenames = df.iloc[:, 0].to_numpy()
    counts = df.iloc[:, 1:].to_numpy(dtype=int)

    return filenames, counts


def load_jpg_folder(path_to_folder):
    """
    Load every *.jpg / *.jpeg image in a directory into one NumPy tensor.

    Parameters
    ----------
    path_to_folder : str
        Directory that contains the JPEG files.

    Returns
    -------
    images : np.ndarray
        Image stack with dtype uint8.  H × W (and channels) are the
        same for **all** images.

    Raises
    ------
    ValueError
        If the folder contains no JPEG files or if the images do not
        all share identical dimensions.
    """
    folder = Path(path_to_folder).expanduser()
    files  = sorted(
        f for f in folder.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg"}
    )

    folder = Path(path_to_folder).expanduser()
    files = sorted(f for f in folder.iterdir() if f.suffix.lower() in {".jpg", ".jpeg"})

    if not files:
        raise ValueError(f"No JPEG files found in {folder!s}")

    # Get dimensions
    ref_img = Image.open(files[0]).convert("RGB")
    ref_arr = np.asarray(ref_img, dtype=np.uint8)
    H, W = ref_arr.shape[:2]
    N = len(files)

    # Preallocate array
    images = np.empty((N, H, W, 3), dtype=np.uint8)
    images[0] = ref_arr

    for i, f in enumerate(files[1:], 1):
        img = Image.open(f).convert("RGB")
        if img.size != (W, H):
            raise ValueError(f"Image {f.name} has size {img.size}, expected {(W, H)}")
        images[i] = np.asarray(img, dtype=np.uint8)

    return images


def plot_image_comparison(img1: np.ndarray, img2: np.ndarray, title: str):
    """
    Plot the original image and its thresholded version

    Args
    ----
    img: np.ndarray (M, N, 3)
        Input image of shape MxNx3.
    title: str
        Title of the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img1)
    axes[1].imshow(img2, interpolation=None)
    [a.axis('off') for a in axes]
    plt.suptitle(title)
    plt.tight_layout()


def plot_batch_comparison(images: np.ndarray, masks: np.ndarray, title: str):
    """
    Plot a batch of image–mask comparisons side-by-side.

    Parameters
    ----------
    images : (N, H, W, 3) or (N, H, W) ndarray
        Original images (RGB or grayscale).
    masks : (N, H, W) ndarray
        Corresponding binary masks.
    title : str
        Global title of the whole plot.
    """
    n = len(images)
    fig, axes = plt.subplots(n, 2, figsize=(10, 3 * n))

    if n == 1:
        axes = np.expand_dims(axes, 0)  # ensure 2D structure if n == 1

    for i in range(n):
        axes[i, 0].imshow(images[i], cmap='gray' if images[i].ndim == 2 else None)
        axes[i, 0].set_title(f"Image {i+1}")
        axes[i, 1].imshow(masks[i], cmap='gray')
        axes[i, 1].set_title(f"Mask {i+1}")
        axes[i, 0].axis('off')
        axes[i, 1].axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_contour_batch(images, contours_list):
    """
    Plot contours on top of their corresponding images/masks.

    Parameters
    ----------
    images : list or array of shape (N, H, W[, 3])
        List or array of N images or masks.
    contours_list : list of (M_i, 2) arrays
        Each element is the (x, y) contour points for that image.
    title : str
        Main title for the figure.
    """
    n = len(images)
    plt.figure(figsize=(10, 3 * n))

    for i in range(n):
        plt.subplot(n, 1, i + 1)
        plt.imshow(images[i], cmap='gray' if images[i].ndim == 2 else None)
        plt.plot(contours_list[i][:, 0], contours_list[i][:, 1], 'r-', linewidth=1)
        plt.title(f"Image {i+1} with Contour")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def plot_fourier_reconstructions(contours_list, x_recon_batch, y_recon_batch):
    """
    Plot original vs reconstructed contours for a batch of objects.

    Parameters
    ----------
    contours_list : list of (K_i, 2) arrays
        Original contours (e.g., from find_contour_single).
    x_recon_batch : (N, n_samples) array
        Reconstructed x coordinates (real part of inverse FFT).
    y_recon_batch : (N, n_samples) array
        Reconstructed y coordinates (imaginary part of inverse FFT).
    """
    n = len(contours_list)
    plt.figure(figsize=(10, 3 * n))

    for i in range(n):
        plt.subplot(n, 2, 2 * i + 1)
        plt.plot(contours_list[i][:, 0], contours_list[i][:, 1], label="Original", lw=1)
        plt.title(f"Original Contour {i+1}")
        plt.gca().invert_yaxis()
        plt.axis('equal')

        plt.subplot(n, 2, 2 * i + 2)
        plt.plot(x_recon_batch[i], y_recon_batch[i], 'g-o', label="Reconstruction")
        plt.title(f"Reconstructed Contour {i+1}")
        plt.gca().invert_yaxis()
        plt.axis('equal')

    plt.tight_layout()
    plt.show()


def show_rgb_histograms(rgb_images, histograms, bins=8):
    n = len(rgb_images)
    for i in range(n):
        plt.figure(figsize=(10, 4))

        # Left: image
        plt.subplot(1, 2, 1)
        plt.imshow(rgb_images[i])
        plt.axis('off')
        plt.title(f"Image {i+1}")

        # Right: histogram
        plt.subplot(1, 2, 2)
        r_hist = histograms[i][:bins]
        g_hist = histograms[i][bins:2*bins]
        b_hist = histograms[i][2*bins:]
        x = np.arange(bins)

        plt.bar(x - 0.2, r_hist, width=0.2, color='r', label='Red')
        plt.bar(x,         g_hist, width=0.2, color='g', label='Green')
        plt.bar(x + 0.2, b_hist, width=0.2, color='b', label='Blue')
        plt.title("Normalized RGB Histogram")
        plt.xlabel("Bin")
        plt.ylabel("Frequency")
        plt.legend()

        plt.tight_layout()
        plt.show()


def show_lbp_histograms(rgb_images, lbp_hists):
    n = len(rgb_images)
    for i in range(n):
        plt.figure(figsize=(10, 4))

        # Left: image
        plt.subplot(1, 2, 1)
        plt.imshow(rgb_images[i])
        plt.axis('off')
        plt.title(f"Image {i+1}")

        # Right: LBP histogram
        plt.subplot(1, 2, 2)
        plt.bar(np.arange(len(lbp_hists[i])), lbp_hists[i], color='gray')
        plt.title("LBP Histogram")
        plt.xlabel("LBP Code")
        plt.ylabel("Frequency")

        plt.tight_layout()
        plt.show()


####################################  2) SEGMENTATION  ############################################


def compute_seeds(
    img,
    *,
    min_area = 10000,          # discard blobs smaller than this (px)
    max_area: int = 500000,  # discard big objects
    gaussian_sigma = 1.0  # pre–blur for noise robustness
):
    """
    Detect candidate seed points for region growing.

    Parameters
    ----------
    img : (H, W, 3) or (H, W) ndarray
        Input image as uint8 / float in [0, 255] or [0, 1].
    min_area : int, optional
        Blobs with fewer pixels are ignored.
    gaussian_sigma : float, optional
        Std-dev used to blur before thresholding.

    Returns
    -------
    seeds : list[tuple[int, int]]
        List of (row, col) seed coordinates.
    """

    # Pre‑blur (suppresses fine texture / noise)
    img_blur = ndi.gaussian_filter(img, sigma=gaussian_sigma)

    # Thresholding
    gray = color.rgb2gray(img_blur) if img_blur.ndim == 3 else img_blur
    t = filters.threshold_otsu(gray)
    mask = gray < t


    #edges = filters.sobel(gray)
    #t_edge = filters.threshold_otsu(edges)
    #mask = edges > t_edge
    #mask = morphology.binary_dilation(mask, morphology.disk(2))
    #mask = ndi.binary_fill_holes(mask)

    # Morphological clean‑up
    mask = morphology.remove_small_objects(mask, min_size=min_area)
    mask = morphology.binary_opening(mask, morphology.disk(3))

    # Connected‑component labelling
    labeled, _ = ndi.label(mask)
    regions = measure.regionprops(labeled)
    print("Number of regions found:", len(regions))

    # Collect seed points (integer centroids)
    seeds = []
    for region in regions:
        print(f"area: {region.area}")
        if min_area <= region.area <= max_area:
            seeds.append(tuple(map(int, region.centroid)))

    return seeds, mask


def compute_seeds_tot(
    imgs: np.ndarray,
    *,
    min_area: int = 10000,
    max_area: int = 500000,
    gaussian_sigma: float = 1.0,
):
    """
    Compute seeds for a *batch* of images.

    Parameters
    ----------
    imgs : (N,H,W[,C]) ndarray
        Batch of N images (gray or RGB) in the same format accepted by
        ``compute_seeds``.
    min_area, max_area, gaussian_sigma : see ``compute_seeds``.

    Returns
    -------
    all_seeds : list[list[(row,col)]]
        all_seeds[i] is the seed list for imgs[i].
    masks : (N,H,W) bool ndarray
        Binary masks produced internally by ``compute_seeds`` for each image.
    """
    all_seeds: List[List[Tuple[int, int]]] = []
    masks: List[np.ndarray] = []

    for img in imgs:
        seeds, mask = compute_seeds(
            img,
            min_area=min_area,
            max_area=max_area,
            gaussian_sigma=gaussian_sigma,
        )
        all_seeds.append(seeds)
        masks.append(mask)

    return all_seeds, np.stack(masks, axis=0)


def region_growing(
        img: np.ndarray,
        seeds,
        tol: float = 55,  # by experimenting: for lot of chocolates: 55 = best (> => shadow also detected) BUT some chocolate handle tol=70
        connectivity: int = 8
    ):
    """Region–growing segmentation.

    This version automatically converts an RGB image to grayscale using
    *skimage.color.rgb2gray*, so you can pass either a 2D grayscale array
    or a 3D (H,W,3) RGB array.

    Parameters
    ----------
    img : ndarray
        2D grayscale or 3D RGB image.  Accepts uint8/16 or float.
    seeds : iterable of (row, col)
        Starting pixels known to belong to the desired region.
    tol : float, default 5.0
        Maximum |I(p) – I(region_mean)| allowed for a pixel to join the
        region.  Adapt *tol* to your image dynamic range.
    connectivity : {4,8}, default 8
        Pixel neighbourhood definition.

    Returns
    -------
    mask : 2D boolean ndarray
        Pixels labelled **True** belong to the grown region.
    """

    # ---- 1.  Ensure grayscale ------------------------------------------------
    if img.ndim == 3 and img.shape[-1] == 3:
        # skimage.rgb2gray returns float32/64 in [0, 1].  Bring back to the
        # original scale if the input was uint‑type for tol consistency.
        gray = rgb2gray(img)
        if img.dtype == np.uint8:
            img = (gray * 255).astype(np.float32)
        elif img.dtype == np.uint16:
            img = (gray * 65535).astype(np.float32)
        else:
            img = gray.astype(np.float32)
    elif img.ndim == 2:
        img = img.astype(np.float32)
    else:
        raise ValueError("img must be 2D grayscale or 3D RGB image")

    h, w = img.shape

    # ---- 2.  Init ------------------------------------------------------------
    seeds = [(int(r), int(c)) for r, c in seeds]
    mask     = np.zeros((h, w), dtype=bool)
    visited  = np.zeros_like(mask)
    q        = deque()

    seed_vals = [img[r, c] for r, c in seeds]
    region_mean = float(np.mean(seed_vals))

    for r, c in seeds:
        q.append((r, c))
        visited[r, c] = True
        mask[r, c]    = True

    # ---- 3.  Choose neighbourhood offsets -----------------------------------
    if connectivity == 4:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == 8:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)]
    else:
        raise ValueError("connectivity must be 4 or 8")

    # ---- 4.  Grow ------------------------------------------------------------
    region_size = len(seeds)
    while q:
        r, c = q.popleft()
        for dr, dc in nbrs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                visited[nr, nc] = True
                if abs(float(img[nr, nc]) - region_mean) <= tol:
                    mask[nr, nc] = True
                    q.append((nr, nc))

                    # online update of the mean intensity
                    region_size += 1
                    region_mean += (float(img[nr, nc]) - region_mean) / region_size

    return mask


def region_growing_tot(
    imgs,
    seeds_tot,
    *,
    tol = 55.0,
    connectivity = 8,
):
    """
    Region-growing for a batch of images.

    Parameters
    ----------
    imgs : (N,H,W[,C]) ndarray
    seeds_tot : list[list[(row,col)]]
        Seed list returned by ``compute_seeds_tot``.
    tol, connectivity : see ``region_growing``.

    Returns
    -------
    masks : (N,H,W) bool ndarray
        Region-growing result for every image.
    """
    grown_masks: List[np.ndarray] = []

    for img, seeds in zip(imgs, seeds_tot):
        grown_masks.append(
            region_growing(img, seeds, tol=tol, connectivity=connectivity)
        )

    return np.stack(grown_masks, axis=0)


def postprocess_region_growing(mask: np.ndarray):
    """
    Hole-filling (or any other post-processing) for mask.

    Parameters
    ----------
    mask : (H,W) bool ndarray
        Output of ``region_growing``.

    Returns
    -------
    processed : (H,W) bool ndarray
        Post-processed mask.
    """
    postprocessed_mask = binary_fill_holes(mask)
    # other postprocess needed???? morpho operations????????? remove_holes / remove_objects?????

    return postprocessed_mask


def postprocess_region_growing_tot(masks: np.ndarray):
    """
    Hole-filling (or any other post-processing) for every mask in a batch.

    Parameters
    ----------
    masks : (N,H,W) bool ndarray
        Output of ``region_growing_tot``.

    Returns
    -------
    processed : (N,H,W) bool ndarray
        Post-processed masks.
    """
    processed: List[np.ndarray] = [
        postprocess_region_growing(mask) for mask in masks
    ]
    return np.stack(processed, axis=0)


#################################  3) OBJECT DESCRIPTION  #########################################


def find_all_contours(images: np.ndarray):
    """
    Find the contours for the set of images
    
    Args
    ----
    images: np.ndarray (N, 28, 28)
        Source images to process

    Return
    ------
    contours: list of np.ndarray
        List of N arrays containing the coordinates of the contour. Each element of the 
        list is an array of 2d coordinates (K, 2) where K depends on the number of elements 
        that form the contour. 
    """
    N, _, _ = np.shape(images)
    contours = [np.array([[0, 0], [1, 1]]) for i in range(N)]

    for i in range(N):
        contours[i] = find_contour_single(images[i])

    return contours

def find_contour_single(image: np.ndarray):
    """
    Find the contours for the image
    
    Args
    ----
    image: np.ndarray (N, 28, 28)
        Source images to process

    Return
    ------
    contours: list of np.ndarray
        List of N arrays containing the coordinates of the contour. Each element of the 
        list is an array of 2d coordinates (K, 2) where K depends on the number of elements 
        that form the contour. 
    """

    # Fill in dummy values (fake points)
    contours = np.array([[0, 0], [1, 1]])

    image_corrected = (image > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(image_corrected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    largest_contour = largest_contour.squeeze()

    return largest_contour


def compute_region_descriptors(binary_mask):
    """
    Compute region-based descriptors from a binary mask of a segmented chocolate.

    Parameters:
        binary_mask (np.ndarray): Binary image (uint8) where the object has value 1 (or 255), and background is 0.

    Returns:
        dict: Dictionary of descriptors: area, elongation, rectangularity, compacity, and Hu moments.
    """
    # Mask should be uint8
    mask_u8 = (binary_mask>0).astype(np.uint8) * 255

    # Find contours for rectangularity calculation
    contour = find_contour_single(binary_mask)

    # Area
    area = cv2.contourArea(contour)

    # Rectangularity
    x, y, w, h = cv2.boundingRect(contour)
    rectangularity = area / (w*h)

    # Compacity (C = 4πA / P²)
    perimeter = cv2.arcLength(contour, True)
    compacity = (4 * np.pi * area) / (perimeter ** 2)

    # Moments and Elongation
    M = cv2.moments(mask_u8, binaryImage=True)
    cov = np.array([[M['mu20'], M['mu11']],
                [M['mu11'], M['mu02']]])
    eig = np.sort(np.linalg.eigvals(cov))[::-1]
    elongation = np.sqrt(eig[0] / eig[1]) if eig[1] else 0 

    hu = cv2.HuMoments(M).flatten().tolist()

    return {
        'area': area,
        'rectangularity': rectangularity,
        'compacity': compacity,
        'elongation': elongation,
        'hu_moments': hu
    }


def compute_all_region_descriptors(binary_masks: np.ndarray):
    """
    Compute region-based descriptors for every mask in a batch.

    Parameters
    ----------
    binary_masks : (N, H, W) or (N, H, W, 1) ndarray
        Batch of N binary masks (non-zero = object).  The datatype can be
        anything; it is thresholded internally.

    Returns
    -------
    descriptors_list : list[dict]
        `descriptors_list[i]` is exactly what `compute_region_descriptors`
        would return for the i-th mask — i.e. a dict containing:
        * area
        * rectangularity
        * compacity
        * elongation
        * hu_moments  (list of 7 floats)
    """
    descriptors_list: List[Dict[str, object]] = []

    for mask in binary_masks:
        descriptors = compute_region_descriptors(mask)
        descriptors_list.append(descriptors)

    return descriptors_list


def linear_interpolation_single(contour: np.ndarray, n_samples: int = 11):
    """
    Perform interpolation/resampling of a single contour across n_samples.

    Args
    ----
    contour: np.ndarray
        Array of 2D coordinates (K, 2) representing the contour, 
        where K is the number of contour points.
    n_samples: int
        Number of samples to consider along the contour.

    Return
    ------
    contour_inter: np.ndarray (n_samples, 2)
        Interpolated contour with n_samples points.
    """

    x = contour[:, 0]
    y = contour[:, 1]
    K = len(contour)
    t = np.zeros(K)

    # Compute cumulative distances (used as "time")
    for j in range(1, K):
        dist = np.sqrt((x[j] - x[j-1])**2 + (y[j] - y[j-1])**2)
        t[j] = t[j-1] + dist

    # Create uniform sampling points along the cumulative length
    t_new = np.linspace(0, t[-1], n_samples)

    # Interpolate x and y coordinates separately
    x_new = np.interp(t_new, t, x)
    y_new = np.interp(t_new, t, y)

    contour_inter = np.stack((x_new, y_new), axis=1)

    return contour_inter


def compute_single_descriptor_interpolated(contour: np.ndarray, n_samples: int = 11):
    """
    Compute Fourier descriptor of a single contour with interpolation 
    when the number of points is less than n_samples.

    Args
    ----
    contour: np.ndarray
        Array of 2D coordinates (K, 2) representing the contour, 
        where K is the number of contour points.
    n_samples: int
        Number of samples to consider. The contour is resampled to exactly n_samples 
        using linear interpolation before computing the FFT.

    Return
    ------
    descriptor: np.ndarray complex (n_samples,)
        Computed complex Fourier descriptor for the given input contour.
    """

    resampled_contour = linear_interpolation_single(contour, n_samples)
    complex_contour = resampled_contour[:, 0] + 1j * resampled_contour[:, 1]

    descriptor = np.fft.fft(complex_contour)

    return descriptor


def compute_descriptor_interpolated(contours, n_samples = 11):
    """
    Compute (interpolated) Fourier descriptors for *multiple* contours.

    Parameters
    ----------
    contours : sequence of (K_i, 2) ndarrays
        Each element is the (x,y) contour of one object.
    n_samples : int, default 11
        Number of samples used in the resampling step (must match whatever
        you used when training / comparing).

    Returns
    -------
    descriptors : (N, n_samples) complex ndarray
        `descriptors[i]` is exactly what `compute_single_descriptor_interpolated`
        would return for `contours[i]`.
    """
    desc_list = [
        compute_single_descriptor_interpolated(c, n_samples=n_samples)
        for c in contours
    ]
    return np.vstack(desc_list)


def compute_reverse_descriptor(descriptor: np.ndarray, n_samples: int = 11):
    """
    Reverse a Fourier descriptor to xy coordinates given a number of samples.
    
    Args
    ----
    descriptor: np.ndarray (D,)
        Complex descriptor of length D.
    n_samples: int
        Number of samples to consider to reverse transformation.

    Return
    ------
    x: np.ndarray complex (n_samples,)
        x coordinates of the contour
    y: np.ndarray complex (n_samples,)
        y coordinates of the contour
    """

    x = np.zeros(n_samples)
    y = np.zeros(n_samples)

    reverse_fourier = np.fft.ifft(descriptor)

    if np.shape(reverse_fourier)[0] > n_samples:
        reverse_fourier = reverse_fourier[:n_samples]

    for i in range(n_samples):
        x[i] = reverse_fourier[i].real
        y[i] = reverse_fourier[i].imag

    return x, y


def compute_reverse_descriptors(descriptors, n_samples = 11):
    """
    Convert a *batch* of Fourier descriptors back to x/y coordinates.

    Parameters
    ----------
    descriptors : (N, D) complex ndarray
        Batch of complex descriptors (length‑D, usually `n_samples`).
    n_samples : int, default 11
        Number of contour points you want back for every descriptor.

    Returns
    -------
    xs : (N, n_samples) float ndarray
    ys : (N, n_samples) float ndarray
        Coordinates of the reconstructed contours.
    """
    xs, ys = [], []
    for d in descriptors:
        x, y = compute_reverse_descriptor(d, n_samples=n_samples)
        xs.append(x)
        ys.append(y)

    return np.vstack(xs), np.vstack(ys)


def color_mean_std(rgb: np.ndarray, mask: np.ndarray):
    """Mean and standard deviation of each RGB channel inside mask.

    Parameters
    ----------
    rgb : ndarray, shape (H, W, 3), uint8 or float
    mask : bool ndarray, shape (H, W)
        True for pixels that belong to the object.

    Returns
    -------
    feat : 1D float array, length 6  = [R̄, Ḡ, B̄, σR, σG, σB]
    """
    assert rgb.ndim == 3 and rgb.shape[-1] == 3, "rgb must be H*W*3"
    pix = rgb[mask]                     # (N, 3)
    mean = pix.mean(axis=0)
    std = pix.std(axis=0)
    return np.concatenate([mean, std])


def color_hist(rgb: np.ndarray, mask: np.ndarray, bins: int = 8):
    """Concatenated, L1-normalised histograms of each colour channel.

    By default returns 24 numbers (8 bins * 3 channels).
    """
    pix = rgb[mask]
    if pix.dtype != np.float32 and pix.dtype != np.float64:
        # scale 0‑255 → 0‑1 for consistent binning
        pix = pix.astype(float) / 255.0
    hists = []
    for ch in range(3):
        hist, _ = np.histogram(pix[:, ch], bins=bins, range=(0.0, 1.0), density=False)
        hist = hist.astype(float)
        hist /= hist.sum() + 1e-12         # L1 normalise
        hists.append(hist)
    return np.concatenate(hists)


def color_descriptors(rgbs, masks: np.ndarray, *, bins = 8):
    """
    Compute colour statistics for a *set* of images.

    Parameters
    ----------
    rgbs  : (N,H,W,3) uint8 / float ndarray
    masks : (N,H,W)   bool ndarray
    bins  : int, default 8
        Number of histogram bins per channel (passed to `color_hist`).

    Returns
    -------
    means_and_stds : (N, 6)  float ndarray
        Row i = [R̄, Ḡ, B̄, σR, σG, σB] of image i.
    histograms     : (N, 3*bins) float ndarray
        Concatenated RGB histograms for every image.
    """
    N = rgbs.shape[0]
    mean_std_list = []
    hist_list     = []

    for i in range(N):
        mstd = color_mean_std(rgbs[i], masks[i])
        hist = color_hist(rgbs[i], masks[i], bins=bins)
        mean_std_list.append(mstd)
        hist_list.append(hist)

    return np.vstack(mean_std_list), np.vstack(hist_list)


def lbp_hist(rgb: np.ndarray, mask: np.ndarray, P: int = 8, R: float = 1.0,
             method: str = "uniform"):
    """Rotation-invariant LBP histogram inside the mask.

    Parameters
    ----------
    P, R : classical LBP neighbourhood parameters.
    method : 'uniform' gives (P+2) unique codes.

    Returns
    -------
    hist : 1D float array, length depends on method (for 'uniform', P+2).
    """
    gray = rgb2gray(rgb)
    lbp = local_binary_pattern(gray, P=P, R=R, method=method)
    lbp = lbp[mask]
    n_bins = int(lbp.max()) + 1
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=False)
    hist = hist.astype(float)
    hist /= hist.sum() + 1e-12
    return hist


def lbp_histograms(rgbs: np.ndarray, masks: np.ndarray, *, P: int = 8, R: float = 1.0,
                    method: str = "uniform"):
    """
    Compute rotation-invariant LBP histograms for a batch.

    Parameters
    ----------
    rgbs  : (N,H,W,3) ndarray
    masks : (N,H,W)   ndarray (bool)
    P, R, method : forwarded to `lbp_hist`.

    Returns
    -------
    lbp_hists : (N, L) float ndarray
        `L = P+2` when `method='uniform'`; otherwise depends on `method`.
        Row i contains the LBP histogram of image i.
    """
    hists = [
        lbp_hist(rgbs[i], masks[i], P=P, R=R, method=method)
        for i in range(rgbs.shape[0])
    ]
    return np.vstack(hists)


def full_descriptor(img, mask, contour, n_samples_fourier=13, use_color=True, use_texture=True):
    # Compute region-based descriptors
    region_desc = compute_region_descriptors(mask)
    region_feat = np.atleast_1d(np.array([
        region_desc["area"],
        region_desc["rectangularity"],
        region_desc["compacity"],
        region_desc["elongation"],
        *region_desc["hu_moments"]
    ]))

    # Fourier descriptor
    fourier_descriptor = compute_single_descriptor_interpolated(contour, n_samples=n_samples_fourier)

    # Initialize descriptor list
    desc = [region_feat, fourier_descriptor]

    if use_color:
        desc.extend([
            color_mean_std(img, mask),
            color_hist(img, mask)
        ])
    if use_texture:
        desc.append(lbp_hist(img, mask))
    return np.concatenate(desc)


def full_descriptor_batch(
    imgs: np.ndarray,                  # (N, H, W, 3)
    masks: np.ndarray,                 # (N, H, W)
    contours_list: list[np.ndarray],   # list of (K_i, 2) arrays
    *,
    use_color: bool = True,
    use_texture: bool = True,
    n_fourier_samples: int = 11
):
    """
    Compute full feature vectors for a batch of images.

    Parameters
    ----------
    imgs : (N, H, W, 3) ndarray
        RGB image batch.
    masks : (N, H, W) ndarray
        Binary masks for each image.
    contours_list : list of (K_i, 2) arrays
        List of contours for each image.
    use_color : bool, default True
        Whether to include color features.
    use_texture : bool, default True
        Whether to include LBP features.
    n_fourier_samples : int, default 11
        Length of the Fourier descriptor.

    Returns
    -------
    features : (N, D) ndarray
        Matrix of feature vectors.
    """

    features = []

    for i in range(len(imgs)):
        feat = full_descriptor(
            img=imgs[i],
            mask=masks[i],
            contour=contours_list[i],
            use_color=use_color,
            use_texture=use_texture,
            n_fourier_samples=n_fourier_samples
        )
        features.append(feat)

    return np.vstack(features)


def concatenate_descriptors_batch(
    region_feats: list[dict],
    fourier_descs: np.ndarray,
    color_descs: np.ndarray = None,
    color_hists: np.ndarray = None,
    lbps: np.ndarray = None
):
    """
    Concatenate multiple descriptor types into full feature vectors for a batch.

    Parameters
    ----------
    region_feats : (N, D1) ndarray
    fourier_descs : (N, D2) ndarray
        Can be complex; real and imaginary parts are concatenated.
    color_descs : (N, D3) ndarray or None
    color_hists : (N, D4) ndarray or None
    lbps : (N, D5) ndarray or None

    Returns
    -------
    features : (N, D_total) float ndarray
        Each row is the concatenated descriptor vector for one image.
    """
    region_features = []
    for desc in region_feats:
        row = [
            desc["area"],
            desc["rectangularity"],
            desc["compacity"],
            desc["elongation"],
            *desc["hu_moments"]
        ]
        region_features.append(row)
    region_feats_np = np.array(region_features, dtype=float)

    descs = [region_feats_np]

    # Convert complex Fourier descriptors to [real | imag] format
    if np.iscomplexobj(fourier_descs):
        real = fourier_descs.real
        imag = fourier_descs.imag
        descs.append(np.hstack([real, imag]))
    else:
        descs.append(np.atleast_2d(fourier_descs))

    if color_descs is not None:
        descs.append(np.atleast_2d(color_descs))
    if color_hists is not None:
        descs.append(np.atleast_2d(color_hists))
    if lbps is not None:
        descs.append(np.atleast_2d(lbps))

    return np.hstack(descs)


##################################  4) CLASSIFICATION  ############################################



##################################  5) CROSS-VALIDATION  ##########################################



########################  6) FINAL TESTING & SUBMISSION FILE CREATION  ############################