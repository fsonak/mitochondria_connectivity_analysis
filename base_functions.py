import networkx as nx
from skimage.graph import route_through_array
from skan import csr
import imageio.v3 as iio
import numpy as np
import napari
import matplotlib.pyplot as plt
from skimage import filters, exposure
from scipy.ndimage import gaussian_filter, median_filter
from skimage.filters import unsharp_mask
from skimage.morphology import skeletonize_3d
import os
import shutil
import cv2
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from skimage.measure import label, regionprops


def load_and_preprocess_image(path, sigma=0.8, median_size=2, clip_limit=0.03, verbose=False):
    """
    Load a 3D grayscale image and apply denoising, sharpening, and contrast enhancement.
    """
    if verbose:
        print("[INFO] Loading image...")
    raw_img = iio.imread(path)
    img_norm = raw_img / np.max(raw_img)
    img_denoised = gaussian_filter(img_norm, sigma=sigma)
    img_denoised = median_filter(img_denoised, size=median_size)
    sharpened = unsharp_mask(img_denoised, radius=1, amount=1.5)
    enhanced_img = exposure.equalize_adapthist(sharpened, clip_limit=clip_limit)
    if verbose:
        print("[INFO] Image loaded and preprocessed.")
    return raw_img, enhanced_img


def binarize_image(enhanced_img, percentile=98, verbose=False):
    """
    Binarize the enhanced image using a percentile threshold.
    """
    if verbose:
        print(f"[INFO] Binarizing image using {percentile}th percentile threshold...")
    threshold_value = np.percentile(enhanced_img, percentile)
    binary = enhanced_img > threshold_value
    return binary


def crop_to_skeleton(raw_img, enhanced_img, skeleton, verbose=False):
    """
    Crop the images and skeleton to the bounding box of the skeleton.
    """
    if verbose:
        print("[INFO] Cropping to skeleton bounding box...")
    if not np.any(skeleton):
        raise ValueError("No skeleton detected. Check thresholding or preprocessing settings.")
    coords = np.argwhere(skeleton)
    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0) + 1
    cropped = {
        'img': raw_img[z_min:z_max, y_min:y_max, x_min:x_max],
        'contrast_enhanced': enhanced_img[z_min:z_max, y_min:y_max, x_min:x_max],
        'skeleton': skeleton[z_min:z_max, y_min:y_max, x_min:x_max],
        'bounds': (z_min, z_max, y_min, y_max, x_min, x_max)
    }
    if verbose:
        print("[INFO] Cropping completed.")
    return cropped


def analyze_skeleton(skeleton, verbose=True):
    from skan import csr

    if np.count_nonzero(skeleton) == 0:
        if verbose:
            print("[INFO] Empty skeleton, skipping analysis.")
        return {
            "total_nodes": 0,
            "total_edges": 0,
            "connected_components": 0,
            "largest_component_pct": 0.0
        }

    skeleton_graph = csr.Skeleton(skeleton)
    stats_table = csr.summarize(skeleton_graph, separator='-')

    total_edges = len(stats_table)
    unique_nodes = set()
    for path in skeleton_graph.paths_list():
        if len(path) >= 3:
            unique_nodes.add(path[1])
            unique_nodes.add(path[2])
        else:
            if verbose:
                print(f"[WARNING] Skipping malformed skeleton path: {path}")
    total_nodes = len(unique_nodes)

    labeled_skeleton = label(skeleton)
    connected_components = len(np.unique(labeled_skeleton)) - (1 if 0 in labeled_skeleton else 0)

    # Compute area of largest connected component relative to total
    labeled = label(skeleton)
    regions = regionprops(labeled)
    areas = [r.area for r in regions]
    largest_area = max(areas) if areas else 0
    total_area = np.sum(areas)
    largest_component_pct = (largest_area / total_area) * 100 if total_area > 0 else 0.0

    if verbose:
        print(f"[INFO] Skeleton analysis: Nodes={total_nodes}, Edges={total_edges}, Connected components={connected_components}")
        print(f"[INFO] Largest component percentage: {largest_component_pct:.2f}%")

    return {
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "connected_components": connected_components,
        "largest_component_pct": largest_component_pct
    }


def visualize_and_animate(raw_img, enhanced_img, skeleton, bounds, out_path="mitochondria_3d_rotation.mp4", open_viewer=True, verbose=False):
    """
    Visualize the images and skeleton, create a rotation video, and save a z-projection.
    """
    if verbose:
        print("[INFO] Starting visualization and animation...")
    viewer = napari.Viewer()
    viewer.add_image(enhanced_img, name="Sharpened & Enhanced", colormap='gray', rendering='mip')
    viewer.add_image(raw_img, name="Original", colormap='gray', rendering='mip')
    viewer.add_labels(skeleton.astype(np.uint8), name="Skeleton", opacity=0.7)

    # Add labeled connected components of the skeleton in color
    from skimage.measure import label
    from napari.utils.colormaps import label_colormap
    labeled_components = label(skeleton, connectivity=3)
    viewer.add_labels(labeled_components, name="Connected Components")

    z_min, z_max, y_min, y_max, x_min, x_max = bounds
    viewer.dims.ndisplay = 3
    viewer.camera.center = ((x_max - x_min) // 2, (y_max - y_min) // 2, (z_max - z_min) // 2)
    viewer.camera.zoom = 2.5
    viewer.camera.angles = (45, 45, 45)

    os.makedirs("rotation_frames", exist_ok=True)
    n_frames = 72
    for i in range(n_frames):
        angle = i * 5
        viewer.camera.angles = (45, 45 + angle, 0)
        viewer.screenshot(f"rotation_frames/frame_{i:03d}.png", canvas_only=True)

    frame0 = cv2.imread("rotation_frames/frame_000.png")
    height, width, _ = frame0.shape
    video = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))
    for i in range(n_frames):
        frame = cv2.imread(f"rotation_frames/frame_{i:03d}.png")
        video.write(frame)
    video.release()
    shutil.rmtree("rotation_frames")
    if verbose:
        print(f"[INFO] MP4 saved as {out_path}")

    # Create z-projection (maximum intensity projection)
    z_proj = enhanced_img.max(axis=0)

    # Save the z-projection in the same directory as the MP4
    video_dir = os.path.dirname(out_path)
    video_base = os.path.splitext(os.path.basename(out_path))[0]
    proj_path = os.path.join(video_dir, f"{video_base}_z_projection.tif")
    iio.imwrite(proj_path, (z_proj * 255).astype(np.uint8))
    if verbose:
        print(f"[INFO] Z-projection saved as {proj_path}")

    if open_viewer:
        napari.run()
    return z_proj








def process_image_folder(input_root, output_root, generate_visualisation=True):
    """
    Batch-process 3D mitochondrial images across subfolders.
    Each subfolder in input_root is treated as a condition.
    The results (MP4 + projection + summary CSV) are saved in the mirrored structure in output_root.
    """
    input_root = Path(input_root)
    output_root = Path(output_root)
    stats_list = []

    image_paths = [(condition.name, image)
                   for condition in input_root.iterdir() if condition.is_dir()
                   for image in condition.glob("*.tif")]

    for condition_name, image_path in tqdm(image_paths, desc="Batch Processing", unit="image"):
        output_condition_folder = output_root / condition_name
        output_condition_folder.mkdir(parents=True, exist_ok=True)
        print(f"[BATCH] Processing {image_path.name} in {condition_name}")
        try:
            raw_img, enhanced_img = load_and_preprocess_image(image_path, verbose=False)
            binary_mask = binarize_image(enhanced_img, verbose=False)
            skeleton = skeletonize_3d(binary_mask.astype(bool))
            cropped = crop_to_skeleton(raw_img, enhanced_img, skeleton, verbose=False)
            stats = analyze_skeleton(cropped['skeleton'], verbose=False)
            out_name = image_path.stem

            # Save visualization + z-projection
            if generate_visualisation:

                out_video = output_condition_folder / f"{out_name}.mp4"

                z_proj = visualize_and_animate(
                    cropped['img'], cropped['contrast_enhanced'], cropped['skeleton'],
                    cropped['bounds'], out_path=str(out_video), open_viewer=False, verbose=False
                )

            stats = analyze_skeleton(cropped['skeleton'], verbose=False)

            # Collect all stats for summary
            summary_row = {
                "condition": condition_name,
                "image_name": image_path.name,
                "nodes": stats.get("total_nodes", 0),
                "edges": stats.get("total_edges", 0),
                "components": stats.get("connected_components", 0),
                "largest_component_pct": stats.get("largest_component_pct", 0.0)
            }
            stats_list.append(summary_row)

        except Exception as e:
            print(f"[ERROR] Failed to process {image_path.name}: {e}")

    df = pd.DataFrame(stats_list)
    df.to_csv(output_root / "summary.csv", index=False)
    print("[BATCH]  Done! Summary saved to:", output_root / "summary.csv")
