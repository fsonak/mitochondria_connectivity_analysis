"""
# Script: sript_just_one_example_file.py

This script demonstrates the full mitochondrial connectivity analysis pipeline
on a single 3D image file. It includes preprocessing, skeletonization, graph-based
analysis, and visualization.

Steps:
- Load and enhance a 3D mitochondrial fluorescence image
- Binarize and skeletonize the structure in 3D
- Crop the region of interest around the skeleton
- Analyze the skeleton for graph-based connectivity metrics
- Visualize the structure and create a 3D movie
- Save results including summary statistics to CSV

Edit the `image_path` and `output_folder` variables as needed.
"""

from base_functions import *
from pathlib import Path
import pandas as pd


# Define input and output paths
image_path = Path("/Users/frederic/Nextcloud/MD_MSC_Project/Jana_Mitograph/Input_pictures/CCCP/CCCP_16.tif")
output_folder = Path("/Users/frederic/Nextcloud/MD_MSC_Project/Jana_Mitograph/Output_summary/CCCP")

output_folder.mkdir(parents=True, exist_ok=True)

# Step 1: Load and preprocess
raw_img, enhanced_img = load_and_preprocess_image(image_path, verbose=True)

# Step 2: Threshold and skeletonize
binary_mask = binarize_image(enhanced_img, verbose=True)
skeleton = skeletonize_3d(binary_mask)

# Step 3: Crop to region of interest
cropped = crop_to_skeleton(raw_img, enhanced_img, skeleton, verbose=True)

# Step 4: Analyze connectivity
stats = analyze_skeleton(cropped["skeleton"], verbose=True)

#Step 5: Visualize and animate
out_video = output_folder / f"{image_path.stem}.mp4"
z_proj = visualize_and_animate(
    cropped["img"],
    cropped["contrast_enhanced"],
    cropped["skeleton"],
    cropped["bounds"],
    out_path=str(out_video),
    open_viewer=True,
    verbose=True
)

# Step 6: Save statistics and 2D projection
df = pd.DataFrame([{
    "condition": image_path.parent.name,
    "image_name": image_path.name,
    "nodes": stats.get("total_nodes", 0),
    "edges": stats.get("total_edges", 0),
    "components": stats.get("connected_components", 0)
}])
df.to_csv(output_folder / "summary_single.csv", index=False)
