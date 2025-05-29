from base_functions import *
from pathlib import Path
import pandas as pd
from skimage.morphology import binary_dilation, ball, remove_small_objects

#code to run for just one image:




# Define input and output paths
image_path = Path("/Users/frederic/Nextcloud/MD_MSC_Project/Jana_Mitograph/Input_pictures/CCCP/CCCP_16.tif")
output_folder = Path("/Users/frederic/Nextcloud/MD_MSC_Project/Jana_Mitograph/Output_summary/CCCP")

# image_path = Path('/Users/frederic/Desktop/untitled folder/Input/021.tif')
# output_folder = Path('/Users/frederic/Desktop/untitled folder/Output')
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
