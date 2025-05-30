"""
# Script: script_batch_processing.py

This script batch-processes a folder of 3D mitochondrial images for connectivity analysis.

Steps:
- Loads all images from `input_dir`
- Skeletonizes and analyzes each image using `process_image_folder`
- Saves outputs including skeleton overlays, 3D movies, and summary CSVs to `output_dir`

Adjust paths and flags below as needed.
"""


from base_functions import *


# Batch-processing code
input_dir = Path('/Users/frederic/Nextcloud/MD_MSC_Project/Jana_Mitograph/Input_pictures')
output_dir = Path('/Users/frederic/Nextcloud/MD_MSC_Project/Jana_Mitograph/Output_summary')

process_image_folder(input_dir, output_dir, generate_visualisation=True, Verbose_for_all_functions=False)


