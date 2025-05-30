Mitochondrial Connectivity Analysis

This repository provides tools for analyzing the 3D connectivity of mitochondrial networks from fluorescence microscopy images. The pipeline uses skeletonization and graph-based analysis to quantify mitochondrial morphology and structure.

Features
	•	3D skeletonization of segmented mitochondrial volumes
	•	Graph construction and connectivity analysis using skan
	•	Quantification of node degrees:
	•	Degree 1 nodes (“dead-ends”)
	•	Degree 2 nodes (“linear connections”)
	•	Degree ≥3 nodes (“branch points”)
	•	CSV export of per-image statistics
	•	Combined summary plots per experimental condition (e.g. pie charts)
	•	Generation of 3D skeleton movies
	•	Interactive visualization using napari
	•	Batch processing for entire folders of images

Workflow
	1.	Load and segment mitochondrial fluorescence images
	2.	Skeletonize the 3D volume using skimage
	3.	Analyze the skeleton with skan to extract nodes and paths
	4.	Calculate per-image and per-condition connectivity metrics
	5.	Visualize results using pie charts, 3D skeleton overlays, and movies

Requirements
	•	Python 3.8+
	•	numpy, pandas, scipy
	•	scikit-image
	•	skan
	•	matplotlib
	•	napari
	•	tqdm

Install with pip or conda.

Outputs
	•	Skeleton overlays and summaries
	•	CSVs with node and edge statistics
	•	Summary pie charts per condition
	•	3D animation of skeletons

Usage

Set your folder paths in script_batch_processing.py, then run the script to analyze images and generate all outputs automatically.

Author

Frédéric Sonak
University of Freiburg

Updated: 30th May 2025


