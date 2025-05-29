from base_functions import *


# Batch-processing code
input_dir = Path('/Users/frederic/Nextcloud/MD_MSC_Project/Jana_Mitograph/Input_pictures')
output_dir = Path('/Users/frederic/Nextcloud/MD_MSC_Project/Jana_Mitograph/Output_summary')

process_image_folder(input_dir, output_dir, generate_visualisation=False)


