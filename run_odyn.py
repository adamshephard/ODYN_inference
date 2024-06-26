"""
Run the complete ODYN pipeline for OED diagnosis and prognosis.

Usage:
  run_odyn.py [options] [--help] [<args>...]
  run_odyn.py --version
  run_odyn.py (-h | --help)
  
Options:
  -h --help                       Show this string.
  --version                       Show version.

  --input_data_file=<string>      Path to csv file containing fold information and targets per slide.
  --input_dir=<string>            Path to input directory containing slides or images.
  --output_dir=<string>           Path to output directory to save results.
  --transformer_weights=<string>  Path to transformer weights.
  --hovernetplus_weights=<string> Path to HoverNet+ weights.
  --mode=<string>                 Tile-level or WSI-level mode. [default: wsi]
  --mlp_weights                   Path to MLP weights.
  --mlp_norm_params               Path to MLP normalization parameters.
  --mlp_cutoff_file               Path to MLP cutoff file.
  --nr_loader_workers=<n>         Number of workers during data loading. [default: 10]
  --nr_post_proc_workers=<n>      Number of workers during post-processing. [default: 10]
  --batch_size=<n>                Batch size for deep learning models. [default: 8]

Use `run_odyn.py --help` to show their options and usage
"""

from docopt import docopt
import os
import glob
import shutil

import numpy as np
import torch
import cv2
from scipy import ndimage
from skimage import morphology
from collections import OrderedDict
from tiatoolbox.utils.misc import imwrite

from dysplasia_segmentation import segment_dysplasia
from epithelium_segmentation import segment_epithelium
from oed_diagnosis import combine_masks, oed_diagnosis
from feature_generation import generate_features
from oed_prognosis import oed_prognosis


if __name__ == '__main__':
    args = docopt(__doc__, help=False, options_first=True, 
                    version='ODYN Inference')

    if args['--help']:
        print(__doc__)
        exit()
        
    if args['--input_data_file']:
        input_data_file = args['--input_data_file']
    else:      
        input_data_file = "/data/ANTICIPATE/github/testdata/sheffield_inference_data.csv" 
    
    if args['--input_dir']:
        input_wsi_dir = args['--input_dir']
    else:      
        input_wsi_dir = "/data/ANTICIPATE/github/testdata/wsis/"
    
    if args['--output_dir']:
        output_dir = args['--output_dir']
    else:
        output_dir = "/data/ANTICIPATE/github/testdata/output/odyn/"
    
    if args['--mode']:
        mode = args['--mode']
        if mode not in ["tile", "wsi"]:
            raise ValueError("Mode must be tile or wsi")
    else:
        mode = "wsi" # or tile
        
    if args['--transformer_weights']:
        transformer_weights = args['--transformer_weights']
    else:
        transformer_weights = "/data/ANTICIPATE/github/ODYN_inference/weights/transunet_external.tar"
        
    if args['--hovernetplus_weights']:
        hovernetplus_weights = args['--hovernetplus_weights']
    else:
        hovernetplus_weights = "/data/ANTICIPATE/github/ODYN_inference/weights/hovernetplus.tar"        
        
    if args['--mlp_weights']:
        mlp_weights = args['--mlp_weights']
    else:
        mlp_weights = "/data/ANTICIPATE/github/ODYN_inference/weights/cross_validation/"        
    
    if args['--mlp_norm_params']:
        mlp_norm_params = args['--mlp_norm_params']
    else:
        mlp_norm_params = "/data/ANTICIPATE/github/ODYN_inference/models/norm_params/"        
   
    if args['--mlp_cutoff_file']:
        mlp_cutoff_file = args['--mlp_cutoff_file']
    else:
        mlp_cutoff_file = "/data/ANTICIPATE/github/ODYN_inference/models/norm_params/cutoff_summary.csv"        
               
    ### 1. Segment Dysplasia ###
    dysp_colour_dict = {
        "nolabel": [0, [0  ,   0,   0]],
        "dysplasia": [1, [255,   0,   0]],
    }
    
    segment_dysplasia(
        input_wsi_dir=input_wsi_dir,
        output_dir=output_dir,
        model_weights=transformer_weights,
        colour_dict=dysp_colour_dict,
        mode=mode,
        nr_loader_workers=int(args['--nr_loader_workers']),
        nr_post_proc_workers=int(args['--nr_post_proc_workers']),
        batch_size=int(args['--batch_size']),
        )

    ### 2. Segment Epithelium/Nuclei ###
    epith_colour_dict = {
        "nolabel": [0, [0  ,   0,   0]],
        "other": [1, [255, 165,   0]],
        "basal": [2, [255, 0,   0]],
        "epith": [3, [0,   255,   0]],
        "keratin": [4, [0,   0,   255]],
    }
    
    segment_epithelium(
        input_wsi_dir=input_wsi_dir,
        output_dir=output_dir,
        model_weights=hovernetplus_weights,
        colour_dict=epith_colour_dict,
        mode=mode,
        nr_loader_workers=int(args['--nr_loader_workers']),
        nr_post_proc_workers=int(args['--nr_post_proc_workers']),
        batch_size=int(args['--batch_size']),
        )
    
    ### 3. Diagnosis, i.e. OED vs normal ###
    oed_diag_threshold = 0.055022543958983 # hardcoded! # dysplasia ratio threshold for determing OED vs normal

    new_colour_dict = {
        "nolabel": [0, [0  ,   0,   0]],
        "other": [1, [255, 165,   0]],
        "dysplasia": [2, [255, 0,   0]],
        "epith": [3, [0,   255,   0]],
    }  
    
    combine_masks(
        input_epith_dir=os.path.join(output_dir, "epith"),
        input_dysp_dir=os.path.join(output_dir, "dysplasia"),
        output_dir=os.path.join(output_dir, "combined"),
        epith_colour_dict=epith_colour_dict,
        dysp_colour_dict=dysp_colour_dict,
        new_colour_dict=new_colour_dict,
    )
    
    oed_diagnosis(
        input_mask_dir=os.path.join(output_dir, "combined"),
        output_dir=os.path.join(output_dir, "diagnoses"),
        colour_dict=new_colour_dict,
        threshold=oed_diag_threshold # hardcoded
        )
    
    
    ### 4. Create patch-level features ###
    generate_features(
        input_wsi_dir=input_wsi_dir,
        mask_dir=os.path.join(output_dir, "combined"),
        nuclei_dir=os.path.join(output_dir, "nuclei"),
        output_dir=os.path.join(output_dir, "features"),
        colour_dict=epith_colour_dict,
        num_processes=int(args['--nr_post_proc_workers']),
    )
    
    ### 5. Prognosis, with MLP ###
    oed_prognosis(
        data_file=input_data_file, 
        data_path=os.path.join(output_dir, "features/0.5-mpp_512_256_dysplasia-0.5/nuclear/pt_files"), 
        norm_parameters=mlp_norm_params, 
        checkpoint_path=mlp_weights,
        output=os.path.join(output_dir, "prognosis"),
        nr_repeats=3, 
        nr_folds=5, 
        aggregation_method="avgtop",
        cutoff_file=mlp_cutoff_file, 
        method="mlp", 
        features="morph_features_104_64ftrs", 
        outcome="transformation",
        )