"""
Run the complete ODYN pipeline for OED diagnosis and prognosis.

Usage:
  run_odyn.py [options] [--help] [<args>...]
  run_odyn.py --version
  run_odyn.py (-h | --help)
  
Options:
  -h --help                   Show this string.
  --version                   Show version.

  --input_dir=<string>        Path to input directory containing slides or images.
  --output_dir=<string>       Path to output directory to save results.
  --mode=<string>             Tile-level or WSI-level mode. [default: wsi]
  --nr_loader_workers=<n>     Number of workers during data loading. [default: 10]
  --nr_post_proc_workers=<n>  Number of workers during post-processing. [default: 10]
  --batch_size=<n>            Batch size for deep learning models. [default: 8]

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
# from oed_diagnosis import oed_diagnosis
from feature_generation import generate_features
# from oed_prognosis import oed_prognosis


if __name__ == '__main__':
    args = docopt(__doc__, help=False, options_first=True, 
                    version='ODYN Inference')

    if args['--help']:
        print(__doc__)
        exit()

    if args['--input_dir']:
        input_wsi_dir = args['--input_dir']
    else:      
        input_wsi_dir = "/data/ANTICIPATE/outcome_prediction/MIL/github_testdata/wsis/"
    
    if args['--output_dir']:
        output_dir = args['--output_dir']
    else:
        output_dir = "/data/ANTICIPATE/outcome_prediction/MIL/github_testdata/output_odyn/"
    
    if args['--mode']:
        mode = args['--mode']
        if mode not in ["tile", "wsi"]:
            raise ValueError("Mode must be tile or wsi")
    else:
        mode = "wsi" # or tile

    ### 1. Segment Dysplasia ###
    transformer_weights = "/data/ANTICIPATE/outcome_prediction/MIL/github_testdata/models/TransUNet_model_best.pth"
    dysp_colour_dict = {
        "nolabel": [0, [0  ,   0,   0]],
        "dysplasia": [1, [255,   0,   0]],
    }
    
    segment_dysplasia(
        input_wsi_dir=input_wsi_dir,
        output_dir=os.path.join(output_dir, "dysplasia"),
        transformer_weights=transformer_weights,
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
        output_dir=os.path.join(output_dir, "hovernetplus"),
        colour_dict=epith_colour_dict,
        mode=mode,
        nr_loader_workers=int(args['--nr_loader_workers']),
        nr_post_proc_workers=int(args['--nr_post_proc_workers']),
        batch_size=int(args['--batch_size']),
        )
    
    ### 3. Diagnosis, i.e. OED vs normal ##
    # odyn_diagnosis = oed_diagnosis()
    # save diagnosis to csv file
    
    ### 4. Create patch-level features ###
    generate_features(
        input_wsi_dir=input_wsi_dir,
        hovernetplus_dir=os.path.join(output_dir, "hovernetplus"),
        output_dir=os.path.join(output_dir, "features"),
        colour_dict=epith_colour_dict,
        num_processes=int(args['--nr_post_proc_workers']),
    )
    
    ### 5. Prognosis, with MLP ###
    # odyn_prognosis = oed_prognosis()
    # odyn_score = odyn_prognosis
    # save prognosis to csv file
