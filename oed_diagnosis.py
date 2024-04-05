"""
Use generated masks to classify an oral tissue slide as OED or normal.

Usage:
  oed_diagnosis.py [options] [--help] [<args>...]
  oed_diagnosis.py --version
  oed_diagnosis.py (-h | --help)
  
Options:
  -h --help                   Show this string.
  --version                   Show version.
  --input_epith=<string>      Path to input directory containing HoVer-Net+ epithelium maps.
  --input_dysplasia=<string>  Path to input directory containing the Transformer dysplasia maps.
  --output_dir=<string>       Path to output directory to save results.

Use `oed_diagnosis.py --help` to show their options and usage
"""

from docopt import docopt
import os
import glob
from tiatoolbox.utils.misc import imread, imwrite
import cv2
import numpy as np
import pandas as pd

from utils.utils import decolourise, colourise


def combine_masks(
    input_epith_dir: str,
    input_dysp_dir: str,
    output_dir: str,
    epith_colour_dict: dict,
    dysp_colour_dict: dict,
    new_colour_dict: dict,
    ) -> None:
    """
    Combine epithelium and dysplasia masks into a new combined mask.
    """
    os.makedirs(output_dir, exist_ok=True)
    for epith_file in sorted(glob.glob(os.path.join(input_epith_dir, "*.png"))):
        # read in images
        basename = os.path.basename(epith_file).split(".")[0]
        dysp_file = os.path.join(input_dysp_dir, basename + ".png")
        epith_img = imread(epith_file)
        dysp_img = imread(dysp_file)
        epith_img = decolourise(epith_img, epith_colour_dict)
        dysp_img = decolourise(dysp_img, dysp_colour_dict)
        
        # combine masks - process at dysplasia resolution
        combined_img = epith_img.copy()
        if dysp_img.shape != combined_img.shape:
            combined_img = cv2.resize(combined_img, (dysp_img.shape[1], dysp_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Commented out uses the union of both methods
        # combined_img[combined_img >= 2] = 3
        # combined_img[dysp_img == 1] = 2
        
        # Use the overlap of both methods
        epith = combined_img.copy()
        epith[epith <= 1] = 0
        epith[epith >= 2] = 1
        dysp_epith = dysp_img*epith

        tissue_img = combined_img.copy()
        tissue_img[tissue_img >= 1] = 1
        tissue_img[dysp_img >= 1] = 1
        
        combined_img = tissue_img.copy()
        combined_img[epith == 1] = 3
        combined_img[dysp_epith == 1] = 2
        
        combined_img_col = colourise(combined_img, new_colour_dict)

        # save combined mask
        imwrite(os.path.join(output_dir, basename + ".png"), combined_img_col)
    
    return
    
    
def oed_diagnosis(
    input_mask_dir: str,
    output_dir: str,
    colour_dict: dict,
    threshold: float,
    ) -> None:
    """
    Classify oral tissue slides as OED or normal.
    """
    os.makedirs(output_dir, exist_ok=True)
    diagnoses = []
    for mask_file in sorted(glob.glob(os.path.join(input_mask_dir, "*.png"))):
        basename = os.path.basename(mask_file).split(".")[0]
        mask = imread(mask_file)
        mask = decolourise(mask, colour_dict)
        
        # classify as OED or normal
        vals, counts = np.unique(mask, return_counts=True)
        dysp_pixels = counts[list(vals).index(new_colour_dict["dysplasia"][0])]
        epith_pixels = counts[list(vals).index(new_colour_dict["epith"][0])]
        
        ratio = dysp_pixels / (epith_pixels + dysp_pixels)
        if ratio > threshold:
            classification = "OED"
        else:
            classification = "normal"
        
        diagnoses.append({
            "slide_name": basename,
            "dysplasia-epithelium_ratio": ratio,
            "classification": classification,
        })
    
    # save classification
    diag = pd.DataFrame(diagnoses)
    diag.to_csv(os.path.join(output_dir, "diagnoses.csv"), index=False)

    return

            
if __name__ == '__main__':
    args = docopt(__doc__, help=False, options_first=True, 
                    version='ODYN Diagnosis')

    if args['--help']:
        print(__doc__)
        exit()

    if args['--input_dysplasia']:
        input_dysp_dir = args['--input_dysplasia']
    else:      
        input_dysp_dir = "/data/ANTICIPATE/github/testdata/output/odyn/dysplasia/"
        
    if args['--input_epith']:
        input_epith_dir = args['--input_epith']
    else:      
        input_epith_dir = "/data/ANTICIPATE/github/testdata/output/odyn/epith/"
            
    if args['--output_dir']:
        output_dir = args['--output_dir']
    else:
        output_dir = "/data/ANTICIPATE/github/testdata/output/odyn/"
    

    threshold = 0.055022543958983 # hardcoded! # dysplasia ratio threshold for determing OED vs normal

    epith_colour_dict = {
        "nolabel": [0, [0  ,   0,   0]],
        "other": [1, [255, 165,   0]],
        "basal": [2, [255, 0,   0]],
        "epith": [3, [0,   255,   0]],
        "keratin": [4, [0,   0,   255]],
    }
    
    dysp_colour_dict = {
        "nolabel": [0, [0  ,   0,   0]],
        "dysplasia": [1, [255, 0,   0]]
    }
    
    new_colour_dict = {
        "nolabel": [0, [0  ,   0,   0]],
        "other": [1, [255, 165,   0]],
        "dysplasia": [2, [255, 0,   0]],
        "epith": [3, [0,   255,   0]],
    }  
    
    combine_masks(
        input_epith_dir,
        input_dysp_dir,
        os.path.join(output_dir, "combined"),
        epith_colour_dict,
        dysp_colour_dict,
        new_colour_dict,
    )
    
    oed_diagnosis(
        input_mask_dir=os.path.join(output_dir, "combined"),
        output_dir=os.path.join(output_dir, "diagnoses"),
        colour_dict=new_colour_dict,
        threshold=threshold
        )
