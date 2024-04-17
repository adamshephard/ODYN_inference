"""
Collate the output from ODYN and supply it in a succint way for visualisation with TIAViz.
NOTE script does not work for holes currently.

Usage:
  visualize_output.py [options] [--help] [<args>...]
  visualize_output.py --version
  visualize_output.py (-h | --help)
  
Options:
  -h --help                       Show this string.
  --version                       Show version.

  --input_dir=<string>            Path to input directory containing slides or images.
  --output_dir=<string>           Path to output directory to save results.
  --transformer_weights=<string>  Path to transformer weights.
  --hovernetplus_weights=<string> Path to HoverNet+ weights.
  --mode=<string>                 Tile-level or WSI-level mode. [default: wsi]

Use `visualize_output.py --help` to show their options and usage
"""

from docopt import docopt
import os
import glob
import shutil

import numpy as np
import torch
import cv2
from scipy import ndimage
from collections import OrderedDict
from tiatoolbox.utils.misc import store_from_dat, imread
from tiatoolbox.annotation.storage import Annotation, SQLiteStore
from tiatoolbox.wsicore.wsireader import WSIReader
from shapely.geometry import shape, Point, Polygon, box, MultiPolygon

from utils.shapely_utils import get_nuclear_polygons, get_mask_polygons
from utils.utils import decolourise


def dat2db(input_wsi_path, input_dat_file, output_db_file, nuc_dict, proc_res=0.5):
    """
    Convert DAT file to annotation store (DB).
    """
    # Get scale factor for annotation store
    wsi = WSIReader.open(input_wsi_path)
    base_dims = wsi.slide_dimensions(units='level', resolution=0)
    proc_dims = wsi.slide_dimensions(units='mpp', resolution=proc_res)

    scale_factor = np.asarray(base_dims) / np.asarray(proc_dims)

    # Convert DAT file to annotation store
    annos_db = store_from_dat(input_dat_file, scale_factor=scale_factor, typedict=nuc_dict)

    # Save annotation store to DB
    annos_db.dump(output_db_file)
    return


def store_from_rgb(wsi_path, mask_path, colour_dict, type_dict):
    """
    Convert RGB image to annotation store (DB).
    """
    
    mask = imread(mask_path)
    mask = decolourise(mask, colour_dict)
    wsi = WSIReader.open(wsi_path)
    
    mask_dims = mask.shape[::-1]
    base_dims = wsi.slide_dimensions(units='level', resolution=0)
    scale_factor = np.asarray(base_dims) / np.asarray(mask_dims)
    
    annotations = []
    for name, m in colour_dict.items():
        if m[0] == 0: # ignore background
            continue
        if m[0] == 1: # ignore connective tissue
            continue 
        properties = {"type": name}
        if m[0] == 3:
            mask_ = (mask == 2).astype(np.uint8) + (mask == 3).astype(np.uint8) # hardcoded
        else:
            mask_ = (mask == m[0]).astype(np.uint8)
        # mask_polygons = get_mask_polygons(wsi, mask_, 0, "level")
        # mask_polys = list(mask_polygons.geoms) if mask_polygons.type is "MultiPolygon" else [mask_polygons]
        # for poly in mask_polys:
        #     annotations.append(
        #         Annotation(poly, properties=properties) # contours have been donwsampled here! No longer...
        #     )
        cntrs, _ = cv2.findContours(mask_, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in cntrs:
            c = np.squeeze(c)
            if len(c) < 4:
                continue
            c[:,0] = (c[:,0] * scale_factor[0]).astype('int')
            c[:,1] = (c[:,1] * scale_factor[1]).astype('int')
            annotations.append(
                Annotation(Polygon(c), properties=properties)
            )
            
    annos_db = SQLiteStore()
    annos_db.append_many(annotations)
    return annos_db
    # return annotations


if __name__ == '__main__':
    args = docopt(__doc__, help=False, options_first=True, 
                    version='Visualisation')

    if args['--help']:
        print(__doc__)
        exit()
    
    if args['--input_dir']:
        input_wsi_dir = args['--input_dir']
    else:      
        input_wsi_dir = "/data/ANTICIPATE/github/testdata/wsis/"
    
    if args['--output_dir']:
        output_dir = args['--output_dir']
    else:
        output_dir = "/data/ANTICIPATE/github/testdata/output/odyn/visualisation"
    
    if args['--mode']:
        mode = args['--mode']
        if mode not in ["tile", "wsi"]:
            raise ValueError("Mode must be tile or wsi")
    else:
        mode = "wsi" # or tile 
        

    nuc_dict = {
        0: "No Label",
        1: "Other Cell",
        2: "Epithelial Cell",
    }
    mask_dict = {
        0: "Background",
        1: "Other Tissue",
        2: "Dysplasia",
        3: "Epithelium",
    }
    mask_colour_dict = {
        "Background": [0, [0  ,   0,   0]],
        "Other Tissue": [1, [255, 165,   0]],
        "Dysplasia": [2, [255, 0,   0]],
        "Epithelium": [3, [0,   255,   0]],
    }
    
    os.makedirs(output_dir, exist_ok=True)
    
    # First prepare nuclei for visualisation 
    nuc_dir = os.path.join("/data/ANTICIPATE/github/testdata/output/odyn/", "nuclei")
    mask_dir =  os.path.join("/data/ANTICIPATE/github/testdata/output/odyn/", "combined")
    
    for input_wsi_path in glob.glob(input_wsi_dir + "*.*"):
        basename = os.path.basename(input_wsi_path).split(".")[0]
        input_dat_file = os.path.join(nuc_dir, basename + ".dat")
        output_db_file = os.path.join(output_dir, basename + ".db")
        print(f"Processing {basename}")

        # Get scale factor for annotation store
        wsi = WSIReader.open(input_wsi_path)
        base_dims = wsi.slide_dimensions(units='level', resolution=0)
        proc_dims = wsi.slide_dimensions(units='mpp', resolution=0.5)

        nuc_scale_factor = np.asarray(base_dims) / np.asarray(proc_dims)

        # Convert DAT file to annotation store
        nuc_db = store_from_dat(input_dat_file, scale_factor=nuc_scale_factor, typedict=nuc_dict)
        
        # RGB image to annotation store
        mask_path = os.path.join(mask_dir, basename + ".png")
        mask_db = store_from_rgb(input_wsi_path, mask_path, mask_colour_dict, mask_dict)
        
        annos_db = SQLiteStore()
        annos_db.append_many(list(nuc_db.values())) 
        annos_db.append_many(list(mask_db.values()))
        
        # Save annotation store to DB
        annos_db.dump(output_db_file)