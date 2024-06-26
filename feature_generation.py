"""
Generate morphological/spatial features for MLP (based on HoVer-Net+/Transformer output).

Usage:
  feature_generation.py [options] [--help] [<args>...]
  feature_generation.py --version
  feature_generation.py (-h | --help)
  
Options:
  -h --help                   Show this string.
  --version                   Show version.

  --input_dir=<string>        Path to input directory containing slides or images.
  --mask_dir=<string>         Path to mask directory.
  --nuclei_dir=<string>       Path to nuclei directory.
  --output_dir=<string>       Path to output directory to save features.

Use `feature_generation.py --help` to show their options and usage
"""

from docopt import docopt
import os
import glob
from torch.multiprocessing import Pool, RLock, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch

from utils.utils import save_hdf5
from utils.patch_generation import create_feature_patches

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def process(
    feature_type,
    wsi_path,
    mask_dir,
    nuclei_dir,
    output_dir,
    mask_colour_dict,
    nuc_colour_dict,
    patch_size,
    stride,
    output_res,
    layer_res,
    epith_thresh,
    viz=True,
    ):
    
    case, _ = os.path.basename(wsi_path).split('.')
    mask_path = os.path.join(mask_dir, f'{case}.png')
    nuclei_path = os.path.join(nuclei_dir, f'{case}.dat')
    output_dir_ftrs = os.path.join(output_dir, f'{output_res}-mpp_{patch_size}_{stride}_dysplasia-{epith_thresh}')

    feature_info = create_feature_patches(
        feature_type,
        wsi_path,
        mask_path,
        nuclei_path,
        mask_colour_dict,
        nuc_colour_dict,
        patch_size,
        stride,
        output_res,
        layer_res,
        epith_thresh,
        output_dir,
        viz=viz,
    )

    if feature_type == "nuclear":
        features_all, coords = feature_info
    elif feature_type == 'deep':
        deep_features_all, deep_coords = feature_info
    elif feature_type == 'both':
        features_all, coords, deep_features_all, deep_coords = feature_info

    if (feature_type == 'nuclear') or (feature_type == 'both'):
        for idx, ftr_df in enumerate(features_all):
            tmp = ftr_df.T
            tmp.insert(loc=0, column='coords', value=f"{coords[idx][0]}_{coords[idx][1]}_{coords[idx][2]}_{coords[idx][3]}")
            if idx == 0:
                ftr_df_all = tmp
            else:
                ftr_df_all = pd.concat([ftr_df_all, tmp], axis=0)
            ftr_df_all.reset_index(drop=True, inplace=True)
        features_all = ftr_df_all.iloc[:, 1:].to_numpy()
        asset_dict = {'features': features_all, 'coords': np.stack(coords)}
        h5_dir = os.path.join(output_dir_ftrs, 'nuclear', 'h5_files')
        pt_dir = os.path.join(output_dir_ftrs, 'nuclear', 'pt_files')
        csv_dir = os.path.join(output_dir_ftrs, 'nuclear', 'csv_files')
        os.makedirs(h5_dir, exist_ok=True)
        os.makedirs(pt_dir, exist_ok=True)
        os.makedirs(csv_dir, exist_ok=True)
        save_hdf5(os.path.join(h5_dir, f'{case}.h5'), asset_dict, attr_dict= None, mode='w')
        # features = torch.from_numpy(features_all)
        # torch.save(features, os.path.join(pt_dir, f'{case}.pt'))
        # No longer save pt with bag of features.   
        # Now save individual pt files for each tile
        for idx, (ftrs, coords) in enumerate(zip(asset_dict["features"], asset_dict["coords"])):
            tile_name = f'{case}_{coords[0]}_{coords[1]}_{coords[2]}_{coords[3]}'
            out_tile_pt_dir = os.path.join(pt_dir, case)
            os.makedirs(out_tile_pt_dir, exist_ok=True)
            torch.save(ftrs, os.path.join(out_tile_pt_dir, f'{tile_name}.pt'))     
        ftr_df_all.to_csv(os.path.join(csv_dir, f'{case}.csv'), index=False) 

    if (feature_type == 'resnet') or (feature_type == 'both'):
        asset_dict = {'features': np.stack(deep_features_all), 'coords': np.stack(deep_coords)}
        h5_dir = os.path.join(output_dir_ftrs, 'resnet', 'h5_files')
        pt_dir = os.path.join(output_dir_ftrs, 'resnet', 'pt_files')
        csv_dir = os.path.join(output_dir_ftrs, 'resnet', 'csv_files')
        os.makedirs(h5_dir, exist_ok=True)
        os.makedirs(pt_dir, exist_ok=True)
        os.makedirs(csv_dir, exist_ok=True)
        save_hdf5(os.path.join(h5_dir, f'{case}.h5'), asset_dict, attr_dict= None, mode='w')
        # features = torch.from_numpy(np.stack(deep_features_all))
        # torch.save(features, os.path.join(pt_dir, f'{case}.pt'))
        # Now save individual pt files for each tile
        for idx, (ftrs, coords) in enumerate(zip(asset_dict["features"], asset_dict["coords"])):
            tile_name = f'{case}_{coords[0]}_{coords[1]}_{coords[2]}_{coords[3]}'
            out_tile_pt_dir = os.path.join(pt_dir, case)
            os.makedirs(out_tile_pt_dir, exist_ok=True)
            torch.save(ftrs, os.path.join(out_tile_pt_dir, f'{tile_name}.pt'))
        deep_df = pd.DataFrame.from_records(deep_features_all)
        deep_coords_names = []
        for coord in deep_coords:
            deep_coords_names.append(f"{coord[0]}_{coord[1]}_{coord[2]}_{coord[3]}")
        deep_df.insert(loc=0, column='coords', value=np.stack(deep_coords_names))
        deep_df.to_csv(os.path.join(csv_dir, f'{case}.csv'), index=False)

    return

def generate_features(
    input_wsi_dir: str,
    mask_dir: str,
    nuclei_dir: str,
    output_dir: str,
    mask_colour_dict: dict,
    nuc_colour_dict: dict,    
    feature_type: str | str = "both", # Choose from: "nuclear", "resnet", "both"
    patch_size: int | int = 512,
    stride: int | int = 256,
    output_res: float | float = 0.5,
    layer_res: float | float = 0.5,
    epith_thresh: float | float = 0.5,
    num_processes: int = 10,
    )-> None:
    """
    Generate morphological/spatial features for MLP (based on HoVer-Net+/Transformer output) for classification.
    """
    
    wsi_file_list = glob.glob(input_wsi_dir + "*.*")
    num_processes = len(wsi_file_list) if len(wsi_file_list) < num_processes else num_processes

    # Start multi-processing
    argument_list = wsi_file_list
    pool = Pool(processes=num_processes, initargs=(RLock(),), initializer=tqdm.set_lock)
    pbar_format = "Processing cases... |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
    pbarx = tqdm(
        total=len(wsi_file_list), bar_format=pbar_format, ascii=True, position=0
    )
    
    def pbarx_update(*a):
        pbarx.update()

    jobs = [pool.apply_async(
            process,
            args=(
                feature_type,
                n,
                mask_dir,
                nuclei_dir,
                output_dir,
                mask_colour_dict,
                nuc_colour_dict,
                patch_size,
                stride,
                output_res,
                layer_res,
                epith_thresh,),
            callback=pbarx_update) for n in argument_list]
    
    pool.close()
    result_list = [job.get() for job in jobs]
    pbarx.close()

if __name__ == "__main__":
    args = docopt(__doc__, help=False, options_first=True)

    if args['--help']:
        print(__doc__)
        exit()

    if args['--input_dir']:
        input_wsi_dir = args['--input_dir']
    else:      
        input_wsi_dir = "/data/ANTICIPATE/github/testdata/wsis/"
    
    if args['--mask_dir']:
        mask_dir = args['--mask_dir']
    else:
        mask_dir = "/data/ANTICIPATE/github/testdata/output/odyn/combined/"
    
    if args['--nuclei_dir']:
        nuclei_dir = args['--nuclei_dir']
    else:
        nuclei_dir = "/data/ANTICIPATE/github/testdata/output/odyn/nuclei/"       
        
    if args['--output_dir']:
        output_dir = args['--output_dir']
    else:
        output_dir = "/data/ANTICIPATE/github/testdata/output/odyn/features/"
    
    ### Input/Output Parameters ###
    num_processes = 4
    feature_type = "both" # Choose from: "nuclear", "resnet", "both"
    mask_colour_dict = {
        "nolabel": [0, [0  ,   0,   0]],
        "other": [1, [255, 165,   0]],
        "dysplasia": [2, [255, 0,   0]],
        "epith": [3, [0,   255,   0]],
    } 
    nuc_colour_dict = {
        "nolabel": [0, [0  ,   0,   0]],
        "other": [1, [255, 165,   0]],
        "epith": [2, [255, 0,   0]],
    }
    patch_size = 512 # desired output patch size
    stride = 256 # stride for sliding window of patches
    output_res = 0.5 # desired resolution of output patches
    layer_res = 0.5 # resolution of layer segmentation from HoVer-Net+ in MPP
    epith_thresh = 0.25 # threshold for ratio of epithelium required to be in a patch to use patch
    
    generate_features(
        input_wsi_dir,
        mask_dir,
        nuclei_dir,
        output_dir,
        mask_colour_dict,
        nuc_colour_dict,
        "both", # Choose from: "nuclear", "resnet", "both"
        patch_size,
        stride,
        output_res,
        layer_res,
        epith_thresh,
        num_processes,
    )

