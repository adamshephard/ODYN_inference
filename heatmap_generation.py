"""
Create heatmaps for ODYN  across the WSI.

Usage:
  heatmap_generation.py [options] [--help] [<args>...]
  heatmap_generation.py --version
  heatmap_generation.py (-h | --help)
  
Options:
  -h --help                   Show this string.
  --version                   Show version.

  --input_dir=<string>        Path to input directory containing slides or images.
  --mask_dir=<string>         Path to mask directory.
  --nuclei_dir=<string>       Path to nuclei directory.
  --checkpoint_path=<string>  Path to MLP checkpoint.
  --output_dir=<string>       Path to output directory to save results.
  --norm_parameters=<string>  Path to file containing normalization parameters.
  --model_checkpoint=<string> Path to model checkpoint.
  --cutoff_file=<string>      Path to file containing cutoffs.  
  --batch_size=<n>            Batch size. [default: 256]

Use `heatmap_generation.py --help` to show their options and usage
"""

from docopt import docopt
import os
import glob
import shutil

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.models as models
from dataloader.mil_reader import featuresdataset_wsi_inference

from tiatoolbox.utils.misc import imwrite

from models.net_desc import MLP
from utils.utils import get_heatmap, build_heatmap
from utils.patch_generation import create_feature_patches, create_image_patches
from feature_generation import generate_features

from tqdm import tqdm

# cnn inference 
def inference(loader, model, batch_size):
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset), 2)
    preds = torch.FloatTensor(len(loader.dataset))
    coords_all = torch.FloatTensor(len(loader.dataset), 4)
    with torch.no_grad():
        with tqdm(total=len(loader)) as pbar:
            for i, (inputs, coords) in tqdm(enumerate(loader)):
                inputs = inputs.cuda()
                coords = coords.cuda()
                output = model(inputs)
                y = F.softmax(output, dim=1)
                _, pr = torch.max(output.data, 1)
                preds[i * batch_size:i * batch_size + inputs.size(0)] = pr.detach().clone()
                probs[i * batch_size:i * batch_size + inputs.size(0)] = y.detach().clone()
                coords_all[i * batch_size:i * batch_size + inputs.size(0)] = coords.detach().clone()
                pbar.update(1)
    return probs.cpu().numpy(), preds.cpu().numpy(), coords_all.cpu().numpy()


def create_patches(
    features: str,
    wsi_path: str,
    mask_dir: str,
    nuc_dir: str,
    output_dir: str,
    mask_colour_dict: dict,
    nuc_colour_dict: dict,
    patch_size: int | int = 512,
    stride: int | int = 128,
    proc_res: float | float = 0.5,
    layer_res: float | float = 0.5,
    epith_thresh: float | float = 0.5,
    ) -> None:
    """
    Create patches for heatmap generation.

    Args:
        features (str): Type of features to use.
        wsi_path (str): Path to WSI.
        mask_dir (str): Path to mask directory.
        nuc_dir (str): Path to nuclei directory.
        mask_colour_dict (dict): Dictionary of mask colours.
        nuc_colour_dict (dict): Dictionary of nuclei colours.
        patch_size (int): Size of patches.
        stride (int): Stride of patches.
        proc_res (float): Processing resolution.
        layer_res (float): Layer resolution.
        epith_thresh (float): Epithelial threshold.
        output_dir (str): Path to output directory to save results.
    """
         
    case = os.path.basename(wsi_path).split(".")[0]
    mask_path = os.path.join(mask_dir, f"{case}.png")
    nuclei_path = os.path.join(nuc_dir, f"{case}.dat")
    
    # Check exist first
    if os.path.exists(os.path.join(output_dir, f'{case}_patches.npy')) and os.path.exists(os.path.join(output_dir, f'{case}_rois.npy')):
        print(f"Patches and ROIs already exist for {case}.")
        return os.path.join(output_dir, f'{case}_patches.npy'), os.path.join(output_dir, f'{case}_rois.npy')

    if features == 'raw_images':
        patches, rois = create_image_patches(
            wsi_path, mask_path, mask_colour_dict, nuc_colour_dict, layer_res, patch_size,
            stride, proc_res, epith_thresh
            )
    else:
        patches, rois = create_feature_patches(
            'nuclear', wsi_path, mask_path, nuclei_path, mask_colour_dict, nuc_colour_dict, 
            patch_size, stride, proc_res, layer_res, epith_thresh, output_dir=None, 
            viz=False
            )
        for idx, ftr_df in enumerate(patches):
            tmp = ftr_df.T
            if idx == 0:
                ftr_df_all = tmp
            else:
                ftr_df_all = pd.concat([ftr_df_all, tmp], axis=0)
            ftr_df_all.reset_index(drop=True, inplace=True)
        patches = ftr_df_all.iloc[:, 0:].to_numpy()
        rois = np.vstack(rois)
    
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f'{case}_patches.npy'), patches)
    np.save(os.path.join(output_dir, f'{case}_rois.npy'), rois)
        
    return os.path.join(output_dir, f'{case}_patches.npy'), os.path.join(output_dir, f'{case}_rois.npy')
    

def create_heatmaps(
    model: nn.Module,
    features: str,
    wsi_path: str,
    input_features_path: str,
    input_rois_path: str,
    input_checkpoint_path: str,
    norm_parameters: tuple,
    output_dir: str,
    proc_res: float | float = 0.5,
    output_res: float | float = 2.0,
    batch_size: int | int = 256,
    ) -> None:
         
    case = os.path.basename(wsi_path).split(".")[0]

    # Load features
    patches = np.load(input_features_path)
    rois = np.load(input_rois_path)  
             
    norm_params, normalize, trans_Valid = norm_parameters
                 
    # Load data
    test_dset = featuresdataset_wsi_inference(
        patches, rois, layer_col_dict=mask_colour_dict, 
        transform=trans_Valid, raw_images=features=='raw_images', norm=norm_params)
    
    print('loaded dataset')
    test_loader = torch.utils.data.DataLoader(
        test_dset,
        batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=False)

    # Loading best checkpoint
    ch = torch.load(input_checkpoint_path)
    state_dict = ch['state_dict']
    # state_dict = convert_state_dict(state_dict)
    model.load_state_dict(state_dict)

    # Run inference
    test_probs, _, test_coords = inference(test_loader, model, batch_size)
    test_probs_1 = test_probs[:, 1]
    print('inferred dataset')

    # Create heatmap
    heatmap = build_heatmap(wsi_path, output_res, proc_res, test_coords, test_probs_1)
    # heatmap_col = get_heatmap(heatmap)
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"{case}_repeat{repeat}_fold{fold}.npy"), heatmap)
    # imwrite(os.path.join(output_dir, f"{case}.png"), heatmap_col)
    return heatmap
    
        
def create_heatmaps_old(
    model: nn.Module,
    features: str,
    wsi_path: str,
    input_checkpoint_path: str,
    mask_dir: str,
    nuc_dir: str,
    output_dir: str,
    mask_colour_dict: dict,
    nuc_colour_dict: dict,
    patch_size: int | int = 512,
    stride: int | int = 128,
    proc_res: float | float = 0.5,
    output_res: float | float = 2.0,
    layer_res: float | float = 0.5,
    epith_thresh: float | float = 0.5,
    batch_size: int | int = 256,
    ) -> None:
         
    case = os.path.basename(wsi_path).split(".")[0]
    mask_path = os.path.join(mask_dir, f"{case}.png")
    nuclei_path = os.path.join(nuc_dir, f"{case}.dat")

    ###### get patches here..... #######
    if features == 'raw_images':
        patches, rois = create_image_patches(
            wsi_path, mask_path, mask_colour_dict, nuc_colour_dict, layer_res, patch_size,
            stride, proc_res, epith_thresh
            )
    else:
        patches, rois = create_feature_patches(
            'nuclear', wsi_path, mask_path, nuclei_path, mask_colour_dict, nuc_colour_dict, 
            patch_size, stride, proc_res, layer_res, epith_thresh, output_dir=None, 
            viz=False
            )
        trans_Valid=None
        for idx, ftr_df in enumerate(patches):
            tmp = ftr_df.T
            if idx == 0:
                ftr_df_all = tmp
            else:
                ftr_df_all = pd.concat([ftr_df_all, tmp], axis=0)
            ftr_df_all.reset_index(drop=True, inplace=True)
        patches = ftr_df_all.iloc[:, 0:].to_numpy()
        rois = np.vstack(rois)
    
    repeat = 1
    fold = 0
    input_checkpoint_path = os.path.join(input_checkpoint_path, f"repeat{repeat}_fold{fold}.pth")
              
    # defining data transform
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.1, 0.1, 0.1])
    trans_Valid = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    # Get norm params
    norm_params = pd.read_csv(os.path.join(norm_parameters, f"repeat{repeat}_fold{fold}.csv"), index_col=0)
    norm_params = [list(norm_params['mean']), list(norm_params['std'])]       
                 
    # Load data
    test_dset = featuresdataset_wsi_inference(
        patches, rois, layer_col_dict=mask_colour_dict, 
        transform=trans_Valid, raw_images=features=='raw_images', norm=norm_params)
    
    print('loaded dataset')
    test_loader = torch.utils.data.DataLoader(
        test_dset,
        batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=False)

    # loading best checkpoint wrt to auroc on test set
    ch = torch.load(input_checkpoint_path)
    state_dict = ch['state_dict']
    # state_dict = convert_state_dict(state_dict)
    model.load_state_dict(state_dict)

     #run inference
    test_probs, _, test_coords = inference(test_loader, model, batch_size)
    test_probs_1 = test_probs[:, 1]
    print('inferred dataset')

    heatmap = build_heatmap(wsi_path, output_res, proc_res, test_coords, test_probs_1)
    # heatmap_col = get_heatmap(heatmap)
    np.save(os.path.join(output_dir, f"{case}_repeat{repeat}_fold{fold}.npy"), heatmap)
    # imwrite(os.path.join(output_dir, f"{case}.png"), heatmap_col)
    return


if __name__ == '__main__':
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
    
    if args['--norm_parameters']:
        norm_parameters = args['--norm_parameters']
    else:
        norm_parameters = "/data/ANTICIPATE/github/ODYN_inference/models/norm_params/"
    
    if args['--checkpoint_path']:
        input_checkpoint_path = args['--checkpoint_path']
    else:
        input_checkpoint_path = "/data/ANTICIPATE/github/ODYN_inference/weights/cross_validation/"
    
    if args['--cutoff_file']:
        cutoff_file = args['--cutoff_file']
    else:
        cutoff_file = "/data/ANTICIPATE/github/ODYN_inference/models/norm_params/cutoff_summary.csv"        
        
    if args['--output_dir']:
        heatmap_output_dir = args['--output_dir']
    else:
        heatmap_output_dir = "/data/ANTICIPATE/github/testdata/output/odyn/heatmaps/"
    
    if args['--batch_size']:
        batch_size = int(args['--batch_size'])
    
    ### Input/Output Parameters ###
    method = "mlp" # alternatives are "resnet34", "resnet34_SA", "resnet34_DA", "resnet34_SA_DA"
    features = 'morph_features_104_64ftrs' # alternatives are "raw_images"
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
    stride = 64 # stride for sliding window of patches. Decrease stride to make smoother heatmaps (but 0.5x stride = 2x comp. time)
    proc_res = 0.5 # resolution of intermediate patches 
    output_res = 2 # desired resolution of output heatmaps
    layer_res = 0.5 # resolution of layer segmentation from HoVer-Net+ in MPP
    epith_thresh = 0.25 # threshold for ratio of epithelium required to be in a patch to use patch
    nr_repeats = 3
    nr_folds = 5
    
    ### Main ###
    os.makedirs(heatmap_output_dir, exist_ok=True)
    torch.cuda.empty_cache()
    
    # Load Model
    if method == 'mlp':
        nr_ftrs = 168
        nr_hidden = 64
        model = MLP(d=nr_ftrs, hidden_d=nr_hidden, nr_classes=2)

    elif 'resnet34' in method:
        model = models.resnet34(True) # pretrained resnet34
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        
    model.cuda()
    cudnn.benchmark = True
    
    for wsi_path in glob.glob(os.path.join(input_wsi_dir, "*.*")):
        # first create patch features
        output_ftrs_dir = os.path.join(heatmap_output_dir, "features", f"{features}_{proc_res}mpp_{patch_size}_{stride}_{epith_thresh}")
        output_ftrs, output_rois = create_patches(
            features,
            wsi_path,
            mask_dir,
            nuclei_dir,
            output_ftrs_dir,
            mask_colour_dict,
            nuc_colour_dict,
            patch_size,
            stride,
            proc_res,
            layer_res,
            epith_thresh,
            )
              
        # then create heatmap per repeat
        heatmaps = []
        heatmap_output_repeat_dir = os.path.join(heatmap_output_dir, "cross_valid", f"{features}_{proc_res}mpp_{patch_size}_{stride}_{epith_thresh}")
        for repeat in range(1, nr_repeats+1):
            for fold in range(nr_folds):
                input_checkpoint_path_ = os.path.join(input_checkpoint_path, f"repeat{repeat}_fold{fold}.pth")
                # Get norm params
                if features == "raw_images":
                    norm_params = None
                    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                    std=[0.1, 0.1, 0.1])
                    trans_Valid = transforms.Compose([
                        transforms.ToTensor(),
                        normalize,
                    ])
                else:
                    norm_params = pd.read_csv(os.path.join(norm_parameters, f"repeat{repeat}_fold{fold}.csv"), index_col=0)
                    norm_params = [list(norm_params['mean']), list(norm_params['std'])] 
                    normalize = None
                    trans_Valid = None
                
                heatmap_repeat = create_heatmaps(
                    model,
                    features,
                    wsi_path,
                    output_ftrs, 
                    output_rois,
                    input_checkpoint_path_,
                    (norm_params, normalize, trans_Valid),
                    heatmap_output_repeat_dir,
                    proc_res,
                    output_res,
                    batch_size,
                    )
                heatmaps.append(heatmap_repeat)
         
        # then create combined heatmap
        slide_name = os.path.basename(wsi_path).split(".")[0]
        combined_heatmap = np.mean(heatmaps, axis=0)
        combined_heatmap = get_heatmap(combined_heatmap)
        imwrite(os.path.join(heatmap_output_dir, f"{slide_name}.png"), combined_heatmap)
        
    # then remove temp directories
    # shutil.rmtree(output_ftrs_dir)
    # shutil.rmtree(heatmap_output_repeat_dir))