"""
Use TIAToolbox Multi-Task Segmentor to get nuclear/epithelial layer segmentations with HoVer-Net+.

Usage:
  epithelium_segmentation.py [options] [--help] [<args>...]
  epithelium_segmentation.py --version
  epithelium_segmentation.py (-h | --help)
  
Options:
  -h --help                   Show this string.
  --version                   Show version.

  --input_dir=<string>        Path to input directory containing slides or images.
  --output_dir=<string>       Path to output directory to save results.
  --mode=<string>             Tile-level or WSI-level mode. [default: wsi]
  --nr_loader_workers=<n>     Number of workers during data loading. [default: 10]
  --nr_post_proc_workers=<n>  Number of workers during post-processing. [default: 10]
  --batch_size=<n>            Batch size. [default: 8]

Use `epithelium_segmentation.py --help` to show their options and usage
"""

from docopt import docopt
import os
import glob
import shutil
from tiatoolbox.models.engine.multi_task_segmentor import MultiTaskSegmentor

# proc functions
# proc functions
def wsi_post_proc(wsi_seg: np.ndarray, fx: int=1) -> np.ndarray:
    """
    Post processing for WSI-level segmentations.
    """
    wsi_seg = wsi_seg[..., 1]
    wsi_seg = np.where(wsi_seg >= 0.5,1,0)

    wsi_seg = np.around(wsi_seg).astype("uint8")  # ensure all numbers are integers
    min_size = int(10000 * fx * fx)
    min_hole = int(15000 * fx * fx)
    kernel_size = int(11*fx)
    if kernel_size % 2 == 0:
        kernel_size += 1
    ep_close = cv2.morphologyEx(
        wsi_seg, cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size))
    ).astype("uint8")
    ep_open = cv2.morphologyEx(
        ep_close, cv2.MORPH_OPEN, np.ones((kernel_size, kernel_size))
    ).astype(bool)
    ep_open2 = morphology.remove_small_objects(
        ep_open.astype('bool'), min_size=min_size
    ).astype("uint8")
    ep_open3 = morphology.remove_small_holes(
        ep_open2.astype('bool'), min_hole).astype('uint8')
    return ep_open3

def process_segmentation(seg_path: str, out_path: str, colour_dict: dict, mode: str) -> None:
    """
    Post-processing for WSI-level segmentations.
    """
    seg = np.load(seg_path)
    if mode == "wsi":
        seg = wsi_post_proc(seg, fx=1)
    else:
        seg = wsi_post_proc(seg, fx=1)#0.5)
    seg_col = np.expand_dims(seg, axis=2)
    seg_col = np.repeat(seg_col, 3, axis=2)
    for key, value in colour_dict.items():
        seg_col[seg == value[0]] = value[1]
    seg_col = seg_col.astype('uint8')
    imwrite(out_path, seg_col)
    return None

def segment_epithelium(
    input_wsi_dir: str,
    output_dir: str,
    colour_dict: dict = None,
    mode: str = "wsi",
    nr_loader_workers: int | int = 10,
    nr_post_proc_workers: int | int = 10,
    batch_size: int | int = 8, 
) -> None:
    
    wsi_file_list = glob.glob(input_wsi_dir + "*")

    multi_segmentor = MultiTaskSegmentor(
        pretrained_model="hovernetplus-oed",
        num_loader_workers=nr_loader_workers,
        num_postproc_workers=nr_post_proc_workers,
        batch_size=batch_size,
        auto_generate_mask=False,
    )

    # WSI prediction
    wsi_output = multi_segmentor.predict(
        imgs=wsi_file_list,
        masks=None,
        save_dir=os.path.join(output_dir, "epith/tmp"),
        mode=mode, #"wsi",
        on_gpu=True,
        crash_on_exception=True,
    )

    # Rename TIAToolbox output files to readability
    layer_dir = os.path.join(output_dir, "epith", "layers")
    nuclei_dir = os.path.join(output_dir, "epith", "nuclei")
    os.makedirs(layer_dir, exist_ok=True)
    os.makedirs(nuclei_dir, exist_ok=True)

    for out in wsi_output:
        basename = os.path.basename(out[0]).split(".")[0]
        outname = os.path.basename(out[1]).split(".")[0]
        shutil.move(
            os.path.join(output_dir, "epith/tmp", f"{outname}.1.npy"),
            os.path.join(layer_dir, basename + ".npy"),
            )   
        shutil.move(
            os.path.join(output_dir, "epith/tmp", f"{outname}.0.dat"),
            os.path.join(nuclei_dir, basename + ".dat"),
            )
    shutil.rmtree(os.path.join(output_dir, "epith/tmp"))

    return
    

if __name__ == '__main__':
    args = docopt(__doc__, help=False, options_first=True, 
                    version='ODYN HoVer-Net+ Inference')

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
        output_dir = "/data/ANTICIPATE/outcome_prediction/MIL/github_testdata/output3/"
    
    if args['--mode']:
        mode = args['--mode']
        if mode not in ["tile", "wsi"]:
            raise ValueError("Mode must be tile or wsi")
    else:
        mode = "wsi" # or tile


    colour_dict = {
        "nolabel": [0, [0  ,   0,   0]],
        "other": [1, [255, 165,   0]],
        "basal": [2, [255, 0,   0]],
        "epith": [3, [0,   255,   0]],
        "keratin": [4, [0,   0,   255]],
    }
    
    segment_epithelium(
        input_wsi_dir=input_wsi_dir,
        output_dir=output_dir,
        colour_dict=colour_dict,
        mode=mode,
        nr_loader_workers=int(args['--nr_loader_workers']),
        nr_post_proc_workers=int(args['--nr_post_proc_workers']),
        batch_size=int(args['--batch_size']),
        )
