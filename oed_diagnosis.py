"""
Use generated masks to classify an oral tissue slide as OED or normal.

Usage:
  oed_diagnosis.py [options] [--help] [<args>...]
  oed_diagnosis.py --version
  oed_diagnosis.py (-h | --help)
  
Options:
  -h --help                   Show this string.
  --version                   Show version.

  --input_dir=<string>        Path to input directory containing slides or images.
  --output_dir=<string>       Path to output directory to save results.
  --mode=<string>             Tile-level or WSI-level mode. [default: wsi]
  --nr_loader_workers=<n>     Number of workers during data loading. [default: 10]
  --nr_post_proc_workers=<n>  Number of workers during post-processing. [default: 10]
  --batch_size=<n>            Batch size. [default: 8]

Use `oed_diagnosis.py --help` to show their options and usage
"""

from docopt import docopt
import os
import glob
import shutil
from tiatoolbox.models.engine.multi_task_segmentor import MultiTaskSegmentor

# proc functions
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
        save_dir=os.path.join(output_dir, "hovernetplus/tmp"),
        mode=mode, #"wsi",
        on_gpu=True,
        crash_on_exception=True,
    )

    # Rename TIAToolbox output files to readability
    layer_dir = os.path.join(output_dir, "hovernetplus", "layers")
    nuclei_dir = os.path.join(output_dir, "hovernetplus", "nuclei")
    os.makedirs(layer_dir, exist_ok=True)
    os.makedirs(nuclei_dir, exist_ok=True)

    for out in wsi_output:
        basename = os.path.basename(out[0]).split(".")[0]
        outname = os.path.basename(out[1]).split(".")[0]
        shutil.move(
            os.path.join(output_dir, "hovernetplus/tmp", f"{outname}.1.npy"),
            os.path.join(layer_dir, basename + ".npy"),
            )   
        shutil.move(
            os.path.join(output_dir, "hovernetplus/tmp", f"{outname}.0.dat"),
            os.path.join(nuclei_dir, basename + ".dat"),
            )
    shutil.rmtree(os.path.join(output_dir, "hovernetplus/tmp"))

    return
    

if __name__ == '__main__':
    args = docopt(__doc__, help=False, options_first=True, 
                    version='ODYN HoVer-Net+ Inference')

    if args['--help']:
        print(__doc__)
        exit()

    if args['--input_wsi_dir']:
        input_wsi_dir = args['--input_wsi_dir']
    else:      
        input_wsi_dir = "/data/ANTICIPATE/outcome_prediction/MIL/github_testdata/wsis/"

    if args['--input_dysplasia_dir']:
        input_dysp_dir = args['--input_dysplasia_dir']
    else:      
        input_dysp_dir = "/data/ANTICIPATE/outcome_prediction/MIL/github_testdata/output3/dysplasia/"
        
    if args['--input_epith_dir']:
        input_dysp_dir = args['--input_epith_dir']
    else:      
        input_dysp_dir = "/data/ANTICIPATE/outcome_prediction/MIL/github_testdata/output3/epith/"
            
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

    threshold = 0.055022543958983 # dysplasia ratio threshold for determing OED vs normal

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
    
    generate_ratio(
        input_wsi_dir=input_wsi_dir,
        output_dir=output_dir,
        colour_dict=colour_dict,
        mode=mode,
        nr_loader_workers=int(args['--nr_loader_workers']),
        nr_post_proc_workers=int(args['--nr_post_proc_workers']),
        batch_size=int(args['--batch_size']),
        )
