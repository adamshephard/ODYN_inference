<p align="center">
  <img src="doc/odyn.png" width='350'>
</p>

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-orange.svg)](https://www.gnu.org/licenses/gpl-3.0)
  <a href="#cite-this-repository"><img src="https://img.shields.io/badge/Cite%20this%20repository-BibTeX-brightgreen" alt="DOI"></a>
<br>

# ODYN: Oral DYsplasia Network Inference

This repository provides the ODYN inference code for the models used within the paper [Development and Validation of an Artificial Intelligence-based Pipeline for the Prediction of Malignant Transformation in Oral Epithelial Dysplasia: A Retrospective Multi-Centric Study](). <br />

The first step in this pipeline is to use HoVer-Net+ (see original paper [here](https://openaccess.thecvf.com/content/ICCV2021W/CDPath/html/Shephard_Simultaneous_Nuclear_Instance_and_Layer_Segmentation_in_Oral_Epithelial_Dysplasia_ICCVW_2021_paper.html)) to segment the epithelium and nuclei. We have used the TIAtoolbox (see paper [here](https://www.nature.com/articles/s43856-022-00186-5)) implementation of HoVer-Net+ in the below scripts. Next, we have used a Transformer-based model to segment the dyplastic regions of the WSIs (see paper [here](https://arxiv.org/abs/2311.05452)).

We determine a slide as being normal or oral epithelial dysplasia (OED), by calculating the proportion of the epithelium that is predicted to be dysplastic. If this is above a certain threshold, we classify the case as OED.

Following this, for OED slides, we generate patch-level morphological and spatial features to use in our ODYN pipeline. We generate an ODYN-score for each slide by passing these patch-level features through a pre-trained multi-layer perceptron (MLP).

Note, this repository is for use with oral tissue H&E-stained WSIs/ROIs alone. We recommend running inference of the ODYN model on the GPU. Nuclear instance segmentation, in particular, will be very slow when run on CPU. 

## TO DO
- [X] Add post processing to epithelium segmentation with HoVer-Net+
- [X] OED diagnosis script
- [X] Feature generation script
- [X] Get feature generation script to create tiles in desired way for MIL model, i.e. per patch
- [X] OED prognosis script
- [X] Run ODYN script and tidy output of odyn score/diagnosis
- [X] Heatmap script
- [X] Add interactive demo
- [ ] Add nuclei (as DBs) and heatmaps, and more slides to interactive demo (see below).
- [X] Viz output demo using TIAViz.
- [ ] Make file to covert images into db files for easy visualisation.
- [X] Add new HoVer-Net+ weights
- [X] Upload CV model weights
- [X] License information
- [ ] Add pre print and citation information
- [ ] Make repo public

## Set Up Environment

We use Python 3.11 with the [tiatoolbox](https://github.com/TissueImageAnalytics/tiatoolbox) package installed. By default this uses PyTorch 2.2.

```
conda create -n odyn python=3.11 cudatoolkit=11.8
conda activate odyn
pip install tiatoolbox
pip uninstall torch
conda install pytorch
pip install h5py
pip install docopt
pip install ml-collections
```

## Repository Structure

Below are the main directories in the repository: 

- `dataloader/`: the data loader and augmentation pipeline
- `doc/`: image files used for rendering the README
- `utils/`: scripts for metric, patch generation
- `models/`: model definition

Below are the main executable scripts in the repository:

- `run_odyn.py`: main inference script for ODYN, runs the below scripts consecutively (except heatmaps)
- `dysplasia_segmentation.py`: transformer inference script
- `epithelium_segmentation.py`: hovernetplus inference script
- `oed_diagnosis.py`: script to diagnose a slide as OED vs normal (using output from above script)
- `feature_generation.py`: script to generate features for the final MLP model (using output from above script)
- `oed_prognosis.py`: main inference script for geenrating the ODYN-score for predicting malignant transformation
- `heatmap_generation.py`: script to generate heatmaps
- `visualize_output.sh`: bash script to load TIAViz for visualising all ODYN output at WSI-level


## Inference

### Data Format
Input: <br />
- WSIs supported by [OpenSlide](https://openslide.org/), including `svs`, `tif`, `ndpi` and `mrxs`.

Output: <br />
- HoVer-Net nuclei and epithelium segmentations as `dat` and `png` files, respectively. These segmentations are saved at 0.5 mpp resolution. Nuclei `dat` files have a key as the ID for each nucleus, which then contain a dictionary with the keys:
  - 'box': bounding box coordinates for each nucleus
  - 'centroid': centroid coordinates for each nucleus
  - 'contour': contour coordinates for each nucleus 
  - 'prob': per class probabilities for each nucleus
  - 'type': prediction of category for each nucleus
- Transformer dysplasia segmentations as `png` files. These segmentations are saved at 1 mpp resolution.
- ODYN diagnosis and prognosis CSV. This CSV will have a row for each input WSI. The columns will then display `slide_name`, `status`, `ODYN-score`. The `status` is whether ODYN has classified the slide as being normal or OED. The `ODYN-score` is whether ODYN has predicted that the slide this lesion is from will progress to malignancy.
- [Optional] ODYN heatmaps as `png` files. These segmentations are saved at 2 mpp resolution.

### Model Weights

We use the following weights in this work. If any of the models or checkpoints are used, please ensure to cite the corresponding paper.

- The Transformer model weights (for dyplasia segmentation) obtained from training on the Sheffield OED dataset: [OED Transformer checkpoint](https://drive.google.com/file/d/1EF3ItKmYhtdOy5aV9CJZ0a-g03LDaVy4/view?usp=sharing). 
- The HoVer-Net+ model weights (for epithelium segmentation) obtained from training on the Sheffield OED dataset: [OED HoVer-Net+ checkpoint](https://drive.google.com/file/d/1D2OQhHv-5e9ncRfjv2QM8HE7PAWoS79h/view?usp=sharing). Note, these weights are updated compared to TIAToolbox's and are those obtained in this [paper](https://arxiv.org/abs/2307.03757).
- The MLP model weights obtained from training on each fold of the Sheffield OED dataset: [OED MLP checkpoints](https://drive.google.com/drive/folders/1pXbIiRpUbgjqcz-I1gIOurTC5ZdRizjM?usp=sharing). 

### Usage

#### ODYN Pipeline

A user can run the ODYN pipeline on all their slides using the below command. This can be quite slow as nuclear segmentation (with HoVer-Net+) is run at 0.5mpp. 

Usage: <br />
```
  python run_odyn.py --input_data_file="/path/to/input/data/file/" --input_dir="/path/to/input/slides/or/images/dir/" --output_dir="/path/to/output/dir/" --transformer_weights="/path/to/hovernetplus/checkpoint/" --hovernetplus_weights="/path/to/hovernetplus/checkpoint/" --mlp_weights="/path/to/mlp/checkpoint/" --mlp_norm_params="/path/to/mlp/norm/params/" --mlp_cutoff_file="/path/to/mlp/cutoffs/"
```

Alternatively, to have more control, a user can run each of the stages used by the ODYN model at a time. These are shown below. We recommend users to do use this method.

#### Dysplasia Segmentation with Transformer

The first stage is to run the Transformer-based model on the WSIs to generate dysplasia segmentations. This is relatively fast and is run at 1.0mpp. Note, the `model_checkpoint` is the path to the Transformer segmentation weights available to download from above.

Usage: <br />
```
  python dysplasia_segmentation.py --input_dir="/path/to/input/slides/or/images/dir/" --output_dir="/path/to/transformer/output/dir/" --model_checkpoint="/path/to/transformer/checkpoint/"
```

#### Epithelium Segmentation with HoVer-Net+

The second stage is to run HoVer-Net+ on the WSIs to generate epithelial and nuclei segmentations. This can be quite slow as run at 0.5mpp. Note, the `model_checkpoint` is the path to the HoVer-Net+ segmentation weights available to download from above. However, if none are provided then the default version of HoVer-Net+ used with TIAToolbox, will be used.

Usage: <br />
```
  python epithelium_segmentation.py --input_dir="/path/to/input/slides/or/images/dir/" --output_dir="/path/to/epithelium/output/dir/" --model_checkpoint="/path/to/hovernetplus/checkpoint/"
```

#### OED Diagnosis with ODYN

The second stage is to classify a slide as being OED vs normal.

Usage: <br />
```
  python oed_diagnosis.py ---input_epith="/path/to/hovernetplus/mask/output/" --input_dysplasia="/path/to/transformer/output/" --output_dir="/path/to/output/dir/"
```

#### Feature Generation with ODYN

The fourth stage is to tesselate the image into smaller patches and generate correpsonding patch-level morphological and spatial features using the nuclei/layer segmentations. Note the `mask_dir` is the epithelial mask output and `nuclei_dir` is the nuclei output directory from the HoVer-Net+ step.

Usage: <br />
```
  python feature_generation.py --input_dir="/path/to/input/slides/or/images/dir/" --mask_dir="/path/to/hovernetplus/mask/output/" --nuclei_dir="/path/to/hovernetplus/nuclei/output/" --output_dir="/path/to/output/feature/dir/"
```

#### OED Prognosis wit ODYN, i.e. the ODYN-score

The final stage is to infer using the MLP on the tiles (and their features) generated in the previous steps. Here, the `input_ftrs_dir` is the directroy containnig the features created in the previous steps. The `model_checkpoint` path is to the weights provided above, and the `input_data_file` is the path to the data file describing the slides to process. An example file is provided in `data_file_template.csv`.

Usage: <br />
```
  python oed_prognosis.py --input_data_file="/path/to/input/data/file/" --input_ftrs_dir="/path/to/input/tile/ftrs/" --model_checkpoint="/path/to/mlp/checkpoint/" --output_dir="/path/to/output/dir/"
```

#### ODYN Heatmaps

We can also generate heatmaps for these images. Change the `stride` within the file from 128 to create smoother images. However, a decreased stride by 2X will increase the processing time by 2X. Note this use the combined mask prdocued by the `oed_diagnosis.py` script.
    
Usage: <br />
```
  python heatmap_generation.py --input_dir="/path/to/input/slides/or/images/dir/" --mask_dir="/path/to/combined/mask/" --nuclei_dir --checkpoint_path="/path/to/mlp/checkpoint/" --output_dir="/path/to/heatmap/output/dir/"
```

#### Visualisation with TIAViz

Below we use the TIAToolbox's TIAViz tool to visualise the model output from OsDYN. Simply ammend the `slide_dir` and `overlay_dir` to the corresponding folders. Note, TIAViz will look two directory levels deep for overlays, prefixed with the name of the slide. You may need to change the permissions of the script to make it executable. If you use TIAViz, then please cite the paper [TIAViz: A Browser-based Visualization Tool for Computational Pathology Models](https://arxiv.org/abs/2402.09990).
```
$ chmod u+wrx ./visualize_output.sh
$ ./visualize_output.sh
```

## Interactive Demo

We have made an interactive demo to help visualise the output of our model. Note, this is not optimised for mobile phones and tablets. The demo was built using the TIAToolbox [tile server](https://tia-toolbox.readthedocs.io/en/latest/_autosummary/tiatoolbox.visualization.tileserver.TileServer.html).

Check out the demo [here](https://tiademos.dcs.warwick.ac.uk/bokeh_app?demo=odyn). 

In the demo, we provide multiple examples of WSI-level results. These include:
- Dysplasia segmentations (using the Transformer model). Here, dysplasia is in red.
- Intra-epithelial layer segmentation (using HoVer-Net+). Here, orange is stroma, red is the basal layer, green the (core) epithelial layer, and blue keratin.
- Nuclei segmentations (using HoVer-Net+). Here, orange is "other" nuclei (i.e. connective/inflammatory), whilst the epithelial nuclei are coloured according to their intra-epithelial layer (see above).
- ODYN heatmaps where red spots show areas of high importance for predicting malignant transformation.

Each histological object can be toggled on/off by clicking the appropriate buton on the right hand side. Also, the colours and the opacity can be altered.

## License

Code is under a GPL-3.0 license. See the [LICENSE](https://github.com/adamshephard/ODYN_inference/blob/main/LICENSE) file for further details.

Model weights are licensed under [Attribution-NonCommercial-ShareAlike 4.0 International](http://creativecommons.org/licenses/by-nc-sa/4.0/). Please consider the implications of using the weights under this license. 

## Cite this repository

TO DO
