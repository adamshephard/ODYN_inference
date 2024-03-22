<p align="center">
  <img src="doc/odyn.png" width='350'>
</p>

# ODYN: Oral DYsplasia Network Inference

This repository provides the ODYN inference code for the models used within the paper [Development and Validation of an Artificial Intelligence-based Pipeline for the Prediction of Malignant Transformation in Oral Epithelial Dysplasia: A Retrospective Multi-Centric Study](). <br />

The first step in this pipeline is to use HoVer-Net+ (see original paper [here](https://openaccess.thecvf.com/content/ICCV2021W/CDPath/html/Shephard_Simultaneous_Nuclear_Instance_and_Layer_Segmentation_in_Oral_Epithelial_Dysplasia_ICCVW_2021_paper.html)) to segment the epithelium and nuclei. We have used the TIAtoolbox (see paper [here](https://www.nature.com/articles/s43856-022-00186-5)) implementation of HoVer-Net+ in the below scripts. Next, we have used a Transformer-based model to segment the dyplastic regions of the WSIs (see paper [here](https://arxiv.org/abs/2311.05452)).

We determine a slide as being normal or oral epithelial dysplasia (OED), by calculating the proportion of the epithelium that is predicted to be dysplastic. If this is above a certain threshold, we classify the case as OED.

Following this, for OED slides, we generate patch-level morphological and spatial features to use in our ODYN pipeline. We generate an ODYN-score for each slide by passing these patch-level features through a pre-trained multi-layer perceptron (MLP).

Note, we recommend running inference of the ODYN model on the GPU. Nuclear instance segmentation, in particular, will be very slow when run on CPU. 

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
- `heatmap_generation.py`: script to generate heatmaps (needs tidying up)


## Inference

### Data Format
Input: <br />
- WSIs supported by [OpenSlide](https://openslide.org/), including `svs`, `tif`, `ndpi` and `mrxs`.

### Model Weights

The Transformer model weights (for dyplasia segmentation) obtained from training on the Sheffield OED dataset: [OED Transformer checkpoint](https://drive.google.com/file/d/1EF3ItKmYhtdOy5aV9CJZ0a-g03LDaVy4/view?usp=sharing). 
The MLP model weights obtained from training on each fold of the Sheffield OED dataset: [OED MLP checkpoint](). 
If any of the models/checkpoints are used, please ensure to cite the corresponding paper.

### Usage

#### ODYN Pipeline

A user can run the ODYN pipeline on all their slides using the below command. This can be quite slow as nuclear segmentation (with HoVer-Net+) is run at 0.5mpp.

Usage: <br />
```
  python run_odyn.py --input_dir="/path/to/input/slides/or/images/dir/" --output_dir="/path/to/output/dir/"
```

Alternatively, to have more control, a user can run each of the stages used by the ODYN model at a time. These are shown below.

#### Dysplasia Segmentation with Transformer

The first stage is to run the Transformer-based model on the WSIs to generate dysplasia segmentations. This is relatively fast and is run at 1.0mpp.

Usage: <br />
```
  python dysplasia_segmentation.py --input_dir="/path/to/input/slides/or/images/dir/" --output_dir="/path/to/transformer/output/dir/"
```
#### Epithelium Segmentation with HoVer-Net+

The second stage is to run HoVer-Net+ on the WSIs to generate epithelial and nuclei segmentations. This can be quite slow as run at 0.5mpp.

Usage: <br />
```
  python epithelium_segmentation.py --input_dir="/path/to/input/slides/or/images/dir/" --output_dir="/path/to/epithelium/output/dir/"
```

#### OED Diagnosis with ODYN

The second stage is to classify a slide as being OED vs normal.

Usage: <br />
```
  python epithelium_segmentation.py --input_dir="/path/to/input/slides/or/images/dir/" --epithelium_dir="/path/to/hovernetplus/output/" --dysplasia_dir="/path/to/transformer/output/" --output_dir="/path/to/output/dir/"
```

#### Feature Generation with ODYN

The fourth stage is to tesselate the image into smaller patches and generate correpsonding patch-level morphological and spatial features using the nuclei/layer segmentations. Note the `epithelium_dir` is the output directory from the previous step.

Usage: <br />
```
  python feature_generation.py --input_dir="/path/to/input/slides/or/images/dir/" --epithelium_dir="/path/to/hovernetplus/output/" --output_dir="/path/to/output/feature/dir/"
```

#### OED Prognosis wit ODYN, i.e. the ODYN-score

The final stage is to infer using the MLP on the tiles (and their features) generated in the previous steps. Here, the `input_ftrs_dir` is the directroy containnig the features created in the previous steps. The `model_checkpoint` path is to the weights provided above, and the `input_data_file` is the path to the data file describing the slides to process. An example file is provided in `data_file_template.csv`.

Usage: <br />
```
  python oed_prognosis.py --input_data_file="/path/to/input/data/file/" --input_ftrs_dir="/path/to/input/tile/ftrs/" --model_checkpoint="/path/to/model/checkpoint/" --output_dir="/path/to/output/dir/"
```

#### ODYN Heatmaps

We can also generate heatmaps for these images. Change the `stride` within the file from 128 to create smoother images. However, a decreased stride by 2X will increase the processing time by 2X.
    
Usage: <br />
```
  python heatmap_generation.py --input_dir="/path/to/input/slides/or/images/dir/" --hovernetplus_dir="/path/to/hovernetplus/output/" --checkpoint_path="/path/to/checkpoint/" --output_dir="/path/to/heatmap/output/dir/"
```

## Interactive Demo

TO DO

## License

TO DO

## Cite this repository

TO DO
