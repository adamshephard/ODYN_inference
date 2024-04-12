"""
Inference script for generating ODYN-scores. This uses repeated cross-validation.

Usage:
  oed_prognosis.py [options] [--help] [<args>...]
  oed_prognosis.py --version
  oed_prognosis.py (-h | --help)
  
Options:
  -h --help                    Show this string.
  --version                    Show version.

  --input_data_file=<string>   Path to csv file containing fold information and targets per slide.
  --input_ftrs_dir=<string>    Path to folder containing features. Stored as indiviudal .tar files containing each tile's features.
  --output_dir=<string>        Path to output directory to save results.
  --norm_parameters=<string>   Path to file containing normalization parameters.
  --model_checkpoint=<string>  Path to model checkpoint.
  --cutoff_file=<string>       Path to file containing cutoffs.    

Use `oed_prognosis.py --help` to show their options and usage
"""


from docopt import docopt
import os
multi_gpu = True
if multi_gpu==False:
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import auc, roc_curve, f1_score, precision_recall_curve, average_precision_score, precision_score, recall_score
from tqdm import tqdm

from dataloader.mil_reader import featuresdataset_inference
from models.net_desc import MLP
from utils.metrics import compute_aggregated_predictions, compute_aggregated_probabilities, group_avg_df, get_topk_patches, get_bottomk_patches
from sklearn.metrics import recall_score


def get_metrics(y_true, y_prob, cutoff):
    y_pred = [1 if i >= cutoff else 0 for i in y_prob]
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    f1 = f1_score(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    average_precision = average_precision_score(y_true, y_prob)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return {
        "f1":f1,
        "roc_auc": roc_auc, 
        "precision": precision, 
        "recall": recall, 
        "average_precision": average_precision
    }

def oed_prognosis(
    data_file: str,
    data_path: str,
    norm_parameters: str,
    checkpoint_path: str, 
    output: str, 
    nr_repeats: int | int = 3, 
    nr_folds: int | int = 5, 
    batch_size: int | int = 256,
    workers: int | int = 8, 
    aggregation_method: str | str = "avgtop", 
    cutoff: str | str = None, 
    method: str | str = "mlp", 
    features: str | str = "morph_features_104_64ftrs",  
    outcome: str | str = "transformation", 
    k: int | int = 5,
    ) -> None:
    """
    Inference script for generating ODYN-scores. This uses repeated cross-validation.

    Args:
        data_file: Path to csv file containing fold information and targets per slide.
        data_path: Path to folder containing features. Stored as indiviudal .tar files containing each tile's features.
        norm_parameters: Path to file containing normalization parameters.
        checkpoint_path: Path to model checkpoint.
        nr_repeats: Number of cross-validation repeats.
        nr_folds: Number of cross-validation folds.
        output: Path to output directory to save results.
        batch_size: Batch size.
        workers: Number of workers.
        aggregation_method: Aggregation method.
        cutoff: Cutoff file.
        method: Method, i.e. MLP.
        features: Features.
        outcome: Outcome.
        k: K top patches.
    """

    # cnn inference 
    def inference(loader, model):
        model.eval()
        probs = torch.FloatTensor(len(loader.dataset), 2)
        preds = torch.FloatTensor(len(loader.dataset))
        with torch.no_grad():
            with tqdm(total=len(loader)) as pbar:
                for i, (inputs, target) in enumerate(loader):
                    inputs = inputs.cuda()
                    target, wsi_name = target
                    target = target.cuda()
                    output = model(inputs)
                    y = F.softmax(output, dim=1)
                    _, pr = torch.max(output.data, 1)
                    preds[i * batch_size:i * batch_size + inputs.size(0)] = pr.detach().clone()
                    probs[i * batch_size:i * batch_size + inputs.size(0)] = y.detach().clone()
                    pbar.update(1)
        return probs.cpu().numpy(), preds.cpu().numpy()

    dataset = pd.read_csv(data_file)
        
    torch.cuda.empty_cache()

    test_data = dataset[dataset['test']==1][['slide_name', outcome, 'cohort']]
    test_data = test_data.rename(columns={"slide_name": "wsi", outcome: "label"})     

    if "raw_images" in features: # e.g raw images to train CNN
        model = models.resnet34(True) # pretrained resnet34
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        # defining data transform
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                        std=[0.1, 0.1, 0.1])
        trans_Valid = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        raw_images = True
        
    else: # e.g. MLP
        if features == 'deep_features':
            nr_ftrs = 1024
            nr_hidden = 512
        elif features == 'morph_features_104_64ftrs':
            nr_ftrs = 168
            nr_hidden = 64
        trans_Valid = None
        raw_images = False

        if method == 'mlp':
            model = MLP(d=nr_ftrs, hidden_d=nr_hidden, nr_classes=2)

    model.cuda()
    cudnn.benchmark = True

    cutoff_df = pd.read_csv(cutoff, index_col=[0])
    
    preds_all = []
    scores_all = []
    # Inference with CV
    for repeat in range(1, nr_repeats):
        for fold in range(nr_folds):
            
            # Get norm params
            norm_params = pd.read_csv(os.path.join(norm_parameters, f"repeat{repeat}_fold{fold}.csv"), index_col=0)
            norm_params = [list(norm_params['mean']), list(norm_params['std'])]
            
            # Load data
            print(f"Processing data with repeat {repeat} fold {fold}")
            test_dset = featuresdataset_inference(data_path=data_path, data_frame=test_data, transform=trans_Valid, raw_images=raw_images, norm=norm_params)
            test_loader = torch.utils.data.DataLoader(
                test_dset,
                batch_size=batch_size, shuffle=False,
                num_workers=workers, pin_memory=False)
                    
            # Inference
            ch = torch.load(os.path.join(checkpoint_path, f"repeat{repeat}_fold{fold}.pth"))
            cutoff = cutoff_df.loc[(cutoff_df["repeat"] == str(float(repeat))) & (cutoff_df["fold"] == str(float(fold)))][aggregation_method].item()
            model.load_state_dict(ch['state_dict'])
            
            test_probs, test_preds = inference(test_loader, model)
            test_probs_1 = test_probs[:, 1]
            top_prob_test = np.nanmax(test_probs, axis=1)

            ## aggregation of tile scores into slide score
            test_slide_mjvt, test_slide_mjvt_raw  = compute_aggregated_predictions(np.array(test_dset.slideIDX), test_preds)
            test_slide_avg, test_slide_max, test_slide_sum, test_slide_md, test_slide_gm, test_slide_avgtop  = compute_aggregated_probabilities(np.array(test_dset.slideIDX), test_probs_1)
            test_slide_topk_patches = get_topk_patches(np.array(test_dset.slideIDX), np.array(test_dset.tiles), test_probs_1, k)
            test_slide_bottomk_patches = get_bottomk_patches(np.array(test_dset.slideIDX), np.array(test_dset.tiles), test_probs_1, k)

            test_slide_avgt5 = group_avg_df(np.array(test_dset.slideIDX), test_probs_1)

            if aggregation_method == 'med':
                test_pred = test_slide_md
            elif aggregation_method == 'mj vote':
                test_pred = test_slide_mjvt
            elif aggregation_method == 'mj vote raw':
                test_pred = test_slide_mjvt_raw
            elif aggregation_method == 'avg prob':
                test_pred = test_slide_avg
            elif aggregation_method == 'max prob':
                test_pred = test_slide_max
            elif aggregation_method == 'sump':
                test_pred = test_slide_sum
            elif aggregation_method == 'gmp':
                test_pred = test_slide_gm
            elif aggregation_method == 'avgtop':
                test_pred = test_slide_avgtop 
            elif aggregation_method == 'top5':
                test_pred = test_slide_avgt5
        
            slides = test_dset.slides
            cohorts = test_dset.cohorts
            slides = [os.path.basename(i) for i in slides]
            y_pred = [1 if i >= cutoff else 0 for i in test_pred]
            y_true = test_dset.targets
            metrics = get_metrics(y_true, test_pred, cutoff)

            preds = {
                "case": slides,
                "cohort": cohorts,
                "y_true": y_true,
                f"y_prob_repeat{repeat}_fold{fold}": test_pred,
                f"y_pred_repeat{repeat}_fold{fold}": y_pred,
            }
            preds_all.append(preds)
            
            scores = {
                "model": f"repeat{repeat}_fold{fold}",
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "average_precision": metrics["average_precision"],
                "cutoff": cutoff,
                "aggregation_method": aggregation_method,
            }
            scores_all.append(scores)
            
    for p, preds in enumerate(preds_all):
        if p == 0:
            preds_all_comb = pd.DataFrame(preds)
        else:
            preds_all_comb = preds_all_comb.merge(pd.DataFrame(preds), on=['case', 'cohort', 'y_true'], how="inner")
        
    scores_all = pd.DataFrame(scores_all)

    # Now combine results....
    # combine via prediction
    preds_all_comb[f"y_pred_prob_ensemble"] = preds_all_comb.filter(regex='_pred_repeat').mean(axis=1)
    preds_all_comb[f"y_pred_pred_ensemble"] = [1 if p >= 0.5 else 0 for p in preds_all_comb[f"y_pred_prob_ensemble"]]
    # combined via probability - this was more successful
    preds_all_comb[f"y_prob_prob_ensemble"] = preds_all_comb.filter(regex='_prob_repeat').mean(axis=1)
    mean_cutoff = cutoff_df.loc['mean'][aggregation_method]
    preds_all_comb[f"y_prob_pred_ensemble"] = [1 if p >= mean_cutoff else 0 for p in preds_all_comb[f"y_pred_prob_ensemble"]]
    
    out_name = os.path.join(output, f'predictions.csv')
    preds_all_comb.to_csv(out_name, index=False)
    
    out_name = os.path.join(output, f'predictions_ensemble.csv')
    preds_ens = preds_all_comb.loc[:,~preds_all_comb.columns.str.contains('repeat')]
    preds_ens.to_csv(out_name, index=False)
    
    # Get ensemble metrics
    metrics_1 = get_metrics(preds_all_comb["y_true"],preds_all_comb["y_pred_prob_ensemble"], 0.5)
    metrics_2 = get_metrics(preds_all_comb["y_true"],preds_all_comb["y_prob_prob_ensemble"], mean_cutoff)
    scores_all = pd.concat([scores_all, pd.DataFrame({
        "model": "y_pred_pred_ensemble",
        "f1": metrics_1["f1"],
        "roc_auc": metrics_1["roc_auc"],
        "precision": metrics_1["precision"],
        "recall": metrics_1["recall"],
        "average_precision": metrics_1["average_precision"],
        "cutoff": "0.5",
        "aggregation_method": aggregation_method,
    }, index=[0])])
    scores_all = pd.concat([scores_all, pd.DataFrame({
        "model": "y_prob_pred_ensemble",
        "f1": metrics_2["f1"],
        "roc_auc": metrics_2["roc_auc"],
        "precision": metrics_2["precision"],
        "recall": metrics_2["recall"],
        "average_precision": metrics_2["average_precision"],
        "cutoff": mean_cutoff,
        "aggregation_method": aggregation_method,
    }, index=[0])]).reset_index(drop=True)
        
    summary_out_name = os.path.join(output, f'summary_metrics.csv')
    scores_all.to_csv(summary_out_name, index=False)
    
    # ensembled output should be a csv with columns: case, cohort, y_true, y_prob, y_pred
    # raw output should be a csv with columns: case, cohort, y_true, y_pred_rep1_fold1, y_prob_rep1_fold1,...., , sy_ensemble_prob, y_ensemble_pred
    


if __name__ == '__main__':
    args = docopt(__doc__, help=False, options_first=True, 
                    version='ODYN Prognosis')

    if args['--help']:
        print(__doc__)
        exit()

    if args['--input_data_file']:
        input_data_file = args['--input_data_file']
    else:      
        input_data_file = "/data/ANTICIPATE/github/testdata/sheffield_inference_data.csv"   
    
    if args['--input_ftrs_dir']:
        input_ftrs_dir = args['--input_ftrs_dir']
    else:
        input_ftrs_dir = "/data/ANTICIPATE/github/testdata/output/odyn/features/0.5-mpp_512_256_dysplasia-0.25/nuclear/pt_files/"
        
    if args['--output_dir']:
        output_dir = args['--output_dir']
    else:
        output_dir = "/data/ANTICIPATE/github/testdata/output/odyn/prognosis/"
    
    if args['--norm_parameters']:
        norm_parameters = args['--norm_parameters']
    else:
        norm_parameters = "/data/ANTICIPATE/github/ODYN_inference/models/norm_params/"
    
    if args['--model_checkpoint']:
        checkpoint_path = args['--model_checkpoint']
    else:
        # checkpoint_path = "/data/ANTICIPATE/outcome_prediction/MIL/idars/output/train-sheffield_test-belfast,birmingham,brazil/transformation/mlp/morph_features_104_64ftrs_50eps_corrected_belfast_train_thr/oed/repeat_2/best0/checkpoint_best_AUC.pth"
        checkpoint_path = "/data/ANTICIPATE/github/ODYN_inference/weights/cross_validation/"
        
    if args['--cutoff_file']:
        cutoff_file = args['--cutoff_file']
    else:
        cutoff_file = "/data/ANTICIPATE/github/ODYN_inference/models/norm_params/cutoff_summary.csv"        
        

    ### Inputs Files and Paramaters ###
    batch_size = 256          
    workers = 6                             # number of data loading workers
    aggregation_method = "avgtop"           # method for aggregating predictions
    method = "mlp"                          # model being used, for paper use MLP, but alternatives are ResNet34
    features = "morph_features_104_64ftrs"  # input features. Could also be deep features e.g. resnet
    outcome = "transformation"              # prediction outcome
    k = 5                                   # top/bottom patches to keep
    nr_repeats = 3
    nr_folds = 5
    
    os.makedirs(output_dir, exist_ok=True)
        
    oed_prognosis(
        input_data_file, input_ftrs_dir, norm_parameters, checkpoint_path, output_dir,
        nr_repeats, nr_folds, batch_size, workers, aggregation_method, cutoff_file,
        method, features, outcome, k
        )