# Reproducible And Clinically Translatable Deep Neural Networks For Cervical Screening

This repository will allow you to reproduce the experiments in our paper titled "Reproducible And Clinically Translatable Deep Neural Networks For Cervical Screening".

## Get started

1. Clone github repository   
```bash
git clone https://github.com/rknahmed0/cervical_cancer.git
```
2. Build docker image  
```bash
cd cervical_cancer/
```
```bash
docker build -t gray_zone docker/
```
3. Run docker container  
```bash
docker run -it -v tunnel/to/local/folder:/tunnel --gpus 0 gray_zone:latest bash
```
4. Run the following command at the root of the repository to install the modules  
```bash
cd path/to/cervical_cancer
```
```bash
pip install -e .
```
5. Train model  
```bash
python run_model.py -o <outpath/path> -p <resources/training_configs/config.json> -d <image/data/path> -c <path/csv/file.csv>
```
For more information on the different flags: `python run_model.py --help`.

## Configuration file (flag -p or --param-path)  
The configuration file is a json file containing the main training parameters.  
A json file example is located in `gray_zone/resources/training_configs/`  

### Required configuration parameters
Sample config file: https://github.com/rknahmed0/cervical_cancer/blob/main/gray_zone/resources/training_configs/sample_config_file_model_36.json

|    Parameter   | Description |  
| -------- | --- |  
| architecture |   Architecture id. Choice among: 'densenet121', 'resnest50', 'resnet50', 'vit' (swin transformer). |
| model_type | Choices among: "classification", "ordinal". |
| loss |  Loss function id. Choices among: 'ce' (Cross entropy), 'foc' (Focal), 'qwk' (Quadratic weighted kappa), 'coral' (Ordinal loss). |  
| foc_gamma | gamma parameter for foc loss (float or None). |
| batch_size | Batch size (int). |  
| lr | Learning rate (float). |  
| n_epochs | Number of training epochs (int). |  
| device | Device id (e.g., 'cuda:0', 'cpu') (str).  |   
| val_metric | Choices among: "custom_cervix" (default: summed AUC), "auc" (average ROC AUC over all classes), "val_loss" (minimum val loss), "kappa" (linear Cohen's kappa). |  
| dropout_rate | Dropout rate (Necessary for Monte Carlo models). A dropout rate of 0 will disable dropout. (float). |  
| is_weighted_loss | Indicates if the loss is weighted by the number of cases by class (bool). |  
| is_weighted_sampling |  Indicates if the sampling is weighted by the number of cases by class (bool). |
| weights |  Indicates preassigned weights to be used for sampling (if any) per ground truth class (list of weights with length(list) = num_class). |  
| seed | Random seed (int).  |  
| train_frac | Fraction of cases used for training if splitting not already done in csv file, or else the parameter is ignored (float). |  
| test_frac | Fraction of cases used for testing if splitting not already done in csv file, or else the parameter is ignored (float). |  
| train_transforms / val_transforms | monai training / validation transforms with parameters. Validation transforms are also used during testing (see https://docs.monai.io/en/latest/transforms.html for transform list)  |


## csv file (flag -c or --csv-path)
The provided csv file contains the filename of the images used for training, GT labels (int from 0-n_class), patient ID 
(str) and split column (containing 'train', 'val' or 'test') (optional). 

Example of csv file with the default column names. If the column names are different from the default values,
the flags `--label-colname`, `--image-colname`, `--patient-colname`, and `--split-colname` can 
be used to indicate the custom column names. There can be more columns in the csv file. All this
metadata will be included in `predictions.csv` and `split_df.csv`.

|    image   | label | patient  |  dataset  |
| :--------: | :---: | :------: |  :------: |  
| patient1_000.jpg |   0   |  patient1  |    train  |
| patient1_001.jpg |   0   |  patient1  |    train  |
| patient2_000.png |   2   |  patient2  |    val  |
| patient2_001.png |   2   |  patient2  |    val  |
| patient2_002.png |   2   |  patient2  |    val  |
| patient3_000.jpg |   1   |  patient3  |    test  |
| patient3_001.jpg |   1   |  patient3  |    test  |

## Output directory (flag -o or --output-path)
```

└── output directory                # Output directory specified with `-o`  
    ├──   checkpoints               # All models (one .pth per epoch)  
    |     ├──  checkpoint0.pth   
    |     ├──  ...  
    |     └──  checkpointn.pth   
    ├──   best_metric_model.pth     # Best model based on validation metric  
    ├──   params.json               # Parameters used for training (configuration file)  
    ├──   predictions.csv           # Test results  
    ├──   predictions_validation.csv           # Validation results  
    ├──   split_df.csv              # csv file containing image filenames, labels, split and patient id  
    └──   train_record.json         # Record of CLI used to train and other info for reproducibility
```
