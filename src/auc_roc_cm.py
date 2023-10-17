#########
# Imports
#########

import ast
import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
import matplotlib.pyplot as plt

###########
# Functions
###########


## Plotting AUC-ROC
def plot_auc_roc(y_true, y_scores, title, save_dir, save_title):

    n_classes = len(np.unique(y_true))

    # Binarize true labels for each class
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    y_scores = np.array(y_scores)

    # Initialize variables to store ROC curves and AUC scores
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Calculate ROC curves and AUC-ROC scores for each class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves for each class
    plt.figure(figsize=(8, 6))
    colors = cycle(['darkorange', 'cornflowerblue', 'purple'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir,save_title))
    plt.close()

## Plotting Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, title, save_dir, save_title):

    n_classes = len(np.unique(y_true))
    # Calculate confusion matrix
    confusion_mat = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', cbar=False, square=True,
                xticklabels=[f'Class {i}' for i in range(n_classes)],
                yticklabels=[f'Class {i}' for i in range(n_classes)])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(title)
    plt.savefig(os.path.join(save_dir,save_title))
    plt.close()

############
# Running it
############

if __name__ == "__main__":

    save_dir = '/mnt/cervical_cancer/analysis/ViT_Downstream'
 
    # Getting csvs
    ## Validation
    validation_csv_path = '/mnt/cervical_cancer/predictions/full_dataset_vit_huge_downstream/predictions_validation.csv'
    validation_csv = pd.read_csv(validation_csv_path)

    ## Testing
    testing_csv_path = '/mnt/cervical_cancer/predictions/full_dataset_vit_huge_downstream/predictions.csv'
    testing_csv = pd.read_csv(testing_csv_path)

    testing_ground_truths_csv_path = '/mnt/cervical_cancer/csvs/model_36_split_df_all_gt.csv' # Needed as the predictions.csv doesn't have the ground truths
    testing_ground_truths_csv = pd.read_csv(testing_ground_truths_csv_path)
    testing_ground_truths_csv = testing_ground_truths_csv[testing_ground_truths_csv['dataset'] == 'test'] # Needed, as there are two test sets (test and test2) in this csv

    # testing2_csv_path = '/mnt/cervical_cancer/predictions/full_dataset_vit_huge_inference_test2/predictions.csv'
    # testing2_csv = pd.read_csv(testing2_csv_path)

    # testing2_ground_truths_csv_path = '/mnt/data/model_36_split_df_all_gt.csv'
    # testing2_ground_truths_csv = pd.read_csv(testing_ground_truths_csv_path)
    # testing2_ground_truths_csv = testing_ground_truths_csv[testing_ground_truths_csv['dataset'] == 'test2'] # Needed, as there are two test sets (test and test2) in this csv

    # Validation
    ground_truth = [int(value) for value in validation_csv['CC_ST']]
    prediction_mc_probs = [ast.literal_eval(value) for value in validation_csv['pred_mc']]
    prediction = [int(value) for value in validation_csv['predicted_class']]

    plot_auc_roc(ground_truth, prediction_mc_probs, 'Multiclass Receiver Operating Characteristic (ROC) Curves for Validation Predictions', save_dir, 'full_dataset_vit_huge_downstream_Val_AUC_ROC.png')
    plot_confusion_matrix(ground_truth, prediction, 'Confusion Matrix for Validation Set on ViT Huge', save_dir, 'full_dataset_vit_huge_downstream_Val_CM.png')

    # Testing
    ground_truth = [int(value) for value in testing_ground_truths_csv['CC_ST']] # Pulling the ground truths from a different file
    prediction_mc_probs = [ast.literal_eval(value) for value in testing_csv['pred_mc']]
    prediction = [int(value) for value in testing_csv['predicted_class']]

    plot_auc_roc(ground_truth, prediction_mc_probs, 'Multiclass Receiver Operating Characteristic (ROC) Curves for Test Predictions', save_dir, 'full_dataset_vit_huge_downstream_Test_AUC_ROC.png')
    plot_confusion_matrix(ground_truth, prediction, 'Confusion Matrix for Validation Set on ViT Huge Huge', save_dir, 'full_dataset_vit_huge_downstream_Test_CM.png')

    # # Testing2
    # ground_truth = [int(value) for value in testing2_csv['CC_ST']]
    # prediction_mc_probs = [ast.literal_eval(value) for value in testing2_csv['pred_mc']]
    # prediction = [int(value) for value in testing2_csv['predicted_class']]

    # plot_auc_roc(ground_truth, prediction_mc_probs, 'Multiclass Receiver Operating Characteristic (ROC) Curves for Test2 Predictions', save_dir, 'ViTMAE_Huge_Finetuned_full_dataset_Test2_AUC_ROC.png')
    # plot_confusion_matrix(ground_truth, prediction, 'Confusion Matrix for Validation Set on Finetuned ViTMAE Huge', save_dir, 'ViTMAE_Huge_Finetuned_full_dataset_Test2_CM.png')