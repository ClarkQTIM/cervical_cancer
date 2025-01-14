import os
import sys
import pandas as pd
import json
import math
import numpy as np
import torch
import matplotlib.image as mpimg
from monai.transforms import Compose, NormalizeIntensity, ScaleIntensity, Resize
from sklearn.model_selection import train_test_split
from transformers import ViTFeatureExtractor, AutoImageProcessor

def transform_dataset(dataset, transformation): # Chris added

    prepared_ds = dataset.with_transform(transformation)

    return prepared_ds

def modify_transforms_feat_extractor_custom_norms(transforms, feat_extractor, custom_norms): # Chris added

    if feat_extractor != None: # Getting the feature extractor means, stds, and sizes
        image_mean, image_std = feat_extractor.image_mean, feat_extractor.image_std

        try: # Issue, some feature extractors have different names for things, so this just checks it
            size = feat_extractor.size["height"]
        except:
            size = feat_extractor.crop_size["height"]

    '''
    11/13: 
    You shouldn't need custom_norms[0], so figure out why that is happening. Also, make sure that the feat_extractor can
    be None and it won't cause issues. You haven't run into this because you have always used one, but make sure.
    '''

    if custom_norms[0] != None: # Changing the means and stds
        image_mean = custom_norms[0]['means']
        image_std = custom_norms[0]['stds']

    normalize = NormalizeIntensity(subtrahend=image_mean, divisor=image_std, channel_wise=True)
    resize = Resize(spatial_size=(size, size))

    updated_transforms = []
    for transform in transforms.transforms:
        if isinstance(transform, Resize):
            updated_transforms.append(resize)
        elif isinstance(transform, ScaleIntensity):
            updated_transforms.append(normalize)
        else:
            updated_transforms.append(transform)

    return Compose(updated_transforms)

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                feature_extractor, # Chris added
                custom_norms,
                df: pd.DataFrame,
                data_path: str,
                transforms: Compose,
                label_colname: str,
                image_colname: str):
        self.feature_extractor = feature_extractor # Chris added
        self.custom_norms = custom_norms,
        self.df = df
        self.data_path = data_path
        self.transforms = transforms
        self.label_name = label_colname
        self.image_name = image_colname

    def __len__(self):
        return len(self.df)

    def __getitem__(self,
                    index: int):
        img_path = os.path.join(self.data_path, self.df[self.image_name].iloc[index])
        if img_path.endswith('.npy'):
            img = np.load(img_path).astype('float32')
        else:
            try:
                img = mpimg.imread(img_path).astype('float32')
            except:
                from PIL import Image, ImageFile
                print(img_path)
                ImageFile.LOAD_TRUNCATED_IMAGES = True
                img = np.array(Image.open(img_path)).astype('float32')               
            
            # Use provided bounding box if available. The bounding box coordinates should be stored in columns named
            # y1, y2, x1, x2.
            if 'y1' in self.df:
                idx_data = self.df.iloc[index]
                img = img[int(idx_data['y1']): int(idx_data['y2']), int(idx_data['x1']): int(idx_data['x2']), :]

                if self.feature_extractor != None or self.custom_norms[0] != None: # If we have a feature extractor, we are going to change the transforms
                    self.transforms = modify_transforms_feat_extractor_custom_norms(self.transforms, self.feature_extractor, self.custom_norms)
                    # Remove center crop if the bounding box is provided
                    self.transforms = Compose([tr for tr in list(self.transforms.transforms)
                                           if 'CenterSpatialCrop' not in str(tr)])
                else: # No feature extractor and no custom_norms
                    # Remove center crop if the bounding box is provided
                    self.transforms = Compose([tr for tr in list(self.transforms.transforms)
                    if 'CenterSpatialCrop' not in str(tr)])

            else: # No bounding box
                if self.feature_extractor != None or self.custom_norms[0] != None: # If we have a feature extractor, we are going to change the transforms
                    self.transforms = modify_transforms_feat_extractor_custom_norms(self.transforms, self.feature_extractor, self.custom_norms)
                
        gt = self.df[self.label_name].iloc[index]

        # Image, label, image filename
        return self.transforms(img), \
               torch.as_tensor(int(gt)) if not math.isnan(gt) else gt, \
               self.df[self.image_name].iloc[index]


def loader(architecture: str, # Chris added
           data_path: str,
           output_path: str,
           train_transforms: Compose,
           val_transforms: Compose,
           metadata_path: str = None,
           train_frac: float = 0.65,
           test_frac: float = 0.25,
           custom_norms = None,
           seed: int = 0,
           batch_size: int = 32,
           balanced: bool = False,
           weights: torch.Tensor = None,
           label_colname: str = 'label',
           image_colname: str = 'image',
           split_colname: str = 'dataset',
           patient_colname: str = 'patient',
           data_origin: str = None):
            # Add an input that referneces the architecture params so I can get the feature extractor from vit-mae
    """
    Inspired by https://github.com/Project-MONAI/tutorials/blob/master/2d_classification/mednist_tutorial.ipynb

    Returns:
        DataLoader, DataLoader, DataLoader, pd.Dataframe: train dataset, validation dataset, val dataset, test dataset,
         test df
    """

    # Load in feature extractor if using a pretrained ViTMAE
    feature_extractor = None
    if 'vit' in architecture: 
        print('We are using a pre-made feature extractor!')
        feature_extractor = ViTFeatureExtractor.from_pretrained(architecture)
    elif 'dinov2' in architecture:
        print('We are using a pre-made feature extractor!')
        feature_extractor = AutoImageProcessor.from_pretrained(architecture)
    elif 'dino_' in architecture:
        print('We are using a pre-made feature extractor from facebook/vit-mae-base!')
        feature_extractor = AutoImageProcessor.from_pretrained('facebook/vit-mae-base') # Hardcoded here so we get ImageNet features
    
    # Removing the feature extractor if we want model_36 training augs/transformations
    if 'no_fe' in architecture:
        print('We are loading in a model, but we are NOT keeping the feature extractor. Defaulting to model_36 training augs/transformations.')
        feature_extractor = None

    # Loading in custom norms
    if custom_norms:
        print(f'Custom normalization: {custom_norms}')

    # Load metadata and create val/train/test split if not already done
    split_df = split_dataset(output_path, train_frac=train_frac, test_frac=test_frac,
                             seed=seed, metadata_path=metadata_path, split_colname=split_colname, image_colname=image_colname,
                             patient_colname=patient_colname, data_origin=data_origin) # added image_colname on 08/21/2022, remove for DC
    train_loader, val_loader, test_loader = None, None, None

    # Training loader
    df_train = split_df[split_df[split_colname] == "train"]
    if len(df_train):
        if data_origin != "None":
            df_train = df_train[df_train['STUDY'] == data_origin]
            print(f'We have subset with {data_origin} and have {len(df_train)} training examples')
        if balanced:
            sampler, weights = get_balanced_sampler(df_train, label_colname, weights)
        shuffle = not balanced

        train_ds = Dataset(feature_extractor, custom_norms, df_train, data_path, train_transforms, label_colname, image_colname)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=10, sampler=sampler if balanced else None)
        
    # Val loader
    df_val = split_df[split_df[split_colname] == "val"]
    if len(df_val):
        if data_origin != "None":
            df_val = df_val[df_val['STUDY'] == data_origin]
        if balanced:
            sampler, _ = get_balanced_sampler(df_val, label_colname, weights)
        val_ds = Dataset(feature_extractor, custom_norms, df_val, data_path, val_transforms, label_colname, image_colname)
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, num_workers=10, sampler=sampler if balanced else None)
            
    # Test loader
    df_test = split_df[split_df[split_colname] == "test"]
    if len(df_test):
        if data_origin != "None":
            df_test = df_test[df_test['STUDY'] == data_origin]
        test_ds = Dataset(feature_extractor, custom_norms, df_test, data_path, val_transforms, label_colname, image_colname)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, num_workers=10)

    return train_loader, val_loader, test_loader, df_val, df_test, weights

def get_unbalanced_loader(feature_extractor, custom_norms, df, data_path, batch_size, transforms, label_colname, image_colname):
    ds = Dataset(feature_extractor, custom_norms, df, data_path, transforms, label_colname, image_colname)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=10, sampler=None)


def get_balanced_sampler(split_df: pd.DataFrame,
                         label_name: str,
                         weights: None):
    """ Balances the sampling of classes to have equal representation. """
    labels, count = np.unique(split_df[label_name], return_counts=True)
    weight_count = (1 / torch.Tensor(count)).float()
    if weights is None:
        weights = weight_count
    else:
        weights *= weight_count
    sample_weights = torch.tensor([weights[int(l)] for l in split_df[label_name]]).float()
    sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(sample_weights))
    return sampler, weights


def split_dataset(output_path: str,
                  metadata_path: str,
                  train_frac: float,
                  test_frac: float,
                  seed: int,
                  split_colname: str,
                  image_colname: str,
                  patient_colname: str,
                  data_origin: str):
    """Load csv file containing metadata (image filenames, labels, patient ids, and val/train/test split)"""
    if data_origin == "None":
        split_df_path = os.path.join(output_path, "split_df_orig.csv")
    else:
        split_df_path = os.path.join(output_path, "split_df_"+data_origin+"_training.csv")

    # If output_path / "split_df.csv" exists use the already split csv
    if os.path.isfile(split_df_path):
        df = pd.read_csv(split_df_path)
        if data_origin != "None":
            df = df[df['STUDY'] == data_origin]
            print(f'We have subset the split_df by {data_origin}')
    # If output_path / "split_df.csv" doesn't exist: split images by patient using the train and test fractions
    else:
        df = pd.read_csv(metadata_path)
        if data_origin != "None":
            df = df[df['STUDY'] == data_origin]
        # If images are not already split into val/train/test, split by patient
        if split_colname not in df:
            print('generating splits based on metrics provided')
            patient_lst = list(set(df[patient_colname].tolist()))
            train_patients, remain_patients = train_test_split(patient_lst, train_size=train_frac, random_state=seed)
            test_patients, val_patients = train_test_split(remain_patients, train_size=test_frac / (1 - train_frac),
                                                           random_state=seed)

            df[split_colname] = None
            df.loc[df[patient_colname].isin(train_patients), split_colname] = 'train'
            df.loc[df[patient_colname].isin(val_patients), split_colname] = 'val'
            df.loc[df[patient_colname].isin(test_patients), split_colname] = 'test'

        df.to_csv(split_df_path)

    return df
