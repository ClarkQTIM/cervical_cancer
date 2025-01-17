import os
import torchvision
import torch
import monai
from monai.transforms import Activations
import torch.nn as nn
from transformers import ViTFeatureExtractor, ViTMAEForPreTraining, ViTForImageClassification, Dinov2ForImageClassification, AutoImageProcessor
from gray_zone.models import dropout_resnet, resnest, vit, vgg
from gray_zone.models.coral import CoralLayer
import collections

# Custom dropout modifier for the ViTMAE
def set_dropout_rate(model, new_dropout_rate): # Chris added
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = new_dropout_rate

def load_vitmae_from_from_pretrained_w_weights(from_pretrained_model_path, weights_path, pretraining, classification, classes):

    feature_extractor = ViTFeatureExtractor.from_pretrained(from_pretrained_model_path)

    if weights_path == 'None' and pretraining:
        print('We are loading in a pretrained reconstruction model directly from the source (no weight switching).')
        model = ViTMAEForPreTraining.from_pretrained(from_pretrained_model_path)
    elif weights_path == 'None' and classification:
        print('We are loading in a pretrained classification model directly from the source (no weight switching).')
        model = ViTForImageClassification.from_pretrained(from_pretrained_model_path, num_labels=classes)

    if weights_path != 'None' and pretraining: # So, in this case, we are going to simply load in the fine-tuned weights to the pretraining
        print('We are loading in a pretrained reconstruction model architecture and then switching it out with fine-tuned weights.')
        model = ViTMAEForPreTraining.from_pretrained(from_pretrained_model_path)
        checkpoint = torch.load(weights_path, map_location='cpu')
        msg = model.load_state_dict(checkpoint, strict=False)
        print(f'Loading checkpoint weights for pretrained message: {msg}')
    elif weights_path != 'None' and classification: # So, in this case, we are going to replace the fine-tuned encoder of the pretrained (with weights loaded in)
        # to serve as the encoder to the classification
        print('We are loading in a pretrained reconstuction model, replacing the weights with a fine-tuned version, and then subsituting the encoder in a pretrained, but un-fine-tuned, classification model.')
        model_pt = ViTMAEForPreTraining.from_pretrained(from_pretrained_model_path)
        checkpoint = torch.load(weights_path, map_location='cpu')
        msg = model_pt.load_state_dict(checkpoint, strict=False)
        print(f'Loading checkpoint weights for pretrained message before switching encoders: {msg}')
        model_clf = ViTForImageClassification.from_pretrained(from_pretrained_model_path, num_labels=classes)
        msg = model_clf.vit.encoder.load_state_dict(model_pt.vit.encoder.state_dict())
        print(f'Switching pretrained/fine-tuned encoder in the classification model message: {msg}')
        model = model_clf

    return feature_extractor, model

def load_dinov2_from_pretrained_w_weights_change_classification_head(from_pretrained_model_path, weights_path, classes):

    feature_extractor = AutoImageProcessor.from_pretrained(from_pretrained_model_path)

    if weights_path == 'None':
        print('We are loading in a pre-trained DINOv2 classification model and changing the classification head.')
        model = Dinov2ForImageClassification.from_pretrained(from_pretrained_model_path)
        # Access the feature dimension from the Dinov2Model
        in_features = model.classifier.in_features
        # Replace the 'classifier' layer with a new linear layer
        model.classifier = nn.Linear(in_features, classes)

    elif weights_path != 'None':
        print('We are loading in a pre-trained DINOv2 classification model and switching the classification head. We are then switching weights to something already trained.')
        checkpoint = torch.load(weights_path, map_location='cpu')
        msg = model.load_state_dict(checkpoint, strict=False)
        print(f'Loading checkpoint weights for pretrained message: {msg}')

    return feature_extractor, model

def load_dinov1_from_pretrained_w_weights_change_classification_head(from_pretrained_model_path, weights_path, classes):

    if 'no_fe' in from_pretrained_model_path:
        from_pretrained_model_path = from_pretrained_model_path.split('no_fe_')[-1]
        
    feature_extractor = AutoImageProcessor.from_pretrained('facebook/vit-mae-base') # Hardcoded here so we get ImageNet features

    if weights_path == 'None':
        print('We are loading in a pre-trained DINOv1 foundational model and adding the classification head.')
        model = torch.hub.load('facebookresearch/dino:main', from_pretrained_model_path)
        # Replacing the 'Identity()' head with a classifier head
        linear_classifier = nn.Sequential(
            nn.Linear(2048, classes)
            )

        model.fc = linear_classifier

    elif weights_path != 'None':
        print('We are loading in a pre-trained DINOv1 foundational model and adding the classification head. We are then switching foundational weights to something already trained.')
        model = torch.hub.load('facebookresearch/dino:main', from_pretrained_model_path)
        checkpoint = torch.load(weights_path, map_location='cpu')

        # Getting the student information from the checkpoint
        student_state_dict = checkpoint['student']  # The fine-tuning script saves several things, so we want only the student
        backbone_finetuned_state_dict = collections.OrderedDict() # We need to run through the 

        # Iterate through the items in student_state_dict and select keys starting with 'module.backbone'
        for key, value in student_state_dict.items():
            if key.startswith('module.backbone.'): # Getting only the module.backbone keys, as those will match with the un-fine-tuned model
                # Remove the 'module.backbone.' prefix to get the corresponding key in the backbone
                new_key = key[len('module.backbone.'):]
                backbone_finetuned_state_dict[new_key] = value

        msg = model.load_state_dict(backbone_finetuned_state_dict, strict=False)
        print(f'Loading checkpoint weights for pretrained message: {msg}')
        # Replacing the 'Identity()' head with a classifier head

        linear_classifier = nn.Sequential(
            nn.Linear(2048, classes)
            )

        model.fc = linear_classifier

    return feature_extractor, model

def get_model(architecture: str,
              model_type: str,
              chkpt_path: str, # Chris added chkpt_path
              dropout_rate: float,
              n_class: int,
              device: str,
              transfer_learning: str,
              img_dim: list or tuple):
    """ Init model """
    feature_extractor = None
    output_channels, act = get_model_type_params(model_type, n_class)

    if 'dino_' in architecture or 'no_fe_dino_' in architecture: # Chris added

        feature_extractor, model = load_dinov1_from_pretrained_w_weights_change_classification_head(architecture, chkpt_path, output_channels)

        set_dropout_rate(model, dropout_rate)

        if "lin_probing" in model_type: # If we are doing Linear Probing, we are freezing everything but the final, classification layer
            print('We are applying Linear Probing. The following parameters will not be frozen:')
            for name, param in model.named_parameters():
                if 'classifier' not in name:  # Freeze all layers except the classifier (final linear layer)
                    param.requires_grad = False
                else:
                    print(name, param)
                    
    elif 'resnet' in architecture:
        resnet = getattr(dropout_resnet, architecture) if float(dropout_rate) > 0 else getattr(torchvision.models,
                                                                                               architecture)
        model = resnet(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, output_channels)

    elif 'resnest' in architecture:
        resnet = getattr(resnest, architecture)
        model = resnet(pretrained=True, final_drop=dropout_rate)
        model.fc = torch.nn.Linear(model.fc.in_features, output_channels)

    elif 'densenet' in architecture:
        densenet = getattr(monai.networks.nets, architecture)
        model = densenet(spatial_dims=2,
                         in_channels=3,
                         out_channels=output_channels,
                         dropout_prob=float(dropout_rate),
                         pretrained=True)

    elif 'vit-mae' in architecture: # Chris added

        feature_extractor, model = load_vitmae_from_from_pretrained_w_weights(architecture, chkpt_path, False, True, output_channels)

        set_dropout_rate(model, dropout_rate)

        if "lin_probing" in model_type: # If we are doing Linear Probing, we are freezing everything but the final, classification layer
            print('We are applying Linear Probing. The following parameters will not be frozen:')
            for name, param in model.named_parameters():
                if 'classifier' not in name:  # Freeze all layers except the classifier (final linear layer)
                    param.requires_grad = False
                else:
                    print(name, param)

    elif 'dinov2' in architecture: # Chris added

        feature_extractor, model = load_dinov2_from_pretrained_w_weights_change_classification_head(architecture, chkpt_path, output_channels)

        set_dropout_rate(model, dropout_rate)

        if "lin_probing" in model_type: # If we are doing Linear Probing, we are freezing everything but the final, classification layer
            print('We are applying Linear Probing. The following parameters will not be frozen:')
            for name, param in model.named_parameters():
                if 'classifier' not in name:  # Freeze all layers except the classifier (final linear layer)
                    param.requires_grad = False
                else:
                    print(name, param)

    elif 'vit' in architecture:
        # model = vit.vit_b16(num_classes=output_channels, image_size=img_dim[1], dropout_rate=dropout_rate)
        feature_extractor, model = load_vitmae_from_from_pretrained_w_weights(architecture, chkpt_path, False, True, output_channels)
        
    elif 'vgg' in architecture:
        vgg_model = getattr(vgg, architecture)
        model = vgg_model(num_classes=output_channels, dropout=dropout_rate)

    else:
        raise ValueError("Only ResNet or Densenet models are available.")

    model = model.to(device)

    # Ordinal model requires a particular last layer to ensure coherent prediction (monotonic prediction)
    if model_type == 'ordinal':
        model = torch.nn.Sequential(
            model,
            CoralLayer(output_channels, n_class)
        )
        model = model.to(device)

    # Transfer weights if transfer learning
    if transfer_learning is not None:
        pretrained_dict = {k: v for k, v in torch.load(transfer_learning, map_location=device).items() if
                           k in model.state_dict()
                           and v.size() == model.state_dict()[k].size()}
        model.state_dict().update(pretrained_dict)
        model.load_state_dict(model.state_dict())

    return model, act, feature_extractor


def get_model_type_params(model_type: str,
                          n_class: int):
    if model_type == 'ordinal':
        # Intermediate number of nodes
        out_channels = 10
        act = Activations(sigmoid=True)
    elif model_type == 'regression':
        out_channels = 1
        act = Activations(other=lambda x: x)
    elif 'classification' in model_type: # Changed here from "elif model_type == 'classification'"" so we can add 'lin_probing', as well.
        # Multiclass model
        if n_class > 2:
            act = torch.nn.Softmax(dim=1)
            out_channels = n_class
        # Binary model
        else:
            act = Activations(sigmoid=True)
            out_channels = 1
    else:
        raise ValueError("Model type needs to be 'ordinal', 'regression' or 'classification'")

    return out_channels, act
