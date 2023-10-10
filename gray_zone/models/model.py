import os
import torchvision
import torch
import monai
from monai.transforms import Activations
import torch.nn as nn
from transformers import ViTFeatureExtractor, ViTMAEForPreTraining, ViTForImageClassification
from gray_zone.models import dropout_resnet, resnest, vit, vgg
from gray_zone.models.coral import CoralLayer

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
    if 'resnet' in architecture:
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
        # model = ViTForImageClassification.from_pretrained(architecture, num_labels=output_channels)
        # feature_extractor = ViTFeatureExtractor.from_pretrained(architecture)
        # if chkpt_path !='None':
        #     print(f'We are loading in {chkpt_path} as fine-tuned weights.')
        #     checkpoint = torch.load(chkpt_path, map_location='cpu')
        #     msg = model.load_state_dict(checkpoint, strict=False)
        #     print(f'Message in model loading: {msg}')

        feature_extractor, model = load_vitmae_from_from_pretrained_w_weights(architecture, chkpt_path, False, True, output_channels)

        set_dropout_rate(model, dropout_rate)

    elif 'vit' in architecture:
        model = vit.vit_b16(num_classes=output_channels, image_size=img_dim[1], dropout_rate=dropout_rate)
        
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
    elif model_type == 'classification':
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
