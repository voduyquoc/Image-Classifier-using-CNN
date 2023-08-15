import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import json
import argparse
import sys

from collections import OrderedDict

# A function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg19(pretrained = True)
    model.class_to_idx = checkpoint['class_to_idx']
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 4096)),
                                            ('relu', nn.ReLU()),
                                            ('drop', nn.Dropout(p = 0.5)),
                                            ('fc2', nn.Linear(4096, 102)),
                                            ('output', nn.LogSoftmax(dim = 1))]))
    model.classifier = classifier
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

# A function that pre-processes the image so it can be used as input for the model.
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image_path)
    
    # Resize with Aspect Ratio maintained
    # First fixing the short axes
    if pil_image.width > pil_image.height:
        (width, height) = (int(pil_image.width / pil_image.height) * 256, 256)
    else:
        (width, height) = (256, int(pil_image.height / pil_image.width) * 256)
    pil_image = pil_image.resize((width, height))
    
    # Crop
    left = (pil_image.width - 224) / 2
    bottom = (pil_image.height - 224) / 2
    right = left + 224
    top = bottom + 224
    
    pil_image = pil_image.crop((left, bottom, right, top))
    
    # Convert to np then Normalize
    np_image = np.array(pil_image) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # Transpose to fit PyTorch Axes
    np_image = np_image.transpose([2, 0, 1])
    
    return np_image


# A function converts a PyTorch tensor and displays it
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    if title is not None:
        ax.set_title(title)
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

# A function for making predictions with model
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Implement the code to predict the class from an image file
    img = process_image(image_path)
    
    # Convert np_img to PT tensor and send to GPU
    pt_img = torch.from_numpy(img).type(torch.cuda.FloatTensor)
    
    # Unsqueeze to get shape of tensor from [Ch, H, W] to [Batch, Ch, H, W]
    pt_img = pt_img.unsqueeze(0)

    # Run the model to predict
    output = model.forward(pt_img)
    
    probs = torch.exp(output)
    
    # Pick out the topk from all classes 
    top_probs, top_indices = probs.topk(topk)
    
    # Convert to list on CPU without grads
    top_probs = top_probs.detach().type(torch.FloatTensor).numpy().tolist()[0]
    top_indices = top_indices.detach().type(torch.FloatTensor).numpy().tolist()[0]
    
    # Invert the class_to_idx dict to a idx_to_class dict
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    
    top_classname = {idx_to_class[index] for index in top_indices}
    
    return top_probs, top_classname


if __name__ == "__main__":

    # Input

    # Class labelling
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Now the device is set to {device}')

    # Load checkpoint
    print('*** Loading checkpoint ***')
    model = load_checkpoint(saved_pth=sys.argv[2])

    # -----------------------------^^^^^^ INFERENCE STAGE ^^^^^^^----------------------------------- #
    print('*** Inference Stage ***')

    # Plot flower input image
    print('*** Input image ***')
    plt.figure(figsize = (6,10))
    plot_1 = plt.subplot(2,1,1)

    image = process_image(sys.argv[1])

    #flower_name = cat_to_name['21']

    imshow(image, plot_1)

    # Prediction with model
    model.to(device)
    probs, classes = predict(image_path=sys.argv[1], model=model, topk=5)   
    print(probs)
    print(classes)

    print('*** Prediction result ***')
    # Convert from the class integer encoding to actual flower names
    flower_names = [cat_to_name[i] for i in classes]

    # Plot the probabilities for the top 5 classes as a bar graph
    plt.subplot(2,1,2)

    sb.barplot(x=probs, y=flower_names, color=sb.color_palette()[0])

    plt.show()