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
from collections import OrderedDict
import json
import argparse
import sys

# Define your transforms for the training, validation, and testing sets
data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Define training model function
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10, device='cpu'):
    train_losses, test_losses = [], []
    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in train_loader:
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
        else:
            test_loss = 0
            accuracy = 0
            with torch.no_grad():                  
                model.eval()
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    test_loss += criterion(logps, labels).item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            model.train()
            
            train_losses.append(running_loss/len(train_loader))
            test_losses.append(test_loss/len(val_loader))
            
            print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                  "Training Loss: {:.3f}.. ".format(train_losses[-1]),
                  "Validation Loss: {:.3f}.. ".format(test_losses[-1]),
                  "Validation Accuracy: {:.3f}".format(accuracy/len(val_loader)))
            
def test_model(model, test_loader, device):
    model.to(device)
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
                        
            test_loss += batch_loss.item()
                        
            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
    print(f"Testing Accuracy: {accuracy/len(test_loader):.3f}")

# Save the checkpoint function
def save_checkpoint(model, optimizer, train_image_datasets, saved_pth):
    # set a new attr to the model object, which holds the class_to_idx conversion
    model.class_to_idx = train_image_datasets.class_to_idx
    
    # Checkpoint is a dictionary, can be modified to hold anything you need in the future
    checkpoint = {'arch': 'vgg19',
                  'class_to_idx': model.class_to_idx,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict()}
    
    # Save with torch.save
    torch.save(checkpoint, saved_pth)

if __name__ == "__main__":
    
    # Input
    

    # Data loading
    data_dir = sys.argv[1]
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Load the datasets with ImageFolder
    train_image_datasets = datasets.ImageFolder(train_dir, transform = data_transforms)
    val_image_datasets = datasets.ImageFolder(valid_dir, transform = test_transforms)
    test_image_datasets = datasets.ImageFolder(test_dir, transform = test_transforms)

    # Using the image datasets and the transform, define the dataloader
    train_loader = torch.utils.data.DataLoader(train_image_datasets, batch_size = 32, shuffle = True)
    val_loader = torch.utils.data.DataLoader(val_image_datasets, batch_size = 32)
    test_loader = torch.utils.data.DataLoader(test_image_datasets, batch_size = 32)

    # Class labelling
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    # Build and train network
    # Use GPU if it's available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Now the device is set to {device}')

    model = models.vgg19(pretrained = True)

    # Freeze parameters so we don't backprop through them
    for params in model.parameters():
        params.requries_grad = False


    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 4096)),
                                            ('relu', nn.ReLU()),
                                            ('drop', nn.Dropout(p = 0.5)),
                                            ('fc2', nn.Linear(4096, 102)),
                                            ('output', nn.LogSoftmax(dim = 1))]))

    model.classifier = classifier

    # Negative Log Likelihood Loss
    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr = 0.001)

    model.to(device)

    # Train model
    print('*** Training model ***')
    train_model(model=model, train_loader=train_loader, val_loader=val_loader, criterion=criterion, optimizer=optimizer, epochs=10, device=device)

    # Test model
    print('*** Testing model on testing dataset ***')
    test_model(model=model, test_loader=test_loader, device=device)

    # Save checkpoint
    print('*** Saving checkpoint ***')
    save_checkpoint(model=model, optimizer=optimizer, train_image_datasets=train_image_datasets, saved_pth='checkpoint.pth')

