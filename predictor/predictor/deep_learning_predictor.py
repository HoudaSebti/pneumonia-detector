import sys
import numpy as np

import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.optim as optim
from utilities import data_utils

from preprocessor import deep_learning_preprocessor
import matplotlib.pyplot as plt


def train_on_batch(batch_images, batch_labels, model, optimizer, criterion, epoch, batch_num, running_loss, confusion_matrix=None):
    optimizer.zero_grad()
    if torch.cuda.is_available():
        batch_images = batch_images.cuda()
        batch_labels = batch_labels.cuda()
    outputs = model(torch.unsqueeze(batch_images, 1).float())
    loss = criterion(outputs, batch_labels)
    loss.backward()
    optimizer.step()
    print(
        'for epoch: %s and batch: %s ' %(
            str(epoch), str(batch_num)
        )
    )
    confusion_matrix += get_confusion_matrix(
        model,
        batch_images,
        batch_labels
    )
    running_loss += loss.item()
    print('the loss:')
    print(loss.item())
    return model, optimizer, running_loss, confusion_matrix

def get_confusion_matrix(model, images, labels):
    confusion_matrix = torch.zeros(2, 2)
    if torch.cuda.is_available():
        images = images.cuda()
        labels = labels.cuda()
    outputs = model(torch.unsqueeze(images, 1).float())
    _, preds = torch.max(outputs, 1)
    for t, p in zip(labels.view(-1), preds.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1
    return confusion_matrix

def train_pytorch_model(model, train_data_generator, optimizer_name, optimizer_params, criterion, epochs_num):
    optimizer = getattr(
        sys.modules['torch.optim'],
        optimizer_name,
    )(params = model.parameters(), **optimizer_params)
    
    accuracies = np.zeros((epochs_num, ))
    f_scores = np.zeros((epochs_num, ))
    recalls = np.zeros((epochs_num, ))
    precisions = np.zeros((epochs_num, ))
    losses = np.zeros((epochs_num, ))
    
    for epoch in range(epochs_num):  # loop over the dataset multiple times
        running_loss = 0.0
        data_size = 0
        confusion_matrix = torch.zeros(2, 2)
        for i, (batch_images, batch_labels) in enumerate(train_data_generator):
            model, optimizer, running_loss, confusion_matrix = train_on_batch(
                batch_images,
                batch_labels,
                model,
                optimizer,
                criterion,
                epoch,
                i,
                running_loss,
                confusion_matrix
            )
            data_size += batch_images.size(0)
        confusion_matrix = confusion_matrix.numpy()
        accuracies[epoch] = (
            confusion_matrix[0, 0] + confusion_matrix[1, 1]
        ) / np.sum(confusion_matrix)
        
        recalls[epoch] = confusion_matrix[0, 0] / (
            confusion_matrix[0, 0] + confusion_matrix[0, 1]
        )
        precisions[epoch] = confusion_matrix[0, 0] / (
            confusion_matrix[0, 0] + confusion_matrix[1, 0]
        )
        f_scores[epoch] = recalls[epoch] * precisions[epoch]
        losses[epoch] = running_loss / data_size
    return model, f_scores, recalls, precisions, accuracies, losses

def predict_with_pytorch(model, train_images, train_labels, test_images, test_labels, optimizer_name, optimizer_params, criterion, epochs_num, batch_size):
    if torch.cuda.is_available():
        model.cuda()
    class_weights = [
        train_images.shape[0] / (2 * np.bincount(train_labels)[i]) for i in range(2)
    ]
    model, f_scores, recalls, precisions, accuracies, losses = train_pytorch_model(
        model,
        torch.utils.data.DataLoader(
            data_utils.Dataset(
                train_images,
                train_labels
            ),
            **{
                'batch_size': batch_size,
                'sampler' : torch.utils.data.WeightedRandomSampler(
                    [
                        class_weights[train_label] for train_label in train_labels
                    ],
                    int(train_labels.shape[0])
                ),
                'num_workers': 6
            }
        ),
        optimizer_name,
        optimizer_params,
        criterion,
        epochs_num
    )
    
    print('Finished Training')
    predict(
        model,
        torch.utils.data.DataLoader(
            data_utils.Dataset(
                test_images,
                test_labels
            ),
            batch_size = batch_size,
            num_workers = 6

        )
    )
    plt.subplot(3, 1, 1)
    plt.plot(list(range(epochs_num)), f_scores)
    plt.plot(list(range(epochs_num)), recalls)
    plt.plot(list(range(epochs_num)), precisions)
    plt.legend(['f_scores', 'recalls', 'precisions'])
    plt.subplot(3, 1, 2)
    plt.plot(list(range(epochs_num)), accuracies)
    plt.legend(['accuracies'])
    plt.subplot(3,1,3)
    plt.plot(list(range(epochs_num)), losses)
    plt.legend(['losses'])
    plt.show()

def predict(model, test_data_generator):
    confusion_matrix = torch.zeros(2, 2)
    print('confusion matrix for testing: ')
    for batch_num, (test_images, test_labels) in enumerate(test_data_generator):
        print('for batch number: %s the confusion matrix is: ' %str(batch_num))
        confusion_matrix +=get_confusion_matrix(model, test_images, test_labels)
        print(confusion_matrix)
        


