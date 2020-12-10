import sys

import torch
import torch.nn as nn
from torchvision import models, transforms
from utilities import data_utils


from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

from preprocessor import deep_learning_preprocessor
import torch.optim as optim


def train_on_batch(batch_images, batch_labels, model, optimizer, criterion, epoch, batch_num, confusion_matrix=None):
    optimizer.zero_grad()
    if torch.cuda.is_available():
        batch_images = batch_images.cuda()
        batch_labels = batch_labels.cuda()
    outputs = model(torch.unsqueeze(batch_images, 1).float())
    loss = criterion(outputs, batch_labels)
    loss.backward()
    optimizer.step()
    if epoch is not None:
        print(
            'for epoch: %s and batch: %s ' %(
                str(epoch), str(batch_num)
            )
        )
    print('the loss:')
    print(loss.item())

def get_confusion_matrix(model, images, labels, confusion_matrix):
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
    
    for epoch in range(epochs_num):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, (batch_images, batch_labels) in enumerate(train_data_generator):
            train_on_batch(
                batch_images,
                batch_labels,
                model,
                optimizer,
                criterion,
                epoch,
                i
            )
    return model

def predict_with_pytorch(model_name, train_images, train_labels, test_images, test_labels, optimizer_name, optimizer_params, criterion, epochs_num, batch_size):
    model = getattr(
        sys.modules['torchvision.models'],
        model_name
    )(
        pretrained=False,
        num_classes=2
    )
    model.features[0] = nn.Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    if torch.cuda.is_available():
        model.cuda()
    model = train_pytorch_model(
        model,
        torch.utils.data.DataLoader(
            data_utils.Dataset(
                train_images,
                train_labels
            ),
            **{
                'batch_size': batch_size,
                'shuffle': True,
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
            )
        ),
        confusion_matrix = torch.zeros(2, 2)
    )

def predict(model, test_data_generator, confusion_matrix):
    print('confusion matrix for testing: ')
    for batch_num, (test_images, test_labels) in enumerate(test_data_generator):
        print('for batch number: %s the confusion matrix is: ' %str(batch_num))
        print(
            get_confusion_matrix(model, test_images, test_labels, confusion_matrix)
        )

def build_model(input_shape):
    #use the sequential thing
    input_img = Input(shape=input_shape, name='ImageInput')
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv1_1')(input_img)
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv1_2')(x)
    x = MaxPooling2D((2,2), name='pool1')(x)
    
    x = SeparableConv2D(128, (3,3), activation='relu', padding='same', name='Conv2_1')(x)
    x = SeparableConv2D(128, (3,3), activation='relu', padding='same', name='Conv2_2')(x)
    x = MaxPooling2D((2,2), name='pool2')(x)
    
    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_1')(x)
    x = BatchNormalization(name='bn1')(x)
    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_3')(x)
    x = MaxPooling2D((2,2), name='pool3')(x)
    
    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_1')(x)
    x = BatchNormalization(name='bn3')(x)
    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_2')(x)
    x = BatchNormalization(name='bn4')(x)
    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_3')(x)
    x = MaxPooling2D((2,2), name='pool4')(x)
    
    x = Flatten(name='flatten')(x)
    x = Dense(1024, activation='relu', name='fc1')(x)
    x = Dropout(0.7, name='dropout1')(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    x = Dropout(0.5, name='dropout2')(x)
    x = Dense(2, activation='softmax', name='fc3')(x)
    
    model = Model(inputs=input_img, outputs=x)
    return model


