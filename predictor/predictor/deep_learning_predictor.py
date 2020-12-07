import sys

import torch
from torchvision import models, transforms

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

def train_pytorch_model(model, train_images_batches, train_labels_batches, optimizer, optimizer_params, criterion, epochs_num):
    getattr(
        sys.modules['torch.optim'],
        optimizer
    )(**optimizer_params)
    for epoch in range(epochs_num):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, (images_batch, labels_batch) in enumerate(zip(train_images_batches, train_labels_batches)):
            optimizer.zero_grad()
            if torch.cuda.is_available():
                images_batch, labels_batch = deep_learning_preprocessor.preprocess_batch(
                    images_batch,
                    labels_batch
                ).to("cuda")
            outputs = model(deep_learning_preprocessor(images_batch))
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    return model

def predict_with_pytorch(model_name, train_images_batches, train_labels_batches, test_images, test_labels, optimizer, optimizer_params, criterion, epochs_num):
    model = getattr(
        sys.modules['torchvision.models'],
        model_name
    )(pretrained = False)
    if torch.cuda.is_available():
        model.to("cuda")
    model = train_pytorch_model(
        model,
        train_images_batches,
        train_labels_batches,
        optimizer,
        optimizer_params,
        criterion,
        epochs_num
    )


    print('Finished Training')

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


