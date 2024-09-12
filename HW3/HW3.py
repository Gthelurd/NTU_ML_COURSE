# Import necessary packages.
import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
# Pretrained model
from torchvision.models import resnet34, resnet50, vgg16, densenet121, alexnet, squeezenet1_0
# This is for the progress bar.
from tqdm.auto import tqdm
import random
# For plotting learning curve
from torch.utils.tensorboard import SummaryWriter
# K-fold cross validation and boosting
from sklearn.model_selection import KFold
from sklearn.ensemble import AdaBoostClassifier
# Optuna
# import optuna
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        self.cnn = nn.Sequential(
            #############################################################
#             BOTTOM
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]
            ######################################################################
#             midium
            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2, 0),      # [256, 16, 16]
            ############################################################################
#             TOP
            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]
            
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 4, 4]
        )
        self.fc = nn.Sequential(
            
#            
            nn.Dropout(p=0.5),
            nn.Linear(512*4*4, 1024),
            nn.LeakyReLU(),
            
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            
            nn.Dropout(0.5),
            nn.Linear(512, 11),
            nn.LogSoftmax()
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

    
class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.model = resnet34(weights=None)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 11)
        self.model.num_classes = 11
        
    def forward(self, x):
        return self.model(x)
    
class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.model = resnet50(weights=None)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 11)
        self.model.num_classes = 11
        
    
    def forward(self, x):
        return self.model(x)

    
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.model = vgg16(weights=None)
        num_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_features, 11)
        self.model.num_classes = 11

    def forward(self, x):
        return self.model(x)


class DenseNet121(nn.Module):
    def __init__(self):
        super(DenseNet121, self).__init__()
        self.model = densenet121(weights=None)
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, 11)
        self.model.num_classes = 11

    def forward(self, x):
        x = self.model(x)
        return x
    

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.model = alexnet(weights=None)
        num_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_features, 11)
        self.model.num_classes = 11

    def forward(self, x):
        x = self.model(x)
        return x
    
    
class SqueezeNet(nn.Module):
    def __init__(self):
        super(SqueezeNet, self).__init__()
        self.model = squeezenet1_0(weights=None)
        # (1): Conv2d(512, 1000, kernel_size=(1, 1), stride=(1, 1))
        num_features = self.model.classifier[1].in_channels
        self.model.classifier[1] = nn.Conv2d(num_features, 11, kernel_size=(1,1), stride=(1,1))
        self.model.num_classes = 11
        
    def forward(self, x):
        x = self.model(x)
        return x
    
# Load the trained model
# model = Classifier().to(device)
# state_dict = torch.load(f"F:\VSC-code\Python\ML\HW3\sample_best.ckpt",map_location=torch.device('cpu'))
# please use torch.load with map_location=torch.device('cpu') to map your storages to the CPU.
# model.load_state_dict(state_dict)
# model.eval()

# print(model)

# study of training set size for an mlp on the circles problem
from sklearn.datasets import make_circles
from keras.layers import Dense
from keras.models import Sequential
from numpy import mean
from matplotlib import pyplot

# create train and test datasets
def create_dataset(n_train, n_test=100000, noise=0.1):
	# generate samples
	n_samples = n_train + n_test
	X, y = make_circles(n_samples=n_samples, noise=noise, random_state=1)
	# split into train and test, first n for test
	trainX, testX = X[n_test:, :], X[:n_test, :]
	trainy, testy = y[n_test:], y[:n_test]
	# return samples
	return trainX, trainy, testX, testy

# evaluate an mlp model
def evaluate_model(trainX, trainy, testX, testy):
	# define model
	model = Sequential()
	model.add(Dense(25, input_dim=2, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit model
	model.fit(trainX, trainy, epochs=500, verbose=0)
	# evaluate the model
	_, test_acc = model.evaluate(testX, testy, verbose=0)
	return test_acc

# repeated evaluation of mlp model with dataset of a given size
def evaluate_size(n_train, n_repeats=5):
	# create dataset
	trainX, trainy, testX, testy = create_dataset(n_train)
	# repeat evaluation of model with dataset
	scores = list()
	for _ in range(n_repeats):
		# evaluate model for size
		score = evaluate_model(trainX, trainy, testX, testy)
		scores.append(score)
	return scores

# define dataset sizes to evaluate
sizes = [100, 1000, 5000, 10000]
score_sets, means = list(), list()
for n_train in sizes:
	# repeated evaluate model with training set size
	scores = evaluate_size(n_train)
	score_sets.append(scores)
	# summarize score for size
	mean_score = mean(scores)
	means.append(mean_score)
	print('Train Size=%d, Test Accuracy %.3f' % (n_train, mean_score*100))
# summarize relationship of train size to test accuracy
pyplot.plot(sizes, means, marker='o')
pyplot.show()
# plot distributions of test accuracy for train size
pyplot.boxplot(score_sets, labels=sizes)
pyplot.show()
pyplot.savefig(".\\train_size.jpg")
# study of test set size for an mlp on the circles problem
from sklearn.datasets import make_circles
from keras.layers import Dense
from keras.models import Sequential
from numpy import mean
from matplotlib import pyplot

# create dataset
def create_dataset(n_test, n_train=1000, noise=0.1):
	# generate samples
	n_samples = n_train + n_test
	X, y = make_circles(n_samples=n_samples, noise=noise, random_state=1)
	# split into train and test, first n for test
	trainX, testX = X[:n_train, :], X[n_train:, :]
	trainy, testy = y[:n_train], y[n_train:]
	# return samples
	return trainX, trainy, testX, testy

# fit an mlp model
def fit_model(trainX, trainy):
	# define model
	model = Sequential()
	model.add(Dense(25, input_dim=2, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit model
	model.fit(trainX, trainy, epochs=500, verbose=0)
	return model

# evaluate a test set of a given size on the fit models
def evaluate_test_set_size(models, n_test):
	# create dataset
	_, _, testX, testy = create_dataset(n_test)
	scores = list()
	for model in models:
		# evaluate the model
		_, test_acc = model.evaluate(testX, testy, verbose=0)
		scores.append(test_acc)
	return scores

# create fixed training dataset
trainX, trainy, _, _ = create_dataset(10)
# fit one model for each repeat
n_repeats = 10
models = [fit_model(trainX, trainy) for _ in range(n_repeats)]
print('Fit %d models' % n_repeats)
# define test set sizes to evaluate
sizes = [100, 1000, 5000, 10000]
score_sets, means = list(), list()
for n_test in sizes:
	# evaluate a test set of a given size on the models
	scores = evaluate_test_set_size(models, n_test)
	score_sets.append(scores)
	# summarize score for size
	mean_score = mean(scores)
	means.append(mean_score)
	print('Test Size=%d, Test Accuracy %.3f' % (n_test, mean_score*100))
# summarize relationship of test size to test accuracy
pyplot.plot(sizes, means, marker='o')
pyplot.show()
# plot distributions of test size to test accuracy
pyplot.boxplot(score_sets, labels=sizes)
pyplot.show()

pyplot.savefig(".\\test_size.jpg")