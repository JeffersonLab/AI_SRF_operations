import sys
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
# from torch.utils.data.sampler import ImbalancedDatasetSampler
from pathlib import Path
from sklearn import preprocessing
from sklearn.utils import shuffle
import random
import pandas as pd
import os
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from LSTM_utils import *
from LSTM_FCN_ResNet_singleTask import *
import _pickle as cPickle
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
import time


if __name__ == '__main__':

    #### model training and testing ####

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(device)

    # Hyper-parameters
    classificationTask = 'cavity' #cavity or fault

    if classificationTask == 'cavity':
        numClasses = 9
    elif classificationTask == 'fault':
        numClasses = 7
    else:
        sys.exit('Please provide the correct task!')

    batch_size = 16
    num_epochs = 100
    learning_rate = 0.0001
    dimension_shuffle = False
    early_stop_patience = 25
    sequence_length=4096
    input_size = 32


    model = LSTM_FCN_ResNet_singleTask(numClasses, sequence_length, input_size, dimension_shuffle,
                                       device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

    loss1 = nn.CrossEntropyLoss()

    # saving outputs results
    system = 'pc'

    # Set path appropriately
    if system == 'pc':
        mainPath = Path('C:\\Monibor\\rfwtools\\All_Data_Extraction') #set directory
    else:
        sys.exit('Please pick the correct system!')

    if not mainPath.exists():
        mainPath.mkdir(parents=True)

    # set output folder based on current experiment
    outputFolder = mainPath / ('Model_cavity') #Name of the output folder
    ### Data Loading##
    data=np.array(cPickle.load(open("C:\\Monibor\\rfwtools\\All_Data_Extraction/cavity_data.pkl", "rb"))) #data directory
    label = np.array(cPickle.load(open("C:\\Monibor\\rfwtools\\All_Data_Extraction/cav_labels.pkl", "rb"))) # label directory
    ######################################
    from scipy import signal

    number, length, channel = data.shape
    resampled_data = np.zeros((number, sequence_length, input_size))
    for i in range(number):
        for j in range(input_size):
            resampled_data[i, 0:sequence_length, j] = signal.resample(data[i, :, j], sequence_length)

    from sklearn import preprocessing
    from scipy import stats

    # stats.zscore(a)

    data_normalized = np.zeros((number, sequence_length, input_size))
    from scipy import stats

    for i in range(number):
        for j in range(input_size):
            if np.max(resampled_data[i, :, j]) == np.min(resampled_data[i, :, j]):
                data_normalized[i, :, j] = 0.001
            else:
                data_normalized[i, :, j] = stats.zscore(resampled_data[i, :, j])

    X_train, X, y_train, y = train_test_split(data_normalized, label, test_size=0.4,stratify=label, random_state=21)
    X_val, X_test, y_val, y_test = train_test_split(X, y, test_size=0.5,stratify=y, random_state=21)
   ###################################################
    faultDict = {
        'Quench100ms': 0,
        'Quench_3ms': 1,
        'E_Quench': 2,
        'Heat Riser Choke': 3,
        'Microphonics': 4,
        'Controls Fault': 5,
        'Single Cav Turn off': 6
    }

#########################

    ####################################################################################################


    # Shuffle training and validation data to mix old and new examples
    X_train, label_train = shuffle(X_train, y_train )
    X_val, label_val = shuffle(X_val, y_val)
    print(label_train.dtype)
    label_train=np.int_(label_train)
    # Convert train and validate data from numpy arrays to torch tensors
    trainSet = torch.from_numpy(X_train).float()
    trainLabels = torch.from_numpy(label_train).long()

    valSet = torch.from_numpy(X_val).float()
    valLabels = torch.from_numpy(label_val).long()

    print('train set size',trainSet.size())

    # construct DataLoader
    train_dataset = torch.utils.data.TensorDataset(trainSet, trainLabels)

    class_sample_count = np.array([len(np.where(label_train == t)[0]) for t in np.unique(label_train)])
    print('class sample count',class_sample_count)
    weights = 1.0 / class_sample_count
    print('weight',weights)
    print(label_train)
    samples_weight = np.array([weights[t] for t in label_train])
    # print(samples_weight)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))
    # print(train_dataset.trainLabels.tolist())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    print(len(train_loader))

    val_dataset = torch.utils.data.TensorDataset(valSet, valLabels)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    print(len(val_loader))

    ##### Train the model ######
    # # Layer freezing
    # for param in model.parameters():
    #     param.requires_grad = False
    # for param in model.FC.parameters():
    #     param.requires_grad = True

    train_stats_df, best_model = trainModel(model, train_loader, val_loader,
                                            num_epochs, optimizer, scheduler, early_stop_patience, loss1, device)

    train_stats_df.to_csv(outputFolder / 'training_statistics.csv', index=False)

    torch.save(best_model.state_dict(), outputFolder / 'best_model.pt')
    torch.save(model.state_dict(), outputFolder / 'final_model.pt')

    # Convert testing data from numpy arrays to torch tensors
    testSet = torch.from_numpy(X_test).float()
    testLabels = torch.from_numpy(y_test).long()

    test_dataset = torch.utils.data.TensorDataset(testSet, testLabels)

    ##### Testing the fully trained model  #####
    MCDropout_UQ = 0  # regular testing:0, testing with MCDroput on for uncertainty quantification: 1
    if MCDropout_UQ:

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

        testPred, testAcc, acc_df, output_df = testModel_MCDropout_V2(
            model, test_loader, faultDict, classificationTask, device, MC_sample_size=64)

        if classificationTask == 'cavity':
            print('Final Model: CavityID Test Accuracy: {}'.format(testAcc))
            acc_df.to_csv(outputFolder / 'CavityID_model_classification_accuracy.csv', index=False)
            output_df.to_csv(outputFolder / 'CavityID_model_classification_outputs_UQ.csv', index=False)
        else:
            print('Final Model: Fault ID Test Accuracy: {}'.format(testAcc))
            acc_df.to_csv(outputFolder / 'FaultID_model_classification_accuracy.csv', index=False)
            output_df.to_csv(outputFolder / 'FaultID_model_classification_outputs_UQ.csv', index=False)
    else:

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

        testPred, testAcc = testModel(model, test_loader, device)
        print('Final Model: ' + classificationTask + 'ID Test Accuracy: {}'.format(testAcc))

    # interpret and save results #

    if classificationTask == 'cavity':
        # cavity classification
        cavReportPath = outputFolder / 'cavReports'
        if not cavReportPath.exists():
            cavReportPath.mkdir()
        cavityNames = ['All Cavities', 'Cavity 1', 'Cavity 2', 'Cavity 3', 'Cavity 4', 'Cavity 5', 'Cavity 6',
                       'Cavity 7', 'Cavity 8']
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8]

        confusion_matrix_title = 'Cavity Classification Final Model'
        saveResults(cavityNames, testLabels, testPred, classes, confusion_matrix_title, cavReportPath,
                    saveName='final_model_cavity')

    if classificationTask == 'fault':
        # Fault classification
        faultReportPath = outputFolder / 'faultReports'
        if not faultReportPath.exists():
            faultReportPath.mkdir()
        faultNames = ['Quench100ms','Quench_3ms', 'E_Quench','Heat Riser Choke', 'Microphonics', 'Controls Fault','Single Cav Turn off']
        classes = [0, 1, 2, 3, 4, 5, 6]

        confusion_matrix_title = 'Fault Classification Final Model'
        saveResults(faultNames, testLabels, testPred, classes, confusion_matrix_title, faultReportPath,
                    saveName='final_model_fault')

    ##### best model testing #####

    best_model = LSTM_FCN_ResNet_singleTask(numClasses, sequence_length, input_size, dimension_shuffle,
                                            device).to(
        device)
    best_model.load_state_dict(torch.load(outputFolder / 'best_model.pt'))

    MCDropout_UQ = 0  # regular testing:0, testing with MCDroput on for uncertainty quantification: 1
    if MCDropout_UQ:

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

        testPred, testAcc, acc_df, output_df = testModel_MCDropout_V2(
            best_model, test_loader, faultDict, classificationTask, device, MC_sample_size=64)

        if classificationTask == 'cavity':
            print('Best Model: CavityID Test Accuracy: {}'.format(testAcc))
            acc_df.to_csv(outputFolder / 'CavityID_BestModel_classification_accuracy.csv', index=False)
            output_df.to_csv(outputFolder / 'CavityID_BestModel_classification_outputs_UQ.csv', index=False)
        else:
            print('Best Model: Fault ID Test Accuracy: {}'.format(testAcc))
            acc_df.to_csv(outputFolder / 'FaultID_BestModel_classification_accuracy.csv', index=False)
            output_df.to_csv(outputFolder / 'FaultID_BestModel_classification_outputs_UQ.csv', index=False)

    else:

        testPred, testAcc = testModel(best_model, test_loader, device)
        print('Best Model: ' + classificationTask + 'ID Test Accuracy: {}'.format(testAcc))

    # interpret and save results #

    if classificationTask == 'cavity':
        # cavity classification
        cavReportPath = outputFolder / 'cavReports'
        if not cavReportPath.exists():
            cavReportPath.mkdir()
        cavityNames = ['All Cavities', 'Cavity 1', 'Cavity 2', 'Cavity 3', 'Cavity 4', 'Cavity 5', 'Cavity 6',
                       'Cavity 7', 'Cavity 8']
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8]

        confusion_matrix_title = 'Cavity Classification Best Model'
        saveResults(cavityNames, testLabels, testPred, classes, confusion_matrix_title, cavReportPath,
                    saveName='best_model_cavity')

    if classificationTask == 'fault':
        # Fault classification
        faultReportPath = outputFolder / 'faultReports'
        if not faultReportPath.exists():
            faultReportPath.mkdir()
        faultNames = ['Quench100ms','Quench_3ms', 'E_Quench','Heat Riser Choke', 'Microphonics', 'Controls Fault','Single Cav Turn off']
        classes = [0, 1, 2, 3, 4, 5, 6]

        confusion_matrix_title = 'Fault Classification Best Model'
        saveResults(faultNames, testLabels, testPred, classes, confusion_matrix_title, faultReportPath,
                    saveName='best_model_fault')

    # timing analysis #

    # _, _, _, _ = model.testModel_timing(testSetNew, testCavLabelsNew, testFaultLabelsNew, 1)
