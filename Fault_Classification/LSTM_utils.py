import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from pathlib import Path
from sklearn import preprocessing
from sklearn.utils import shuffle
import random
import pandas as pd
import os
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(cm, target_names, title, savePath, rotAngle, cmap=None, normalize=False):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=rotAngle)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.tight_layout()
    plt.show()
    fig.savefig(savePath)
    plt.close()


def data_split(examples, cavLabels, faultLabels, train_frac, random_state=0):
    ''' https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    param data:       Data to be split
    param train_frac: Ratio of train set to whole dataset

    Randomly split dataset, based on these ratios:
        'train': train_frac
        'valid': (1-train_frac) / 2
        'test':  (1-train_frac) / 2

    Eg: passing train_frac=0.8 gives a 80% / 10% / 10% split
    '''

    assert train_frac >= 0 and train_frac <= 1, "Invalid training set fraction"

    X_train, X_tmp, cav_train, cav_tmp, fault_train, fault_tmp = train_test_split(
        examples, cavLabels, faultLabels, train_size=train_frac, random_state=random_state, stratify=faultLabels)

    X_val, X_test, cav_val, cav_test, fault_val, fault_test = train_test_split(
        X_tmp, cav_tmp, fault_tmp, train_size=0.5, random_state=random_state, stratify=fault_tmp)

    return X_train, X_val, X_test, cav_train, cav_val, cav_test, fault_train, fault_val, fault_test


def data_split_singleTask(examples, labels, train_frac, random_state=0):
    ''' https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    param data:       Data to be split
    param train_frac: Ratio of train set to whole dataset

    Randomly split dataset, based on these ratios:
        'train': train_frac
        'valid': (1-train_frac) / 2
        'test':  (1-train_frac) / 2

    Eg: passing train_frac=0.8 gives a 80% / 10% / 10% split
    '''

    assert train_frac >= 0 and train_frac <= 1, "Invalid training set fraction"

    X_train, X_tmp, label_train, label_tmp, = train_test_split(
        examples, labels, train_size=train_frac, random_state=random_state, stratify=labels)

    X_val, X_test, label_val, label_test, = train_test_split(
        X_tmp, label_tmp, train_size=0.5, random_state=random_state, stratify=label_tmp)

    return X_train, X_val, X_test, label_train, label_val, label_test


def data_split_binary(examples, Labels, train_frac, random_state=0):
    ''' https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    param data:       Data to be split
    param train_frac: Ratio of train set to whole dataset

    Randomly split dataset, based on these ratios:
        'train': train_frac
        'valid': (1-train_frac) / 2
        'test':  (1-train_frac) / 2

    Eg: passing train_frac=0.8 gives a 80% / 10% / 10% split
    '''

    assert train_frac >= 0 and train_frac <= 1, "Invalid training set fraction"

    X_train, X_tmp, labels_train, labels_tmp = train_test_split(
        examples, Labels, train_size=train_frac, random_state=random_state, stratify=Labels)

    X_val, X_test, labels_val, labels_test = train_test_split(
        X_tmp, labels_tmp, train_size=0.5, random_state=random_state, stratify=labels_tmp)

    return X_train, X_val, X_test, labels_train, labels_val, labels_test


def load_data(filePath, faultDict, remove_faults):
    df = pd.read_pickle(filePath)
    if remove_faults:
        for fault in remove_faults:
            df.drop(df[df['fault-label'] == fault].index, inplace=True)
        df.reset_index(drop=True, inplace=True)

    samples = []
    cryomodules = []
    cav_labels = []
    fault_labels = []

    for i in range(df.shape[0]):
        samples.append(df['signals'][i])
        cav_labels.append(int(df['cavity-label'][i]))
        fault_labels.append(faultDict[df['fault-label'][i]])
        cryomodules.append(df['zone'][i])

    # print(np.shape(samples))
    return np.stack(samples, axis=0), np.array(cav_labels), np.array(fault_labels), cryomodules


def formatData(df, classDict, signalSet, remove_faults, n_samples):
    # cavities = ['1', '2', '3', '4', '5', '6', '7', '8']
    # signals = ['GMES', 'CRFP', 'DETA2', 'GASK']
    # signalSet = [c + '_' + s for c in cavities for s in signals]
    signalCols = ['Sample_' + str(sample + 1) + '_' + signal for signal in signalSet for sample in range(0, n_samples)]
    # signalCols = [str(i) for i in range(len(signalSet) * n_samples)]
    data = []
    cav_labels = []
    fault_labels = []
    with tqdm(total=df.shape[0]) as pbar:
        for i in range(0, df.shape[0]):
            if df.loc[i, 'fault_label'] not in remove_faults:
                tempEvent = []
                # for cavity in cavities:
                #     for signal in signals:
                #         #colnames = ['Sample_' + str(sample + 1) + '_' + cavity + '_' + signal for sample in range(0, n_samples)]
                #         tempEvent.append(np.expand_dims(df.loc[i, colnames].values, axis=1))
                # for signal in signalSet:
                #     colnames = ['Sample_' + str(sample + 1) + '_' + signal for sample in range(0, n_samples)]
                #     tempEvent.append(np.expand_dims(df.loc[i, colnames].values, axis=1))
                tempEvent = np.transpose(np.reshape(df.loc[i, signalCols].values, (len(signalSet), n_samples)))

                data.append(tempEvent)
                # labels.append(classDict[df.loc[i, 'window_label']])
                cav_labels.append(int(df.loc[i, 'cavity_label']))
                fault_labels.append(classDict[df.loc[i, 'fault_label']])
                pbar.update(1)
        # if df.loc[i, 'window_label'] == 'stable':
        #     labels.append(0)
        # elif df.loc[i, 'window_label'] == 'impending':
        #     labels.append(1)
        # else:
        #     sys.exit('Check labels!')

    return np.stack(data, axis=0).astype('float'), np.array(cav_labels), np.array(fault_labels)


def formatData_faulty_cavity(df, classDict, n_samples=500):
    # cavities = ['1', '2', '3', '4', '5', '6', '7', '8']
    df.drop(df[df['cavity_label'] == 0].index, inplace=True)
    df.reset_index(drop=True, inplace=True)
    signals = ['GMES', 'CRFP', 'DETA2', 'GASK']
    # signalSet = [c + '_' + s for c in cavities for s in signals]
    # signalCols = ['Sample_' + str(sample + 1) + '_' + signal for signal in signalSet for sample in range(0, n_samples)]
    data = []
    labels = []
    with tqdm(total=df.shape[0]) as pbar:
        for i in range(0, df.shape[0]):
            cavity = df.loc[i, 'cavity_label']
            signalCols = ['Sample_' + str(sample + 1) + '_' + str(cavity) + '_' + signal for signal in signals for
                          sample in
                          range(0, n_samples)]
            tempEvent = []
            # for cavity in cavities:
            #     for signal in signals:
            #         #colnames = ['Sample_' + str(sample + 1) + '_' + cavity + '_' + signal for sample in range(0, n_samples)]
            #         tempEvent.append(np.expand_dims(df.loc[i, colnames].values, axis=1))
            # for signal in signalSet:
            #     colnames = ['Sample_' + str(sample + 1) + '_' + signal for sample in range(0, n_samples)]
            #     tempEvent.append(np.expand_dims(df.loc[i, colnames].values, axis=1))
            tempEvent = np.transpose(np.reshape(df.loc[i, signalCols].values, (len(signals), n_samples)))

            data.append(tempEvent)
            labels.append(int(classDict[df.loc[i, 'window_label']]))
            pbar.update(1)
        # if df.loc[i, 'window_label'] == 'stable':
        #     labels.append(0)
        # elif df.loc[i, 'window_label'] == 'impending':
        #     labels.append(1)
        # else:
        #     sys.exit('Check labels!')

    return np.stack(data, axis=0).astype('float'), np.array(labels)


def get_fault(faultClass, faultDict):
    for key, value in faultDict.items():
        # print([key, value, faultClass])
        if faultClass == value:
            return key


def saveResults(classNames, labels, predictions, classes, confusion_title, savePath, saveName):
    report = classification_report(labels, predictions, labels=classes, target_names=classNames, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print(report_df)
    report_df.to_csv(savePath / (saveName + '_classification_report.csv'))

    cm = confusion_matrix(labels, predictions, labels=classes)
    # print(cm)
    plot_confusion_matrix(cm, normalize=0, target_names=classNames,
                          title=confusion_title,
                          savePath=savePath / (saveName + '_confusion_matrix.jpg'), rotAngle=90)


def saveClassStatsBinary(df, outputPath):
    # TP_df = df.loc[df['cavity_label'] == df['cavity_choice_1']]
    TP_df = df.loc[(df['window_label'] == 'impending') & (df['prediction'] == 'impending')]
    plotFrequencyCounts(TP_df, columnName='fault_label', plot_title='True Positive Fault Representation',
                        xlabel='Fault classes', ylabel='Number of examples',
                        savePath=outputPath / 'TP_frequencyPlot.jpg')

    TN_df = df.loc[(df['window_label'] == 'stable') & (df['prediction'] == 'stable')]
    plotFrequencyCounts(TN_df, columnName='fault_label', plot_title='True Negative Fault Representation',
                        xlabel='Fault classes', ylabel='Number of examples',
                        savePath=outputPath / 'TN_frequencyPlot.jpg')

    FP_df = df.loc[(df['window_label'] == 'stable') & (df['prediction'] == 'impending')]
    plotFrequencyCounts(FP_df, columnName='fault_label', plot_title='False Positive Fault Representation',
                        xlabel='Fault classes', ylabel='Number of examples',
                        savePath=outputPath / 'FP_frequencyPlot.jpg')

    FN_df = df.loc[(df['window_label'] == 'impending') & (df['prediction'] == 'stable')]
    plotFrequencyCounts(FN_df, columnName='fault_label', plot_title='False Negative Fault Representation',
                        xlabel='Fault classes', ylabel='Number of examples',
                        savePath=outputPath / 'FN_frequencyPlot.jpg')

    # frequency counts


def plotFrequencyCounts(df, columnName, plot_title, xlabel, ylabel, savePath):
    fig1 = df[columnName].value_counts().plot(kind='bar', figsize=(14, 7), rot=90, fontsize=14)
    fig1.set_title(plot_title)
    fig1.set_ylabel(ylabel)
    fig1.set_xlabel(xlabel)

    for i in fig1.patches:
        fig1.text(i.get_x() + 0.06, i.get_height() + 1.5, str(i.get_height()), fontsize=14)

    plt.tight_layout()
    plt.savefig(savePath)
    plt.close()


def saveUQplots_binary(df, faults, outputPath):
    correct_df = df.loc[df['window_label'] == df['prediction']]

    incorrect_df = df.loc[df['window_label'] != df['prediction']]

    violin_plots_UQ(correct_df, incorrect_df, 'All Data UQ Plots', outputPath, saveName='fullData_violinPlots.jpg')

    for fault in faults:
        temp_correct_df = correct_df.loc[correct_df['fault_label'] == fault]
        temp_incorrect_df = incorrect_df.loc[incorrect_df['fault_label'] == fault]

        violin_plots_UQ(temp_correct_df, temp_incorrect_df, fault + ' UQ Plots', outputPath,
                        saveName=fault + '_violinPlots.jpg')

    ##### fault threshold plots ######
    xAxis = np.linspace(0.0, 1.0, num=100)
    correctAgg = []
    incorrectAgg = []
    print(xAxis)
    for i in range(0, 100):
        tempCorrect = correct_df.loc[correct_df['prediction_mean'] >= xAxis[i]]
        correctAgg.append((tempCorrect.shape[0] / correct_df.shape[0]) * 100)

        tempIncorrect = incorrect_df.loc[incorrect_df['prediction_mean'] >= xAxis[i]]
        incorrectAgg.append((tempIncorrect.shape[0] / incorrect_df.shape[0]) * 100)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.yaxis.set_ticks_position('right')
    print(correctAgg)
    plt.plot(xAxis, np.array(incorrectAgg))
    plt.plot(xAxis, np.array(correctAgg))
    plt.xlabel('Threshold Confidence')
    plt.ylabel('Events Above Threshold (%)')
    plt.legend(['disagree', 'agree'])
    plt.title('Binary Classification: CQ-Mean')
    # plt.show()
    plt.savefig(outputPath / 'CQ-Mean.jpg', format='jpg')
    plt.close(fig)

    # print(correct_df.head())
    varianceMax = df['prediction_mean_variance'].max()
    print(varianceMax)

    xAxis = np.linspace(0.0, varianceMax, num=100)
    correctAgg = []
    incorrectAgg = []
    print(xAxis)
    for i in range(0, 100):
        tempCorrect = correct_df.loc[correct_df['prediction_mean_variance'] <= xAxis[i]]
        correctAgg.append((tempCorrect.shape[0] / correct_df.shape[0]) * 100)

        tempIncorrect = incorrect_df.loc[incorrect_df['prediction_mean_variance'] <= xAxis[i]]
        incorrectAgg.append((tempIncorrect.shape[0] / incorrect_df.shape[0]) * 100)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.yaxis.set_ticks_position('right')
    print(correctAgg)
    print(incorrectAgg)
    plt.plot(xAxis, np.array(incorrectAgg))
    plt.plot(xAxis, np.array(correctAgg))
    plt.xlabel('Uncertainty Threshold')
    plt.ylabel('Events Below Threshold (%)')
    plt.legend(['disagree', 'agree'])
    plt.title('Binary Classification: UQ-Variance')
    # plt.show()
    plt.savefig(outputPath / 'UQ-Variance.jpg', format='jpg')
    plt.close(fig)

    entropyMax = df['prediction_entropy'].max()
    print(entropyMax)

    xAxis = np.linspace(0.0, entropyMax, num=100)
    correctAgg = []
    incorrectAgg = []
    print(xAxis)
    for i in range(0, 100):
        tempCorrect = correct_df.loc[correct_df['prediction_entropy'] <= xAxis[i]]
        correctAgg.append((tempCorrect.shape[0] / correct_df.shape[0]) * 100)

        tempIncorrect = incorrect_df.loc[incorrect_df['prediction_entropy'] <= xAxis[i]]
        incorrectAgg.append((tempIncorrect.shape[0] / incorrect_df.shape[0]) * 100)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.yaxis.set_ticks_position('right')
    print(correctAgg)
    plt.plot(xAxis, np.array(incorrectAgg))
    plt.plot(xAxis, np.array(correctAgg))
    plt.xlabel('Uncertainty Threshold')
    plt.ylabel('Events Below Threshold (%)')
    plt.legend(['disagree', 'agree'])
    plt.title('Binary Classification: UQ-Predictive Entropy')
    # plt.show()
    plt.savefig(outputPath / 'UQ-Predictive Entropy.jpg', format='jpg')
    plt.close(fig)


def violin_plots_UQ(correct_df, incorrect_df, figure_title, outputPath, saveName):
    data1 = pd.concat([incorrect_df['prediction_mean'], correct_df['prediction_mean']], axis=1,
                      ignore_index=True)
    data1.columns = ['Incorrect Classification', 'Correct Classification']
    data1 = data1.assign(Location='mean')

    data2 = pd.concat([incorrect_df['prediction_mean_variance'], correct_df['prediction_mean_variance']], axis=1,
                      ignore_index=True)
    data2.columns = ['Incorrect Classification', 'Correct Classification']
    data2 = data2.assign(Location='variance')
    data3 = pd.concat([incorrect_df['prediction_entropy'], correct_df['prediction_entropy']], axis=1,
                      ignore_index=True)
    data3.columns = ['Incorrect Classification', 'Correct Classification']
    data3 = data3.assign(Location='entropy')

    fig = plt.figure()
    plt.subplot(3, 1, 1)
    # data1.boxplot(column=['Incorrect Classification', 'Correct Classification'])
    # data1.violinplot(column=['Incorrect Classification', 'Correct Classification'])
    ax = sns.violinplot(data=data1, orient='h', palette=['y', 'g'])
    plt.title('Mean')
    # plt.tight_layout()
    # fig.savefig(mainFolder / 'cavViolinplot_mean.jpg', format='jpg')
    # plt.close(fig)

    # fig = plt.figure()
    plt.subplot(3, 1, 2)
    # cavBoxplot2 = data2.boxplot(column=['Incorrect Classification', 'Correct Classification'])
    ax = sns.violinplot(data=data2, orient='h', palette=['y', 'g'])
    plt.title('Variance')
    # plt.tight_layout()
    # plt.savefig(mainFolder / 'cavViolinplot_variance.jpg')
    # plt.close(fig)

    # fig = plt.figure()
    plt.subplot(3, 1, 3)
    # cavBoxplot3 = data3.boxplot(column=['Incorrect Classification', 'Correct Classification'])
    ax = sns.violinplot(data=data3, orient='h', palette=['y', 'g'])
    plt.title('Entropy')
    # plt.tight_layout()

    fig.suptitle(figure_title)
    plt.tight_layout()
    plt.savefig(outputPath / saveName)
    plt.close(fig)
