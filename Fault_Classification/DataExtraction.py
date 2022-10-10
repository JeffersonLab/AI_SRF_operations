import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
# from torchsampler import ImbalancedDatasetSampler
from pathlib import Path
from sklearn import preprocessing
import random
import pandas as pd
import os
from rfwtools.config import Config
from rfwtools.example import ExampleType
from rfwtools.example_validator import WindowedExampleValidator
from rfwtools.data_set import DataSet
from rfwtools.extractor.windowing import window_extractor, window_extractor_metadata

if __name__ == '__main__':
    # print('Hello')
    #### RFWTools to generate the dataset #####
    # set file locations #
    Config().label_dir = 'C:\\Monibor\\rfwtools\\All_Data_Extraction/labels' # Directory of the label file
    Config().output_dir = 'C:\\Monibor\\rfwtools\\All_Data_Extraction'  #Output directory
    produce_features = 1
    produce_examples = 1

    cavities = ['1', '2', '3', '4', '5', '6', '7', '8']
    waveforms = ['GMES', 'GASK', 'CRFP', 'DETA2' ]


    signalSet = [c + '_' + s for c in cavities for s in waveforms]

    if produce_features:
        example_set_path = 'C:\\Monibor\\rfwtools\\All_Data_Extraction//example_set_test.csv' #Directory of the example set [create an empty csv file in the following directory]
        ### get windowed datasets ###
        if produce_examples:
            label_files = ['test.txt']  #label file name

            # This tells the DataSet that you will want to work with WindowedExamples
            e_type = ExampleType.WINDOWED_EXAMPLE

            # These parameters will be passed to the Example objects upon construction, e.g., all example will have the same
            # window.
            e_kw = {"start": -1533.4, "n_samples": 7680}  #Input the time (ms) before fault and number of samples
            ev = None  # WindowedExampleValidator()
            ds = DataSet(label_files=label_files, e_type=e_type, example_validator=ev, example_kwargs=e_kw)

            ds.produce_example_set(report=True)
            ds.save_example_set_csv(example_set_path)

        ds = DataSet()
        ds.load_example_set_csv(example_set_path)

        ds.produce_feature_set(window_extractor, windows={'pre-fault': -1533.4}, n_samples=7680,
                               standardize=False, downsample=False, ds_kwargs={'num': 7680, 'axis': 0}, verbose=True,
                               max_workers=4) #input of the start time, number of sample [didn't apply any preprocessing]


        ds.feature_set.update_metadata_columns(window_extractor_metadata)

        ds.feature_set.save_csv('Full_data_test.csv')  #Name of the output csv file





