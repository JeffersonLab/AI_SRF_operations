Step 1: DataExtraction.py
The purpose of this step is to extract raw data using rfwtools (we did not apply any of the preprocessing steps available in rfwtools). The following are changes that need to be made:

line 22: directory of the label file
line 23: output directory
line 34: directory of the example set [create an empty .csv file in the following directory]
line 37: label file name
line 44: input the time (ms) before fault and number of samples
line 54: input of the start time, number of sample [did not apply any preprocessing]
line 61: name of the output .csv file

Step 2: DataFormating.py
The purpose of this step is to format the data. Two types of data (cavity or fault) will be generated based on the input in line 28.

line 28: cavity or fault 
line 61: directory of the dataset
line 66-67: start and end date of the dataset
line 77-79: save the dataset, cavity, and fault information of the data

Step 3: ModelBuilding.py
The purpose of this step is to load cavity/fault signals, apply resampling, perform z-score normalization, split the data, train, and test the model. The saved model can then load, fine-tune and/or test anytime. 

line 34: cavity or fault
line 65: set directory
line 73: name of the output folder
line 75-76: data and label directory
line 154-157: activate (layer freezing, if required)
