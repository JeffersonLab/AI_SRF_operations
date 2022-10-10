import sys
from LSTM_utils import *
import time

def formatData(df, classDict, signalSet, remove_faults, n_samples):
    signalCols = ['Sample_' + str(sample + 1) + '_' + signal for signal in signalSet for sample in range(0, n_samples)]
    data = []
    cav_labels = []
    fault_labels = []
    with tqdm(total=df.shape[0]) as pbar:
        for i in range(0, df.shape[0]):
            if df.loc[i, 'fault_label'] not in remove_faults:
                tempEvent = []
                tempEvent = np.transpose(np.reshape(df.loc[i, signalCols].values, (len(signalSet), n_samples)))
                try:
                    tempEvent.astype('float16', copy=False)
                    data.append(tempEvent)
                    cav_labels.append(int(df.loc[i, 'cavity_label']))
                    fault_labels.append(classDict[df.loc[i, 'fault_label']])
                except:
                    print('Error:', i)
                pbar.update(1)

    return np.stack(data, axis=0).astype('float16'), np.array(cav_labels), np.array(fault_labels)


if __name__ == '__main__':
    classificationTask = 'cavity' #cavity or fault
    if classificationTask == 'cavity':
        numClasses = 9
    elif classificationTask == 'fault':
        numClasses = 7
    else:
        sys.exit('Please provide the correct task!')

    # Load Dataset
    cavities = ['1', '2', '3', '4', '5', '6', '7', '8']
    waveforms = ['GMES', 'GASK', 'CRFP', 'DETA2']

    signalSet = [c + '_' + s for c in cavities for s in waveforms]

    # dictionary of faults considered for analysis. Reconstruct dictionary when adding/removing fault types
    faultDict = {
        'Single Cav Turn off': 6,
        'Microphonics': 4,
        'Quench_100ms': 0,
        'Controls Fault': 5,
        'E_Quench': 2,
        'Quench_3ms': 1,
        'Heat Riser Choke': 3,
        'Heat Riser': 3,
        'Multi Cav turn off': 7,
        'Multi Cav turn Off': 7,
        'Unknown': 8,
    }
    if classificationTask == 'fault':
        remove_faults = ['Unknown', 'Multi Cav turn Off', 'Multi Cav turn off']
    else:
        remove_faults = []

    full_df = pd.read_csv('C:\\Monibor\\rfwtools\All_Data_Extraction\\Full_data_test.csv',
                          comment='#', skip_blank_lines=True) # directory of the dataset
    full_df = full_df[full_df['fault_label'] != 'Unknown'].reset_index(drop=True, inplace=False)
    full_df['datetime'] = pd.to_datetime(full_df['dtime'])
    # print(full_df.shape)
    dataset_mask = (full_df['datetime'] > pd.to_datetime('2021-12-01')) & \
                       (full_df['datetime'] < pd.to_datetime('2021-12-03')) #start and end date of the dataset
    feature_df = full_df.loc[dataset_mask]
    # print('dataset_mask', feature_df.shape)

    feature_df.reset_index(drop=True, inplace=True)

    data, cav_labels, fault_labels = formatData(feature_df, faultDict, signalSet, remove_faults,
                                                n_samples=7680)
    print(data.shape)
    import _pickle as cPickle
    cPickle.dump(data, open("C:\\Monibor\\rfwtools\\All_Data_Extraction/cavity_data.pkl", "wb")) # Dataset directory
    cPickle.dump(cav_labels, open("C:\\Monibor\\rfwtools\\All_Data_Extraction/cav_labels.pkl", "wb")) # cavity information
    cPickle.dump(fault_labels, open("C:\\Monibor\\rfwtools\\All_Data_Extraction/fault_labels.pkl", "wb")) # Fault information

