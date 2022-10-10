import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import torch.optim as optimfcx
import pandas as pd
from LSTM_utils import get_fault
# from ignite.contrib.engines.tbptt import create_supervised_tbptt_trainer
import time
import copy


class ResNet_Block(nn.Module):
    """
    ResNet Block
    """

    def __init__(self, in_channels, out_channels, stride=1, enable_SE=True):

        super(ResNet_Block, self).__init__()

        self.enable_SE = enable_SE
        if self.enable_SE:
            self.SE = SE_Block(out_channels, r=16)
        self.skip = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.skip = None

        self.block = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1,
                      stride=stride, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),

            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1,
                      stride=1, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):

        out = self.block(x)
        if self.enable_SE:
            out = self.SE(out)

        out += (x if self.skip is None else self.skip(x))

        out = self.relu(out)
        return out


class SE_Block(nn.Module):
    """
    Squeeze-and-Excitation (SE) block from https://arxiv.org/abs/1709.01507
    """

    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1)
        return x * y.expand_as(x)


class LSTM_FCN_ResNet_singleTask(nn.Module):
    def __init__(self, NumClassesOut, N_time, N_Features, dimension_shuffle, device, N_LSTM_Out=128,
                 N_LSTM_layers=2, Conv1_NF=128, Conv2_NF=256, Conv3_NF=128, lstmDropP=0.8, FC_DropP=0.1):
        super(LSTM_FCN_ResNet_singleTask, self).__init__()

        self.N_time = N_time
        self.N_Features = N_Features
        self.NumClassesOut = NumClassesOut
        self.dimension_shuffle = dimension_shuffle
        self.device = device
        self.N_LSTM_Out = N_LSTM_Out
        self.N_LSTM_layers = N_LSTM_layers
        self.Conv1_NF = Conv1_NF
        self.Conv2_NF = Conv2_NF
        self.Conv3_NF = Conv3_NF

        if self.dimension_shuffle:
            self.lstm = nn.LSTM(self.N_time, self.N_LSTM_Out, self.N_LSTM_layers, dropout=0.5, batch_first=True)
        else:
            self.lstm = nn.LSTM(self.N_Features, self.N_LSTM_Out, self.N_LSTM_layers, dropout=0.5, batch_first=True)

        # self.C1 = nn.Conv1d(self.N_Features, self.Conv1_NF, 9, padding=4)
        # self.se1 = SE_Block(self.Conv1_NF, r=16)
        # self.C2 = nn.Conv1d(self.Conv1_NF, self.Conv2_NF, 5, padding=2)
        # self.se2 = SE_Block(self.Conv2_NF, r=16)
        # self.C3 = nn.Conv1d(self.Conv2_NF, self.Conv3_NF, 3, padding=1)
        # self.BN1 = nn.BatchNorm1d(self.Conv1_NF)
        # self.BN2 = nn.BatchNorm1d(self.Conv2_NF)
        # self.BN3 = nn.BatchNorm1d(self.Conv3_NF)
        self.C1 = ResNet_Block(self.N_Features, self.Conv1_NF, stride=1, enable_SE=True)
        self.C2 = ResNet_Block(self.Conv1_NF, self.Conv2_NF, stride=1, enable_SE=True)
        self.C3 = ResNet_Block(self.Conv2_NF, self.Conv3_NF, stride=1, enable_SE=True)

        # self.relu = nn.ReLU()
        self.lstmDrop = nn.Dropout(lstmDropP)
        self.ConvDrop = nn.Dropout(FC_DropP)

        self.FC = nn.Linear(self.Conv3_NF + self.N_LSTM_Out * 1, self.NumClassesOut)

        # self.lsm = nn.LogSoftmax(dim=1)

    def init_hidden(self):
        h0 = torch.zeros(self.N_LSTM_layers, self.N_time, self.N_LSTM_Out).to(self.device)
        c0 = torch.zeros(self.N_LSTM_layers, self.N_time, self.N_LSTM_Out).to(self.device)
        return h0, c0

    def forward(self, x):
        # input x should be in size [B,T,F] , where B = Batch size
        #                                         T = Time sampels
        #                                         F = features

        # h0, c0 = self.init_hidden()
        if self.dimension_shuffle:
            x1 = x.transpose(2, 1)
            x1, _ = self.lstm(x1)
        else:
            x1, _ = self.lstm(x)
        x1 = x1[:, -1, :]

        x1 = self.lstmDrop(x1)

        x2 = x.transpose(2, 1)
        x2 = self.ConvDrop(self.C1(x2))
        x2 = self.ConvDrop(self.C2(x2))
        x2 = self.ConvDrop(self.C3(x2))

        x2 = torch.mean(x2, 2)

        x_all = torch.cat((x1, x2), dim=1)
        x_out = self.FC(x_all)

        return x_out


def trainModel(model, train_loader, val_loader,
               num_epochs, optimizer, scheduler, early_stop_patience, loss1, device):
    columnNames = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc']
    train_stats_df = pd.DataFrame(columns=columnNames)

    best_model = copy.deepcopy(model)

    bestValAcc = 0
    last_val_loss = 1e9
    early_stop_count = 0
    ### training and validation ###
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        epochTrainAcc = 0.0

        # training #
        for batchInputs, batchLabels in train_loader:
            batchInputs = batchInputs.to(device)
            batchLabels = batchLabels.to(device)
            # print('batchlabels',batchLabels)

            # print(batchLabels)
            # Forward Pass
            outputs = model.forward(batchInputs)
            # print('outputs',outputs)

            loss = loss1(outputs, batchLabels)
            # print('loss',loss)

            train_loss += loss.item()
            # Backward pass and Optimize
            optimizer.zero_grad()
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)

            epochTrainAcc += 100 * (predicted == batchLabels).sum().item() / batchLabels.size(0)

        epoch_val_loss = 0.0
        epochValAcc = 0.0

        # validation #
        for batchInputs, batchLabels in val_loader:
            batchInputs = batchInputs.to(device)
            batchLabels = batchLabels.to(device)

            with torch.no_grad():
                model.eval()

                outputs = model.forward(batchInputs)

                val_loss = loss1(outputs, batchLabels)

                _, predicted = torch.max(outputs.data, 1)

                epoch_val_loss += val_loss.item()

                epochValAcc += 100 * (predicted == batchLabels).sum().item() / batchLabels.size(0)

        epoch_stat = [epoch + 1, train_loss / len(train_loader),
                      epochTrainAcc / len(train_loader),
                      epoch_val_loss / len(val_loader),
                      epochValAcc / len(val_loader)]
        train_stats_df.loc[epoch] = epoch_stat

        print('Epoch [{}/{}], Train Loss: {:.4f}, Train Accuracy: {}, '
              'Validation Accuracy: {}'.format(
            epoch + 1, num_epochs, train_loss / len(train_loader),
            epochTrainAcc / len(train_loader),
            epochValAcc / len(val_loader)))

        scheduler.step(epoch_val_loss / len(val_loader))

        if epochValAcc / len(val_loader) > bestValAcc:
            best_model = copy.deepcopy(model)
            bestValAcc = epochValAcc / len(val_loader)

        # early stopping
        if epoch_val_loss / len(val_loader) > last_val_loss:
            early_stop_count += 1
            print('early stop count: ', early_stop_count)

            if early_stop_count >= early_stop_patience:
                print('Early stopping!')
                break
        else:
            early_stop_count = 0
            last_val_loss = epoch_val_loss / len(val_loader)

    return train_stats_df, best_model


def testModel(model, test_loader, device):
    # Initialize metrics #
    testAcc = 0.0
    testLabels = np.array([])
    testPred = np.array([])

    for batchInputs, batchLabels in test_loader:
        testLabels = np.concatenate([testLabels, batchLabels.data.numpy()])
        batchInputs = batchInputs.to(device)
        batchLabels = batchLabels.to(device)

        # Run model without gradient computing in eval mode for testing #
        with torch.no_grad():
            model.eval()
            # start1 = time.time()
            # print('start1', start1)

            outputs = model.forward(batchInputs)
            examples=len(outputs.data)
            # for ex in range(examples):
            #     p = outputs.data[ex].cpu().numpy()
            #     # print(p)
            #     print(np.exp(p)/np.sum(np.exp(p)))
                # pred=np.exp(p) / np.sum(np.exp(p))
                # print(torch.max)
            # p=outputs.data[0].cpu().numpy()
            # output.cpu().data.numpy()
            # print(p)
            # print(outputs.data[0])
            # print(outputs.data)
            # Check predictions against GT, and compute accuracies #
            _, predicted = torch.max(outputs.data, 1)
            # print(predicted)
            # print(batchLabels)
            # end1 = time.time()
            # print('end1', end1)
            # print(f"Runtime of the program is {end1 - start1}")
            testPred = np.concatenate([testPred, predicted.cpu().data.numpy()])

            testAcc += 100 * (predicted == batchLabels).sum().item() / batchLabels.size(0)

    return testPred, testAcc / len(test_loader)


def testModel_MCDropout_V2(model, test_loader, faultDict, classificationTask, device, MC_sample_size):
    # Initialize metrics #
    # cavTop1Count, faultTop1Count = 0, 0
    # cavTop3Count, faultTop3Count = 0, 0
    # testCavPred, testFaultPred = [], []
    top1Count, top3Count = 0, 0
    testPred = []

    if classificationTask == 'cavity':
        columnNames = ['timestamp', 'zone', 'cavity_label', 'cavity_choice_1', 'cavity_choice_1_mean',
                       'cavity_choice_1_variance',
                       'cavity_choice_2', 'cavity_choice_2_mean', 'cavity_choice_2_variance',
                       'cavity_choice_3', 'cavity_choice_3_mean', 'cavity_choice_3_variance',
                       'cavity_mean_variance', 'cavity_entropy']
    else:
        columnNames = ['timestamp', 'zone', 'fault_label', 'fault_choice_1', 'fault_choice_1_mean',
                       'fault_choice_1_variance', 'fault_choice_2', 'fault_choice_2_mean', 'fault_choice_2_variance',
                       'fault_choice_3', 'fault_choice_3_mean', 'fault_choice_3_variance',
                       'fault_mean_variance', 'fault_entropy']

    output_df = pd.DataFrame(columns=columnNames)

    # For each example in the testing set, run MC sampling with dropout
    for input, label in test_loader:
        # example =
        output, outVar, outMeanVar, outEntropy = MonteCarloTest(model, input, device, num_samples=MC_sample_size)

        values, indices = torch.topk(output, 3)
        testPred.append(indices[0])
        if indices[0] == label:
            top1Count += 1
            top3Count += 1
        elif label in indices:
            top3Count += 1

        # faultValues, faultIndices = torch.topk(faultOutput, 3)
        # testFaultPred.append(faultIndices[0])
        # if faultIndices[0] == faultDict[test_df['fault-label'][i]]:
        #     faultTop1Count += 1
        #     faultTop3Count += 1
        # elif faultDict[test_df['fault-label'][i]] in faultIndices:
        #     faultTop3Count += 1
        if classificationTask == 'cavity':
            output_entry = ['not available', 'not available',
                            'cavity ' + str(label.item()),
                            'cavity ' + str(indices[0].item()), values[0].item(), outVar[indices[0]].item(),
                            'cavity ' + str(indices[1].item()), values[1].item(), outVar[indices[1]].item(),
                            'cavity ' + str(indices[2].item()), values[2].item(), outVar[indices[2]].item(),
                            outMeanVar.item(), outEntropy.item()]
        else:
            output_entry = ['not available', 'not available',
                            get_fault(label.item(), faultDict),
                            get_fault(indices[0], faultDict), values[0].item(),
                            outVar[indices[0]].item(),
                            get_fault(indices[1], faultDict), values[1].item(),
                            outVar[indices[1]].item(),
                            get_fault(indices[2], faultDict), values[2].item(),
                            outVar[indices[2]].item(),
                            outMeanVar.item(), outEntropy.item()]

        output_df.loc[len(output_df)] = output_entry

    top1acc = 100 * top1Count / len(test_loader)
    top3acc = 100 * top3Count / len(test_loader)
    # faultTop1Acc = 100 * faultTop1Count / test_df.shape[0]
    # faultTop3Acc = 100 * faultTop3Count / test_df.shape[0]

    # print('CavityID Test Accuracy Top1 and top3: {} and {}, Fault ID Test Accuracy top1 and top3: {} and {}'.format(
    #     cavTop1Acc,
    #     cavTop3Acc, faultTop1Acc, faultTop3Acc))
    if classificationTask == 'cavity':
        accuracy_df = pd.DataFrame({'cavity top1 Accuracy': ['{:.2f}'.format(top1acc)],
                                    'cavity top3 Accuracy': ['{:.2f}'.format(top3acc)]})
    else:
        accuracy_df = pd.DataFrame({'cavity top1 Accuracy': ['{:.2f}'.format(top1acc)],
                                    'cavity top3 Accuracy': ['{:.2f}'.format(top3acc)]})

    return testPred, top1acc, accuracy_df, output_df


def apply_dropout(m):
    for module in m.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()


def MonteCarloTest(model, X, device, num_samples):
    # Replicate the example based on the number of MC samples
    # print(X.size())
    # input = X.unsqueeze(0).repeat(num_samples, 1, 1).to(self.device)
    input = X.repeat(num_samples, 1, 1).to(device)
    # print(input.size())
    # m = nn.LogSoftmax(dim=1)
    m = nn.Softmax(dim=1)
    # Set model to train mode to enable random dropout
    model.eval()
    apply_dropout(model)
    with torch.no_grad():  # Run model without gradient calculations for speed
        outputs = m(model.forward(input))
        # print(faultOutputs)
        # Compute cavity classification uncertainty stats
        output = torch.mean(outputs, dim=0)
        outVar = torch.var(outputs, dim=0)
        outMeanVar = torch.mean(outVar)
        outEntropy = Categorical(probs=output).entropy()

        # Compute fault classification uncertainty stats
        # faultOutput = torch.mean(torch.exp(faultOutputs), dim=0)
        # faultVar = torch.var(torch.exp(faultOutputs), dim=0)
        # faultMeanVar = torch.mean(faultVar)
        # faultEntropy = Categorical(probs=faultOutput).entropy()

    return output.cpu(), outVar.cpu(), outMeanVar.cpu(), outEntropy.cpu()


def testModel_timing(self, testSet, testCavLabels, testFaultLabels, batch_size=1):
    # Split testing data into batches #
    testBatches = torch.split(testSet, batch_size, dim=0)
    testCavLabelBatches = torch.split(testCavLabels, batch_size, dim=0)
    testFaultLabelBatches = torch.split(testFaultLabels, batch_size, dim=0)

    # Initialize metrics #
    testCavAcc, testFaultAcc = 0.0, 0.0
    testCavLabels, testFaultLabels = np.array([]), np.array([])
    testCavPred, testFaultPred = np.array([]), np.array([])

    for batch in range(0, len(testBatches)):
        # start = time.process_time()
        batchInputs = testBatches[batch].to(self.device)
        batchCavLabels = testCavLabelBatches[batch].to(self.device)
        testCavLabels = np.concatenate([testCavLabels, testCavLabelBatches[batch].data.numpy()])
        batchFaultLabels = testFaultLabelBatches[batch].to(self.device)
        testFaultLabels = np.concatenate([testFaultLabels, testFaultLabelBatches[batch].data.numpy()])

        # Run model without gradient computing in eval mode for testing #
        with torch.no_grad():
            self.eval()

            cavOutputs, faultOutputs = self.forward(batchInputs)

            # print(time.process_time() - start)
            # Check predictions against GT, and compute accuracies #
            _, cavPredicted = torch.max(cavOutputs.data, 1)
            _, faultPredicted = torch.max(faultOutputs.data, 1)

            testCavPred = np.concatenate([testCavPred, cavPredicted.cpu().data.numpy()])
            testFaultPred = np.concatenate([testFaultPred, faultPredicted.cpu().data.numpy()])

            testCavAcc += 100 * (cavPredicted == batchCavLabels).sum().item() / batchCavLabels.size(0)
            testFaultAcc += 100 * (faultPredicted == batchFaultLabels).sum().item() / batchFaultLabels.size(0)

    return testCavPred, testFaultPred, testCavAcc / len(testBatches), testFaultAcc / len(testBatches)
