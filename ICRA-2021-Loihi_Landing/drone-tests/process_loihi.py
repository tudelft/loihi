import csv
import seaborn
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

DIM_OUTPUT_LAYER = 5
DIM_HIDDEN_LAYER = 10
PRINT = False

def convSpikesList2IndicesEvents(spikes, dim):
    idxEvents = list()
    for i in range(len(spikes)):
        l = list()
        for j in range(dim):
            if spikes[i][j] == 1.0 or spikes[i][j] == 1:
                l.append(j)
        if len(l) > 0:
            idxEvents.append(l)
        else:
            l.append(-1)
            idxEvents.append(l)
    return idxEvents

def spikeSequenceMatching(s1, s2):
    s1 = np.array(s1)
    s2 = np.array(s2)
    s = abs(s1 - s2)
    return 100 * (1 - s.sum() / (len(s1) * len(s1[0])))

def getInfill(s):
    return 100 * s.sum() / (len(s) * len(s[0]))

def getNbrofSpikes(s):
    return s.sum()

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

if __name__ == "__main__":

    InfillLoihi         = list()
    SpikesCountLoihi    = list()
    DivergenceLoihi     = list()
    ThrustLoihi         = list()
    IdxOutputLoihi      = list()

    cpt = 0

    for idx in range(20):
        if idx != 5:
            with open("loihi_logs/output_spikes_rec_file_"+str(idx)+".txt", "r") as f:
                data_loihi = list()
                for line in f:
                    D = list()
                    a = line.split(',')
                    for i in range(DIM_OUTPUT_LAYER):
                        D.append((int)(a[i]))
                    data_loihi.append(D)
                data_loihi = np.array(data_loihi).T

            with open("loihi_logs/thrust_rec_file_"+str(idx)+".txt", "r") as f:
                thrust_loihi = list()
                for line in f:
                    thrust_loihi.append((float)(line))
                ThrustLoihi.append(thrust_loihi)

            with open("loihi_logs/input_divergence_rec_file_"+str(idx)+".txt", "r") as f:
                divergence = list()
                for line in f:
                    divergence.append((float)(line))
                divergence_loihi = np.array(divergence)
                DivergenceLoihi.append(divergence_loihi)

            InfillLoihi.append(getInfill(data_loihi))
            SpikesCountLoihi.append(getNbrofSpikes(data_loihi))
            IdxOutputLoihi.append(convSpikesList2IndicesEvents(data_loihi.T, DIM_OUTPUT_LAYER))

            if PRINT:
                print("\tLoihi output infill:\t\t" + "{:.2f}".format(InfillLoihi[cpt])  + "%\n")

            cpt = cpt + 1

    print("------------------------------------------")
    print("Output infill\t Loihi")
    print("------------------------------------------")
    print("Average          " + "{:.2f}".format(np.mean(InfillLoihi))  + "%")
    print("Median           " + "{:.2f}".format(np.median(InfillLoihi))  + "%")
    print("Variance          " + "{:.2f}".format(np.var(InfillLoihi))  + "%")
    print("Standard dev.     " + "{:.2f}".format(np.std(InfillLoihi))  + "%")
    print("Max              " + "{:.2f}".format(np.amax(InfillLoihi))  + "%")
    print("Min              " + "{:.2f}".format(np.amin(InfillLoihi))  + "%")
    print("------------------------------------------\n")


    # Output infill looks Gaussian. Let's prove that distributions are normal by using the Kolmogorov-Smirnov (KS) test: 
    _, p_val_loihi = scipy.stats.kstest((np.array(InfillLoihi) - np.mean(InfillLoihi)) / np.std(InfillLoihi), 'norm')
    print("P-value for the Kolmogorov-Smirnov normality test")
    print("        [Loihi] " + "{:.2f}".format(p_val_loihi) + "\n")



