import csv
import seaborn
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

DIM_OUTPUT_LAYER = 5
DIM_HIDDEN_LAYER = 10
DELAY = 2

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
    
    N = 67

    with open("loihi_logs/output_spikes_rec_file_"+str(N)+".txt", "r") as f:
        count = 0
        data_loihi = list()
        for line in f:
            D = list()
            a = line.split(',')
            for i in range(DIM_OUTPUT_LAYER):
                D.append((int)(a[i]))
            if count > DELAY-1:
                data_loihi.append(D)
            count = count + 1
        data_loihi = np.array(data_loihi).T

    with open("loihi_logs/hidden_spikes_rec_file_"+str(N)+".txt", "r") as f:
        count = 0
        data_hid_loihi = list()
        for line in f:
            D = list()
            a = line.split(',')
            for i in range(DIM_HIDDEN_LAYER):
                D.append((int)(a[i]))
            if count > DELAY-1:
                data_hid_loihi.append(D)
            count = count + 1
        data_hid_loihi = np.array(data_hid_loihi).T
        
    with open("optitrack_paparazzi_logs/run_raw"+str(N)+".csv", newline='') as f:
        reader = csv.reader(f)
        dataList2 = list(reader)
        data = np.array(dataList2[1:]).T
        divergence_sim = data[2][:].astype(np.float)
        divergence_sim = divergence_sim[0:len(divergence_sim)-DELAY]
        thrust_sim = data[3][:].astype(np.float)
        thrust_sim = thrust_sim[0:len(thrust_sim)-DELAY]

    with open("loihi_logs/thrust_rec_file_"+str(N)+".txt", "r") as f:
        thrust_loihi = list()
        for line in f:
            thrust_loihi.append((float)(line))
        t_loihi = np.array(thrust_loihi)
        ThrustLoihi = t_loihi[DELAY:]

    IdxOutputLoihi = convSpikesList2IndicesEvents(data_loihi.T, DIM_OUTPUT_LAYER)
    IdxHiddenLoihi = convSpikesList2IndicesEvents(data_hid_loihi.T, DIM_HIDDEN_LAYER)

    Time = np.arange(0,len(ThrustLoihi)/25,1/25)
    
    plt.subplot(5,1,1)
    plt.plot(Time,divergence_sim,'steelblue')
    plt.grid(True)
    plt.ylabel("Divergence error [1/s]")
    plt.xlabel("Time [s]")

    plt.subplot(5,1,4)
    plt.plot(Time,ThrustLoihi,'steelblue')
    plt.grid(True)
    plt.ylabel("Thrust setpoint [g]")
    plt.xlabel("Time [s]")

    plt.subplot(5,1,5)
    plt.plot(thrust_sim - ThrustLoihi,'r')
    plt.grid(True)
    plt.ylabel("Thrust error [g]")
    plt.xlabel("Time [s]")

    plt.subplot(5,1,2)
    for i in range(len(divergence_sim)):
        plt.plot(Time[i]*np.ones(len(IdxHiddenLoihi[i])), IdxHiddenLoihi[i][:], 'k.')
    plt.grid(True)
    plt.ylim(-0.5, 9.5)
    plt.ylabel("Hidden neurons")
    plt.xlabel("Time [s]")

    plt.subplot(5,1,3)
    for i in range(len(divergence_sim)):
        plt.plot(Time[i]*np.ones(len(IdxOutputLoihi[i])), IdxOutputLoihi[i][:], 'k.')
    plt.grid(True)
    plt.ylim(-0.5, 4.5)
    plt.ylabel("Output neurons")
    plt.xlabel("Time [s]")

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.35, hspace=0.25)
    fig = plt.gcf()
    fig.set_size_inches(12.0,18.0)
    plt.savefig("fig_6.svg")
    plt.clf()

