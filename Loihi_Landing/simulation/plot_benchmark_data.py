import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

DIM_OUTPUT_LAYER = 5

def convSpikesList2IndicesEvents(spikes):
    idxEvents = list()
    for i in range(len(spikes)):
        l = list()
        for j in range(DIM_OUTPUT_LAYER):
            if spikes[i][j] == 1.0 or spikes[i][j] == 1:
                l.append(j)
        if len(l) > 0:
            idxEvents.append(l)
        else:
            l.append(-1)
            idxEvents.append(l)
    return idxEvents

for idx in range(100):
    with open("optitrack_paparazzi_logs/run_raw_out_spikes"+str(idx)+".csv", newline='') as f:
        reader = csv.reader(f)
        dataList = list(reader)
        data_sim = np.array(dataList[1:]).T
        data_sim = data_sim[1:].astype(np.float)

    with open("loihi_logs/output_spikes_rec_file_"+str(idx)+".txt", "r") as f:
        data_loihi = list()
        for line in f:
            D = list()
            a = line.split(',')
            for i in range(DIM_OUTPUT_LAYER):
                D.append((int)(a[i]))
            data_loihi.append(D)
        data_loihi = np.array(data_loihi).T

    with open("optitrack_paparazzi_logs/run_raw"+str(idx)+".csv", newline='') as f:
        reader = csv.reader(f)
        dataList2 = list(reader)
        data = np.array(dataList2[1:]).T
        divergence_sim = data[2][:].astype(np.float)
        thrust_sim = data[3][:].astype(np.float)

    with open("loihi_logs/thrust_rec_file_"+str(idx)+".txt", "r") as f:
        thrust_loihi = list()
        for line in f:
            thrust_loihi.append((float)(line))

    with open("loihi_logs/input_divergence_rec_file_"+str(idx)+".txt", "r") as f:
        divergence = list()
        for line in f:
            divergence.append((float)(line))
        divergence_loihi = np.array(divergence)

    idxEventsSim = convSpikesList2IndicesEvents(data_sim.T)
    idxEventsLoihi = convSpikesList2IndicesEvents(data_loihi.T)

    plt.subplot(2,3,1)
    plt.plot(divergence_sim)
    plt.plot(range(len(divergence_sim)), np.zeros(len(divergence_sim)))
    plt.ylim(-3, 3)
    plt.title("Expected input divergence")

    plt.subplot(2,3,4)
    plt.plot(divergence_loihi)
    plt.plot(range(len(divergence_loihi)), np.zeros(len(divergence_loihi)))
    plt.ylim(-3, 3)
    plt.title("Loihi input divergence")

    plt.subplot(2,3,2)
    for i in range(len(idxEventsSim)):
        plt.plot(i*np.ones(len(idxEventsSim[i][:])), idxEventsSim[i][:], 'k.')
    plt.ylim(-0.5, DIM_OUTPUT_LAYER - 0.5)
    plt.title("Expected output spikes")

    plt.subplot(2,3,5)
    for i in range(len(idxEventsLoihi)):
        plt.plot(i*np.ones(len(idxEventsLoihi[i][:])), idxEventsLoihi[i][:], 'k.')
    plt.ylim(-0.5, DIM_OUTPUT_LAYER - 0.5)
    plt.title("Loihi output spikes")

    plt.subplot(2,3,3)
    plt.plot(thrust_sim)
    plt.ylim(-0.5,0.5)
    plt.title("Expected output thrust")

    plt.subplot(2,3,6)
    plt.plot(thrust_loihi)
    plt.ylim(-0.5, 0.5)
    plt.title("Loihi output thrust")

    fig = plt.gcf()
    fig.set_size_inches(14.0,8.0)

    plt.savefig("images/results_"+str(idx)+".png")
    plt.clf()
