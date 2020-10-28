import csv
import numpy as np
import matplotlib.pyplot as plt

DIM_INPUT_LAYER  = 20
DIM_HIDDEN_LAYER = 10
DIM_OUTPUT_LAYER = 5

NB_TESTS = 20

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

for idx in range(NB_TESTS):
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

        with open("loihi_logs/input_divergence_rec_file_"+str(idx)+".txt", "r") as f:
            divergence = list()
            for line in f:
                divergence.append((float)(line))
            divergence_loihi = np.array(divergence)

        idxEventsLoihi = convSpikesList2IndicesEvents(data_loihi.T)

        plt.subplot(3,1,1)
        plt.plot(divergence_loihi)
        plt.ylim(-3, 5)
        plt.title("Loihi input divergence")
        plt.grid(True)

        plt.subplot(3,2,(3,5))
        for i in range(len(idxEventsLoihi)):
            plt.plot(i*np.ones(len(idxEventsLoihi[i][:])), idxEventsLoihi[i][:], 'k.')
        plt.ylim(-0.5, DIM_OUTPUT_LAYER - 0.5)
        plt.title("Loihi output spikes")
        plt.grid(True)

        plt.subplot(3,2,(4,6))
        plt.plot(thrust_loihi)
        plt.ylim(-0.6, 0.2)
        plt.title("Loihi output thrust")
        plt.grid(True)

        fig = plt.gcf()
        fig.set_size_inches(18.0,12.0)

        plt.savefig("images/loihi_raw_output_"+str(idx)+".svg")
        plt.clf()