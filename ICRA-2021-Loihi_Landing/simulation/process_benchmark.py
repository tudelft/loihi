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
    OutputMatchingPerformance = list()
    HiddenMatchingPerformance = list()

    InfillSim           = list()
    SpikesCountSim      = list()
    DivergenceSim       = list()
    ThrustSim           = list()
    IdxOutputSim        = list()
    IdxHiddenSim        = list()
    HiddenInfillSim     = list()

    InfillLoihi         = list()
    SpikesCountLoihi    = list()
    DivergenceLoihi     = list()
    ThrustLoihi         = list()
    IdxOutputLoihi      = list()
    IdxHiddenLoihi      = list()
    HiddenInfillLoihi   = list()

    ThrustRMSE          = list()
    IdxOutputDiff       = list()
    IdxHiddenDiff       = list()

    for idx in range(100):
        with open("optitrack_paparazzi_logs/run_raw_out_spikes"+str(idx)+".csv", newline='') as f:
            reader = csv.reader(f)
            dataList = list(reader)
            data_sim2 = np.array(dataList[1:])
            data_sim2 = data_sim2.T
            data_sim2 = data_sim2[1:].astype(np.float)
            data_sim = np.zeros((DIM_OUTPUT_LAYER,len(data_sim2[0])-DELAY))
            for i in range(DIM_OUTPUT_LAYER):
                data_sim[i] = data_sim2[i][0:len(data_sim2[i])-DELAY]

        with open("optitrack_paparazzi_logs/run_raw_hid_spikes"+str(idx)+".csv", newline='') as f:
            reader = csv.reader(f)
            dataList = list(reader)
            data_hid_sim2 = np.array(dataList[1:])
            data_hid_sim2 = data_hid_sim2.T
            data_hid_sim2 = data_hid_sim2[1:].astype(np.float)
            data_hid_sim = np.zeros((DIM_HIDDEN_LAYER,len(data_hid_sim2[0])-DELAY))
            for i in range(DIM_HIDDEN_LAYER):
                data_hid_sim[i] = data_hid_sim2[i][0:len(data_hid_sim2[i])-DELAY]

        with open("loihi_logs/output_spikes_rec_file_"+str(idx)+".txt", "r") as f:
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

        with open("loihi_logs/hidden_spikes_rec_file_"+str(idx)+".txt", "r") as f:
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

        with open("optitrack_paparazzi_logs/run_raw"+str(idx)+".csv", newline='') as f:
            reader = csv.reader(f)
            dataList2 = list(reader)
            data = np.array(dataList2[1:]).T
            divergence_sim = data[2][:].astype(np.float)
            DivergenceSim.append(divergence_sim[0:len(divergence_sim)-DELAY])
            thrust_sim = data[3][:].astype(np.float)
            ThrustSim.append(thrust_sim[0:len(thrust_sim)-DELAY])

        with open("loihi_logs/thrust_rec_file_"+str(idx)+".txt", "r") as f:
            thrust_loihi = list()
            for line in f:
                thrust_loihi.append((float)(line))
            t_loihi = np.array(thrust_loihi)
            ThrustLoihi.append(t_loihi[DELAY:])

        with open("loihi_logs/input_divergence_rec_file_"+str(idx)+".txt", "r") as f:
            divergence = list()
            for line in f:
                divergence.append((float)(line))
            divergence_loihi = np.array(divergence)
            DivergenceLoihi.append(divergence_loihi[DELAY:])

        OutputMatchingPerformance.append(spikeSequenceMatching(data_sim, data_loihi))
        HiddenMatchingPerformance.append(spikeSequenceMatching(data_hid_sim, data_hid_loihi))

        InfillLoihi.append(getInfill(data_loihi))
        InfillSim.append(getInfill(data_sim))

        HiddenInfillLoihi.append(getInfill(data_hid_loihi))
        HiddenInfillSim.append(getInfill(data_hid_sim))

        SpikesCountSim.append(getNbrofSpikes(data_sim))
        SpikesCountLoihi.append(getNbrofSpikes(data_loihi))

        IdxOutputSim.append(convSpikesList2IndicesEvents(data_sim.T, DIM_OUTPUT_LAYER))
        IdxOutputLoihi.append(convSpikesList2IndicesEvents(data_loihi.T, DIM_OUTPUT_LAYER))
        IdxOutputDiff.append(convSpikesList2IndicesEvents(data_sim.T - data_loihi.T, DIM_OUTPUT_LAYER))

        IdxHiddenSim.append(convSpikesList2IndicesEvents(data_hid_sim.T, DIM_HIDDEN_LAYER))
        IdxHiddenLoihi.append(convSpikesList2IndicesEvents(data_hid_loihi.T, DIM_HIDDEN_LAYER))
        IdxHiddenDiff.append(convSpikesList2IndicesEvents(data_hid_sim.T - data_hid_loihi.T, DIM_HIDDEN_LAYER))

        ThrustRMSE.append(rmse(thrust_sim[0:len(thrust_sim)-DELAY], thrust_loihi[DELAY:]))

        if PRINT:
            print("[" + "{:02d}".format(idx) + "] Matching performance in output spike sequence: " + "{:.2f}".format(OutputMatchingPerformance[idx])  + "%")
            print("\tLoihi output infill:\t\t" + "{:.2f}".format(InfillLoihi[idx])  + "%")
            print("\tSimulation output infill:\t" + "{:.2f}".format(InfillSim[idx])  + "%\n")

    print("------------------------------------------")
    print("Hidden infill\t Loihi \t\tSimulation")
    print("------------------------------------------")
    print("Average            " + "{:.2f}".format(np.mean(HiddenInfillLoihi))  + "%" + "\t " + "{:.2f}".format(np.mean(HiddenInfillSim))  + "%")
    print("Median             " + "{:.2f}".format(np.median(HiddenInfillLoihi))  + "%" + "\t " + "{:.2f}".format(np.median(HiddenInfillSim))  + "%")
    print("Variance           " + "{:.2f}".format(np.var(HiddenInfillLoihi))  + "%" + "\t " + "{:.2f}".format(np.var(HiddenInfillSim))  + "%")
    print("Standard dev.      " + "{:.2f}".format(np.std(HiddenInfillLoihi))  + "%" + "\t " + "{:.2f}".format(np.std(HiddenInfillSim))  + "%")
    print("Max               " + "{:.2f}".format(np.amax(HiddenInfillLoihi))  + "%" + "\t" + "{:.2f}".format(np.amax(HiddenInfillSim))  + "%")
    print("Min                " + "{:.2f}".format(np.amin(HiddenInfillLoihi))  + "%" + "\t " + "{:.2f}".format(np.amin(HiddenInfillSim))  + "%")
    print("------------------------------------------\n")

    print("------------------------------------------")
    print("Hidden spikes sequence - Matching scores")
    print("------------------------------------------")
    print("Average          " + "{:.2f}".format(np.mean(HiddenMatchingPerformance))  + "%")
    print("Median           " + "{:.2f}".format(np.median(HiddenMatchingPerformance))  + "%")
    print("Variance          " + "{:.2f}".format(np.var(HiddenMatchingPerformance))  + "%")
    print("Standard dev.     " + "{:.2f}".format(np.std(HiddenMatchingPerformance))  + "%")
    print("Max              " + "{:.2f}".format(np.amax(HiddenMatchingPerformance))  + "%")
    print("Min              " + "{:.2f}".format(np.amin(HiddenMatchingPerformance))  + "%")
    print("------------------------------------------\n")

    # Hidden infill looks Gaussian. Let's prove that distributions are normal by using the Kolmogorov-Smirnov (KS) test:
    _, p_val_sim = scipy.stats.kstest((np.array(HiddenInfillSim) - np.mean(HiddenInfillSim)) / np.std(HiddenInfillSim), 'norm')
    _, p_val_loihi = scipy.stats.kstest((np.array(HiddenInfillLoihi) - np.mean(HiddenInfillLoihi)) / np.std(HiddenInfillLoihi), 'norm')
    print("P-value for the Kolmogorov-Smirnov normality test")
    print("   [Simulation] " + "{:.2f}".format(p_val_sim))
    print("        [Loihi] " + "{:.2f}".format(p_val_loihi) + "\n")
    # All p-values are >> 0.5, meaning hypothesis is confirmed.
    # Let's now prove that both distributions (sim & loihi) are not statistically different.
    # Two-sample Kolmogorov-Smirnov test:
    _, p_val_compare = scipy.stats.ks_2samp(
        (np.array(HiddenInfillSim) - np.mean(HiddenInfillSim)) / np.std(HiddenInfillSim),
        (np.array(HiddenInfillLoihi) - np.mean(HiddenInfillLoihi)) / np.std(HiddenInfillLoihi),
        alternative='two-sided')
    print("P-value for the two-sided Kolmogorov-Smirnov between infill distributions: " + "{:.2f}".format(p_val_compare) + "\n")
    # Confirmed.

    print("------------------------------------------")
    print("Output infill\t Loihi \t\tSimulation")
    print("------------------------------------------")
    print("Average          " + "{:.2f}".format(np.mean(InfillLoihi))  + "%" + "\t\t" + "{:.2f}".format(np.mean(InfillSim))  + "%")
    print("Median           " + "{:.2f}".format(np.median(InfillLoihi))  + "%" + "\t\t" + "{:.2f}".format(np.median(InfillSim))  + "%")
    print("Variance          " + "{:.2f}".format(np.var(InfillLoihi))  + "%" + "\t\t " + "{:.2f}".format(np.var(InfillSim))  + "%")
    print("Standard dev.     " + "{:.2f}".format(np.std(InfillLoihi))  + "%" + "\t\t " + "{:.2f}".format(np.std(InfillSim))  + "%")
    print("Max              " + "{:.2f}".format(np.amax(InfillLoihi))  + "%" + "\t\t" + "{:.2f}".format(np.amax(InfillSim))  + "%")
    print("Min              " + "{:.2f}".format(np.amin(InfillLoihi))  + "%" + "\t\t" + "{:.2f}".format(np.amin(InfillSim))  + "%")
    print("------------------------------------------\n")

    print("------------------------------------------")
    print("Output spikes sequence - Matching scores")
    print("------------------------------------------")
    print("Average          " + "{:.2f}".format(np.mean(OutputMatchingPerformance))  + "%")
    print("Median           " + "{:.2f}".format(np.median(OutputMatchingPerformance))  + "%")
    print("Variance          " + "{:.2f}".format(np.var(OutputMatchingPerformance))  + "%")
    print("Standard dev.     " + "{:.2f}".format(np.std(OutputMatchingPerformance))  + "%")
    print("Max              " + "{:.2f}".format(np.amax(OutputMatchingPerformance))  + "%")
    print("Min              " + "{:.2f}".format(np.amin(OutputMatchingPerformance))  + "%")
    print("------------------------------------------\n")

    # Violin plots for spikes infill
    plt.subplot(1,3,1)
    seaborn.set_theme(style="whitegrid")
    seaborn.violinplot(data=pd.DataFrame({'Simulation': np.array(HiddenInfillSim),'Loihi': np.array(HiddenInfillLoihi)}, columns=['Simulation','Loihi']))
    # plt.ylim(5, 15)
    plt.grid(None, which='major', axis='x')
    plt.ylabel("Hidden spikes infill (%)")

    plt.subplot(1,3,2)
    seaborn.set_theme(style="whitegrid")
    seaborn.violinplot(data=pd.DataFrame({'Simulation': np.array(InfillSim),'Loihi': np.array(InfillLoihi)}, columns=['Simulation','Loihi']))
    # plt.ylim(5, 15)
    plt.grid(None, which='major', axis='x')
    plt.ylabel("Output spikes infill (%)")

    plt.subplot(1,3,3)
    plt.plot(ThrustRMSE, 'k')
    plt.xlim(0, len(ThrustRMSE))
    plt.xlabel("Test")
    plt.grid(True, which='minor', axis='both')
    plt.ylabel("Thrust RMSE")

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.35, hspace=0.25)
    fig = plt.gcf()
    fig.set_size_inches(14.0,8.0)
    plt.savefig("stats.png")
    plt.clf()

    # Output infill looks Gaussian. Let's prove that distributions are normal by using the Kolmogorov-Smirnov (KS) test: 
    _, p_val_sim = scipy.stats.kstest((np.array(InfillSim) - np.mean(InfillSim)) / np.std(InfillSim), 'norm')
    _, p_val_loihi = scipy.stats.kstest((np.array(InfillLoihi) - np.mean(InfillLoihi)) / np.std(InfillLoihi), 'norm')
    print("P-value for the Kolmogorov-Smirnov normality test")
    print("   [Simulation] " + "{:.2f}".format(p_val_sim))
    print("        [Loihi] " + "{:.2f}".format(p_val_loihi) + "\n")
    # All p-values are >> 0.5, meaning hypothesis is confirmed.
    # Let's now prove that both distributions (sim & loihi) are not statistically different.
    # Two-sample Kolmogorov-Smirnov test:
    _, p_val_compare = scipy.stats.ks_2samp(
        (np.array(InfillSim) - np.mean(InfillSim)) / np.std(InfillSim),
        (np.array(InfillLoihi) - np.mean(InfillLoihi)) / np.std(InfillLoihi),
        alternative='two-sided')
    print("P-value for the two-sided Kolmogorov-Smirnov between infill distributions: " + "{:.2f}".format(p_val_compare) + "\n")
    # Confirmed.

    N = 67
    print("\nRMSE for test #" + str(N) + ": {:.2f}".format(ThrustRMSE[N]))
    print("\nAverage RMSE: {:.4f}".format(np.mean(ThrustRMSE)))
    print("\nSTD RMSE: {:.4f}".format(np.std(ThrustRMSE)) + "\n")

    plt.subplot(3,4,(1,5))
    plt.plot(DivergenceSim[N],'steelblue')
    plt.ylim(-2, 2)
    plt.xlim(0, len(DivergenceSim[N]))
    plt.xlabel("Time-step")

    plt.subplot(3,4,4)
    plt.plot(ThrustSim[N],'steelblue')
    plt.ylim(-0.3, 0.1)
    plt.xlim(0, len(DivergenceSim[N]))
    plt.xlabel("Time-step")

    plt.subplot(3,4,8)
    plt.plot(ThrustLoihi[N],'steelblue')
    plt.ylim(-0.3, 0.1)
    plt.xlim(0, len(DivergenceSim[N]))
    plt.grid(False)
    plt.xlabel("Time-step")

    plt.subplot(3,4,12)
    plt.plot(ThrustSim[N] - ThrustLoihi[N],'r')
    # plt.ylim(-1e-4, 1e-4)
    plt.xlim(0, len(DivergenceSim[N]))
    plt.grid(False)
    plt.xlabel("Time-step")

    plt.subplot(3,4,2)
    idxEventsSim = IdxHiddenSim[N]
    for i in range(len(idxEventsSim)):
        plt.plot(i*np.ones(len(idxEventsSim[i][:])), idxEventsSim[i][:], 'k.')
    plt.ylim(-0.5, DIM_HIDDEN_LAYER + 0.5)
    plt.xlim(0, len(idxEventsSim))
    plt.grid(None, which='major', axis='x')
    plt.xlabel("Time-step")

    plt.subplot(3,4,6)
    idxEventsLoihi = IdxHiddenLoihi[N]
    for i in range(len(idxEventsLoihi)):
        plt.plot(i*np.ones(len(idxEventsLoihi[i][:])), idxEventsLoihi[i][:], 'k.')
    plt.ylim(-0.5, DIM_HIDDEN_LAYER + 0.5)
    plt.xlim(0, len(idxEventsSim))
    plt.grid(None, which='major', axis='x')
    plt.xlabel("Time-step")

    plt.subplot(3,4,10)
    idxEventsDiff = IdxHiddenDiff[N]
    for i in range(len(idxEventsDiff)):
        plt.plot(i*np.ones(len(idxEventsDiff[i][:])), idxEventsDiff[i][:], 'r.')
    plt.ylim(-0.5, DIM_HIDDEN_LAYER + 0.5)
    plt.xlim(0, len(idxEventsSim))
    plt.grid(None, which='major', axis='x')
    plt.xlabel("Time-step")

    plt.subplot(3,4,3)
    idxEventsSim = IdxOutputSim[N]
    for i in range(len(idxEventsSim)):
        plt.plot(i*np.ones(len(idxEventsSim[i][:])), idxEventsSim[i][:], 'k.')
    plt.ylim(-0.5, DIM_OUTPUT_LAYER + 0.5)
    plt.xlim(0, len(idxEventsSim))
    plt.grid(None, which='major', axis='x')
    plt.xlabel("Time-step")

    plt.subplot(3,4,7)
    idxEventsLoihi = IdxOutputLoihi[N]
    for i in range(len(idxEventsLoihi)):
        plt.plot(i*np.ones(len(idxEventsLoihi[i][:])), idxEventsLoihi[i][:], 'k.')
    plt.ylim(-0.5, DIM_OUTPUT_LAYER + 0.5)
    plt.xlim(0, len(idxEventsSim))
    plt.grid(None, which='major', axis='x')
    plt.xlabel("Time-step")

    plt.subplot(3,4,11)
    idxEventsDiff = IdxOutputDiff[N]
    for i in range(len(idxEventsDiff)):
        plt.plot(i*np.ones(len(idxEventsDiff[i][:])), idxEventsDiff[i][:], 'r.')
    plt.ylim(-0.5, DIM_OUTPUT_LAYER + 0.5)
    plt.xlim(0, len(idxEventsSim))
    plt.grid(None, which='major', axis='x')
    plt.xlabel("Time-step")

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.35, hspace=0.25)
    fig = plt.gcf()
    fig.set_size_inches(18.0,12.0)
    plt.savefig("example_sample_"+str(N)+".svg")
    plt.clf()

