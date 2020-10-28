import csv
import numpy as np
import matplotlib.pyplot as plt

DIM_INPUT_LAYER  = 20
DIM_HIDDEN_LAYER = 10
DIM_OUTPUT_LAYER = 5

MED_FILT_WINDOW_1 = 51
MED_FILT_WINDOW_2 = 201

NB_TESTS = 20

Divergence  = list()
Thrust      = list()
Velocity    = list()
Altitude    = list()
Time        = list()

LINEWIDTH = 0.8
COLOR = "#00A6D6" # TUD colors

def medfilt (x, k):
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros ((len (x), k), dtype=x.dtype)
    y[:,k2] = x
    for i in range (k2):
        j = k2 - i
        y[j:,i] = x[:-j]
        y[:j,i] = x[0]
        y[:-j,-(i+1)] = x[j:]
        y[-j:,-(i+1)] = x[-1]
    return np.median (y, axis=1)

for idx in range(NB_TESTS):
    with open("optitrack_paparazzi_logs/run"+str(idx)+".csv", newline='') as f:
        reader = csv.reader(f)
        dataList = list(reader)
        data = np.array(dataList[1:]).T
        Time.append(data[0].astype(np.float))
        Altitude.append(data[1].astype(np.float))
        Velocity.append(data[2].astype(np.float))
        Divergence.append(data[3].astype(np.float))
        Thrust.append(data[14].astype(np.float))

plt.subplot(2,2,1)
# Divergence error
for i in range(NB_TESTS):
    plt.plot(
        Time[i], 
        medfilt(Divergence[i] - np.ones(len(Divergence[i][:])), MED_FILT_WINDOW_1), 
        COLOR, 
        linewidth=LINEWIDTH
    )
plt.title("Divergence [1/s]")
plt.xlabel("Time [s]")
plt.grid(True)

plt.subplot(2,2,2)
# Thrust setpoint
for i in range(NB_TESTS):
    plt.plot(
        Time[i], 
        medfilt(Thrust[i], MED_FILT_WINDOW_1), 
        COLOR, 
        linewidth=LINEWIDTH
    )
plt.title("Thrust setpoint [g]")
plt.xlabel("Time [s]")
plt.grid(True)

plt.subplot(2,2,3)
# Vertical velocity
for i in range(NB_TESTS):
    plt.plot(
        Time[i], 
        medfilt(-Velocity[i], MED_FILT_WINDOW_2),
        COLOR, 
        linewidth=LINEWIDTH
    )
plt.title("Vertical speed [m/s]")
plt.xlabel("Time [s]")
plt.grid(True)

plt.subplot(2,2,4)
# Altitude
for i in range(NB_TESTS):
    plt.plot(
        Time[i], 
        medfilt(-Altitude[i], MED_FILT_WINDOW_2),
        COLOR, 
        linewidth=LINEWIDTH
    )
plt.title("Altitude [m]")
plt.xlabel("Time [s]")
plt.grid(True)

fig = plt.gcf()
fig.set_size_inches(18.0,12.0)

plt.savefig("optitrack_output.svg")
plt.clf()
