import matplotlib.pyplot as plt
import numpy as np


xlabels = [1, 2, 4, 8, 12, 16, 20, 24]
l = [[32.554, 17.0245, 10.6109, 7.20303, 5.38142, 5.26352, 4.65435, 4.36338],
     [358.949, 185.369, 104.376, 65.7268, 57.6382, 47.1132, 39.8151, 36.5777],
     [3969.48, 2127.69, 1147.71, 650.821, 555.986, 509.343, 456.077, 403.246],
     [43334.5, 22104.1, 12499.9, 7751.46, 5883.65, 5372.53, 4521.28, 3830.43],
     [499912, 219262, 145033, 88060.3, 68244.8, 57395.8, 55914, 40673.8]
    ]
plt.plot(l[4], label='$n = 10^9$')
plt.xticks(np.arange(start=0, step=1, stop=8), xlabels)
plt.xlabel("Number of CPU cores")
plt.ylabel("Runtime (in ms)")
plt.title("Runtime v/s Number of CPUs")
plt.legend()
plt.savefig("plot_10_9.png")