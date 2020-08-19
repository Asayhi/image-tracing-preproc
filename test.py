# %%
import matplotlib.pyplot as plt
%pylab inline

# %%
import numpy as np
f = open("logs/timelog.txt", "r")
times = f.read().splitlines()
tarray = np.asarray(times).astype(np.float)
# %%
fig, ax = plt.subplots()
ax.boxplot(tarray)
ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
            alpha=0.5)
ax.set_xticklabels(" ")
fig.suptitle("Duration of processing pipe process")
# %%
