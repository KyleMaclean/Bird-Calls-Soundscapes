import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import tikzplotlib
from cycler import cycler

df = pd.read_csv('scores-and-times-for-ensembles.csv')

high_nc = df.loc[((df['diversity'] == 'high') & (df['detector'] == 'nc'))]
high_nnc = df.loc[((df['diversity'] == 'high') & (df['detector'] == 'nnc'))]


fig, ax1 = plt.subplots()

ax1.plot(high_nc['size'], high_nc['public'], 'o', color='black', label='nc')
ax1.plot(high_nnc['size'], high_nnc['public'], 'o', color='grey', label='$\\neg$nc')

plt.legend(ncol=2, loc='lower left')
ax1.set_xlabel('Size of ensemble with high diversity')
ax1.set_ylabel('Kaggle Public Leaderboard F1 score')
ax1.set_ylim(0.3, 0.8)

ax2 = ax1.twinx()

ax2.set_ylabel('Kaggle runtime in seconds')

ax2.plot(high_nc['size'], high_nc['runtime'], 'x', color='black', label='nc')
ax2.plot(high_nnc['size'], high_nnc['runtime'], 'x', color='grey', label='$\\neg$nc')
ax2.set_ylim(300, 800)

plt.legend(ncol=2, loc='lower right')
plt.xticks([1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40])
tikzplotlib.save('kaggle-runtime-accuracy.tex')
plt.show()
