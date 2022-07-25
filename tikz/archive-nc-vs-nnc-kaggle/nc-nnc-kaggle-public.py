import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import tikzplotlib
from cycler import cycler

public = pd.read_csv('nc-nnc-kaggle-public.csv')
public_nc = public[public['detector'] == 'nc']
public_nnc = public[public['detector'] == 'nnc']

bar_width = 0.35
br1 = np.arange(len(public_nc))
br2 = [x + bar_width + 0.15 for x in br1]

plt.bar(br1, public_nc['public'], color='lightgrey', edgecolor='black', width=bar_width, label='nc')
plt.bar(br2, public_nnc['public'], color='black', width=bar_width, label='$\\neg$nc')

plt.ylabel('Kaggle Public Leaderboard F1 score')
plt.xlabel('Medium-diversity ensemble size')

# hyperparameters: ma gba valid freefield

plt.xticks([r + bar_width/2 for r in range(len(public_nc))], range(1, 11))
plt.legend(ncol=4)
plt.gca().set_ylim([0.5, 0.8])
# plt.legend(bbox_to_anchor=(-1, 1, 1, 0), loc="lower left", mode="expand", ncol=2)  # bbox_to_anchor=(0, 1, 1, 0)

tikzplotlib.save('nc-vs-nnc-kaggle-scores.tex')
plt.show()
