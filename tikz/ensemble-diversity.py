import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import tikzplotlib
from cycler import cycler

df = pd.read_csv('scores-and-times-for-ensembles.csv')

low_nc = df.loc[((df['diversity'] == 'low') & (df['detector'] == 'nc') & (df['size'] <= 20))]
low_nnc = df.loc[((df['diversity'] == 'low') & (df['detector'] == 'nnc') & (df['size'] <= 20))]
med_nc = df.loc[((df['diversity'] == 'med') & (df['detector'] == 'nc') & (df['size'] <= 20))]
med_nnc = df.loc[((df['diversity'] == 'med') & (df['detector'] == 'nnc') & (df['size'] <= 20))]
high_nc = df.loc[((df['diversity'] == 'high') & (df['detector'] == 'nc') & (df['size'] <= 20))]
high_nnc = df.loc[((df['diversity'] == 'high') & (df['detector'] == 'nnc') & (df['size'] <= 20))]

plt.plot(low_nc['size'], low_nc['public'], 'x', color='black', label='l, nc')
plt.plot(low_nnc['size'], low_nnc['public'], 'x', color='grey', label='l, $\\neg$nc')
plt.plot(med_nc['size'], med_nc['public'], 'o', color='black', label='m, nc')
plt.plot(med_nnc['size'], med_nnc['public'], 'o', color='grey', label='m, $\\neg$nc')
plt.plot(high_nc['size'], high_nc['public'], '^', color='black', label='h, nc')
plt.plot(high_nnc['size'], high_nnc['public'], '^', color='grey', label='h, $\\neg$nc')
plt.legend(ncol=2, loc='lower right')
plt.xlabel('Ensemble size')
plt.ylabel('Kaggle Public Leaderboard F1 score')
plt.xticks([1,2,3,4,5,6,7,8,9,10,12,14,16,18,20])
tikzplotlib.save('ensemble-diversity.tex')
plt.show()

# bar_width = 0.35
# br1 = np.arange(len(public_nc))
# br2 = [x + bar_width + 0.15 for x in br1]
#
# plt.bar(br1, public_nc['public'], color='lightgrey', edgecolor='black', width=bar_width, label='nc')
# plt.bar(br2, public_nnc['public'], color='black', width=bar_width, label='$\\neg$nc')
#
# plt.ylabel('Kaggle Public Leaderboard F1 score')
# plt.xlabel('Medium-diversity ensemble size')
#
# # hyperparameters: ma gba valid freefield
#
# plt.xticks([r + bar_width/2 for r in range(len(public_nc))], range(1, 11))
# plt.legend(ncol=4)
# plt.gca().set_ylim([0.5, 0.8])
# # plt.legend(bbox_to_anchor=(-1, 1, 1, 0), loc="lower left", mode="expand", ncol=2)  # bbox_to_anchor=(0, 1, 1, 0)
#
# tikzplotlib.save('nc-vs-nnc-kaggle-scores.tex')
# plt.show()
