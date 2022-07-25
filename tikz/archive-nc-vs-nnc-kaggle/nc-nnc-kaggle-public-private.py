import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import tikzplotlib
from cycler import cycler

public = pd.read_csv('nc-nnc-kaggle-public.csv')
public_nc = public[public['detector'] == 'nc']
public_nnc = public[public['detector'] == 'nnc']
private = pd.read_csv('nc-nnc-kaggle-private.csv')
private_nc = private[private['detector'] == 'nc']
private_nnc = private[private['detector'] == 'nnc']

fig, (ax1, ax2) = plt.subplots(1, 2)
# fig.suptitle('Kaggle Performance Evaluation')

bar_width = 0.4
br1 = np.arange(len(public_nc))
br2 = [x + bar_width for x in br1]

# bar_cycle = (cycler('hatch', ['///', '--', '...','\///', 'xxx', '\\\\']) * cycler('color', 'w')*cycler('zorder', [10]))
# {'/', '\', '|', '-', '+', 'x', 'o', 'O', '.', '*'}
bar_cycle = (cycler('hatch', ['.', 'x']) * cycler('color', 'w')*cycler('zorder', [10]))
styles = bar_cycle()

ax1.bar(br1, public_nc['public'], color='lightgrey', edgecolor='black', width=bar_width, label='nc')
ax1.bar(br2, public_nnc['public'], color='black', width=bar_width, label='$\\neg$nc')
ax2.bar(br1, private_nc['private'], color='lightgrey', edgecolor='black', width=bar_width, label='nc')
ax2.bar(br2, private_nnc['private'], color='black', width=bar_width, label='$\\neg$nc')

ax1.set(ylabel='Public F1 Score')
ax2.yaxis.set_label_position('right')
ax2.yaxis.tick_right()
ax2.set(ylabel='Private F1 Score')
fig.text(0.5, 0.04, 'Number of ensembled models', ha='center')

# hyperparameters: ma gba valid freefield

ax1.set_xticks([r + bar_width for r in range(len(public_nc))], public_nc['ensemble-id'])
ax2.set_xticks([r + bar_width for r in range(len(public_nc))], public_nc['ensemble-id'])
# plt.gca().set_ylim([0.4, 0.8])

plt.legend(bbox_to_anchor=(-0.6, 1, 1, 0), loc="lower left", mode="expand", ncol=2)

tikzplotlib.save('nc-vs-nnc-kaggle-scores.tex')
plt.show()
