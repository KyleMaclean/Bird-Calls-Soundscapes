import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib
import tikzplotlib
import matplotlib.pyplot as plt

train = pd.read_csv('../input/birdclef-2021/train_metadata.csv',)
species = train['primary_label'].value_counts()
print(len(species))
species = species[::5]
# print(species.keys())
# print(species.values)
plt.bar(x=species.keys(), height=species.values, color='black')
ax = plt.gca()
# ax.set_xticks(ax.get_xticks()[::5])
# ax.axes.xaxis.set_ticklabels([])
plt.xlabel('Species')
plt.ylabel('Number of instances')
tikzplotlib.save('long-tail.tex')
plt.show()

'''
xtick={},
xticklabels={},
'''