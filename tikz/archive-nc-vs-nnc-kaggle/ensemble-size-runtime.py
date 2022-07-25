import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import tikzplotlib

runtime = pd.read_csv('nc-nnc-kaggle-runtime.csv')
runtime_wide = runtime.pivot(index='ensemble-id', columns='detector', values='runtime')
runtime_wide['avg'] = runtime_wide[['nc', 'nnc']].mean(axis=1)

br1 = np.arange(len(runtime_wide))
plt.bar(x=br1, height=runtime_wide['nnc'], color='lightgrey', edgecolor='black', label='$\\neg$nc')

plt.xlabel('Medium-diversity ensemble size')
plt.ylabel('Runtime during Kaggle testing (seconds)')

x = range(1, 11)
plt.xticks([r for r in range(len(runtime_wide))], x)
# data labels
plt.legend(bbox_to_anchor=(-1, 1, 1, 0), loc="lower left", mode="expand", ncol=4) # bbox_to_anchor=(0, 1, 1, 0)

x = range(0, 10)
m, b = np.polyfit(x, runtime_wide['nnc'], 1)
plt.plot(x, m*x+b, color='black')
tikzplotlib.save('ensemble-size-runtime.tex')
plt.show()
