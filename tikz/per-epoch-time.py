import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tikzplotlib

df = pd.read_csv("per-epoch-time.csv")
# X , X = freefield1010, valid2020
ff = df.loc[(df['data_config'] == 'ff')]['secs']
ft = df.loc[(df['data_config'] == 'ft')]['secs']
tf = df.loc[(df['data_config'] == 'tf')]['secs']
tt = df.loc[(df['data_config'] == 'tt')]['secs']

plt.hist(x=[99], hatch='x', label='$\\neg$f, $\\neg$v', color='white', edgecolor='black')
plt.hist(x=[99], label='$\\neg$f, v', color='white', edgecolor='black')
plt.hist(x=[99], label='f, $\\neg$v', color='black', edgecolor='black')
plt.hist(x=[99], label='f, v', color='lightgrey', edgecolor='black')
plt.legend(ncol=4)


width = 0.2
x = np.arange(len(ff))

# {'/', '\', '|', '-', '+', 'x', 'o', 'O', '.', '*'}
ff_plot = plt.bar(x-(width*2), ff, color='white', edgecolor='black', hatch='x', width=width/1.5)
ft_plot = plt.bar(x-width, ft, color='white', edgecolor='black', width=width/1.5)
tf_plot = plt.bar(x, tf, color='black', edgecolor='black', width=width/1.5)
tt_plot = plt.bar(x+width, tt, color='lightgrey', edgecolor='black', width=width/1.5)

# plt.legend(bbox_to_anchor=(-1, 1, 1, 0), loc="lower left", mode="expand", ncol=4)
plt.gca().set_xlim([-0.6, 2.4])
plt.gca().set_ylim([0, 440])
# plt.legend()
plt.xticks([val-0.1 for val in x], ['resnest26', 'resnest50', 'efficientnet'])
plt.ylabel('Training time per epoch in seconds')
# plt.legend(['$\\neg$f, $\\neg$v', '$\\neg$f, v', 'f, $\\neg$v', 'f, v'], ncol=4)
# plt.legend([ff_plot["boxes"][0], ft_plot["boxes"][0], tf_plot["boxes"][0], tt_plot["boxes"][0]],
#            ['$\\neg$f, $\\neg$v', '$\\neg$f, v', 'f, $\\neg$v', 'f, v'],
#            bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=4)
tikzplotlib.save("per-epoch-time.tex")
plt.show()

