import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tikzplotlib

df = pd.read_csv("ff1010-valid2020-comparison.csv")
# X , X = freefield1010, valid2020
ff = [df['resnest26d_FF_f1'], df['resnest50_FF_f1'], df['efficientnet_FF_f1']]
ft = [df['resnest26d_FT_f1'], df['resnest50_FT_f1'], df['efficientnet_FT_f1']]
tf = [df['resnest26d_TF_f1'], df['resnest50_TF_f1'], df['efficientnet_TF_f1']]
tt = [df['resnest26d_TT_f1'], df['resnest50_TT_f1'], df['efficientnet_TT_f1']]

ticks = ['resnest26', 'resnest50', 'efficientnet']

plt.hist(x=[99], hatch='x', label='$\\neg$f, $\\neg$v', color='white', edgecolor='black')
plt.hist(x=[99], label='$\\neg$f, v', color='white', edgecolor='black')
plt.hist(x=[99], label='f, $\\neg$v', color='black', edgecolor='black')
plt.hist(x=[99], label='f, v', color='lightgrey', edgecolor='black')
plt.legend(ncol=4)
# plt.legend(bbox_to_anchor=(-1, 1, 1, 0), loc="lower left", mode="expand", ncol=4) # bbox_to_anchor=(0, 1, 1, 0)

# {'/', '\', '|', '-', '+', 'x', 'o', 'O', '.', '*'}
ff_plot = plt.boxplot(ff, patch_artist=True, positions=np.array(np.arange(len(ff))) * 2.0 - 0.8, widths=0.4)
# ff_plot = plt.boxplot(ff, patch_artist=True, positions=np.array(np.arange(len(ff))) * 2.0 - 0.8, widths=0.4,
#                       notch=True, labels=['foo','bar','foobar'])
for box in ff_plot['boxes']:
    box.set(hatch='x', fill=False)
# plt.plot([0.55], [0.5], patch_artist=True, hatch='+', label='foo')
ft_plot = plt.boxplot(ft, patch_artist=True, positions=np.array(np.arange(len(ft))) * 2.0 - 0.25, widths=0.4)
for box in ft_plot['boxes']:
    box.set(fill=True, color='white', edgecolor='black')
tf_plot = plt.boxplot(tf, patch_artist=True, positions=np.array(np.arange(len(tf))) * 2.0 + 0.25, widths=0.4)
for box in tf_plot['boxes']:
    box.set(fill=True, color='black', edgecolor='black')
tt_plot = plt.boxplot(tt, patch_artist=True, positions=np.array(np.arange(len(tt))) * 2.0 + 0.8, widths=0.4)
for box in tt_plot['boxes']:
    box.set(fill=True, color='lightgrey', edgecolor='black')
# plt.legend([ff_plot["boxes"][0], ft_plot["boxes"][0], tf_plot["boxes"][0], tt_plot["boxes"][0]],
#            ['$\\neg$f, $\\neg$v', '$\\neg$f, v', 'f, $\\neg$v', 'f, v'])

# plt.legend(bbox_to_anchor=(-1, 1, 1, 0), loc="lower left", mode="expand", ncol=4)
plt.gca().set_xlim([-1.2, 5.1])
plt.gca().set_ylim([0.485, 0.56])
# plt.legend()
# plt.ylabel('Local cross-validation F1 score')
plt.xticks(np.arange(0, len(ticks) * 2, 2), ticks)
# plt.gca().yaxis.set_label_position("right")
# plt.gca().yaxis.tick_right()
# plt.legend([ff_plot["boxes"][0], ft_plot["boxes"][0], tf_plot["boxes"][0], tt_plot["boxes"][0]],
#            ['$\\neg$f, $\\neg$v', '$\\neg$f, v', 'f, $\\neg$v', 'f, v'],
#            bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=4)
# plt.tick_params(bottom=False)
# plt.gca().get_yaxis().set_visible(False)

plt.ylabel('Local cross-validation F1 score')
tikzplotlib.save("ff1010-valid2020-comparison.tex")
plt.show()


"""

\draw[draw=black,fill=white,postaction={pattern=grid}] (axis cs:-0.5,0) rectangle (axis cs:-0.4,0);
\addlegendimage{ybar,ybar legend,draw=black,fill=white,postaction={pattern=grid}}
\addlegendentry{$\neg$f, $\neg$v}

\draw[draw=black,fill=white,postaction={pattern=grid}] (axis cs:-0.4,0) rectangle (axis cs:-0.3,0);
\draw[draw=black,fill=white,postaction={pattern=grid}] (axis cs:-0.3,0) rectangle (axis cs:-0.2,0);
\draw[draw=black,fill=white,postaction={pattern=grid}] (axis cs:-0.2,0) rectangle (axis cs:-0.1,0);
\draw[draw=black,fill=white,postaction={pattern=grid}] (axis cs:-0.1,0) rectangle (axis cs:0,0);
\draw[draw=black,fill=white,postaction={pattern=grid}] (axis cs:0,0) rectangle (axis cs:0.1,1);
\draw[draw=black,fill=white,postaction={pattern=grid}] (axis cs:0.1,0) rectangle (axis cs:0.2,0);
\draw[draw=black,fill=white,postaction={pattern=grid}] (axis cs:0.2,0) rectangle (axis cs:0.3,0);
\draw[draw=black,fill=white,postaction={pattern=grid}] (axis cs:0.3,0) rectangle (axis cs:0.4,0);
\draw[draw=black,fill=white,postaction={pattern=grid}] (axis cs:0.4,0) rectangle (axis cs:0.5,0);
\draw[draw=black,fill=white,postaction={pattern=north east lines}] (axis cs:-0.5,0) rectangle (axis cs:-0.4,0);
\addlegendimage{ybar,ybar legend,draw=black,fill=white,postaction={pattern=north east lines}}
\addlegendentry{$\neg$f, v}

\draw[draw=black,fill=white,postaction={pattern=north east lines}] (axis cs:-0.4,0) rectangle (axis cs:-0.3,0);
\draw[draw=black,fill=white,postaction={pattern=north east lines}] (axis cs:-0.3,0) rectangle (axis cs:-0.2,0);
\draw[draw=black,fill=white,postaction={pattern=north east lines}] (axis cs:-0.2,0) rectangle (axis cs:-0.1,0);
\draw[draw=black,fill=white,postaction={pattern=north east lines}] (axis cs:-0.1,0) rectangle (axis cs:0,0);
\draw[draw=black,fill=white,postaction={pattern=north east lines}] (axis cs:0,0) rectangle (axis cs:0.1,1);
\draw[draw=black,fill=white,postaction={pattern=north east lines}] (axis cs:0.1,0) rectangle (axis cs:0.2,0);
\draw[draw=black,fill=white,postaction={pattern=north east lines}] (axis cs:0.2,0) rectangle (axis cs:0.3,0);
\draw[draw=black,fill=white,postaction={pattern=north east lines}] (axis cs:0.3,0) rectangle (axis cs:0.4,0);
\draw[draw=black,fill=white,postaction={pattern=north east lines}] (axis cs:0.4,0) rectangle (axis cs:0.5,0);
\draw[draw=black,fill=white,postaction={pattern=crosshatch dots}] (axis cs:-0.5,0) rectangle (axis cs:-0.4,0);
\addlegendimage{ybar,ybar legend,draw=black,fill=white,postaction={pattern=crosshatch dots}}
\addlegendentry{f, $\neg$v}

\draw[draw=black,fill=white,postaction={pattern=crosshatch dots}] (axis cs:-0.4,0) rectangle (axis cs:-0.3,0);
\draw[draw=black,fill=white,postaction={pattern=crosshatch dots}] (axis cs:-0.3,0) rectangle (axis cs:-0.2,0);
\draw[draw=black,fill=white,postaction={pattern=crosshatch dots}] (axis cs:-0.2,0) rectangle (axis cs:-0.1,0);
\draw[draw=black,fill=white,postaction={pattern=crosshatch dots}] (axis cs:-0.1,0) rectangle (axis cs:0,0);
\draw[draw=black,fill=white,postaction={pattern=crosshatch dots}] (axis cs:0,0) rectangle (axis cs:0.1,1);
\draw[draw=black,fill=white,postaction={pattern=crosshatch dots}] (axis cs:0.1,0) rectangle (axis cs:0.2,0);
\draw[draw=black,fill=white,postaction={pattern=crosshatch dots}] (axis cs:0.2,0) rectangle (axis cs:0.3,0);
\draw[draw=black,fill=white,postaction={pattern=crosshatch dots}] (axis cs:0.3,0) rectangle (axis cs:0.4,0);
\draw[draw=black,fill=white,postaction={pattern=crosshatch dots}] (axis cs:0.4,0) rectangle (axis cs:0.5,0);
\draw[draw=black,fill=white,postaction={pattern=fivepointed stars}] (axis cs:-0.5,0) rectangle (axis cs:-0.4,0);
\addlegendimage{ybar,ybar legend,draw=black,fill=white,postaction={pattern=fivepointed stars}}
\addlegendentry{f, v}

\draw[draw=black,fill=white,postaction={pattern=fivepointed stars}] (axis cs:-0.4,0) rectangle (axis cs:-0.3,0);
\draw[draw=black,fill=white,postaction={pattern=fivepointed stars}] (axis cs:-0.3,0) rectangle (axis cs:-0.2,0);
\draw[draw=black,fill=white,postaction={pattern=fivepointed stars}] (axis cs:-0.2,0) rectangle (axis cs:-0.1,0);
\draw[draw=black,fill=white,postaction={pattern=fivepointed stars}] (axis cs:-0.1,0) rectangle (axis cs:0,0);
\draw[draw=black,fill=white,postaction={pattern=fivepointed stars}] (axis cs:0,0) rectangle (axis cs:0.1,1);
\draw[draw=black,fill=white,postaction={pattern=fivepointed stars}] (axis cs:0.1,0) rectangle (axis cs:0.2,0);
\draw[draw=black,fill=white,postaction={pattern=fivepointed stars}] (axis cs:0.2,0) rectangle (axis cs:0.3,0);
\draw[draw=black,fill=white,postaction={pattern=fivepointed stars}] (axis cs:0.3,0) rectangle (axis cs:0.4,0);
\draw[draw=black,fill=white,postaction={pattern=fivepointed stars}] (axis cs:0.4,0) rectangle (axis cs:0.5,0);
\path [draw=black, postaction={pattern=grid}]
"""