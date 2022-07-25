import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tikzplotlib

def plot(dic, name):
    ticks = ['$\\alpha$=0.5, g', '$\\alpha$=0.5, $\\neg$g', '$\\alpha$=5.0, g', '$\\alpha$=5.0, $\\neg$g']
    widths = 0.3
    l = [-0.5, 1.3, 3.4, 5.3]

    plt.hist(x=[99], hatch='x', label='$\\neg$f, $\\neg$v', color='white', edgecolor='black')
    plt.hist(x=[99], label='$\\neg$f, v', color='white', edgecolor='black')
    plt.hist(x=[99], label='f, $\\neg$v', color='black', edgecolor='black')
    plt.hist(x=[99], label='f, v', color='lightgrey', edgecolor='black')
    plt.legend(ncol=4)

    ff_plot = plt.boxplot(dic['ff'], patch_artist=True, positions=l, widths=widths)
    for box in ff_plot['boxes']:
        box.set(hatch='x', fill=False)
    ft_plot = plt.boxplot(dic['ft'], patch_artist=True, positions=[x + 0.4 for x in l], widths=widths)
    for box in ft_plot['boxes']:
        box.set(fill=True, color='white', edgecolor='black')
    tf_plot = plt.boxplot(dic['tf'], patch_artist=True, positions=[x + 0.8 for x in l], widths=widths)
    for box in tf_plot['boxes']:
        box.set(fill=True, color='black', edgecolor='black')
    tt_plot = plt.boxplot(dic['tt'], patch_artist=True, positions=[x + 1.2 for x in l], widths=widths)
    for box in tt_plot['boxes']:
        box.set(fill=True, color='lightgrey', edgecolor='black')

    plt.gca().set_xlim([-1, 7])
    plt.gca().set_ylim([0.36, 0.62])

    plt.xticks(np.arange(0, len(ticks) * 2, 2), ticks)
    plt.ylabel('Local cross-validation F1 score')
    tikzplotlib.save(name)
    plt.show()

f = plt.figure()
df = pd.read_csv('hyper-param-local.csv')

nnc_05_T = df.loc[((df['mixup-alpha'] == 0.5) & (df['grouped-by-author'] == 1) & (df['call-detector'] == 'nnc'))]
nnc_05_F = df.loc[((df['mixup-alpha'] == 0.5) & (df['grouped-by-author'] == 0) & (df['call-detector'] == 'nnc'))]

nnc_50_T = df.loc[((df['mixup-alpha'] == 5.0) & (df['grouped-by-author'] == 1) & (df['call-detector'] == 'nnc'))]
nnc_50_F = df.loc[((df['mixup-alpha'] == 5.0) & (df['grouped-by-author'] == 0) & (df['call-detector'] == 'nnc'))]

###

nc_05_T = df.loc[((df['mixup-alpha'] == 0.5) & (df['grouped-by-author'] == 1) & (df['call-detector'] == 'nc'))]
nc_05_F = df.loc[((df['mixup-alpha'] == 0.5) & (df['grouped-by-author'] == 0) & (df['call-detector'] == 'nc'))]

nc_50_T = df.loc[((df['mixup-alpha'] == 5.0) & (df['grouped-by-author'] == 1) & (df['call-detector'] == 'nc'))]
nc_50_F = df.loc[((df['mixup-alpha'] == 5.0) & (df['grouped-by-author'] == 0) & (df['call-detector'] == 'nc'))]

# X , X = freefield1010, valid2020
nnc = {
    'ff': [
        nnc_05_T.loc[(nnc_05_T['freefield1010'] == 0) & (nnc_05_T['valid2020'] == 0)]['f1-score'].values,
        nnc_05_F.loc[(nnc_05_F['freefield1010'] == 0) & (nnc_05_F['valid2020'] == 0)]['f1-score'].values,
        nnc_50_T.loc[(nnc_50_T['freefield1010'] == 0) & (nnc_50_T['valid2020'] == 0)]['f1-score'].values,
        nnc_50_F.loc[(nnc_50_F['freefield1010'] == 0) & (nnc_50_F['valid2020'] == 0)]['f1-score'].values,
    ],

    'ft': [
        nnc_05_T.loc[(nnc_05_T['freefield1010'] == 0) & (nnc_05_T['valid2020'] == 1)]['f1-score'].values,
        nnc_05_F.loc[(nnc_05_F['freefield1010'] == 0) & (nnc_05_F['valid2020'] == 1)]['f1-score'].values,
        nnc_50_T.loc[(nnc_50_T['freefield1010'] == 0) & (nnc_50_T['valid2020'] == 1)]['f1-score'].values,
        nnc_50_F.loc[(nnc_50_F['freefield1010'] == 0) & (nnc_50_F['valid2020'] == 1)]['f1-score'].values,
    ],

    'tf': [
        nnc_05_T.loc[(nnc_05_T['freefield1010'] == 1) & (nnc_05_T['valid2020'] == 0)]['f1-score'].values,
        nnc_05_F.loc[(nnc_05_F['freefield1010'] == 1) & (nnc_05_F['valid2020'] == 0)]['f1-score'].values,
        nnc_50_T.loc[(nnc_50_T['freefield1010'] == 1) & (nnc_50_T['valid2020'] == 0)]['f1-score'].values,
        nnc_50_F.loc[(nnc_50_F['freefield1010'] == 1) & (nnc_50_F['valid2020'] == 0)]['f1-score'].values,
    ],

    'tt': [
        nnc_05_T.loc[(nnc_05_T['freefield1010'] == 1) & (nnc_05_T['valid2020'] == 1)]['f1-score'].values,
        nnc_05_F.loc[(nnc_05_F['freefield1010'] == 1) & (nnc_05_F['valid2020'] == 1)]['f1-score'].values,
        nnc_50_T.loc[(nnc_50_T['freefield1010'] == 1) & (nnc_50_T['valid2020'] == 1)]['f1-score'].values,
        nnc_50_F.loc[(nnc_50_F['freefield1010'] == 1) & (nnc_50_F['valid2020'] == 1)]['f1-score'].values,
    ]
}

###
nc = {
    'ff': [
        nc_05_T.loc[(nc_05_T['freefield1010'] == 0) & (nc_05_T['valid2020'] == 0)]['f1-score'].values,
        nc_05_F.loc[(nc_05_F['freefield1010'] == 0) & (nc_05_F['valid2020'] == 0)]['f1-score'].values,
        nc_50_T.loc[(nc_50_T['freefield1010'] == 0) & (nc_50_T['valid2020'] == 0)]['f1-score'].values,
        nc_50_F.loc[(nc_50_F['freefield1010'] == 0) & (nc_50_F['valid2020'] == 0)]['f1-score'].values,
    ],

    'ft': [
        nc_05_T.loc[(nc_05_T['freefield1010'] == 0) & (nc_05_T['valid2020'] == 1)]['f1-score'].values,
        nc_05_F.loc[(nc_05_F['freefield1010'] == 0) & (nc_05_F['valid2020'] == 1)]['f1-score'].values,
        nc_50_T.loc[(nc_50_T['freefield1010'] == 0) & (nc_50_T['valid2020'] == 1)]['f1-score'].values,
        nc_50_F.loc[(nc_50_F['freefield1010'] == 0) & (nc_50_F['valid2020'] == 1)]['f1-score'].values,
    ],

    'tf': [
        nc_05_T.loc[(nc_05_T['freefield1010'] == 1) & (nc_05_T['valid2020'] == 0)]['f1-score'].values,
        nc_05_F.loc[(nc_05_F['freefield1010'] == 1) & (nc_05_F['valid2020'] == 0)]['f1-score'].values,
        nc_50_T.loc[(nc_50_T['freefield1010'] == 1) & (nc_50_T['valid2020'] == 0)]['f1-score'].values,
        nc_50_F.loc[(nc_50_F['freefield1010'] == 1) & (nc_50_F['valid2020'] == 0)]['f1-score'].values,
    ],

    'tt': [
        nc_05_T.loc[(nc_05_T['freefield1010'] == 1) & (nc_05_T['valid2020'] == 1)]['f1-score'].values,
        nc_05_F.loc[(nc_05_F['freefield1010'] == 1) & (nc_05_F['valid2020'] == 1)]['f1-score'].values,
        nc_50_T.loc[(nc_50_T['freefield1010'] == 1) & (nc_50_T['valid2020'] == 1)]['f1-score'].values,
        nc_50_F.loc[(nc_50_F['freefield1010'] == 1) & (nc_50_F['valid2020'] == 1)]['f1-score'].values,
    ]
}

plot(nnc, 'hyper-param-local-nnc.tex')
plot(nc, 'hyper-param-local-nc.tex')
