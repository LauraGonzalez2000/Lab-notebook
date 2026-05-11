# %% [markdown]
# # Implementing a Nearest-Neighbor Decoder of Neural Activity Patterns
#

# %%
import sys, os
import numpy as np
from sklearn import linear_model, model_selection
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

sys.path += ['../physion/src'] # add src code directory for physion
import physion
import physion.utils.plot_tools as pt
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from physion.analysis.episodes.build import EpisodeData
pt.set_style('manuscript')

from pathlib import Path

# %%
datafolder = os.path.join(Path("E:/"), 'DATA', 'In_Vivo_experiments','NDNF-old-protocol', 'NDNF-WT-Dec-2022','NWBs_rebuilt')
SESSIONS = scan_folder_for_NWBfiles(datafolder)

#%%
filename = SESSIONS['files'][0]
data = Data(filename)
data.build_dFoF()

# Natural Images : 5 images, 20 trials each (100 trials in total)

#ep = EpisodeData(data, protocol_name='Natural-Images-4-repeats',
#                 quantities=['dFoF'])


# Natural Images : 4 images, 10 trials each (40 trials in total)
ep = EpisodeData(data, protocol_name='drifting-gratings',
                 quantities=['dFoF'])

# %% [markdown]
# ## Building patterns
# And transforming to the sklearn `X`, `y` variables

# %%
#  ---------------------------------------------- #
# Transforming to the sklearn `X`, `y` variables
#  ---------------------------------------------- #
# X is the list of matrice response 
# (rows = Ntrials, columns = NRois)
# y is the label of all trials 
# (rows = Ntrials, columns = 1(corresponding label))

t_window = 1.5
t0 = max([0, ep.time_duration[0]-t_window])
averaging_window = [t0, t0+t_window]
averaging_window_cond = (ep.t>averaging_window[0]) &\
                        (ep.t<averaging_window[1])

X = ep.dFoF[:, :, averaging_window_cond].mean(axis=2)

y = np.array([f"Image-{id}"for id in getattr(ep, 'Image-ID')])


X = pd.DataFrame(X,
                columns=[f'ROI{i}' for i in range(data.nROIs)])
y = pd.DataFrame(y,columns=['Stim-ID'])

#%%
print(X)
print(y)
# %%
# normalization of input data ? --> can be a good idea !!
normed = True
if normed:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X, y)


#%%
print(X)
print(y)
# %%
# visualize patterns

fig, AX = pt.figure(axes=(1, len(np.unique(y))), ax_scale=(2,.6))
for id, ax in zip(np.unique(y), AX):

    ax.bar(range(data.nROIs), 
           X[y['Stim-ID']==id].mean(axis=0), 
           yerr = X[y['Stim-ID']==id].std(axis=0), 
           color=None)

    pt.set_plot(ax, 
                ylabel='$\\Delta$F/F',
                xlabel='' if ax!=AX[-1] else 'ROIs',
                xticks_labels=[] if ax!=AX[-1] else None)
    
    pt.annotate(ax, id, (0,1), va='top')

    pt.annotate(ax, '(%i trials)' % np.sum(y['Stim-ID']==id), 
                (1,1), va='top', ha='right', fontsize='small')
pt.set_common_ylims(AX)

# %%
# train-test split using stratified strategy (to always have the same number of a given image in the train and test sets)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    random_state=42,
                                                    test_size=0.5,
                                                    stratify=y)
# check that the values are indeed balanced across classes:
y_test.value_counts()

# %%
# possiblity --> denoising the training set by averaging
averaging = False
if averaging:
    X_train = pd.DataFrame(\
        np.array([X_train[y_train['image-ID']==id].mean(axis=0)\
                        for id in y_train['image-ID'].unique()]),
                 columns=['ROI%i' % i for i in range(data.nROIs)])
    y_train = pd.DataFrame(\
        np.array([id\
                        for id in y_train['image-ID'].unique()]),
                 columns=['image-ID'])

# %%
fig, AX = pt.figure(axes=(1, len(np.unique(y))+1), 
                    ax_scale=(2,.6))

for x in np.array(X_test):
    AX[0].plot(range(data.nROIs), x, lw=0.1, color=None)
pt.annotate(AX[0], 'Test set: (%i single trials)' % len(X_test), 
            (0,1), va='top')
# AX[0].plot(range(data.nROIs), X[y['image-ID']==id].mean(axis=0), color=None)

for id, ax in zip(np.unique(y), AX[1:]):

    ax.bar(range(data.nROIs), 
           X_train[y_train['Stim-ID']==id].mean(axis=0), 
           yerr=X_train[y_train['Stim-ID']==id].std(axis=0), 
           color=None)
    pt.annotate(ax, f'Training Set: {id}', (0,1), va='top')
    pt.annotate(ax, '(%i samples)' % np.sum(y_train['Stim-ID']==id), 
                (1,1), va='top', ha='right', fontsize='small')

for ax in AX:
    pt.set_plot(ax, 
                ylabel='$\\Delta$F/F',
                xlabel='' if ax!=AX[-1] else 'ROIs',
                xticks_labels=[] if ax!=AX[-1] else None)

# %%
# Decoding is implemented as a Nearest-Neighbor Classifier
 
model = KNeighborsClassifier(n_neighbors=2)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

score = accuracy_score(y_test, y_pred)
chance = 1./len(y['Stim-ID'].unique())

fig, ax = pt.figure(ax_scale=(0.8,1.))
ax.bar([0], [score], label='single NN-decoder', color='grey')
ax.plot([-1, 1], [chance, chance], ':', label='chance')
ax.legend(loc=(1., 0.0), frameon=False)
pt.set_plot(ax, xticks=[], ylabel='accuracy', ylim = [0,0.8])
print("accuracy score : ", score)

# %%
