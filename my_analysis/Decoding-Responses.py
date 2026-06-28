# %% [markdown]
# # Implementing a Nearest-Neighbor Decoder of Neural Activity Patterns on Yann's data NDNF
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

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#%% FUNCTIONS
def plot_dataset_responses(X, y, nROIs):
    fig, AX = pt.figure(axes=(1, len(np.unique(y))), ax_scale=(2,.6))
    for id, ax in zip(np.unique(y), AX):

        ax.bar(range(nROIs), 
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

def plot_train_test_responses(X_train, y_train, X_test, y, nROIs):
    fig, AX = pt.figure(axes=(1, len(np.unique(y))+1), 
                    ax_scale=(2,.6))

    for x in np.array(X_test):
        AX[0].plot(range(nROIs), x, lw=0.1, color=None)
    pt.annotate(AX[0], 'Test set: (%i single trials)' % len(X_test), 
                (0,1), va='top')
    
    for id, ax in zip(np.unique(y), AX[1:]):

        ax.bar(range(nROIs), 
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

def plot_accuracy(score, chance):
    fig, ax = pt.figure(ax_scale=(0.8,1.))
    ax.bar([0], [score], label='single NN-decoder', color='grey')
    ax.plot([-1, 1], [chance, chance], ':', label='chance')
    ax.legend(loc=(1., 0.0), frameon=False)
    pt.set_plot(ax, xticks=[], ylabel='accuracy', ylim = [0,0.8])
    ax.set_title(f'Accuracy = {score:.2f}')
    return 0

def plot_confusion_matrix(cm, y, score, ax_scale=(0.8,1.)):
    fig, ax = pt.figure(ax_scale=ax_scale)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=np.unique(y['Stim-ID']))
    disp.plot(ax=ax, colorbar=False, 
                                  xticks_rotation='vertical')
    ax.set_title(f'Confusion Matrix\nAccuracy = {score:.2f}')

def get_X_y_data(SESSIONS, protocols, index = None, substim=True, normed=True):
    #best to exclude files where not all trials were done, 
    # or to truncate the the minimum common trials all files???

    
    #  ---------------------------------------------- #
    # Transforming to the sklearn `X`, `y` variables
    #  ---------------------------------------------- #
    # X is the list of matrice response 
    # (rows = Ntrials, columns = NRois)
    # y is the label of all trials 
    # (rows = Ntrials, columns = 1(corresponding label))

    X_multiprotocols_s =  []
    y_multiprotocols_ref = [] 
    nROIs_tot = 0
    if index != None : 
        filename = SESSIONS['files'][index]
        data = Data(filename)
        data.build_dFoF()
        X_multiprotocols =  []
        y_multiprotocols = np.array([])
        nROIs_tot = data.nROIs
        for protocol in protocols: 
        
            ep = EpisodeData(data, 
                            protocol_name=protocol,
                            quantities=['dFoF'])

            t_window = 1.0
            t0 = max([0, ep.time_duration[0]-t_window])
            averaging_window = [t0, t0+t_window]
            averaging_window_cond = (ep.t>averaging_window[0]) &\
                                    (ep.t<averaging_window[1])

            X = ep.dFoF[:, :, averaging_window_cond].mean(axis=2)
            if substim:
                y = np.array([f"{protocol}-{id}"for id in getattr(ep, 'index')])
            else: 
                y = np.repeat(protocol, X.shape[0])

            X_multiprotocols.append(X)
            y_multiprotocols = np.append(y_multiprotocols, y)

        X_multiprotocols_flatten = np.concatenate(X_multiprotocols)

        print(len(X_multiprotocols_flatten))
        print(len(y_multiprotocols))

        X = pd.DataFrame(X_multiprotocols_flatten,
                        columns=[f'ROI{i}' for i in range(data.nROIs)])
        y = pd.DataFrame(y_multiprotocols,columns=['Stim-ID'])

        if normed:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X = scaler.fit_transform(X, y)
            
    else :    
        for i in range(len(SESSIONS['files'])):
            filename = SESSIONS['files'][i]
            data = Data(filename)
            data.build_dFoF()

            X_multiprotocols =  []
            y_multiprotocols = [] #np.array([])

            for protocol in protocols: 
            
                ep = EpisodeData(data, 
                                protocol_name=protocol,
                                quantities=['dFoF'])

                t_window = 1.0
                t0 = max([0, ep.time_duration[0]-t_window])
                averaging_window = [t0, t0+t_window]
                averaging_window_cond = (ep.t>averaging_window[0]) &\
                                        (ep.t<averaging_window[1])

                X = ep.dFoF[:, :, averaging_window_cond].mean(axis=2)
                X_multiprotocols.append(X)
                
                if substim:
                    y_ref = np.array([f"{protocol}-{id}"for id in getattr(ep, 'index')])
                    y_multiprotocols.append(y_ref)
                else: 
                    y_ref = np.repeat(protocol, X.shape[0])
                    y_multiprotocols.append(y_ref)

            X_multiprotocols_flatten = np.concatenate(X_multiprotocols)
            y_multiprotocols_flatten = np.concatenate(y_multiprotocols)

            if i ==0 :
                X_multiprotocols_flatten_shape_ref = X_multiprotocols_flatten.shape[0]
                y_multiprotocols_ref = y_multiprotocols_flatten

            if X_multiprotocols_flatten.shape[0]==X_multiprotocols_flatten_shape_ref:
                X_multiprotocols_s.append(X_multiprotocols_flatten)
                nROIs_tot += data.nROIs

        X_multiprotocols_s_flatten = np.concatenate(X_multiprotocols_s, axis=1)

        X = pd.DataFrame(X_multiprotocols_s_flatten,
                        columns=[f'ROI{i}' for i in range(nROIs_tot)])
        y = pd.DataFrame(y_multiprotocols_ref, 
                        columns=['Stim-ID'])

        if normed:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X = scaler.fit_transform(X, y)

    return X, y, nROIs_tot

# %%
datafolder = os.path.join(Path("E:/"), 'DATA', 'In_Vivo_experiments','NDNF-old-protocol', 'NDNF-WT-Dec-2022','NWBs_rebuilt')
SESSIONS = scan_folder_for_NWBfiles(datafolder)

############################################################################
###############  DECODING WITHIN EACH STIMULUS #############################
############################################################################
#%% Per FOV
#protocol = 'static-patch' #2 angles, 10 trials each (40 trials in total)
#protocol = 'drifting-gratings' #4 directions, 10 trials each (40 trials in total)
protocol = 'Natural-Images-4-repeats' #5 images, 20 trials each (100 trials in total)

substim = True #should always be true in this case
normed = True # normalization of input data ? --> can be a good idea !!
averaging = False #possiblity --> denoising the training set by averaging


for i in range(len(SESSIONS['files'])):
    X, y , nROIs = get_X_y_data(SESSIONS, protocols = [protocol], index=i, substim=substim, normed=normed )

    plot_dataset_responses(X=X, y=y, nROIs=nROIs)

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        random_state=42,
                                                        test_size=0.5,
                                                        stratify=y)
    y_test.value_counts() # check that the values are indeed balanced across classes:
    
    if averaging:
        X_train = pd.DataFrame(\
            np.array([X_train[y_train['image-ID']==id].mean(axis=0)\
                            for id in y_train['image-ID'].unique()]),
                    columns=['ROI%i' % i for i in range(nROIs)])
        y_train = pd.DataFrame(\
            np.array([id\
                            for id in y_train['image-ID'].unique()]),
                    columns=['image-ID'])

    plot_train_test_responses(X_train, y_train, X_test, y, nROIs=nROIs)
    
    # Decoding is implemented as a Nearest-Neighbor Classifier
    model = KNeighborsClassifier(n_neighbors=2)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Plot accuracy
    score = accuracy_score(y_test, y_pred)
    chance = 1./len(y['Stim-ID'].unique())
    plot_accuracy(score, chance)

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred,
                          labels=np.unique(y['Stim-ID']))
    plot_confusion_matrix(cm, y, score)
    


#%% Taking all cells : 

#protocol = 'static-patch' #2 angles, 10 trials each (20 trials in total)
#protocol = 'drifting-gratings' #4 directions, 10 trials each (40 trials in total)
protocol = 'Natural-Images-4-repeats' #5 images, 20 trials each (100 trials in total)

substim = True #(should always be true here because we are only checking substim within a stim)
normed = True # normalization of input data ? --> can be a good idea !!
averaging = False #possiblity --> denoising the training set by averaging

X, y, nROIs = get_X_y_data(SESSIONS, protocols= [protocol], index = None, substim=substim, normed=normed)

plot_dataset_responses(X, y, nROIs=nROIs)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    random_state=42,
                                                    test_size=0.5,
                                                    stratify=y)
# check that the values are indeed balanced across classes:
#y_test.value_counts()

if averaging:
    X_train = pd.DataFrame(\
        np.array([X_train[y_train['image-ID']==id].mean(axis=0)\
                        for id in y_train['image-ID'].unique()]),
                columns=['ROI%i' % i for i in range(nROIs)])
    y_train = pd.DataFrame(\
        np.array([id\
                        for id in y_train['image-ID'].unique()]),
                columns=['image-ID'])

plot_train_test_responses(X_train, y_train, X_test, y, nROIs=nROIs)

# Decoding is implemented as a Nearest-Neighbor Classifier
model = KNeighborsClassifier(n_neighbors=2)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Plot accuracy
score = accuracy_score(y_test, y_pred)
chance = 1./len(y['Stim-ID'].unique())
plot_accuracy(score, chance)

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred,
                        labels=np.unique(y['Stim-ID']))
plot_confusion_matrix(cm, y, score)

#%%
############################################################################
###############  DECODING BETWEEN STIMULI ##################################
############################################################################
#%% Per FOV
protocols = ['moving-dots', 
             'random-dots',
             'static-patch',
             'looming-stim', 
             'Natural-Images-4-repeats',
             'drifting-gratings']

normed = True # normalization of input data ? --> can be a good idea !!
averaging = False # possiblity --> denoising the training set by averaging
substim = False # model decoding between stimuli (False) or between stimuli and substimuli
index = 4

X, y, nROIs = get_X_y_data(SESSIONS, protocols, index=index, substim=substim, normed=normed)

# visualize patterns
plot_dataset_responses(X, y, nROIs=nROIs)

# train-test split using stratified strategy 
# (to always have the same number of a given image 
# in the train and test sets)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    random_state=42,
                                                    test_size=0.5,
                                                    stratify=y)
# check that the values are indeed balanced across classes:
#y_test.value_counts()

if averaging:
    X_train = pd.DataFrame(\
        np.array([X_train[y_train['image-ID']==id].mean(axis=0)\
                        for id in y_train['image-ID'].unique()]),
                columns=['ROI%i' % i for i in range(nROIs)])
    y_train = pd.DataFrame(\
        np.array([id\
                        for id in y_train['image-ID'].unique()]),
                columns=['image-ID'])

plot_train_test_responses(X_train, y_train, X_test, y, nROIs = nROIs)

# Decoding is implemented as a Nearest-Neighbor Classifier
model = KNeighborsClassifier(n_neighbors=2)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Plot accuracy
score = accuracy_score(y_test, y_pred)
if substim:
    chance = 1./len(y['Stim-ID'].unique())
else: 
    chance = 1./len(protocols)    
plot_accuracy(score, chance)

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred,
                        labels=np.unique(y['Stim-ID']))
plot_confusion_matrix(cm, y, score, ax_scale=(5,5))


#%% Taking all cells : ###########################
##################################################

protocols = ['moving-dots', 
             'random-dots',
             'static-patch',
             'looming-stim', 
             'Natural-Images-4-repeats',
             'drifting-gratings']

normed = True # normalization of input data ? --> can be a good idea !!
averaging = False # possiblity --> denoising the training set by averaging
substim = False # model decoding between stimuli (False) or between stimuli and substimuli


X, y, nROIs = get_X_y_data(SESSIONS, protocols, substim=True, normed=True)

# visualize patterns
plot_dataset_responses(X, y, nROIs=nROIs)

# train-test split using stratified strategy 
# (to always have the same number of a given image 
# in the train and test sets)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    random_state=42,
                                                    test_size=0.5,
                                                    stratify=y)
# check that the values are indeed balanced across classes:
#y_test.value_counts()

if averaging:
    X_train = pd.DataFrame(np.array([X_train[y_train['image-ID']==id].mean(axis=0)\
                            for id in y_train['image-ID'].unique()]),
                            columns=['ROI%i' % i for i in range(nROIs)])
    y_train = pd.DataFrame(np.array([id\
                            for id in y_train['image-ID'].unique()]),
                            columns=['image-ID'])

plot_train_test_responses(X_train, y_train, X_test, y, nROIs = nROIs)

# Decoding is implemented as a Nearest-Neighbor Classifier
model = KNeighborsClassifier(n_neighbors=2)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Plot accuracy
score = accuracy_score(y_test, y_pred)
if substim:
    chance = 1./len(y['Stim-ID'].unique())
else: 
    chance = 1./len(protocols)    
plot_accuracy(score, chance)

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred,
                        labels=np.unique(y['Stim-ID']))
plot_confusion_matrix(cm, y, score, ax_scale=(5,5))
