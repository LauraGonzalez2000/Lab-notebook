# %% [markdown]

# # Implementing a linear SVC Decoder of Neural Activity Patterns on Yann's data NDNF

# %%
import sys, os
import numpy as np
import pandas as pd
from pathlib import Path

os.chdir(r'C:\Users\laura.gonzalez\Programming\Lab-notebook\my_analysis')
sys.path.insert(0, os.getcwd())

sys.path += ['../physion/src'] # add src code directory for physion
import physion.utils.plot_tools as pt
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from physion.analysis.episodes.build import EpisodeData
from utils_.General_overview_episodes import compute_high_arousal_cond

# For decoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

pt.set_style('manuscript')

#%% FUNCTIONS
def plot_dataset_responses(X, y, nROIs):
    fig, AX = pt.figure(axes=(1, len(np.unique(y))+1), 
                        ax_scale=(4,.6))

    for x in np.array(X):
        AX[0].plot(range(nROIs), x, lw=0.2, color=None)
    pt.annotate(AX[0], 'Full train set', 
                    (0,1), va='top')
    pt.annotate(AX[0], '(%i trials)' % len(X), 
                    (1,1), va='top', ha='right', fontsize='small')

    for id, ax in zip(np.unique(y), AX[1:]):

        means = X[y['Stim-ID']==id].mean(axis=0)
        stds = X[y['Stim-ID']==id].std(axis=0)
        colors = ['red' if val > 0 else 'blue' for val in means]

        ax.bar(range(nROIs), 
            means, 
            yerr = stds, 
            color=colors, 
            width=1, 
            error_kw={'elinewidth':0.2})

        pt.set_plot(ax, 
                    ylabel='$\\Delta$F/F',
                    xlabel='' if ax!=AX[-1] else 'ROIs',
                    xticks_labels=[] if ax!=AX[-1] else None)
        pt.annotate(ax, id[0:10]+id[24:], (0,1), va='top') #for natural images
        pt.annotate(ax, '(%i trials)' % np.sum(y['Stim-ID']==id), 
                    (1,1), va='top', ha='right', fontsize='small')

    pt.set_common_ylims(AX)

    return fig, AX

def plot_train_test_responses(X_train, y_train, X_test, y, nROIs):
    fig, AX = pt.figure(axes=(1, len(np.unique(y))+1), 
                    ax_scale=(4,.6))

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

def plot_accuracy_comb(accuracies, chance):
    """
    accuracies: list or array of length 3
                [Any, Active, Rest]
    """
    labels = ['', 'Any', 'Active', 'Rest']
    colors = ['grey', 'orange', 'brown']

    fig, ax = pt.figure(figsize=(10, 10), 
                        wspace=1.4, hspace=1.8)
    pt.bar(ax=ax,y=accuracies, COLORS=colors)
    pt.set_plot(ax, xticks_labels=labels, ylabel='Accuracy', ylim = [0,0.8],
                xticks_rotation=90)
    pt.annotate(stuff=ax,
                s=f'{accuracies[0]}',
                xy=(0.05,accuracies[0]+0.05))
    pt.annotate(stuff=ax,
                s=f'{accuracies[1]}',
                xy=(0.35,accuracies[1]+0.05))
    pt.annotate(stuff=ax,
                s=f'{accuracies[2]}',
                xy=(0.65,accuracies[2]+0.08))
    pt.plt.axhline(chance, color='black', linestyle='--', label='chance level')
    #fig.savefig(f'decoding-between-accuracies.png', 
    #            format='png', dpi=600, transparent=True)
    
    return 0

def plot_confusion_matrix(cm, y, score, ax_scale=(0.8,1.), labels=['','','','','']):
    fig, ax = pt.figure(ax_scale=ax_scale)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=np.unique(y['Stim-ID']))
    disp.plot(ax=ax, 
              colorbar=False, 
              xticks_rotation='vertical')
    # Tick label font size
    ax.tick_params(axis='both', labelsize=15)

    ax.set_xticklabels(labels, 
                       rotation=90,
                       fontsize=15)
    ax.set_yticklabels(labels,
                       fontsize=15)
    ax.set_xlabel("Predicted label", fontsize=15)
    ax.set_ylabel("True label", fontsize=15)
    #fig.savefig(f'decoding-between-nosubstim-any.png', 
    #            format='png', dpi=300, transparent=True)
    ax.grid(False)
    return fig, ax

def plot_hist(scores, chance):
    fig, ax = pt.figure()
    pt.plt.hist(scores, bins=np.arange(0, 1.05, 0.05), edgecolor='grey', color="grey")
    pt.set_plot(ax, 
                ylabel='Number of\n sessions', 
                yticks = [1, 2, 3, 4])

    pt.plt.axvline(chance, color='black', linestyle='--', label='chance level')
    #fig.savefig(f'decoding-between-nosubstim-hist-None.png', 
    #            format='png', dpi=600, transparent=True)
    return fig, ax

def get_X_y_data(data, protocols, substim=True, normed=True, state=None):

    #  ---------------------------------------------- #
    # Transforming to the sklearn `X`, `y` variables
    #  ---------------------------------------------- #
    # X is the list of matrice response 
    # (rows = Ntrials, columns = NRois)
    # y is the label of all trials 
    # (rows = Ntrials, columns = 1(corresponding label))

    data.build_dFoF()
    print("n ROIS in this FOV ", data.nROIs)
    X_multiprotocols =  []
    y_multiprotocols = np.array([])
    for protocol in protocols: 
    
        ep = EpisodeData(data, 
                        protocol_name=protocol,
                        quantities=['dFoF', 'running'])
        
        behav_cond = compute_high_arousal_cond(ep, 
                            pre_stim = 1,
                            running_speed_threshold = 0.1, 
                            metric = "locomotion")
            
        t_window = 1.0
        t0 = max([0, ep.time_duration[0]-t_window])
        averaging_window = [t0, t0+t_window]
        averaging_window_cond = (ep.t>averaging_window[0]) &\
                                (ep.t<averaging_window[1])
        
        temp = ep.dFoF[:, :, averaging_window_cond]
        temp_act = temp[behav_cond,:,:]
        temp_rest = temp[~behav_cond,:,:]
        
        if state == None: 
            X = temp.mean(axis=2)
        elif state == 'act':
            X = temp_act.mean(axis=2)
        elif state == 'rest':
            X = temp_rest.mean(axis=2)
        

        if substim:
            #y = np.array([f"{protocol}-{id}"for id in getattr(ep, 'index')])
            y = np.array([f"{protocol}-{id}"for id in getattr(ep, 'Image-ID')])
            #y = np.array([f"{protocol}-{id}-{c}" for id,c in zip(getattr(ep, 'Image-ID'),getattr(ep, 'contrast'))])

        else: 
            y = np.repeat(protocol, temp.shape[0])

        
        if state == None: 
            y = y
        elif state == 'act':
            y = y[behav_cond]
        elif state == 'rest':
            y = y[~behav_cond]

        X_multiprotocols.append(X)
        y_multiprotocols = np.append(y_multiprotocols, y)

    X_multiprotocols_flatten = np.concatenate(X_multiprotocols)

    X = pd.DataFrame(X_multiprotocols_flatten,
                    columns=[f'ROI{i}' for i in range(data.nROIs)])
    y = pd.DataFrame(y_multiprotocols,columns=['Stim-ID'])

    if normed:
        scaler = StandardScaler()
        X = scaler.fit_transform(X, y)
        
    return X, y

def get_X_y_data_sessions(SESSIONS, protocols, index = None, substim=True, normed=True, state=None):
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
        print("n ROIS in this FOV ", data.nROIs)
        X_multiprotocols =  []
        y_multiprotocols = np.array([])
        nROIs_tot = data.nROIs
        for protocol in protocols: 
        
            ep = EpisodeData(data, 
                            protocol_name=protocol,
                            quantities=['dFoF', 'running'])

            print(ep.running)
            
            behav_cond = compute_high_arousal_cond(ep, 
                              pre_stim = 1,
                              running_speed_threshold = 0.1, 
                              metric = "locomotion")
                
            print("behav condition", len(behav_cond))

            t_window = 1.0
            t0 = max([0, ep.time_duration[0]-t_window])
            averaging_window = [t0, t0+t_window]
            averaging_window_cond = (ep.t>averaging_window[0]) &\
                                    (ep.t<averaging_window[1])
            
            temp = ep.dFoF[:, :, averaging_window_cond]
            temp_act = temp[behav_cond,:,:]
            temp_rest = temp[~behav_cond,:,:]
           
            if state == None: 
                X = temp.mean(axis=2)
            elif state == 'act':
                X = temp_act.mean(axis=2)
            elif state == 'rest':
                X = temp_rest.mean(axis=2)
            

            if substim:
                y = np.array([f"{protocol}-{id}"for id in getattr(ep, 'index')])
            else: 
                y = np.repeat(protocol, temp.shape[0])

            
            if state == None: 
                y = y
            elif state == 'act':
                y = y[behav_cond]
            elif state == 'rest':
                y = y[~behav_cond]


            X_multiprotocols.append(X)
            y_multiprotocols = np.append(y_multiprotocols, y)

        X_multiprotocols_flatten = np.concatenate(X_multiprotocols)

        print(len(X_multiprotocols_flatten))
        print(len(y_multiprotocols))

        X = pd.DataFrame(X_multiprotocols_flatten,
                        columns=[f'ROI{i}' for i in range(data.nROIs)])
        y = pd.DataFrame(y_multiprotocols,columns=['Stim-ID'])

        if normed:
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
                                quantities=['dFoF', 'running'])
                
                behav_cond = compute_high_arousal_cond(ep, 
                              pre_stim = 1,
                              running_speed_threshold = 0.1, 
                              metric = "locomotion")
                
                print("behav condition", len(behav_cond))

                t_window = 1.0
                t0 = max([0, ep.time_duration[0]-t_window])
                averaging_window = [t0, t0+t_window]
                averaging_window_cond = (ep.t>averaging_window[0]) &\
                                        (ep.t<averaging_window[1])
                
                temp = ep.dFoF[:, :, averaging_window_cond]
                temp_act = temp[behav_cond,:,:]
                temp_rest = temp[~behav_cond,:,:]
            
                if state == None: 
                    X = temp.mean(axis=2)
                elif state == 'act':
                    X = temp_act.mean(axis=2)
                elif state == 'rest':
                    X = temp_rest.mean(axis=2)

                

                if substim:
                    y = np.array([f"{protocol}-{id}"for id in getattr(ep, 'index')])
                else: 
                    y = np.repeat(protocol, temp.shape[0])
                
                if state == None: 
                    y = y
                elif state == 'act':
                    y = y[behav_cond]
                elif state == 'rest':
                    y = y[~behav_cond]


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

            print("ref : ", X_multiprotocols_flatten_shape_ref)
            print("file :", X_multiprotocols_flatten.shape[0])
            if X_multiprotocols_flatten.shape[0]==X_multiprotocols_flatten_shape_ref:
                X_multiprotocols_s.append(X_multiprotocols_flatten)
                nROIs_tot += data.nROIs

        X_multiprotocols_s_flatten = np.concatenate(X_multiprotocols_s, axis=1)

        X = pd.DataFrame(X_multiprotocols_s_flatten,
                        columns=[f'ROI{i}' for i in range(nROIs_tot)])
        y = pd.DataFrame(y_multiprotocols_ref, 
                        columns=['Stim-ID'])

        if normed:
            scaler = StandardScaler()
            X = scaler.fit_transform(X, y)
    
    #y = y['Stim-ID']

    return X, y, behav_cond, nROIs_tot

# %%
# 5 natural images 
# 5 Natural Images, 20 trials each (in larger protocol)
#datafolder = os.path.join(Path("C:/"),'Users','laura.gonzalez', 'DATA', 'In_Vivo_experiments','NDNF-old-protocol', 'NDNF-WT-Dec-2022','NWBs_rebuilt')
#SESSIONS = scan_folder_for_NWBfiles(datafolder)
#labels = ["Img 1", "Img 2", "Img 3", "Img 4", "Img 5"] 
#protocol = 'Natural-Images-4-repeats'

# 10 Natural Images - 2 contrasts - 10 trials each 
#NDNF cells (1 session)
datafolder = os.path.join(Path("E:/"), 'DATA', 'In_Vivo_experiments','10NatIm2contrasts', 'NDNF-Cre', 'NWBs') #1 file
SESSIONS = scan_folder_for_NWBfiles(datafolder)
# filenames = [\
#     '2026_09_01-15-02-47',
#     ]
# SESSIONS = scan_folder_for_NWBfiles(datafolder, from_filenames=filenames)

#Pyr cells (3+3 sessions)
#datafolder = os.path.join(Path("E:/"), 'DATA', 'In_Vivo_experiments','10NatIm2contrasts', 'C57BL6J', 'NWBs') #3 files
#datafolder = os.path.join(Path("E:/"), 'DATA', 'In_Vivo_experiments','10NatIm2contrasts', 'Thy1GCaMP6s-NDNF-Cre', 'NWBs') #3 files
#SESSIONS = scan_folder_for_NWBfiles(datafolder)
# filenames = [\
#     '2026_08_31-15-08-11',
#     '2026_10_05-15-07-46',
#     '2026_10_06-11-07-43',
#     ]
# SESSIONS = scan_folder_for_NWBfiles(datafolder, from_filenames=filenames)
labels = ["Img 1 c=0.5", "Img 1 c=1   ", 
          "Img 2 c=0.5", "Img 2 c=1   ",
          "Img 3 c=0.5", "Img 3 c=1   ",
          "Img 4 c=0.5", "Img 4 c=1   ",
          "Img 5 c=0.5", "Img 5 c=1   ",
          "Img 6 c=0.5", "Img 6 c=1   ",
          "Img 7 c=0.5", "Img 7 c=1   ",
          "Img 8 c=0.5", "Img 8 c=1   ",
          "Img 9 c=0.5", "Img 9 c=1   ",
          "Img 10 c=0.5", "Img 10 c=1   "] 
labels = ["Img 1","Img 2","Img 3","Img 4","Img 5", 
         "Img 6", "Img 7", "Img 8", "Img 9" , "Img 10"] 


############################################################################
###############  DECODING WITHIN Natural Images ############################
############################################################################

#%% [markdown]
## Decoding within Natural Images


#%% [markdown]
## Linear SVC
#%%
index = 0
filename = SESSIONS['files'][index]
data = Data(filename, verbose=True, with_visual_stim=True)

protocol ='10NaturalImages-2contrasts-10repeats'

ep = EpisodeData(data, 
                protocol_name=protocol,
                quantities=['dFoF', 'running'])

y = np.array([f"{protocol}-{id}-{c}" for id,c in zip(getattr(ep, 'Image-ID'),getattr(ep, 'contrast'))])
#%%
index = 0
filename = SESSIONS['files'][index]
data = Data(filename, verbose=True, with_visual_stim=True)

protocol ='10NaturalImages-2contrasts-10repeats'

X, y = get_X_y_data(data, protocols = [protocol], substim=True, normed=True, state=None)

# 1. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y,  
                        random_state=42, 
                        test_size=0.2, #80 trials for training, 20 for test
                        stratify=y) 

# Extract the 1D target arrays directly for scikit-learn compatibility
y_train_labels = y_train['Stim-ID']
y_test_labels = y_test['Stim-ID']

# 2. Build a Pipeline (Scales -> Reduces Dimensions -> Classifies)
pipeline = Pipeline([('scaler', StandardScaler()), 
                    ('pca', PCA(random_state=42)), 
                    ('classifier', LinearSVC(class_weight='balanced', 
                                            max_iter=10000, 
                                            random_state=42))])

# 3. Grid Search for Hyperparameters
param_grid = {'pca__n_components': [5, 10, 20, 30, 40, 50, 60], # Tests how many ROIs to keep principal components
            'classifier__C': [0.01, 0.1, 1, 10]}  # Tests different regularization strengths for SVM

gs = GridSearchCV(pipeline, 
                param_grid, 
                scoring='accuracy', 
                cv=5)
gs.fit(X_train, y_train_labels)

print("Best hyperparameters: ", gs.best_params_)
print("Cross-validation accuracy: ", gs.best_score_)

# 4. Test the optimal model on the untouched test set
best_model = gs.best_estimator_
y_pred = best_model.predict(X_test)

# Plot accuracy
score = accuracy_score(y_test_labels, y_pred)
chance = 1. / len(np.unique(y['Stim-ID']))
print(f"Test Set Accuracy: {score:.2f} (Chance: {chance:.2f})")
plot_accuracy(score, chance) 
cm = confusion_matrix(y_test_labels, y_pred, 
                    labels=np.unique(y['Stim-ID']),  
                    normalize='true') 

fig, ax = plot_confusion_matrix(cm, y, score, ax_scale=(10,10), labels=labels)
# %%
