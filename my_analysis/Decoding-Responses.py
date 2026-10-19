# %% [markdown]
# # Implementing a Nearest-Neighbor Decoder of Neural Activity Patterns on Yann's data NDNF
#

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
from sklearn.neighbors import KNeighborsClassifier
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
datafolder = os.path.join(Path("C:/"),'Users','laura.gonzalez', 'DATA', 
                          'In_Vivo_experiments','NDNF-old-protocol', 
                          'NDNF-WT-Dec-2022','NWBs_rebuilt')
SESSIONS = scan_folder_for_NWBfiles(datafolder)

############################################################################
###############  DECODING WITHIN EACH STIMULUS #############################
############################################################################

#%% [markdown]

## Decoding within each stimulus

## Main example with Natural Images

# 5 Natural Images, 20 trials each

#%%
index = 4
filename = SESSIONS['files'][index]
data = Data(filename, verbose=True, with_visual_stim=True)
protocol = 'Natural-Images-4-repeats'
X, y = get_X_y_data(data, protocols = [protocol], substim=True, normed=True, state=None)
#%%
#plot responses 
nROIs = data.nROIs
plot_dataset_responses(X=X, y=y, nROIs=nROIs)
#plot average responses
X_ = pd.DataFrame(\
    np.array([X[y['Stim-ID']==id].mean(axis=0)\
                    for id in y['Stim-ID'].unique()]),
            columns=['ROI%i' % i for i in range(nROIs)])
y_ = pd.DataFrame(\
    np.array([id\
                    for id in y['Stim-ID'].unique()]),
            columns=['Stim-ID'])
plot_dataset_responses(X=X_, y=y_, nROIs=nROIs)
#%%
#splitting train and test sets : 
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    random_state=42,
                                                    test_size=0.2,
                                                    stratify=y)
#y_test.value_counts() #check 
#%%
#plot responses
averaging = False #possiblity --> denoising the training set by averaging
if averaging:
    X_train = pd.DataFrame(\
        np.array([X_train[y_train['Stim-ID']==id].mean(axis=0)\
                        for id in y_train['Stim-ID'].unique()]),
                columns=['ROI%i' % i for i in range(nROIs)])
    y_train = pd.DataFrame(\
        np.array([id\
                        for id in y_train['Stim-ID'].unique()]),
                columns=['Stim-ID'])

plot_train_test_responses(X_train, y_train, X_test, y, nROIs=nROIs)
plot_dataset_responses(X=X_train, y=y_train, nROIs=nROIs)

#%% [markdown]

# Choosing hyperparameters <br>
# k
#%%
k_values = list(range(1, 31)) # k from 1 to 30
acc_scores = []

for k in k_values:
  KNN = KNeighborsClassifier(n_neighbors=k)

  scores = cross_val_score(KNN, X_train, y_train,
                           scoring='accuracy', cv=5)

  acc_scores.append(np.mean(scores))

pt.plt.style.use("ggplot")
pt.plt.plot(k_values, acc_scores, marker='o', color='blue')
pt.plt.xlabel("Neighbors - K Value")
pt.plt.ylabel("Accuracy")
pt.plt.title("Effect of K")

#%%
# using grid search
k_values = list(range(1, 31)) # From 1 to 30
KNN = KNeighborsClassifier()
hyperparameter_space = {'n_neighbors':k_values}
gs = GridSearchCV(KNN, param_grid=hyperparameter_space,
                  scoring='accuracy', cv=5)
gs.fit(X_train, y_train)
print("Best value of K: ", gs.best_params_)
print("Mean CV accuracy of best K-value: ", gs.best_score_)

#%%
#with NumPy argmax() function

k_values = list(range(1, 31)) # From 1 to 30
acc_scores = []
for k in k_values:
  KNN = KNeighborsClassifier(n_neighbors=k)
  scores = cross_val_score(KNN, X_train, y_train,
                           scoring='accuracy', cv=5)
  acc_scores.append(np.mean(scores))

# Using NumPy argmax()
best_k_index = np.argmax(acc_scores)
best_k = k_values[best_k_index]

print("Best k:", best_k)
print("Score:", acc_scores[best_k_index])

#%%
#check the validation curve
# Importing Yellowbrick's validation curve visualizer
from yellowbrick.model_selection import ValidationCurve

k_values = list(range(1, 31)) # From 1 to 30

# Creating the validation curve
viz = ValidationCurve(KNeighborsClassifier(), 
                      param_name="n_neighbors",
                      param_range=k_values, 
                      cv=5, scoring="accuracy")

viz.fit(X_train, y_train)

# Saving plot in PNG format
viz.show(outpath="Validation_Curve.png", dpi=80)

#%%
#weighted distances + visual identification
k_values = list(range(1, 31)) # From 1 to 30
acc_scores = []

for k in k_values:
  KNN = KNeighborsClassifier(n_neighbors=k, weights='distance')
  scores = cross_val_score(KNN, X_train, y_train,
                           scoring='accuracy', cv=5)
  acc_scores.append(np.mean(scores))

pt.plt.style.use("ggplot")
pt.plt.plot(k_values, acc_scores, marker='o', color='blue')
pt.plt.xlabel("Neighbors - K Value")
pt.plt.ylabel("Accuracy")
pt.plt.title("Effect of k with weighted distances")

#%% [markdown]
## Test optimal model!
#%%
N_neighbor = 4
# Decoding is implemented as a Nearest-Neighbor Classifier
model = KNeighborsClassifier(n_neighbors=N_neighbor,
                                 metric = "minkowski",
                                 weights='distance')
                             #class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Plot accuracy
score = accuracy_score(y_test, y_pred)
chance = 1./len(y['Stim-ID'].unique())
plot_accuracy(score, chance)

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred,
                        labels=np.unique(y['Stim-ID']), 
                        normalize='true')
labels = ["Img 1", "Img 2", "Img 3", "Img 4", "Img 5"]
fig, ax = plot_confusion_matrix(cm, y, score, ax_scale=(10,10), labels=labels)



#############################################################################
#%% [markdown]
## Linear SVC
#%%
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
labels = ["Img 1", "Img 2", "Img 3", "Img 4", "Img 5"] 
fig, ax = plot_confusion_matrix(cm, y, score, ax_scale=(10,10), labels=labels)



#%% ############################################################################################################
# divide by state!! Train normal model, then 

X, y = get_X_y_data(data, protocols = [protocol], substim=True, normed=True, state=None)
#%%
print(y.value_counts())
#%%
ep = EpisodeData(data, 
                protocol_name=protocol,
                quantities=['dFoF', 'running'])
behav_cond = compute_high_arousal_cond(ep, 
                            pre_stim = 1,
                            running_speed_threshold = 0.1, 
                            metric = "locomotion")

state_strings = np.where(behav_cond, "act", "rest")
combined_labels = y['Stim-ID'].astype(str) + "_" + state_strings
#%%
# 1. Train/Test Split
X_train, X_test, y_train, y_test, b_train, b_test = train_test_split(X, y,  behav_cond,
                        random_state=42, 
                        test_size=0.2, #80 trials for training, 20 for test
                        stratify=combined_labels) 
#%%
print(y_train)
#print(y_test)
#print(b_test)
#%%
# Extract the 1D target arrays directly for scikit-learn compatibility
y_train_labels = y_train['Stim-ID']
y_test_labels = y_test['Stim-ID']

# 2. Build a Pipeline (Scales -> Reduces Dimensions -> Classifies)
pipeline = Pipeline([('scaler', StandardScaler()), 
                     ('pca', PCA(random_state=42)), 
                     #('classifier', KNeighborsClassifier(n_neighbors=N_neighbor,
                     #            metric = "minkowski",
                     #            weights='distance'))])
                     ('classifier', LinearSVC(class_weight='balanced', 
                                              max_iter=10000, 
                                              random_state=42, 
                                              penalty='l2'))])

# 3. Grid Search for Hyperparameters
param_grid = {'pca__n_components': [5, 10, 20, 30, 40, 50, 60, 70], # Tests how many ROIs to keep principal components
              #'classifier__n_neighbors': [1,2,3,4,5,10,11,12,20]}
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
labels = ["Img 1", "Img 2", "Img 3", "Img 4", "Img 5"] 
fig, ax = plot_confusion_matrix(cm, y, score, ax_scale=(10,10), labels=labels)

best_model.named_steps['classifier'].coef_


#%%
import matplotlib.pyplot as plt

# Extract the coefficients (should be a 2D array of shape 5 x 40)
coefs = best_model.named_steps['classifier'].coef_
n_categories, n_features = coefs.shape

# Generate feature names (or use X_test.columns if available)
if hasattr(X_test, 'columns'):
    feature_names = X_test.columns
else:
    feature_names = [f"ROI_{i+1}" for i in range(n_features)]

# Create a figure with 5 rows and 1 column, sharing the X-axis
fig, axes = plt.subplots(nrows=n_categories, ncols=1, figsize=(16, 12), sharex=True)
fig.suptitle('Coefficients across 5 Categories and 40 ROIs', fontsize=18)

# Loop through each category to create its respective row plot
for i in range(n_categories):
    ax = axes[i]
    category_coefs = coefs[i]
    
    # Assign green for positive coefficients and red for negative
    colors = ['#5A9F5F' if c > 0 else '#C95A5A' for c in category_coefs]
    
    # Plot the bar chart for the current category
    ax.bar(feature_names, category_coefs, color=colors)
    
    # Reference line at zero
    ax.axhline(0, color='black', linestyle='--', linewidth=1.2)
    
    # Subplot formatting
    ax.set_title(f'Category {i+1} Coefficients', fontsize=14)
    ax.set_ylabel('Coefficient Value', fontsize=11)
    ax.grid(axis='y', linestyle='-', alpha=0.4)
    
    # Reduce whitespace on the far left and right ends of the bars
    ax.margins(x=0.01) 

# Format the shared X-axis labels on the bottom plot
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.xlabel('Feature Name (ROIs)', fontsize=14)

# Adjust layout to prevent overlapping titles and labels
plt.tight_layout()
plt.show()
#%% ############################################################################################################
#%% Per FOV
#%%
#Example file
#protocol = 'static-patch' #2 angles, 10 trials each (40 trials in total)
#protocol = 'drifting-gratings' #4 directions, 10 trials each (40 trials in total)
protocol = 'Natural-Images-4-repeats' #5 images, 20 trials each (100 trials in total)

substim = True #should always be true in this case
normed = True # normalization of input data ? --> can be a good idea !!
averaging = False #possiblity --> denoising the training set by averaging
state = None #'rest'#'act'#None#'rest'
N_neighbor = 5

i = 4
X, y , behav, nROIs = get_X_y_data_sessions(SESSIONS, 
                                   protocols = [protocol], 
                                   index=i, substim=substim, 
                                   normed=normed, state=state)

#%%
plot_dataset_responses(X=X, y=y, nROIs=nROIs)
#%%
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
model = KNeighborsClassifier(n_neighbors=N_neighbor,
                                 metric = "minkowski")
                             #class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Plot accuracy
score = accuracy_score(y_test, y_pred)
chance = 1./len(y['Stim-ID'].unique())
plot_accuracy(score, chance)

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred,
                        labels=np.unique(y['Stim-ID']), 
                        normalize='true')
labels = ["Img 1", "Img 2", "Img 3", "Img 4", "Img 5"]
fig, ax = plot_confusion_matrix(cm, y, score, ax_scale=(10,10), labels=labels)


#%%
plot_accuracy_comb(accuracies=[0.50, 0.52, 0.65], chance=0.2)
#%% each file one by one

#protocol = 'static-patch' #2 angles, 10 trials each (40 trials in total)
#protocol = 'drifting-gratings' #4 directions, 10 trials each (40 trials in total)
protocol = 'Natural-Images-4-repeats' #5 images, 20 trials each (100 trials in total)

substim = True #should always be true in this case
normed = True # normalization of input data ? --> can be a good idea !!
averaging = False #possiblity --> denoising the training set by averaging
state = None#'act'

scores = []
for i in range(len(SESSIONS['files'])):
    X, y , behavcond, nROIs = get_X_y_data(SESSIONS, 
                                           protocols = [protocol], 
                                           index=i, substim=substim, 
                                           normed=normed, state=state )

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

    scores.append(score)
    
#%%
plot_hist(scores, chance)
#%% Taking all cells : 

#protocol = 'static-patch' #2 angles, 10 trials each (20 trials in total)
#protocol = 'drifting-gratings' #4 directions, 10 trials each (40 trials in total)
protocol = 'Natural-Images-4-repeats' #5 images, 20 trials each (100 trials in total)

substim = True #(should always be true here because we are only checking substim within a stim)
normed = True # normalization of input data ? --> can be a good idea !!
averaging = False #possiblity --> denoising the training set by averaging

X, y, behavcond, nROIs  = get_X_y_data(SESSIONS, protocols= [protocol], index = None, substim=substim, normed=normed)

#%%
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
#example file

protocols = ['moving-dots', 
             'random-dots',
             'static-patch',
             'looming-stim', 
             'Natural-Images-4-repeats',
             'drifting-gratings']

normed = True # normalization of input data ? --> can be a good idea !!
averaging = False # possiblity --> denoising the training set by averaging
substim = False # model decoding between stimuli (False) or between stimuli and substimuli
state = 'rest'#'rest'#'rest'#'rest'
index = 4

X, y_df, behav, nROIs = get_X_y_data(SESSIONS, protocols, index=index, substim=substim, normed=normed, state=state)

# visualize patterns
#plot_dataset_responses(X, y, nROIs=nROIs)

y = y_df['Stim-ID']
print(y.shape)

seed = 2
scores = []
N_neighbor = 5

for seed in range(1000):

    np.random.seed(seed)

    # initialize a new training/test set
    X_train, y_train, train_indices = [], [], np.zeros(len(y), dtype=bool)
    # loop over different stim
    for yval in np.unique(y):
        # randomly pick an index
        index = np.random.choice(np.flatnonzero(yval==y), N_neighbor)
        # we add this to the training set
        train_indices[index] = True

    model = KNeighborsClassifier(n_neighbors=N_neighbor,
                                 metric = "minkowski")
    model.fit(X[train_indices], y[train_indices])
    X_test = X[~train_indices]
    y_test = y[~train_indices]

    y_pred = model.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))

print(' %.2f +/- %.2f (chance = %.2f)' % (\
    np.mean(scores), np.std(scores), 1./len(np.unique(y))))

plot_accuracy(score = np.mean(scores), chance=1./len(np.unique(y)))
cm = confusion_matrix(y_test, y_pred,
                        labels=np.unique(y_df), 
                        normalize='true')
if substim:
    labels = ["Nat-Im 1", "Nat-Im 2","Nat-Im 3","Nat-Im 4","Nat-Im 5",
              "Drift-grating 1", "Drift-grating 2","Drift-grating 3","Drift-grating 4",
              "Looming",
              "mvg-dots 1",
              "mvg-dots 2", 
              "Rndm-dots 1",
              "Rndm-dots 2",
              "Rndm-dots 3",
              "Rndm-dots 4",
              "Static-patch 1",
              "Static-patch 2"]
else: 
    labels = ["Nat-Im",
              "Drift-grating", 
              "Looming",
              "mvg-dots",
              "Rndm-dots",
              "Static-patch"] 
plot_confusion_matrix(cm, y_df, np.mean(scores), ax_scale=(12,12), labels=labels)
#%%
plot_accuracy_comb(accuracies=[0.5, 0.49, 0.61], chance = 0.1666)
#%%all files : 
protocols = ['moving-dots', 
             'random-dots',
             'static-patch',
             'looming-stim', 
             'Natural-Images-4-repeats',
             'drifting-gratings']

normed = True # normalization of input data ? --> can be a good idea !!
averaging = False # possiblity --> denoising the training set by averaging
substim = False # model decoding between stimuli (False) or between stimuli and substimuli
state ='rest'#'rest'#'rest'#'rest'

scores_s = []

for index in range(len(SESSIONS)):
    X, y_df, behav, nROIs = get_X_y_data(SESSIONS, protocols, index=index, substim=substim, normed=normed, state=state)

    # visualize patterns
    #plot_dataset_responses(X, y, nROIs=nROIs)

    y = y_df['Stim-ID']
    print(y.shape)

    seed = 2
    scores = []
    N_neighbor = 5

    for seed in range(1000):

        np.random.seed(seed)

        # initialize a new training/test set
        X_train, y_train, train_indices = [], [], np.zeros(len(y), dtype=bool)
        # loop over different stim
        for yval in np.unique(y):
            # randomly pick an index
            index = np.random.choice(np.flatnonzero(yval==y), N_neighbor)
            # we add this to the training set
            train_indices[index] = True

        model = KNeighborsClassifier(n_neighbors=N_neighbor,
                                    metric = "minkowski")
        model.fit(X[train_indices], y[train_indices])
        X_test = X[~train_indices]
        y_test = y[~train_indices]

        y_pred = model.predict(X_test)
        scores.append(accuracy_score(y_test, y_pred))
    
    scores_s.append(np.mean(scores))

    print(' %.2f +/- %.2f (chance = %.2f)' % (\
        np.mean(scores), np.std(scores), 1./len(np.unique(y))))

    plot_accuracy(score = np.mean(scores), chance=1./len(np.unique(y)))
    cm = confusion_matrix(y_test, y_pred,
                            labels=np.unique(y_df), 
                            normalize='true')
    if substim:
        labels = ["Nat-Im 1", "Nat-Im 2","Nat-Im 3","Nat-Im 4","Nat-Im 5",
                "Drift-grating 1", "Drift-grating 2","Drift-grating 3","Drift-grating 4",
                "Looming",
                "mvg-dots 1",
                "mvg-dots 2", 
                "Rndm-dots 1",
                "Rndm-dots 2",
                "Rndm-dots 3",
                "Rndm-dots 4",
                "Static-patch 1",
                "Static-patch 2"]
    else: 
        labels = ["Nat-Im",
                "Drift-grating", 
                "Looming",
                "mvg-dots",
                "Rndm-dots",
                "Static-patch"] 
    #plot_confusion_matrix(cm, y_df, np.mean(scores), ax_scale=(12,12), labels=labels)
    
    print('end loop')
#%%
chance=0.055
print("l", scores_s)
plot_hist(scores_s, chance)

# %%
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    random_state=42,
                                                    test_size=0.5,
                                                    stratify=y)
# check that the values are indeed balanced across classes:
#y_test.value_counts()

if averaging:
    X_train = pd.DataFrame(\
        np.array([X_train[y_train['Stim-ID']==id].mean(axis=0)\
                        for id in y_train['Stim-ID'].unique()]),
                columns=['ROI%i' % i for i in range(nROIs)])
    y_train = pd.DataFrame(\
        np.array([id\
                        for id in y_train['Stim-ID'].unique()]),
                columns=['Stim-ID'])

plot_train_test_responses(X_train, y_train, X_test, y, nROIs = nROIs)

# Decoding is implemented as a Nearest-Neighbor Classifier
model = KNeighborsClassifier(n_neighbors=N_neighbor,
                                 metric = "minkowski")
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

'''
#%%
#all files one by one 
protocols = ['moving-dots', 
             'random-dots',
             'static-patch',
             'looming-stim', 
             'Natural-Images-4-repeats',
             'drifting-gratings']

normed = True # normalization of input data ? --> can be a good idea !!
averaging = False # possiblity --> denoising the training set by averaging
substim = False # model decoding between stimuli (False) or between stimuli and substimuli
state = None
#index = 4

scores = []
for index in range(len(SESSIONS)):
    X, y, behavcond, nROIs = get_X_y_data(SESSIONS, protocols, index=index, substim=substim, normed=normed, state=state)

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

    scores.append(score)

#%%
fig, ax = pt.figure()
pt.plt.hist(scores, bins=np.arange(0, 1.05, 0.05), edgecolor='grey', color="grey")
pt.set_plot(ax, 
            ylabel='Number of\n sessions', 
            yticks = [1, 2, 3, 4])
pt.plt.axvline(chance, color='black', linestyle='--', label='chance level')
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


X, y, behavcond, nROIs = get_X_y_data(SESSIONS, protocols, substim=True, normed=True)

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
#%%