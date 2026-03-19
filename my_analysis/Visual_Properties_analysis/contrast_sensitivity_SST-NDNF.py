# %% Packages 

import sys, os
import numpy as np

sys.path += ['../../physion/src'] # add src code directory for physion
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from physion.analysis.episodes.build import EpisodeData
import physion.utils.plot_tools as pt
from physion.analysis.protocols.contrast_sensitivity_new import plot_contrast_sensitivity, plot_contrast_responsiveness

sys.path += ['..']
from utils_.Responsiveness_methods import calc_responsiveness2, plot_contrast_responsiveness_

from scipy import stats

pt.set_style('manuscript')

#%%
def plot_contrast_responsiveness_(keys,
                                  Responsive,
                                  sign='positive',
                                  colors=None,
                                  with_label=True,
                                  fig_args={'right':25}, 
                                  angle = "0.0", 
                                  ylim=[0,100], 
                                  means="session"):

        if colors is None:
            colors = pt.plt.rcParams['axes.prop_cycle'].by_key()['color']

        if type(keys)==str:
                keys, colors = [keys], [colors[0]]

        fig, ax = pt.figure(**fig_args)
        inset = pt.inset(ax, [1.7,0.1,0.5,0.8])

        for i, (cell_type, color) in enumerate(zip(keys, colors)):

            if means=="ROI":
                #merging all ROIs together
                values_all = np.concatenate(Responsive[cell_type][angle], axis=1)[0]  #shape #totROIS x 8
                values_per = 100 * (sum(values_all)/len(values_all))
                y = values_per #no need to mean
                sy = 0 # there is no error bar if we take all ROIs

                Gains = []
                slope, _ = np.polyfit(range(len(values_per)), values_per, 1)
                Gains.append(slope)


            if means=='session':
                #calculating per session and average
                values_per = []
                Gains = []
                for file_i in range(len(Responsive[cell_type][angle])):
                    values_file = Responsive[cell_type][angle][file_i] #shape nROIs x 8
                    values_per_ = 100 * (sum(values_file)/len(values_file))
                    values_per_file = np.mean(values_per_, axis=0)
                    values_per.append(values_per_file)

                    y = values_per_file
                    x = np.arange(len(y))   
                    slope, _ = np.polyfit(x, 
                                          y, 
                                          deg=1)
                    Gains.append(slope)   

                values_per_m = np.mean(values_per, axis=0)
                y = values_per_m
                sy = stats.sem(values_per, axis=0)
                #sy = np.std(values_per, axis=0, ddof=1)
            
            x = np.arange(len(y)) + 0.4*i
            print("values to plot : ", y)
            print("values to plot sem : ", sy)
            print("values to plot gains :", Gains)

            pt.bar(y=y,
                sy=sy,
                x=x,
                width=0.5/len(keys),
                color=color,
                ax=ax)

            pt.violin(Gains, x=i, color=color, ax=inset)
            pt.bar([np.mean(Gains)], x=[i], color=color, ax=inset, alpha=0.1) #looks confusing to me

            if with_label:
                    annot = i*'\n'+' %.1f$\\pm$%.1f, ' %(\
                            np.mean(Gains), stats.sem(Gains))
                    annot += 'N=%02d %s, ' % (len(Responsive[cell_type][angle]), 'sessions') + cell_type

            pt.annotate(inset, annot, (1., 0.9), va='top', color=color)
                
        
        pt.set_plot(ax, 
                    ylabel='%% responsive \n %s' % sign,
                    xlabel='\ncontrast', 
                    xticks=[0,1,2,3,4,5,6,7], 
                    xticks_labels=[0.05,0.19,0.32,0.46,0.59,0.73,0.86,1.0], 
                    yticks= np.arange(0, ylim[1], 10), 
                    xticks_rotation=90, 
                    ylim=ylim)
        
        
        pt.set_plot(inset, ['left'],
                    title='gain',
                    ylabel='%resp. / contrast')
        
        if means=="session":
            num_sessions = [sum(len(Responsive[cell_type][angle][i]) for i in range(len(Responsive[cell_type][angle])))][0]
            ax.annotate(xy = (-0.5,ylim[1]), text = f"mean by {means}s ({num_sessions})")
        elif means=="ROI":
            ax.annotate(xy = (-0.5,ylim[1]), text = f"{len(values_all)} {means}s")
        
        return fig, ax

def fill_Responsive(Responsive, group_name, data_s, repetition_keys=['repeat', 'angle']):

    for data_i, data in enumerate(data_s):
       
        ep = EpisodeData(data, 
                         protocol_name="ff-gratings-2orientations-8contrasts-15repeats", 
                         quantities=['dFoF'])
    
        resp_cond, pos_cond, neg_cond = calc_responsiveness2(ep, data.nROIs, repetition_keys= repetition_keys)
        
        print("len pos cond", len(pos_cond))
        print("pos cond", pos_cond)
        #percentage_pos = 100*(sum(pos_cond)/data.nROIs)
        #percentage_neg = 100*(sum(neg_cond)/data.nROIs)

        #for param_i in range(len(resp_cond[0])):
        if len(resp_cond[0])==16:
            print("angles not merged")
            if param_i<8:
                    Responsive["Pos"][group_name]["0.0"][data_i].append(pos_cond[:][:8])
                    Responsive["Neg"][group_name]["0.0"][data_i].append(neg_cond[:][:8])
            elif 7<param_i:
                    Responsive["Pos"][group_name]["90.0"][data_i].append(pos_cond[:][8:16])
                    Responsive["Neg"][group_name]["90.0"][data_i].append(neg_cond[:][8:16])

        elif len(resp_cond[0])==8: 
            print("angles merged")
            Responsive["Pos"][group_name]["both"][data_i].append(pos_cond)
            Responsive["Neg"][group_name]["both"][data_i].append(neg_cond)

    return Responsive


#%% Data
datafolder_NDNF = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','Ori-contrasts', 'NDNF-Cre','NWBs_contrasts')
datafolder_SST = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','Ori-contrasts', 'SST-cells_WT_Adult_V1', 'NWBs_contrast')

SESSIONS_SST = scan_folder_for_NWBfiles(datafolder_SST)
SESSIONS_SST['nwbfiles'] = [os.path.basename(f) for f in SESSIONS_SST['files']]

SESSIONS_NDNF = scan_folder_for_NWBfiles(datafolder_NDNF)
SESSIONS_NDNF['nwbfiles'] = [os.path.basename(f) for f in SESSIONS_NDNF['files']]

dFoF_options = {
        'roi_to_neuropil_fluo_inclusion_factor': 1.0,
        'method_for_F0': 'sliding_percentile',
        'sliding_window': 300.,
        'percentile': 10.,
        'neuropil_correction_factor': 0.8}
#%%
data_s_SST = []
for i in range(len(SESSIONS_SST['files'])):
    data = Data(SESSIONS_SST['files'][i], verbose=False)
    data.build_dFoF(**dFoF_options, verbose=False)
    data.build_running_speed()
    data.build_facemotion()
    data.build_pupil_diameter()
    data_s_SST.append(data)

data_s_NDNF = []
for i in range(len(SESSIONS_NDNF['files'])):
    data = Data(SESSIONS_NDNF['files'][i], verbose=False)
    data.build_dFoF(**dFoF_options, verbose=False)
    data.build_running_speed()
    data.build_facemotion()
    data.build_pupil_diameter()
    data_s_NDNF.append(data)

#%%
Responsive = {"Pos" : {"SST-Cre" : {"0.0" : [[] for _ in range(len(data_s_SST))],
                                    "90.0": [[] for _ in range(len(data_s_SST))], 
                                    "both": [[] for _ in range(len(data_s_SST))]}, 
                       "NDNF-Cre" : {"0.0" : [[] for _ in range(len(data_s_NDNF))],
                                     "90.0": [[] for _ in range(len(data_s_NDNF))], 
                                     "both": [[] for _ in range(len(data_s_NDNF))]}},

              "Neg" : {"SST-Cre" : {"0.0" : [[] for _ in range(len(data_s_SST))],
                                    "90.0": [[] for _ in range(len(data_s_SST))], 
                                    "both": [[] for _ in range(len(data_s_SST))]}, 
                       "NDNF-Cre" : {"0.0" : [[] for _ in range(len(data_s_NDNF))],
                                     "90.0": [[] for _ in range(len(data_s_NDNF))], 
                                     "both": [[] for _ in range(len(data_s_NDNF))]}}}


repetition_keys = ['repeat', 'angle']
Responsive = fill_Responsive(Responsive, group_name="SST-Cre",  data_s = data_s_SST, repetition_keys=repetition_keys)
Responsive = fill_Responsive(Responsive, group_name="NDNF-Cre", data_s = data_s_NDNF, repetition_keys=repetition_keys)


#%% PLOT
#keys = ['SST-Cre'   , 'NDNF-Cre']
keys = ['SST-Cre']
#keys = ['NDNF-Cre']  

#colors = ['tab:orange','tab:green']
colors = ['tab:orange']
#colors = ['tab:green']

#angle = "0.0"
#angle = "90.0"
angle = "both"
 
for sign in ["Pos", "Neg"]:
    fig, ax = plot_contrast_responsiveness_(keys=keys, 
                                            Responsive=Responsive[sign], 
                                            sign=sign,
                                            colors = colors,
                                            fig_args=dict(ax_scale=(1.5,2)), 
                                            angle=angle, 
                                            ylim=[0,101],  
                                            means="session")
    fig.savefig(os.path.expanduser(f'~/Output_expe/In_Vivo/ANR-NDNF/responsiveness_{sign}.svg'))
    






#%%
############################################################# OLD - from summary
##############################################################

def count_animals_from_npy(folder, dataset_name):
    # Find file matching dataset name
    matches = [f for f in os.listdir(folder)
               if dataset_name in f and f.endswith(".npy")]
    
    if len(matches) == 0:
        raise FileNotFoundError(f"No file found for dataset {dataset_name}")
    if len(matches) > 1:
        raise ValueError(f"Multiple matches for {dataset_name}: {matches}")
    
    fpath = os.path.join(folder, matches[0])
    
    # Load the object array
    arr = np.load(fpath, allow_pickle=True)

    # ---------- ADDED: ROI counts ----------
    n_rois_original = arr[0]["nROIs_original"]   # ORIGINAL ROI count
    n_rois_final = arr[0]["nROIs_final"]         # FINAL ROI count
    # ---------------------------------------

    # Extract subjects from each entry (one per session)
    subjects = [entry['subject'] for entry in arr]
    
    # Unique animal IDs
    unique_subjects = sorted(set(subjects))
    
    return len(unique_subjects), unique_subjects, n_rois_original, n_rois_final

def compute_session_gain(entry):
    """
    Gain = slope between lowest and highest contrast:
    (mean response at max contrast - mean response at min contrast) / Δcontrast
    Assumes contrasts normalized 0→1 (Δcontrast = 1).
    """
    R = entry["Responses"]  # shape (n_cells, n_contrasts)
    mean_low = R[:, 0].mean()
    mean_high = R[:, -1].mean()
    return mean_high - mean_low

def extract_gains_from_dataset(folder, dataset_name):
    matches = [f for f in os.listdir(folder)
               if dataset_name in f and f.endswith(".npy")]
    if len(matches) == 0:
        raise FileNotFoundError(dataset_name)
    if len(matches) > 1:
        raise ValueError(matches)

    arr = np.load(os.path.join(folder, matches[0]), allow_pickle=True)
    
    gains = [compute_session_gain(entry) for entry in arr]
    return np.array(gains)

#%% LOAD DATA
summary_folder = os.path.expanduser('~/DATA/summary')
DATASETS = ['SST-cells_WT_Adult_V1_angle-90.0',
            'SST-cells_WT_Adult_V1_angle-0.0']


#%% d dFOF vs contrasts ; gain
fig, ax = plot_contrast_sensitivity(DATASETS,
                                    average_by='sessions',
                                    colors = ['tab:red', 'tab:green'],
                                    path=summary_folder)
                                    #gain_plot='bar')

animal_counts = {}
gain_dict = {} # Mean gain per dataset and statistical comparison
for d in DATASETS:   
    n_animals, subjects, n_rois_orig, n_rois_final = count_animals_from_npy(summary_folder, f"Sensitivities_{d}")
    animal_counts[d] = (n_animals, subjects)
    print(f"{d}:\n {n_animals} animals → {subjects} |\n "
          f"nROIs_original = {n_rois_orig}\n nROIs_final = {n_rois_final}")
    gains = extract_gains_from_dataset(summary_folder, d)
    gain_dict[d] = gains

    mean_gain = gains.mean()
    sem_gain = gains.std() / np.sqrt(len(gains))

    print(f" mean gain = {mean_gain:.3f}  ± {sem_gain:.3f}  (n={len(gains)})\n")

    
#%% STAT Test gains
print("\n=== Mann–Whitney U tests (gain) ===")
names = list(gain_dict.keys())

for i in range(len(names)):
    for j in range(i+1, len(names)):
        d1, d2 = names[i], names[j]
        U, p = mannwhitneyu(gain_dict[d1], gain_dict[d2], alternative='two-sided')
        print(f"{d1}  vs  {d2}:  U={U:.1f},  p={p:.5f}")


# %% % responsiveness vs contrast ; gain
for sign in ["positive", "negative"]:
    fig, ax = plot_contrast_responsiveness(DATASETS,\
                            sign=sign,
                            nROIs='final', # "original" or "final", before/after dFoF criterion
                            colors = ['tab:red', 'tab:green'],
                            path=summary_folder,
                            fig_args=dict(ax_scale=(1.5,2)))
    ax.set_ylim([0,100])

#%%




Sensitivities = \
                    np.load(os.path.join(summary_folder, 'Sensitivities_SST-cells_WT_Adult_V1_angle-0.0.npy'), 
                            allow_pickle=True)

print(Sensitivities[0].keys())

print(Sensitivities[0]["contrast"])


#%%
#%% TO DELETE
Responsive_pos = {"SST-Cre" : {"0.0" : [[] for _ in range(len(data_s_SST))],
                               "90.0": [[] for _ in range(len(data_s_SST))], 
                               "both": [[] for _ in range(len(data_s_SST))]}, 
                  "NDNF-Cre" : {"0.0" : [[] for _ in range(len(data_s_NDNF))],
                                "90.0": [[] for _ in range(len(data_s_NDNF))], 
                                "both": [[] for _ in range(len(data_s_NDNF))]}}

Responsive_neg = {"SST-Cre" : {"0.0" : [[] for _ in range(len(data_s_SST))],
                               "90.0": [[] for _ in range(len(data_s_SST))], 
                               "both": [[] for _ in range(len(data_s_SST))]}, 
                  "NDNF-Cre" : {"0.0" : [[] for _ in range(len(data_s_NDNF))],
                                "90.0": [[] for _ in range(len(data_s_NDNF))], 
                                "both": [[] for _ in range(len(data_s_NDNF))]}}

repetition_keys = ['repeat', 'angle']
#repetition_keys = ['repeat'] #to plot by angle

nROIs_SST=0
for data_i, data in enumerate(data_s_SST):
    nROIs_SST += data.nROIs
    ep = EpisodeData(data, 
                     protocol_name="ff-gratings-2orientations-8contrasts-15repeats", 
                     quantities=['dFoF'])
   
    varied_params = list(ep.varied_parameters.keys())
    param_values_angles = ep.varied_parameters[varied_params[0]]
    param_values_contrasts = ep.varied_parameters[varied_params[1]]
    resp_cond, pos_cond, neg_cond = calc_responsiveness2(ep, data.nROIs, repetition_keys=repetition_keys)
    print(data.nROIs)
    print("pos_cond", pos_cond)
    percentage_pos = 100*(sum(pos_cond)/data.nROIs)
    percentage_neg = 100*(sum(neg_cond)/data.nROIs)

    print("percentage pos", percentage_pos)
    
    for param_i in range(len(resp_cond[0])):
        if len(resp_cond[0])==16:
            print("angles not merged")
            if param_i<8:
                    Responsive_pos["SST-Cre"]["0.0"][data_i].append(percentage_pos[param_i])
                    Responsive_neg["SST-Cre"]["0.0"][data_i].append(percentage_neg[param_i])
            elif 7<param_i:
                    Responsive_pos["SST-Cre"]["90.0"][data_i].append(percentage_pos[param_i])
                    Responsive_neg["SST-Cre"]["90.0"][data_i].append(percentage_neg[param_i])
        elif len(resp_cond[0])==8: 
            print("angles merged")    
            Responsive_pos["SST-Cre"]["both"][data_i].append(percentage_pos[param_i])
            Responsive_neg["SST-Cre"]["both"][data_i].append(percentage_neg[param_i])


#%%
Responsive_pos = {"SST-Cre" : {"0.0" : [[] for _ in range(len(data_s_SST))],
                               "90.0": [[] for _ in range(len(data_s_SST))], 
                               "both": [[] for _ in range(len(data_s_SST))]}, 
                  "NDNF-Cre" : {"0.0" : [[] for _ in range(len(data_s_NDNF))],
                                "90.0": [[] for _ in range(len(data_s_NDNF))], 
                                "both": [[] for _ in range(len(data_s_NDNF))]}}

Responsive_neg = {"SST-Cre" : {"0.0" : [[] for _ in range(len(data_s_SST))],
                               "90.0": [[] for _ in range(len(data_s_SST))], 
                               "both": [[] for _ in range(len(data_s_SST))]}, 
                  "NDNF-Cre" : {"0.0" : [[] for _ in range(len(data_s_NDNF))],
                                "90.0": [[] for _ in range(len(data_s_NDNF))], 
                                "both": [[] for _ in range(len(data_s_NDNF))]}}

repetition_keys = ['repeat', 'angle']
#repetition_keys = ['repeat'] #to plot by angle
nROIs_SST=0
for data_i, data in enumerate(data_s_SST):
    nROIs_SST += data.nROIs
    ep = EpisodeData(data, 
                     protocol_name="ff-gratings-2orientations-8contrasts-15repeats", 
                     quantities=['dFoF'])
   
    varied_params = list(ep.varied_parameters.keys())
    param_values_angles = ep.varied_parameters[varied_params[0]]
    param_values_contrasts = ep.varied_parameters[varied_params[1]]
    resp_cond, pos_cond, neg_cond = calc_responsiveness2(ep, data.nROIs, repetition_keys=repetition_keys)
    print("pos cond :", pos_cond[0])
    #percentage_pos = 100*(sum(pos_cond)/data.nROIs)
    #percentage_neg = 100*(sum(neg_cond)/data.nROIs)
    '''
    for param_i in range(len(resp_cond[0])):
        if len(resp_cond[0])==16:
            print("angles not merged")
            if param_i<8:
                    Responsive_pos["SST-Cre"]["0.0"][data_i].append(pos_cond[:][param_i])
                    Responsive_neg["SST-Cre"]["0.0"][data_i].append(neg_cond[:][param_i])
            elif 7<param_i:
                    Responsive_pos["SST-Cre"]["90.0"][data_i].append(pos_cond[:][param_i])
                    Responsive_neg["SST-Cre"]["90.0"][data_i].append(neg_cond[:][param_i])

            Responsive_pos["SST-Cre"]["both"][data_i].append(pos_cond[:][param_i])
            Responsive_neg["SST-Cre"]["both"][data_i].append(neg_cond[:][param_i])
        elif len(resp_cond[0])==8: 
    '''
    #if len(resp_cond[0])==8: 
    print("angles merged")
    print("here", pos_cond)
    Responsive_pos["SST-Cre"]["both"][data_i].append(pos_cond)
    Responsive_neg["SST-Cre"]["both"][data_i].append(neg_cond)



#Responsive_pos["SST-Cre"]["both"] = np.stack(Responsive_pos["SST-Cre"]["both"], axis=0)
#Responsive_neg["SST-Cre"]["both"] = np.stack(Responsive_neg["SST-Cre"]["both"], axis=0)


nROIs_NDNF=0
for data_i, data in enumerate(data_s_NDNF):
    nROIs_NDNF += data.nROIs 

    ep = EpisodeData(data, 
                     protocol_name="ff-gratings-2orientations-8contrasts-15repeats", 
                     quantities=['dFoF'])
   
    varied_params = list(ep.varied_parameters.keys())
    param_values_angles = ep.varied_parameters[varied_params[0]]
    param_values_contrasts = ep.varied_parameters[varied_params[1]]
    resp_cond, pos_cond, neg_cond = calc_responsiveness2(ep, data.nROIs, repetition_keys= repetition_keys)
    
    print("pos cond :", pos_cond[0])

    #percentage_pos = 100*(sum(pos_cond)/data.nROIs)
    #percentage_neg = 100*(sum(neg_cond)/data.nROIs)
    '''
    for param_i in range(len(resp_cond[0])):
        if len(resp_cond[0])==16:
            print("angles not merged")
            if param_i<8:
                    Responsive_pos["NDNF-Cre"]["0.0"][data_i].append(pos_cond[:][param_i])
                    Responsive_neg["NDNF-Cre"]["0.0"][data_i].append(neg_cond[:][param_i])
            elif 7<param_i:
                    Responsive_pos["NDNF-Cre"]["90.0"][data_i].append(pos_cond[:][param_i])
                    Responsive_neg["NDNF-Cre"]["90.0"][data_i].append(neg_cond[:][param_i])

            Responsive_pos["NDNF-Cre"]["both"][data_i].append(pos_cond[:][param_i])
            Responsive_neg["NDNF-Cre"]["both"][data_i].append(neg_cond[:][param_i])
    '''
    #    elif len(resp_cond[0])==8: 
    print("angles merged")
    Responsive_pos["NDNF-Cre"]["both"][data_i].append(pos_cond)
    Responsive_neg["NDNF-Cre"]["both"][data_i].append(neg_cond)


#Responsive_pos["NDNF-Cre"]["both"] = np.stack(Responsive_pos["NDNF-Cre"]["both"], axis=0)
#Responsive_neg["NDNF-Cre"]["both"] = np.stack(Responsive_neg["NDNF-Cre"]["both"], axis=0)


Responsive = [Responsive_pos, Responsive_neg]


#%%
Responsive_pos.shape
# (files, 1, nROIs, contrasts)

result = np.squeeze(Responsive_pos, axis=1)

Responsive_pos.shape
# (files, nROIs, contrasts)
#%%
print(nROIs_SST)
print(nROIs_NDNF)
#%% PLOT
#keys = ['SST-Cre'   , 'NDNF-Cre']
keys = ['SST-Cre']
#keys = ['NDNF-Cre']  

#colors = ['tab:orange','tab:green']
colors = ['tab:orange']
#colors = ['tab:green']

#angle = "0.0"
#angle = "90.0"
angle = "both"
 
for i, sign in enumerate(["positive", "negative"]):
    fig, ax = plot_contrast_responsiveness_(keys=keys, 
                                            Responsive=Responsive[i], 
                                            sign=sign,
                                            colors = colors,
                                            fig_args=dict(ax_scale=(1.5,2)), 
                                            angle=angle, 
                                            ylim=[0,101], 
                                            means='session')
    
    fig.savefig(os.path.expanduser(f'~/Output_expe/In_Vivo/ANR-NDNF/responsiveness_{sign}.svg'))
    
def fill_Responsive(Responsive, group_name, data_s, repetition_keys=['repeat', 'angle']):

    nROIs=0
    for data_i, data in enumerate(data_s):
        nROIs += data.nROIs 

        ep = EpisodeData(data, 
                        protocol_name="ff-gratings-2orientations-8contrasts-15repeats", 
                        quantities=['dFoF'])
    
        resp_cond, pos_cond, neg_cond = calc_responsiveness2(ep, data.nROIs, repetition_keys= repetition_keys)
        
        percentage_pos = 100*(sum(pos_cond)/data.nROIs)
        percentage_neg = 100*(sum(neg_cond)/data.nROIs)

        for param_i in range(len(resp_cond[0])):
            if len(resp_cond[0])==16:
                print("angles not merged")
                if param_i<8:
                        Responsive["Pos"][group_name]["0.0"][data_i].append(percentage_pos[param_i])
                        Responsive["Neg"][group_name]["0.0"][data_i].append(percentage_neg[param_i])
                elif 7<param_i:
                        Responsive["Pos"][group_name]["90.0"][data_i].append(percentage_pos[param_i])
                        Responsive["Neg"][group_name]["90.0"][data_i].append(percentage_neg[param_i])

            elif len(resp_cond[0])==8: 
                print("angles merged")
                Responsive["Pos"][group_name]["both"][data_i].append(percentage_pos[param_i])
                Responsive["Neg"][group_name]["both"][data_i].append(percentage_neg[param_i])

    return Responsive


   #contrasts = [0.05,0.18571429,0.32142857,0.45714286,
            #                0.59285714, 0.72857143,0.86428571, 1.]
            #Gains = []
            #for r in Responsive[cell_type][angle]:
            #    temp = [r_ / contrast for r_, contrast in zip(r, contrasts)]
            #    Gains.append(np.mean(temp))
