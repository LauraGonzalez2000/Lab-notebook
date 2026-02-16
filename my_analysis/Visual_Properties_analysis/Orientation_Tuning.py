# %% [markdown]
# # Compute Orientation Selectivity

# %% [markdown]
# The core functions to analyze the orientation selectivity:
#
# - `selectivity_index`
# - `shift_orientation_according_to_pref`
# - `compute_tuning_response_per_cells`
#
# are implemented in the script [../src/physion/analysis/protocols/orientation_tuning.py](./../src/physion/analysis/protocols/orientation_tuning.py)

# %%
import sys, os, tempfile
import numpy as np
from scipy import stats

sys.path += ['../../physion/src'] # add src code directory for physion
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from physion.analysis.episodes.build import EpisodeData
from physion.analysis.protocols.orientation_tuning import compute_tuning_response_per_cells,\
                                                          fit_gaussian, \
                                                          plot_orientation_tuning_curve, \
                                                          plot_selectivity, \
                                                          plot_responsiveness

sys.path += ['../../physion/utils']
import physion.utils.plot_tools as pt

pt.set_style('manuscript')

#%%
# functions
def get_subject_art(f):
        print("f", f)
        if f == "C:\\Users\\laura.gonzalez\\DATA\\In_Vivo_experiments\\Ori-contrasts\\NDNF-Cre\\NWBs_orientations\\2025_12_30-14-28-22.nwb" or "2025_12_30-14-28-22.nwb":
              id = 2
        elif f == "C:\\Users\\laura.gonzalez\\DATA\\In_Vivo_experiments\\Ori-contrasts\\NDNF-Cre\\NWBs_orientations\\2025_12_31-14-31-29.nwb" or "2025_12_31-14-31-29.nwb":
              id = 2
        elif f == "C:\\Users\\laura.gonzalez\\DATA\\In_Vivo_experiments\\Ori-contrasts\\NDNF-Cre\\NWBs_orientations\\2026_01_02-14-32-01.nwb" or "2026_01_02-14-32-01.nwb":
                id = 5
        elif f == "C:\\Users\\laura.gonzalez\\DATA\\In_Vivo_experiments\\Ori-contrasts\\NDNF-Cre\\NWBs_orientations\\2026_01_16-15-28-33.nwb" or "2026_01_16-15-28-33.nwb":
              id = 2
        return id 

def compute_tunings(files, stat_test_props, response_significance_threshold):
        for contrast in [0.5, 1.0]:
                Tunings = []
                for f in files:
                        print(' - analyzing file: %s  [...] ' % f)
                        data = Data(f, verbose=False)
                        data.build_dFoF(neuropil_correction_factor=0.9,
                                        percentile=10., 
                                        verbose=False)

                        protocol_name=[p for p in data.protocols if 'ff-gratings' in p][0]
                        Episodes = EpisodeData(data, 
                                        quantities=['dFoF'], 
                                        protocol_name=protocol_name, 
                                        verbose=False)
                        
                        Tuning = compute_tuning_response_per_cells(data, 
                                                                   Episodes, 
                                                                   quantity='dFoF', 
                                                                   stat_test_props=stat_test_props, 
                                                                   response_significance_threshold = response_significance_threshold, 
                                                                   contrast=contrast)
                        Tuning['nROIs_original'] = data.original_nROIs
                        Tuning['nROIs_final'] = data.nROIs
                        Tuning['nROIs_responsive'] = np.sum(Tuning['significant_ROIs'])
                        Tuning['subject'] = get_subject_art(f)  ## correct subject id demo mouse!!
                        #Tuning['subject'] = data.nwbfile.subject.subject_id
                        Tunings.append(Tuning)
                # saving data
                np.save(os.path.join(tempfile.tempdir, 'Tunings_WT_contrast-%.1f.npy' % contrast),Tunings)
        return 0

def compute_tunings2(data_s, stat_test_props, response_significance_threshold):
        for contrast in [0.5, 1.0]:
                Tunings = []
                for data in data_s:
                        protocol_name=[p for p in data.protocols if 'ff-gratings' in p][0]
                        Episodes = EpisodeData(data, 
                                        quantities=['dFoF'], 
                                        protocol_name=protocol_name, 
                                        verbose=False)
                        
                        Tuning = compute_tuning_response_per_cells(data, 
                                                                   Episodes, 
                                                                   quantity='dFoF', 
                                                                   stat_test_props=stat_test_props, 
                                                                   response_significance_threshold = response_significance_threshold, 
                                                                   contrast=contrast)
                        Tuning['nROIs_original'] = data.original_nROIs
                        Tuning['nROIs_final'] = data.nROIs
                        Tuning['nROIs_responsive'] = np.sum(Tuning['significant_ROIs'])
                        print(data.filename)
                        Tuning['subject'] = get_subject_art(data.filename)  ## correct subject id demo mouse!!
                        #Tuning['subject'] = data.nwbfile.subject.subject_id
                        Tunings.append(Tuning)
                # saving data
                np.save(os.path.join(tempfile.tempdir, 'Tunings_WT_contrast-%.1f.npy' % contrast),Tunings)
        return 0

# %% LOAD DATA
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','Ori-contrasts', 'NDNF-Cre', 'NWBs_orientations')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
SESSIONS['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]

dFoF_options = {
        'roi_to_neuropil_fluo_inclusion_factor': 1.0,
        'method_for_F0': 'sliding_percentile',
        'sliding_window': 300.,
        'percentile': 10.,
        'neuropil_correction_factor': 0.8}

stat_test_props = dict(interval_pre=[-1.,0],                                   
                       interval_post=[1.,2.],                                   
                       test='ttest',                                            
                       sign='positive')

response_significance_threshold = 0.05 # very very conservative

# %% [markdown]
# ## Compute Responses of Visually-Reponsive Cells (i.e. Significantly-Modulated)
# N.B. This considers only significantly responsive cells !!

#%% EXAMPLE FILE  ################################################
index = 1
filename = SESSIONS['files'][index]
data = Data(filename, verbose=False)
data.build_dFoF(**dFoF_options, verbose=False)
Episodes = EpisodeData(data,
                       quantities=['dFoF', 'running_speed'],
                       protocol_name=[p for p in data.protocols if 'ff-gratings' in p][0],
                       verbose=False)

stat_test_props = dict(interval_pre=[-1.,0],                                   
                       interval_post=[1.,2.],                                   
                       test='ttest',                                            
                       sign='positive')

# possibility to **FILTER THE EPISODES** by behavioral state !
#stim_cond = (Episodes.t>0) & (Episodes.t<Episodes.time_duration[0])
#filtering_cond = Episodes.running_speed[:,stim_cond].mean(axis=1)<0.1

Tuning = compute_tuning_response_per_cells(data, 
                                           Episodes,
                                           #filtering_cond=filtering_cond,
                                           quantity='dFoF',
                                           stat_test_props=stat_test_props,
                                           response_significance_threshold = response_significance_threshold,
                                           contrast=1,
                                           verbose=True)

# %% [markdown]
# ## Plot Individual Responses

# %%
fig, AX = pt.figure(axes=(5, int(data.nROIs/5)+1), ax_scale=(1,.7), hspace=1.5)

for i, ax in enumerate(pt.flatten(AX)):
    if i<data.nROIs:
        pt.annotate(ax, ' ROI #%i \n (SI=%.2f)' % (1+i, Tuning['selectivities'][i]), (1,1), 
                    va='center', ha='right', fontsize=7,
                    color='tab:green' if Tuning['significant_ROIs'][i] else 'tab:red')
        # ax.plot(Tuning['shifted_angle'], Tuning['Responses'][i], 'k')
        pt.plot(Tuning['shifted_angle'], Tuning['Responses'][i], 
                sy=Tuning['semResponses'][i], ax=ax)
        ax.plot(Tuning['shifted_angle'], 0*Tuning['shifted_angle'], 'k:', lw=0.5)
        pt.set_plot(ax, xticks=Tuning['shifted_angle'], 
                    ylabel='(post - pre)\n$\\delta$ $\\Delta$F/F' if i%5==0 else '',
                    xlabel='angle ($^o$) from preferred orientation' if i==(data.nROIs-1) else '',
                    xticks_labels=['%i' % a if (a in [0, 90])\
                                    else '' for a in Tuning['shifted_angle'] ])
    else:
        ax.axis('off')

# %% [markdown]
# ## Single Session Summary

# %%
# Gaussian Fit
_, func = fit_gaussian(Tuning['shifted_angle'], np.mean([r/r[1] for r in Tuning['Responses']], axis=0))

# Plot

fig, AX = pt.figure(axes=(3, 1), ax_scale=(1.2, 1), wspace=1.5)

pt.plot(Tuning['shifted_angle'], np.mean(Tuning['Responses'][Tuning['significant_ROIs'],:], axis=0), 
        sy=np.std(Tuning['Responses'][Tuning['significant_ROIs'],:], axis=0), ax=AX[0])

pt.plot(Tuning['shifted_angle'], np.mean([r/r[1] for r in Tuning['Responses'][Tuning['significant_ROIs'],:]], axis=0), 
        sy=np.std([r/r[1] for r in Tuning['Responses'][Tuning['significant_ROIs'],:]], axis=0), ax=AX[1])

pt.scatter(Tuning['shifted_angle'], np.mean([r/r[1] for r in Tuning['Responses'][Tuning['significant_ROIs'],:]], axis=0), 
        sy=stats.sem([r/r[1] for r in Tuning['Responses'][Tuning['significant_ROIs'],:]], axis=0), 
        ax=AX[2], ms=2)

x = np.linspace(-30, 180-30, 100)

AX[2].plot(x, func(x), lw=2, alpha=.5, color='dimgrey')

pt.set_plot(AX[0], xticks=Tuning['shifted_angle'], 
            ylabel='(post - pre)\n$\\delta$ $\\Delta$F/F',  xlabel='angle ($^o$) from pref.',
            xticks_labels=['%i' % a if (a in [0, 90]) else '' for a in Tuning['shifted_angle'] ])

pt.set_plot(AX[1], xticks=Tuning['shifted_angle'], yticks=np.arange(3)*0.5, ylim=[-0.05, 1.05],
            ylabel='norm. $\\delta$ $\\Delta$F/F',  xlabel='angle ($^o$) from pref.',
            xticks_labels=['%i' % a if (a in [0, 90]) else '' for a in Tuning['shifted_angle'] ])

pt.set_plot(AX[2], xticks=Tuning['shifted_angle'], yticks=np.arange(3)*0.5, ylim=[-0.05, 1.05],
            ylabel='norm. $\\delta$ $\\Delta$F/F',  xlabel='angle ($^o$) from pref.',
            xticks_labels=['%i' % a if (a in [0, 90]) else '' for a in Tuning['shifted_angle'] ])

fig.suptitle(' session: %s ' % os.path.basename(data.filename), fontsize=7);

# %% [markdown]
# ALL FILES #############################################
#########################################################

# %%


# %%
# compute tunings
compute_tunings(files = SESSIONS['files'], stat_test_props=stat_test_props, response_significance_threshold=response_significance_threshold)
#%%
# loading data
Tunings = np.load(os.path.join(tempfile.tempdir, 'Tunings_WT_contrast-0.5.npy'), allow_pickle=True)

#%%
# mean significant responses per session
Responses = [np.mean(Tuning['Responses'][Tuning['significant_ROIs'],:], axis=0) for Tuning in Tunings]
# Gaussian Fit
C, func = fit_gaussian(Tunings[0]['shifted_angle'], np.mean([r/r[1] for r in Responses], axis=0))
# Plot
fig, ax = pt.figure(ax_scale=(1.2, 1))

pt.scatter(Tunings[0]['shifted_angle'], np.mean([r/r[1] for r in Responses], axis=0), 
        sy=stats.sem([r/r[1] for r in Responses], axis=0), ax=ax, ms=3)

x = np.linspace(-30, 180-30, 100)
ax.plot(x, func(x), lw=2, alpha=.5, color='dimgrey')

pt.annotate(ax, 'N=%i sessions' % len(Responses), (0.8,1))
pt.annotate(ax, 'SI=%.2f' % (1-C[2]), (1., 0.9), ha='right', va='top')

pt.set_plot(ax, xticks=Tunings[0]['shifted_angle'], 
            yticks=np.arange(3)*0.5, 
            ylim=[-0.05, 1.05],
            ylabel='norm. $\delta$ $\Delta$F/F',  
            xlabel='angle ($^o$) from pref.',
            xticks_labels=\
                ['%i' % a if (a in [0, 90]) else ''\
                  for a in Tunings[0]['shifted_angle'] ])

# %%
######### SELECTIVITY, TUNING CURVE, RESPONSIVENESS #########################
################# for different contrasts ###################################

# average by "sessions", "subjects", "ROIs"

fig, ax = plot_selectivity(['WT_contrast-1.0', 
                            'WT_contrast-0.5'],
                           average_by='ROIs',
                           #using='fit',
                           path=tempfile.tempdir)
    
fig, ax = plot_orientation_tuning_curve(['WT_contrast-1.0', 
                                         'WT_contrast-0.5'],
                                        average_by='ROIs',
                                        path=tempfile.tempdir)
    
fig, ax = plot_responsiveness(['WT_contrast-1.0', 
                               'WT_contrast-0.5'],
                              average_by='ROIs', #issue !!  #not average by subject
                              path=tempfile.tempdir)
