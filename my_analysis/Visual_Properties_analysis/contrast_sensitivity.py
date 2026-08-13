# %% [markdown]
# STUDIES CONTRAST SENSITIVITY ON DIFFERENT DATASETS

#%%
import os, sys
from pathlib import Path

sys.path += ['../physion/src'] # add src code directory for physion
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles

sys.path += ['../']
from utils_.contrast_angle_sensitivity_methods import\
            generate_figures, \
            create_PDF, \
            generate_figures_GROUP, \
            create_group_PDF

from utils_.Responsiveness_methods import calc_responsiveness2

from utils_.General_overview_episodes import plot_dFoF_of_protocol


from physion.analysis.episodes.build import EpisodeData

 
# Variables : 
test = "contrast"

#%% #############################################################################
########################################       STATIC GRATINGS      #############
######################################## 2 ORIENTATIONS 8 contrasts #############
########################################            NDNF            #############
#################################################################################
#%% LOAD DATA
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 
                          'In_Vivo_experiments','Ori-contrasts', 
                          'NDNF-Cre', 'NWBs_8contrasts2ori')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
SESSIONS['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]

dFoF_options = {'roi_to_neuropil_fluo_inclusion_factor': 1.0,
                'method_for_F0': 'sliding_percentile',
                'sliding_window': 300.,
                'percentile': 10.,
                'neuropil_correction_factor': 0.8}

data_s_con = []
nROIS = 0
for idx, filename in enumerate(SESSIONS['files']):
    data = Data(filename, verbose=False)
    data.build_dFoF(**dFoF_options)
    data.build_running_speed()
    data.build_facemotion()
    data.build_pupil_diameter()
    data_s_con.append(data)
    print(idx, data.protocols)
    nROIS += data.nROIs

#%%
nROIS
#%% [markdown]
## All individual files

#%%
#plot traces
fig, ax = plot_dFoF_of_protocol(data_s=data_s_con, protocol=data.protocols[0], ylim=[-0.15,0.15], subset_rois=None, color_trace='black')
#%%
#fig.savefig(f'staticgrating_traces.png', format='png', dpi=600, transparent=True)
#%%

#%%
#dict_annotation, fig1, fig2, fig3, \
#fig4, fig5, fig6, fig7, fig8, fig9, fig10 = generate_figures(data_s_con, 
#                                                             subplots_n=16)
#create_PDF(dict_annotation, fig1, fig2, fig3, 
#           fig4, fig5, fig6, fig7, fig8, fig9, fig10, cell_type='NDNF')
#%%
resp_cond_s, pos_cond_s, neg_cond_s = [], [], []
for filename in SESSIONS['files']:
    data = Data(filename, verbose=False)
    data.build_dFoF()
    protocol = 'ff-gratings-2orientations-8contrasts-15repeats'
    stat_test_props = dict(interval_pre=[-1.,0],                                   
                        interval_post=[1,2],                                   
                        test='ttest')
    ep = EpisodeData(data,
                    protocol_name=protocol,
                    quantities=['dFoF'])
    resp_cond, pos_cond, neg_cond = calc_responsiveness2(ep, nROIs=data.nROIs)
    resp_cond_s.append(resp_cond)
    pos_cond_s.append(pos_cond)
    neg_cond_s.append(neg_cond)

#%% ALL cells

#fig_traces, _ = plot_dFoF_of_protocol(data_s=data_s_con, 
#                                      protocol='ff-gratings-2orientations-8contrasts-15repeats', 
#                                      subset_rois = None, 
#                                      ylim=[-0.15, 0.15],
#                                      norm=True)
#fig_traces, _ = plot_dFoF_of_protocol(data_s=data_s_con, 
#                                      protocol='ff-gratings-2orientations-8contrasts-15repeats', 
#                                      subset_rois = None, 
#                                      ylim=[0.25, 0.55], 
#                                      norm=False)
fig1, fig2, fig3, fig4, fig5 = generate_figures_GROUP(data_s_con, 
                                                      subplots_n=16, 
                                                      test=test, 
                                                      means='session', 
                                                      ylim=[-0.15,0.15], 
                                                      subset_rois= None)

#%% Only responsive cells 
subset_cells_s = []
for resp_cond in resp_cond_s:
    subset_cells = []
    for i, ROI_i in enumerate(resp_cond):
        if all(ROI_i[k] == False for k in range(16)): 
            continue
        else: 
            subset_cells.append(i)
    subset_cells_s.append(subset_cells)

print(sum([len(subset_cells) for subset_cells in subset_cells_s]))
#%%
#fig_traces, _ = plot_dFoF_of_protocol(data_s=data_s_con, 
#                                      protocol='ff-gratings-2orientations-8contrasts-15repeats', 
#                                      subset_rois =subset_cells_s, 
#                                      ylim=[-0.25, 0.2])
fig1, fig2, fig3, fig4, fig5 = generate_figures_GROUP(data_s_con, 
                                                      subplots_n=16, 
                                                      test=test, 
                                                      means='session', 
                                                      ylim=[-0.25,0.2], 
                                                      subset_rois= subset_cells_s, 
                                                      color_trace='green')
#%% Only pos resp cells to first contrast
subset_cells_s = []
for pos_cond in pos_cond_s:
    subset_cells = []
    for i, ROI_i in enumerate(pos_cond):
        if ROI_i[0]==True or ROI_i[8]==True:
            subset_cells.append(i)
    subset_cells_s.append(subset_cells)

print(sum([len(subset_cells) for subset_cells in subset_cells_s]))
#%%
#fig_traces, _ = plot_dFoF_of_protocol(data_s=data_s_con, 
#                                      protocol='ff-gratings-2orientations-8contrasts-15repeats', 
#                                      subset_rois =subset_cells_s, 
#                                      ylim=[-0.25, 0.2])
fig1, fig2, fig3, fig4, fig5 = generate_figures_GROUP(data_s_con, 
                                                      subplots_n=16, 
                                                      test=test, 
                                                      means='session', 
                                                      ylim=[-0.25,0.2], 
                                                      subset_rois= subset_cells_s, 
                                                      color_trace='firebrick')
#%% only neg resp cells to last contrast
subset_cells_s = []
for neg_cond in neg_cond_s:
    subset_cells = []
    for i, ROI_i in enumerate(neg_cond):
        if ROI_i[7]==True or ROI_i[15]==True:
            subset_cells.append(i)
    subset_cells_s.append(subset_cells)
print(sum([len(subset_cells) for subset_cells in subset_cells_s]))
#fig_traces, _ = plot_dFoF_of_protocol(data_s=data_s_con, 
#                                      protocol='ff-gratings-2orientations-8contrasts-15repeats', 
#                                      subset_rois = subset_cells_s, 
#                                      ylim=[-0.25, 0.2])

#%%
fig1, fig2, fig3, fig4, fig5 = generate_figures_GROUP(data_s_con, 
                                                      subplots_n=16, 
                                                      test=test, 
                                                      means='session', 
                                                      ylim=[-0.25,0.2], 
                                                      subset_rois= subset_cells_s, 
                                                      color_trace="mediumblue")
#%%
create_group_PDF(fig1, fig2, fig3, fig4, fig5, cell_type='NDNF', test=test)

#%%
#####################################################################################################################
########################################       STATIC GRATINGS      #################################################
######################################## 2 ORIENTATIONS 8 contrasts #################################################
########################################     SST (CIBELE's DATA)    #################################################
#####################################################################################################################
#%% SST CIBELE DATA
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','Ori-contrasts', 'SST-cells_WT_Adult_V1', 'NWBs_contrast')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
SESSIONS['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]

dFoF_options = {'roi_to_neuropil_fluo_inclusion_factor': 1.0,
                'method_for_F0': 'sliding_percentile',
                'sliding_window': 300.,
                'percentile': 10.,
                'neuropil_correction_factor': 0.8}

data_s_con_SST = []
nROIS = 0
for idx, filename in enumerate(SESSIONS['files']):
    data = Data(filename, verbose=False)
    data.build_dFoF(**dFoF_options)
    data.build_running_speed()
    data.build_facemotion()
    data.build_pupil_diameter()
    data_s_con_SST.append(data)
    nROIS += data.nROIs

#%%
nROIS

#%%
fig1, fig2, fig3, fig4, fig5 = generate_figures_GROUP(data_s_con_SST, 
                                                      subplots_n=16, 
                                                      test=test, 
                                                      means='session', 
                                                      subset_rois = None, 
                                                      color_trace = "k")
#%%
create_group_PDF(fig1, fig2, fig3, fig4, fig5, cell_type='SST', test=test)

#%%
#####################################################################################################################
########################################     DRIFTING GRATINGS   ####################################################
######################################## 1 DIRECTION 3 CONTRASTS ####################################################
########################################           NDNF          ####################################################
#####################################################################################################################
#%%
datafolder = os.path.join(Path("E:/"), 'DATA', 'In_Vivo_experiments','Vision-survey', 'NDNF-Cre','NWBs')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
SESSIONS['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]

dFoF_options = {'roi_to_neuropil_fluo_inclusion_factor': 1.0,
                'method_for_F0': 'sliding_percentile',
                'sliding_window': 300.,
                'percentile': 10.,
                'neuropil_correction_factor': 0.8}

data_s_con = []
for idx, filename in enumerate(SESSIONS['files']):
    data = Data(filename, verbose=False)
    data.build_dFoF(**dFoF_options)
    data.build_running_speed()
    data.build_facemotion()
    data.build_pupil_diameter()
    data_s_con.append(data)
    print(idx, data.protocols)


#%% ALL Cells
fig1, fig2, fig3, fig4, fig5 = generate_figures_GROUP(data_s_con, 
                                                      subplots_n=3, 
                                                      test=test, 
                                                      means='session', 
                                                      protocols = ["drifting-grating"], 
                                                      subset_rois = None, 
                                                      color_trace = "k")


#%% Subset cells
resp_cond_s, pos_cond_s, neg_cond_s = [], [], []
for filename in SESSIONS['files']:
    data = Data(filename, verbose=False)
    data.build_dFoF()
    protocol = "drifting-grating"
    stat_test_props = dict(interval_pre=[-1.,0],                                   
                        interval_post=[1,2],                                   
                        test='ttest')
    ep = EpisodeData(data,
                    protocol_name=protocol,
                    quantities=['dFoF'])
    resp_cond, pos_cond, neg_cond = calc_responsiveness2(ep, nROIs=data.nROIs)
    resp_cond_s.append(resp_cond)
    pos_cond_s.append(pos_cond)
    neg_cond_s.append(neg_cond)

#%% Only responsive cells 
subset_cells_s = []
for resp_cond in resp_cond_s:
    subset_cells = []
    for i, ROI_i in enumerate(resp_cond):
        if all(ROI_i[k] == False for k in range(3)): 
            continue
        else: 
            subset_cells.append(i)
    subset_cells_s.append(subset_cells)

print(sum([len(subset_cells) for subset_cells in subset_cells_s]))
print(subset_cells_s)
#%%

fig1, fig2, fig3, fig4, fig5 = generate_figures_GROUP(data_s_con, 
                                                      subplots_n=16, 
                                                      test=test, 
                                                      means='session', 
                                                      ylim=[-0.25,0.2], 
                                                      subset_rois= subset_cells_s, 
                                                      color_trace='green')

#%% Only pos resp cells to first contrast
subset_cells_s = []
for pos_cond in pos_cond_s:
    subset_cells = []
    for i, ROI_i in enumerate(pos_cond):
        if ROI_i[0]==True:
            subset_cells.append(i)
    subset_cells_s.append(subset_cells)

print(sum([len(subset_cells) for subset_cells in subset_cells_s]))

fig1, fig2, fig3, fig4, fig5 = generate_figures_GROUP(data_s_con, 
                                                      subplots_n=16, 
                                                      test=test, 
                                                      means='session', 
                                                      ylim=[-0.25,0.2], 
                                                      subset_rois= subset_cells_s, 
                                                      color_trace='firebrick')

#%% only neg resp cells to last contrast
subset_cells_s = []
for neg_cond in neg_cond_s:
    subset_cells = []
    for i, ROI_i in enumerate(neg_cond):
        if ROI_i[2]==True:
            subset_cells.append(i)
    subset_cells_s.append(subset_cells)
print(sum([len(subset_cells) for subset_cells in subset_cells_s]))

fig1, fig2, fig3, fig4, fig5 = generate_figures_GROUP(data_s_con, 
                                                      subplots_n=16, 
                                                      test=test, 
                                                      means='session', 
                                                      ylim=[-0.25,0.2], 
                                                      subset_rois= subset_cells_s, 
                                                      color_trace="mediumblue")
#%%
create_group_PDF(fig1, fig2, fig3, fig4, fig5, cell_type='NDNF', test=test)

#%%
#####################################################################################################################
########################################     NATURAL IMAGES      ####################################################
########################################   2 IMAGES 8 CONTRASTS  ####################################################
########################################           NDNF          ####################################################
#####################################################################################################################
#%%
datafolder = os.path.join(Path("E:/"), 'DATA', 'In_Vivo_experiments','2NatIm8contrasts', 'NDNF-Cre','NWBs')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
SESSIONS['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]

dFoF_options = {'roi_to_neuropil_fluo_inclusion_factor': 1.0,
                'method_for_F0': 'sliding_percentile',
                'sliding_window': 300.,
                'percentile': 10.,
                'neuropil_correction_factor': 0.8}

data_s_con_natIm = []
for idx, filename in enumerate(SESSIONS['files']):
    data = Data(filename, verbose=False)
    data.build_dFoF(**dFoF_options)
    data.build_running_speed()
    data.build_facemotion()
    data.build_pupil_diameter()
    data_s_con_natIm.append(data)
    print(idx, data.protocols)

#%%
fig1, fig2, fig3, fig4, fig5 = generate_figures_GROUP(data_s_con_natIm, 
                                                      subplots_n=16, 
                                                      test=test,
                                                      ylim=[-0.15,0.15])
#%%
create_group_PDF(fig1, fig2, fig3, fig4, fig5, cell_type='NDNF', test=test)
#%%
#%% SUBSET OF CELLS
resp_cond_s, pos_cond_s, neg_cond_s = [], [], []
for filename in SESSIONS['files']:
    data = Data(filename, verbose=False)
    data.build_dFoF()
    protocol = '2NaturalImages-8contrasts-15repeats'
    stat_test_props = dict(interval_pre=[-1.,0],                                   
                        interval_post=[1,2],                                   
                        test='ttest')
    ep = EpisodeData(data,
                    protocol_name=protocol,
                    quantities=['dFoF'])
    resp_cond, pos_cond, neg_cond = calc_responsiveness2(ep, nROIs=data.nROIs)
    resp_cond_s.append(resp_cond)
    pos_cond_s.append(pos_cond)
    neg_cond_s.append(neg_cond)

#%% ALL cells

#fig_traces, _ = plot_dFoF_of_protocol(data_s=data_s_con, 
#                                      protocol='ff-gratings-2orientations-8contrasts-15repeats', 
#                                      subset_rois = None, 
#                                      ylim=[-0.15, 0.15],
#                                      norm=True)
#fig_traces, _ = plot_dFoF_of_protocol(data_s=data_s_con, 
#                                      protocol='ff-gratings-2orientations-8contrasts-15repeats', 
#                                      subset_rois = None, 
#                                      ylim=[0.25, 0.55], 
#                                      norm=False)
fig1, fig2, fig3, fig4, fig5 = generate_figures_GROUP(data_s_con_natIm, 
                                                      subplots_n=16, 
                                                      test=test, 
                                                      means='session', 
                                                      ylim=[-0.15,0.15], 
                                                      subset_rois= None)
#%% Only responsive cells 
subset_cells_s = []
for resp_cond in resp_cond_s:
    subset_cells = []
    for i, ROI_i in enumerate(resp_cond):
        if all(ROI_i[k] == False for k in range(16)): 
            continue
        else: 
            subset_cells.append(i)
    subset_cells_s.append(subset_cells)

print(sum([len(subset_cells) for subset_cells in subset_cells_s]))
#%%
#fig_traces, _ = plot_dFoF_of_protocol(data_s=data_s_con, 
#                                      protocol='ff-gratings-2orientations-8contrasts-15repeats', 
#                                      subset_rois =subset_cells_s, 
#                                      ylim=[-0.25, 0.2])
fig1, fig2, fig3, fig4, fig5 = generate_figures_GROUP(data_s_con_natIm, 
                                                      subplots_n=16, 
                                                      test=test, 
                                                      means='session', 
                                                      ylim=[-0.25,0.2], 
                                                      subset_rois= subset_cells_s, 
                                                      color_trace='green')