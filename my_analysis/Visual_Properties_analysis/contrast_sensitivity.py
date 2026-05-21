# %% [markdown]
# STUDIES CONTRAST SENSITIVITY ON DIFFERENT DATASETS

#%%
import os, sys
from pathlib import Path

sys.path += ['../../physion/src'] # add src code directory for physion
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles

sys.path += ['../']
from utils_.contrast_angle_sensitivity_methods import\
            generate_figures, \
            create_PDF, \
            generate_figures_GROUP, \
            create_group_PDF

#%% 
# Variables : 
test = "contrast"

#%% ##############################################################################################################
########################################       STATIC GRATINGS      ##############################################
######################################## 2 ORIENTATIONS 8 contrasts ##############################################
########################################            NDNF            ##############################################
##################################################################################################################
#%% LOAD DATA
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','Ori-contrasts', 'NDNF-Cre', 'NWBs_8contrasts2ori')
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
#%% [markdown]
## All individual files
#%%
dict_annotation, fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10 = generate_figures(data_s_con, subplots_n=16)
create_PDF(dict_annotation, fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, cell_type='NDNF')
#%%
fig1, fig2, fig3, fig4, fig5 = generate_figures_GROUP(data_s_con, subplots_n=16, test=test, means='session')
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
for idx, filename in enumerate(SESSIONS['files']):
    data = Data(filename, verbose=False)
    data.build_dFoF(**dFoF_options)
    data.build_running_speed()
    data.build_facemotion()
    data.build_pupil_diameter()
    data_s_con_SST.append(data)

#%%
fig1, fig2, fig3, fig4, fig5 = generate_figures_GROUP(data_s_con_SST, subplots_n=16, test=test, means='session')
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

#%%
fig1, fig2, fig3, fig4, fig5 = generate_figures_GROUP(data_s_con, subplots_n=3, test=test, means='ROI', protocols = ["drifting-grating"])
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
fig1, fig2, fig3, fig4, fig5 = generate_figures_GROUP(data_s_con_natIm, subplots_n=16, test=test)
#%%
create_group_PDF(fig1, fig2, fig3, fig4, fig5, cell_type='NDNF', test=test)