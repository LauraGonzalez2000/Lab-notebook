# %% [markdown]
# FUNCTONS USEFUL TO PLOT RESPONSIVENESS

# %% PACKAGES
import os, sys, tempfile
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

sys.path += ['../../physion/src'] # add src code directory for physion
from physion.analysis.episodes.build import EpisodeData
from physion.analysis.episodes.trial_statistics import pre_post_statistics
from physion.dataviz.imaging import show_CaImaging_FOV
from physion.dataviz.raw import plot as plot_raw
from physion.utils import plot_tools as pt
from physion.analysis.protocols.orientation_tuning import\
        plot_orientation_tuning_curve, \
        plot_selectivity, \
        plot_responsiveness
from physion.analysis.protocols.orientation_tuning import compute_tuning_response_per_cells,\
                                                          plot_orientation_tuning_curve, \
                                                          plot_selectivity, \
                                                          plot_responsiveness
from physion.analysis.read_NWB import Data


from utils_.General_overview_episodes import\
        plot_dFoF_of_protocol,\
        plot_dFoF_of_protocol2
from utils_.Responsiveness_methods import\
        plot_protocol_responsiveness,\
        plot_resp_vs_param

#from Visual_Properties_analysis.Orientation_Tuning import compute_tunings2

from PDF_layout import PDF, PDF2, PDF3, PDF_ori_tuning, PDF_contrast_sensitivity
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
import random
import time



#%% FUNCTIONS

def find_available_settings(data, debug=False):

    settings = {'Locomotion': {'fig_fraction': 1,
                                                   'subsampling': 1,
                                                   'color': '#1f77b4'},
                        'FaceMotion': {'fig_fraction': 1,
                                                   'subsampling': 1,
                                                   'color': 'purple'},
                        'Pupil': {'fig_fraction': 2,
                                                  'subsampling': 1,
                                                  'color': '#d62728'},
                        'CaImaging': {'fig_fraction': 10,
                                                   'subsampling': 1,
                                                   'subquantity': 'dF/F',
                                                   'color': '#2ca02c'}}

    attributes = ['facemotion', 'pupil_diameter', 'dFoF']
    
    missing = [attr for attr in attributes if not hasattr(data, attr)]  # for objects

    if debug:
        if missing:
            print(f"Missing attributes: {missing}")
        else:
            print("All attributes exist")
    
    if missing==['pupil_diameter']:
        if debug:
            print("only pupil diameter missing")
        settings = {'Locomotion': {'fig_fraction': 1,
                                                   'subsampling': 1,
                                                   'color': '#1f77b4'},
                        'FaceMotion': {'fig_fraction': 1,
                                                   'subsampling': 1,
                                                   'color': 'purple'},
                        'CaImaging': {'fig_fraction': 10,
                                                   'subsampling': 1,
                                                   'subquantity': 'dF/F',
                                                   'color': '#2ca02c'}}
    
    if missing==['facemotion']:
        if debug:
            print("only pupil diameter missing")
        settings = {'Locomotion': {'fig_fraction': 1,
                                               'subsampling': 1,
                                               'color': '#1f77b4'},
                        'Pupil': {'fig_fraction': 2,
                                          'subsampling': 1,
                                          'color': '#d62728'},
                        'CaImaging': {'fig_fraction': 10,
                                               'subsampling': 1,
                                               'subquantity': 'dF/F',
                                               'color': '#2ca02c'}}
        
    if missing==['dFoF']:
        if debug:
            print("only Ca imaging missing")
        settings = {'Locomotion': {'fig_fraction': 1,
                                               'subsampling': 1,
                                               'color': '#1f77b4'},
                    'FaceMotion': {'fig_fraction': 1,
                                                   'subsampling': 1,
                                                   'color': 'purple'},
                    'Pupil': {'fig_fraction': 2,
                                          'subsampling': 1,
                                          'color': '#d62728'}}
                        
    
    if missing==['facemotion', 'pupil_diameter']:
        if debug:
            print('facemotion and pupil diameter missing')
        settings = {'Locomotion': {'fig_fraction': 1,
                                               'subsampling': 1,
                                               'color': '#1f77b4'},
                    'CaImaging': {'fig_fraction': 10,
                                               'subsampling': 1,
                                               'subquantity': 'dF/F',
                                               'color': '#2ca02c'}}
        
    return settings

def figure_to_array(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()

    buf = np.asarray(canvas.buffer_rgba())
    # drop alpha channel → RGB
    fig_arr = buf[:, :, :3].copy()

    return fig_arr

def get_roiIndex(data, type='pos', protocol="Natural-Images-4-repeats"):
    
    session_summary = {'significant':[], 'value':[]}

    ep = EpisodeData(data,
                        protocol_name=protocol,
                        quantities=['dFoF'], 
                        verbose=False)
    t0 = max([0, ep.time_duration[0]-1.5])
    stat_test_props = dict(interval_pre=[-1.5,0],                                   
                                interval_post=[t0, t0+1.5],                                   
                                test='ttest', 
                                sign='both')

    for roi_n in range(data.nROIs):
        
        roi_summary_data = pre_post_statistics(ep,
                                               episode_cond = ep.find_episode_cond(),
                                               response_args = dict(roiIndex=roi_n),
                                               response_significance_threshold=0.05,
                                               stat_test_props=stat_test_props,
                                               repetition_keys=list(ep.varied_parameters.keys()))
        
        session_summary['significant'].append(bool(roi_summary_data['significant']))
        session_summary['value'].append(roi_summary_data['value'])

    resp_cond = session_summary['significant']
    pos_cond = resp_cond.copy()
    neg_cond = resp_cond.copy()
 
    for i in range(len(resp_cond)):
        if  resp_cond[i]==True:
            if session_summary['value'][i]>0:
                pos_cond[i]=True
                neg_cond[i]=False
            else: 
                pos_cond[i]=False
                neg_cond[i]=True

    pos_roi = np.where(pos_cond)[0]
    neg_roi = np.where(neg_cond)[0]
    ns_roi = np.where(~np.array(resp_cond))[0]

    found = True

    if type=='pos':
        if len(pos_roi) > 0:
            roiIndex = random.choice(pos_roi)
        else:
            print("No positive ROIs found — choosing from non-significant set instead.")
            found = False
            roiIndex = random.choice(ns_roi)

    elif type=='neg':
        if len(neg_roi) > 0:
            roiIndex = random.choice(neg_roi)
        else:
            print("No negative ROIs found — choosing from non-significant set instead.")
            roiIndex = random.choice(ns_roi)
            found = False

    elif type=='ns':
        if len(ns_roi) >0: 
            roiIndex = random.choice(ns_roi) 
        else : 
            print("No ns ROIs found — choosing from significant set instead.")
            try : 
                roiIndex = random.choice(pos_roi)
                found = False
            except : 
                roiIndex = random.choice(neg_roi)
                found = False


    return roiIndex, found

def plot_protocol_ddFoF(ep, AX, idx, protocol="", subplots_n=16):

    ax = AX[idx] if idx is not None else AX

    t0 = max([0, ep.time_duration[0]-1.5])
    stat_test_props = dict(interval_pre=[-1.5,0],                                   
                            interval_post=[t0, t0+1.5],                                   
                            test='ttest', 
                            sign='both')
    
    summary_data = pre_post_statistics(ep,
                                       episode_cond = ep.find_episode_cond(),
                                       response_args = {},
                                       response_significance_threshold=0.05,
                                       stat_test_props=stat_test_props,
                                       repetition_keys=['repeat'],
                                       nMin_episodes=5,
                                       multiple_comparison_correction=True,
                                       loop_over_cells=False,
                                       verbose=True)
    
    mean_vals = [float(np.ravel(v)[0]) if np.size(v) > 0 else np.nan for v in summary_data['value']]

    target_len = subplots_n #generalize
    if len(mean_vals) < target_len:
        mean_vals.extend([np.nan] * (target_len - len(mean_vals)))
    else:
        mean_vals = mean_vals[:target_len]


    x = np.arange(target_len) #generalize
    ax.bar(x, mean_vals, alpha=0.8, capsize=4)
    ax.set_xticks(x)
    ax.set_title(f"{protocol.replace('Natural-Images-4-repeats','natural-images')}")
    ax.axhline(0, color='black', linewidth=0.8)

    if idx==0:
        AX[0].set_ylabel('variation dFoF')

def generate_figures(data_s, subplots_n=16):
    start_time = time.time()

    for data in data_s:
        dict_annotation = {'name': data.filename,
                           'Subject_ID': data.metadata['subject_ID'],
                           'protocol': data.metadata['protocol']}

        settings = find_available_settings(data)
        protocols = [p for p in data.protocols if (p != 'grey-10min') and (p != 'black-2min') and (p != 'quick-spatial-mapping')]

        ################################# FIGURES ###############################################################
        
        fig1, AX1 = pt.figure(axes=(3,1), figsize=(1.4,3), wspace=0.15)
        show_CaImaging_FOV(data, key='meanImg',cmap=pt.get_linear_colormap('k', 'tab:green'),NL=2, ax=AX1[0])
        show_CaImaging_FOV(data, key='max_proj',cmap=pt.get_linear_colormap('k', 'tab:green'),NL=2, ax=AX1[1])
        show_CaImaging_FOV(data, key='meanImg',cmap=pt.get_linear_colormap('k', 'tab:green'),NL=2,  ax=AX1[2])
        

        if hasattr(data, "dFoF") and data.dFoF is not None and len(data.dFoF) > 0:
            zoom_area = [[(2/20)*data.t_dFoF[-1], (3/20)*data.t_dFoF[-1]],
                         [(15/20)*data.t_dFoF[-1], (16/20)*data.t_dFoF[-1]]]
            
            fig2, _ = plot_raw(data, 
                                tlim=[0, data.t_dFoF[-1]], 
                                settings=settings, 
                                figsize=(9,3),
                                zoom_area=[((2/20)*data.t_dFoF[-1], (3/20)*data.t_dFoF[-1]),
                                                ((15/20)*data.t_dFoF[-1], (16/20)*data.t_dFoF[-1])])
        
            fig3, _ = plot_raw(data, 
                               tlim = zoom_area[0],
                               settings=settings, 
                               figsize=(9,3))
            
            fig4, _ = plot_raw(data, 
                               tlim=zoom_area[1],
                               settings=settings, 
                               figsize=(9,3))
            
          
        fig5, _ = plot_dFoF_of_protocol(data_s=[data], protocol=protocols[0], subplots_n=subplots_n)
        
        fig10, AX10 = pt.figure(axes = (len(protocols),1),figsize=(1.4,8) )

        ep = EpisodeData(data, protocol_name=protocols[0], quantities=['dFoF'])
        plot_protocol_ddFoF(ep=ep, AX=AX10, idx=None, protocol=protocols[0], subplots_n=subplots_n)
        
        roiIndex, found = get_roiIndex(data, type='pos', protocol=data.protocols[0])
        fig6, _ = plot_dFoF_of_protocol2(data_s=[data], roiIndex=roiIndex, found=found)
        
        roiIndex, found = get_roiIndex(data, type='neg', protocol=data.protocols[0])
        fig7, _ = plot_dFoF_of_protocol2(data_s=[data], roiIndex=roiIndex, found=found)
        
        roiIndex, found = get_roiIndex(data, type='ns', protocol=data.protocols[0])
        fig8, _ = plot_dFoF_of_protocol2(data_s=[data], roiIndex=roiIndex, found=found)
        
        fig9, AX9 = pt.figure(axes = (len(protocols),1))
        nROIs = data.nROIs
        plot_protocol_responsiveness(ep, nROIs, AX=AX9, protocol=protocols[0])
        
        fig1 = figure_to_array(fig1)
        fig2 = figure_to_array(fig2)
        fig3 = figure_to_array(fig3)
        fig4 = figure_to_array(fig4)
        fig5 = figure_to_array(fig5)
        fig6 = figure_to_array(fig6)
        fig7 = figure_to_array(fig7)
        fig8 = figure_to_array(fig8)
        fig9 = figure_to_array(fig9)
        fig10 = figure_to_array(fig10)

        #create_PDF(dict_annotation, fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, cell_type)

    elapsed = time.time() - start_time
    print(f"Execution time: {elapsed:.2f} seconds")
        
    return dict_annotation, fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10    

def create_PDF(dict_annotation, fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, cell_type):
    try: 
        pdf1 = PDF()
        pdf1.fill_PDF(dict_annotation, fig1, fig2, fig3, fig4)
        fig_p1 = pdf1.fig

        pdf2 = PDF2()
        pdf2.fill_PDF2(fig5, fig10)
        fig_p2 = pdf2.fig

        pdf3 = PDF3()
        pdf3.fill_PDF3(fig6, fig7, fig8, fig9)
        fig_p3 = pdf3.fig

        output_path = f"C:/Users/laura.gonzalez/Output_expe/In_Vivo/{cell_type}/Summary_PDF/{os.path.splitext(dict_annotation['name'])[0]}_summary.pdf"

        with PdfPages(output_path) as pdf:
                pdf.savefig(fig_p1, dpi=300, bbox_inches="tight")  # Page 1
                pdf.savefig(fig_p2, dpi=300, bbox_inches="tight")  # Page 2
                pdf.savefig(fig_p3, dpi=300, bbox_inches="tight")  # Page 3

        print("Individual PDF File saved successfully ")
        
    except Exception as e:
        print(f"Error creating the individual PDF file : {e}")

    return 0

def plot_barplot2_per_protocol(data_s, AX,idx,  p, subplots_n):
    mean_vals_s = []  # store per-session mean responses

    for data in data_s:
        ep = EpisodeData(data, protocol_name=p, quantities=['dFoF'])

        t0 = max([0, ep.time_duration[0] - 1.5])
        stat_test_props = dict(
            interval_pre=[-1.5, 0],
            interval_post=[t0, t0 + 1.5],
            test='ttest',
            sign='both')

        summary_data = pre_post_statistics(ep,
                                           episode_cond = ep.find_episode_cond(),
                                           response_args = {},
                                           response_significance_threshold=0.05,
                                           stat_test_props=stat_test_props,
                                           repetition_keys=['repeat'])
        
        # Extract ROI mean values
        mean_vals = [float(np.ravel(v)[0]) if np.size(v) > 0 else np.nan for v in summary_data['value']]

        # Pad/truncate to 5 elements
        target_len = subplots_n
        mean_vals = (mean_vals + [np.nan] * target_len)[:target_len]

        mean_vals_s.append(mean_vals)

    # Compute session-aggregated mean and SEM
    values = np.nanmean(mean_vals_s, axis=0)
    yerr = stats.sem(mean_vals_s, axis=0, nan_policy='omit')

    # Plot directly
    x = np.arange(len(values))
    AX[idx].bar(
        x, values, yerr=yerr,
        alpha=0.8, capsize=0,
        error_kw=dict(linewidth=0.6)
    )
    AX[idx].set_xticks(x)
    AX[idx].set_title(f"{p.replace('Natural-Images-4-repeats','natural-images')}")
    AX[idx].axhline(0, color='black', linewidth=0.8)
    
    if idx==0:
        AX[0].set_ylabel('variation dFoF')
    
    return 0

def plot_barplot2_of_protocol(data_s, AX, idx,  p, subplots_n, subset_rois=None):
    mean_vals_s = []  # store per-session mean responses

    for data in data_s:
        print("protocol ", p[0])
        ep = EpisodeData(data, protocol_name=p[0], quantities=['dFoF'])

        t0 = max([0, ep.time_duration[0] - 1.5])
        stat_test_props = dict(
            interval_pre=[-1.5, 0],
            interval_post=[t0, t0 + 1.5],
            test='ttest',
            sign='both')

        summary_data = pre_post_statistics(ep,
                                           episode_cond = ep.find_episode_cond(),
                                           response_args = {},
                                           response_significance_threshold=0.05,
                                           stat_test_props=stat_test_props,
                                           repetition_keys=['repeat'])
        print("summary data !!", summary_data)
        
        # Extract ROI mean values
        mean_vals = [float(np.ravel(v)[0]) if np.size(v) > 0 else np.nan for v in summary_data['value']]
        print("mean vals!", mean_vals)
        # Pad/truncate to subplots_n elements
        target_len = subplots_n
        mean_vals = (mean_vals + [np.nan] * target_len)[:target_len]

        mean_vals_s.append(mean_vals)
        """
        if p == 'ff-gratings-8orientation-2contrasts-15repeats':
            #reorder mean_vals
            mean_vals_ = np.zeros(16)
            mean_vals_[0] = mean_vals[0]
            mean_vals_[1] = mean_vals[2]
            mean_vals_[2] = mean_vals[4]
            mean_vals_[3] = mean_vals[6]
            mean_vals_[4] = mean_vals[8]
            mean_vals_[5] = mean_vals[10]
            mean_vals_[6] = mean_vals[12]
            mean_vals_[7] = mean_vals[14]

            mean_vals_[8] = mean_vals[1]
            mean_vals_[9] = mean_vals[3]
            mean_vals_[10] = mean_vals[5]
            mean_vals_[11] = mean_vals[7]
            mean_vals_[12] = mean_vals[9]
            mean_vals_[13] = mean_vals[11]
            mean_vals_[14] = mean_vals[13]
            mean_vals_[15] = mean_vals[15]

            mean_vals = mean_vals_

        mean_vals_s.append(mean_vals)
        """

    ep0 = EpisodeData(data_s[0], protocol_name=p[0], quantities=['dFoF'])

    varied_keys = list(ep0.varied_parameters.keys())
    angles = ep0.varied_parameters[varied_keys[0]]
    contrasts = ep0.varied_parameters[varied_keys[1]]

    print("protocol:", p[0])

    if p[0] == 'ff-gratings-8orientation-2contrasts-15repeats':
        param_values = [f"a={a:.1f}° , C={c:.1f}" for c in contrasts for a in angles]
    elif p[0] == 'ff-gratings-2orientations-8contrasts-15repeats':
        param_values = [f"a={a:.1f}° , C={c:.2f}" for a in angles for c in contrasts]
    elif p[0]== '2NaturalImages-8contrasts-15repeats':
        param_values = [f"Img={a:.1f}° , C={c:.2f}" for a in angles for c in contrasts]
    else: 
        contrasts = ep0.varied_parameters[varied_keys[0]]
        param_values = [f"C={c:.2f}" for c in contrasts]

    # Compute session-aggregated mean and SEM
    print("mean vals s \n",mean_vals_s)
    values = np.nanmean(mean_vals_s, axis=0)
    yerr = stats.sem(mean_vals_s, axis=0, nan_policy='omit')
    x = np.arange(len(values))
    print(x)
    print(values)
    # Plot directly
    AX.bar(x, values, 
           yerr=yerr,
           alpha=0.8, 
           capsize=0,
           error_kw=dict(linewidth=0.6))
    AX.set_xticks(x)
    print("param values : ", param_values)
    print("actual ticks ", x)
    print("actual values? ", values)
    AX.set_xticklabels(param_values,rotation=90, ha="center")
    AX.axhline(0, color='black', linewidth=0.8)
    
    if idx==0:
        AX.set_ylabel('variation \ndFoF')
    
    return 0

def generate_figures_GROUP(data_s, subplots_n, test="angle", means='ROI', protocols = [], ylim= [-0.2,1.0], subset_rois=None):
    start_time = time.time()  

    if protocols == []:
        protocols = [p for p in data_s[0].protocols 
                        if (p != 'grey-10min') and (p != 'black-2min') and (p != 'quick-spatial-mapping')]

    fig_traces, _     = plot_dFoF_of_protocol(data_s=data_s, protocol=protocols[0], ylim=ylim, subset_rois=subset_rois)
    elapsed = time.time() - start_time
    print(f"Fig 1 ok: {elapsed:.2f} seconds")

    fig_vdFoF, AX2  = pt.figure(axes = (1,1), ax_scale=(3, 1))
    plot_barplot2_of_protocol(data_s, AX2, 0, protocols, subplots_n, subset_rois = subset_rois)

    fig_resp_vs_param, AX3 = pt.figure(axes = (1,1),figsize=(2,2), ax_scale=(2, 5))
    fig_resp_vs_param2, AX4 = pt.figure(axes = (1,1),figsize=(2,2), ax_scale=(2, 5))
    fig_resp_vs_param3, AX5 = pt.figure(axes = (1,1),figsize=(2,2), ax_scale=(2, 5))

    
    if protocols[0] == "2NaturalImages-8contrasts-15repeats":
        plot_resp_vs_param(data_s, p=protocols[0], AX=AX3, test = test, repetition_keys = ["repeat", "Image-ID"], means='ROI', param=1.0, param_type="Image-ID", subset_rois = subset_rois)
        plot_resp_vs_param(data_s, p=protocols[0], AX=AX4, test = test, repetition_keys = ["repeat", "Image-ID"], means='ROI', param=2.0, param_type="Image-ID", subset_rois = subset_rois)
        plot_resp_vs_param(data_s, p=protocols[0], AX=AX5, test = test, repetition_keys = ["repeat", "Image-ID"], means='ROI', param=None, param_type="Image-ID", subset_rois = subset_rois)

    elif protocols[0] == 'ff-gratings-2orientations-8contrasts-15repeats':
        plot_resp_vs_param(data_s, p=protocols[0], AX=AX3, test = test, repetition_keys = ["repeat", "angle"], means=means, param=0.0, param_type="angle", subset_rois = subset_rois)
        plot_resp_vs_param(data_s, p=protocols[0], AX=AX4, test = test, repetition_keys = ["repeat", "angle"], means=means, param=90.0, param_type="angle", subset_rois = subset_rois)
        plot_resp_vs_param(data_s, p=protocols[0], AX=AX5, test = test, repetition_keys = ["repeat", "angle"], means=means, param=None, param_type="angle", subset_rois = subset_rois)
    
    elif protocols[0] == 'ff-gratings-8orientation-2contrasts-15repeats':
        plot_resp_vs_param(data_s, p=protocols[0], AX=AX3, test = test, repetition_keys = ["repeat"], means=means, param=0.5, param_type="contrast", subset_rois = subset_rois)
        plot_resp_vs_param(data_s, p=protocols[0], AX=AX4, test = test, repetition_keys = ["repeat"], means=means, param=1.0, param_type="contrast", subset_rois = subset_rois)
        plot_resp_vs_param(data_s, p=protocols[0], AX=AX5, test = test, repetition_keys = ["repeat", "contrast"], means=means, param=None, param_type="contrast", subset_rois = subset_rois)

    elif protocols[0] == 'drifting-grating':
        plot_resp_vs_param(data_s, p=protocols[0], AX=AX3, test = test, repetition_keys = ["repeat", "angle"], means=means, param=None)


    fig1 = figure_to_array(fig_traces)
    fig2 = figure_to_array(fig_vdFoF)
    fig3 = figure_to_array(fig_resp_vs_param)
    fig4 = figure_to_array(fig_resp_vs_param2)
    fig5 = figure_to_array(fig_resp_vs_param3)
    print(f"Fig 2-3-4-5 ok: {elapsed:.2f} seconds")

    if test =='contrast':
        return fig1, fig2, fig3, fig4, fig5

    stat_test_props = dict(interval_pre=[-1.,0],                                   
                       interval_post=[1.,2.],                                   
                       test='ttest',                                            
                       sign='positive')
    
    response_significance_threshold=0.05
    
    compute_tunings2(data_s = data_s, 
                    stat_test_props=stat_test_props, 
                    response_significance_threshold=response_significance_threshold)
    

    fig_resp_per_param, AX6 = plot_responsiveness(['WT_contrast-1.0', 'WT_contrast-0.5'],
                                    average_by='ROIs', #issue !!  #not average by subject
                                    path=tempfile.tempdir, 
                                    fig_args={'figsize':  (2,1.), 
                                              'ax_scale': (1.,1.)})
    

    fig_selectivity, AX7 = plot_selectivity(['WT_contrast-1.0', 'WT_contrast-0.5'],
                                 average_by='ROIs',
                                 #using='fit',
                                 path=tempfile.tempdir)
    
    fig_tuning_curve, AX8 = plot_orientation_tuning_curve(['WT_contrast-1.0', 'WT_contrast-0.5'],
                                              average_by='ROIs',
                                              path=tempfile.tempdir)
        
    elapsed = time.time() - start_time
    print(f"Fig 6-7-8 ok: {elapsed:.2f} seconds")

    
    fig6 = figure_to_array(fig_resp_per_param)
    fig7 = figure_to_array(fig_selectivity)
    fig8 = figure_to_array(fig_tuning_curve)

    elapsed = time.time() - start_time
    print(f"Total time: {elapsed:.2f} seconds")

    if test=='angle':
        return fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8
    
def create_group_PDF(fig1, fig2, fig3, fig4, fig5, fig6=None, fig7=None, fig8=None, cell_type='', test='angle'):
    try: 
        if test=="angle":
            pdf1 = PDF_ori_tuning() #PDF_angle_contrast()
            pdf1.fill_PDF(fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8)
        
        if test=="contrast":
            pdf1 = PDF_contrast_sensitivity()
            pdf1.fill_PDF(fig1, fig2, fig3, fig4, fig5)

        fig_p1 = pdf1.fig
        output_path = f"C:/Users/laura.gonzalez/Output_expe/In_Vivo/{cell_type}/Summary_PDF/GROUP_summary.pdf"
        with PdfPages(output_path) as pdf:
                pdf.savefig(fig_p1, dpi=300, bbox_inches="tight")  # Page 1

        print("GROUP PDF File saved successfully ")
    except Exception as e:
        print(f"Error creating GROUP PDF file : {e}")

    return 0

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
                        Tuning['subject'] = data.nwbfile.subject.subject_id
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
                        Tuning['subject'] = data.nwbfile.subject.subject_id
                        Tunings.append(Tuning)
                # saving data
                np.save(os.path.join(tempfile.tempdir, 'Tunings_WT_contrast-%.1f.npy' % contrast),Tunings)
        return 0