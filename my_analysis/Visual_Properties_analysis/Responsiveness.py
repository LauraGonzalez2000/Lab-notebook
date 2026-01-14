# %% [markdown]
# # Responsiveness

#%%
#why are Data created??
import sys, os
import numpy as np
import matplotlib.pyplot as plt

sys.path += ['../physion/src'] # add src code directory for physion
from physion.analysis.read_NWB import Data
from physion.utils import plot_tools as pt
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from physion.analysis.episodes.build import EpisodeData

from General_summary.Generate_PDF_summary_opti import plot_responsiveness2_per_protocol, plot_responsiveness_per_protocol

from collections import Counter, defaultdict
#%%
#%%
############################################################################
######################## FUNCTIONS #########################################
############################################################################ 

def study_responsiveness(SESSIONS, index, protocol):

    filename = SESSIONS['files'][index]
    data = Data(filename, verbose=False)
    data.build_dFoF()
   
    protocol = protocol
    stat_test_props = dict(interval_pre=[-1.,0],                                   
                        interval_post=[1.,2.],                                   
                        test='ttest')

    ep = EpisodeData(data,
                    protocol_name=protocol,
                    quantities=['dFoF'])

    print("dFoF shape : ", ep.dFoF.shape)
    print("varied parameters : ", ep.varied_parameters)

    values = []
    significance = []
    colors = []

    for roi_n in range(data.nROIs):
        ep = EpisodeData(data,
                        protocol_name=protocol,
                        quantities=['dFoF'], 
                        verbose=False)

        summary_data = ep.compute_summary_data(stat_test_props,
                                            exclude_keys=['repeat', 'angle', 'contrast'],
                                            response_significance_threshold=0.05,
                                            response_args=dict(roiIndex=roi_n))
        
        if summary_data['significant']: 
            if summary_data['value'] < 0: color = 'red'
            else: color = 'green'
            colors.append(color)
        else: 
            if summary_data['value'] < 0: color = 'pink'
            else: color = 'lime'
            colors.append(color)

        values.append(summary_data['value'].flatten())
        significance.append(summary_data['significant'].flatten())

    fig, AX = plt.subplots(1, 1, figsize=(1, 1))
    x= np.arange(0,len(values),1)
    y = [float(value) for value in values]
    AX.bar(x, y, color=colors)
    AX.set_xlabel('ROI #')
    AX.set_ylabel('Responsiveness')
    AX.set_title(f'Session #{index}')
    print(significance)
    true_indexes = [i for i, val in enumerate(significance) if val]
    false_indexes = [i for i, val in enumerate(significance) if not val]
    print(true_indexes)
    print(f'{len(true_indexes)} significant ROI out of {len(significance)} ROIs')
    return 0

def study_responsiveness_all(SESSIONS, protocol):

    Colors = []
    Values = []
    Significance = []

    for rec in range(len(SESSIONS['files'])):
        filename = SESSIONS['files'][rec]
        data = Data(filename, verbose=False)
        data.build_dFoF()

        protocol = "static-patch"
        stat_test_props = dict(interval_pre=[-1.,0],                                   
                            interval_post=[1.,2.],                                   
                            test='ttest')
        
        ep = EpisodeData(data,
                        protocol_name=protocol,
                        quantities=['dFoF'], 
                        verbose=False)
        
        values = []
        significance = []
        colors = []

        for roi_n in range(data.nROIs):
            
            summary_data = ep.compute_summary_data(stat_test_props,
                                                exclude_keys=['repeat', 'angle', 'contrast'],
                                                response_significance_threshold=0.05,
                                                response_args=dict(roiIndex=roi_n))
            
            if summary_data['significant']: 
                if summary_data['value'] < 0: color = 'red'
                else: color = 'green'
                colors.append(color)
            else: 
                if summary_data['value'] < 0: color = 'pink'
                else: color = 'lime'
                colors.append(color)

            values.append(summary_data['value'].flatten())
            significance.append(summary_data['significant'].flatten())
        
        Colors.append(colors)
        Values.append(values)
        Significance.append(significance)

    
    fig, AX = plt.subplots(5, 5, figsize=(9, 9))

    i,j = 0,0

    for rec in range(len(SESSIONS['files'])):

        x= np.arange(0,len(Values[rec]),1)
        y = [float(value) for value in Values[rec]]    

        AX[i][j].bar(x, y, color=Colors[rec])
        AX[i][j].set_xlabel('ROI #')
        AX[i][j].set_ylabel('Responsiveness')
        AX[i][j].set_title(f'Session #{rec}')

        # ---- PIE CHART ----
        # Count responsive cells
        n_total = len(Significance[rec])
        true_indexes = [i for i, val in enumerate(Significance[rec]) if val]
        false_indexes = [i for i, val in enumerate(Significance[rec]) if not val]
        
        excit_indexes = sum(1 for v in true_indexes if v > 0)
        inhib_indexes = sum(1 for v in true_indexes if v < 0)
        n_nonresponsive = len(false_indexes)
        n_total = len(Significance[rec])
        
        
        pie_counts = [excit_indexes, inhib_indexes, n_nonresponsive]
        pie_colors = ['g', 'r', 'gray']
        
        # Add as inset
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        ax_inset = inset_axes(AX[i][j], width="30%", height="30%", loc='upper right')
        ax_inset.pie(pie_counts, colors=pie_colors)#, autopct='%1.0f%%')
        #ax_inset.set_title('Responsive cells', fontsize=8)

        #update position next graph
        if j<4:
            j+=1
        else: 
            i+=1
            j=0

    AX = AX.flatten()
    for idx in range(len(SESSIONS['files']), len(AX)):
        fig.delaxes(AX[idx])
    fig.tight_layout()
    return 0

def plot_responsiveness_per_protocol(ep, nROIs, AX, idx, p, Resp_ROI_dict):
    
    session_summary = {'significant':[], 'value':[]}

    for roi_n in range(nROIs):

        t0 = max([0, ep.time_duration[0]-1.5])

        stat_test_props = dict(interval_pre=[-1.5,0],                                   
                                interval_post=[t0, t0+1.5],                                   
                                test='ttest', 
                                sign='both')

        roi_summary_data = ep.compute_summary_data(stat_test_props=stat_test_props,
                                                   #exclude_keys=['repeat'],
                                                   exclude_keys= list(ep.varied_parameters.keys()), # we merge different stimulus properties as repetitions of the stim. type  
                                                   response_significance_threshold=0.05,
                                                   response_args=dict(roiIndex=roi_n))
        
        session_summary['significant'].append(bool(roi_summary_data['significant'][0]))
        session_summary['value'].append(roi_summary_data['value'][0])
        print("roi :", roi_n, ", resp : ",  bool(roi_summary_data['significant'][0]), ", value : ",roi_summary_data['value'][0], "\n")
        if bool(roi_summary_data['significant'][0])==False:
            category = 'NS'
        else: 
            if roi_summary_data['value'][0]>0:
                category = "Positive"
            else: 
                category = "Negative"

        Resp_ROI_dict[f"ROI_{roi_n}"][p] = category

    resp_cond = np.array(session_summary['significant'])                     
    pos_cond = resp_cond & ([session_summary['value'][i]>0 for i in range(len(session_summary['value']))])
    neg_cond = resp_cond & ([session_summary['value'][i]<0 for i in range(len(session_summary['value']))])

    print(f'Protocol {p} : {sum(resp_cond)} significant ROI ({np.sum(pos_cond)} positive, {np.sum(neg_cond)} negative) out of {len(session_summary['significant'])} ROIs')

    pos_frac = np.sum(pos_cond)/nROIs
    neg_frac = np.sum(neg_cond)/nROIs
    ns_frac = 1-pos_frac-neg_frac

    colors = ['green', 'red', 'grey']

    pt.pie(data=[pos_frac, neg_frac, ns_frac], ax = AX[idx], COLORS = colors)#, pie_labels = ['%.1f%%' % (100*pos),'%.1f%%' % (100*neg), '%.1f%%' % (100*ns)] )
    
    AX[idx].set_title(f'{p.replace('Natural-Images-4-repeats','natural-images')}')
    pt.annotate(AX[idx], '+ resp=%.1f%% ' % (100*pos_frac), (1, 0), ha='right', va='top')
    pt.annotate(AX[idx], '- resp=%.1f%%' % (100*neg_frac), (1, -0.2), ha='right', va='top')
    pt.annotate(AX[0], f'{nROIs} ROIs', (1, -0.4), ha='right', va='top')
    return 0

#%%
############################
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-WT-Dec-2022','NWBs')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
SESSIONS['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]

dFoF_options = {
        'roi_to_neuropil_fluo_inclusion_factor': 1.0,
        'method_for_F0': 'sliding_percentile',
        'sliding_window': 300.,
        'percentile': 10.,
        'neuropil_correction_factor': 0.8}

data_s = []
for idx, filename in enumerate(SESSIONS['files']):
    data = Data(filename, verbose=False)
    data.build_dFoF(**dFoF_options, verbose=False)
    data.build_running_speed()
    data.build_facemotion()
    data.build_pupil_diameter()
    data_s.append(data)

#%%
protocols = ["static-patch"]#, "drifting-gratings", "Natural-Images-4-repeats"]
fig, AX  = pt.figure(axes = (len(protocols),1))
for idx, p in enumerate(protocols):
    plot_responsiveness2_per_protocol(data_s, AX, idx, p, type='ROI')


# STUDY evolution responsiveness
#%%
nROIs = data.nROIs  

Resp_ROI_dict = {
    f"ROI_{i}": {
        "static-patch": None,
        "drifting-gratings": None,
        "Natural-Images-4-repeats": None
    }
    for i in range(nROIs)
}

states = ["Positive", "NS", "Negative"]
colors = {"Positive": "green","NS": "grey","Negative": "red"}


totals = {cond: Counter(roi[cond] for roi in Resp_ROI_dict.values()) for cond in protocols}
print(totals)
transitions = []
for roi in Resp_ROI_dict.values():
    for c1, c2 in zip(protocols[:-1], protocols[1:]):
        transitions.append((c1, roi[c1], c2, roi[c2]))

transition_counts = Counter(transitions)

y_base = {}
for cond in protocols:
    y = 0
    for state in states:
        y_base[(cond, state)] = y
        y += totals[cond][state]

y_offset = defaultdict(int)

# Plot
fig, ax = plt.subplots(figsize=(6, 4))

x_pos = {cond: i for i, cond in enumerate(protocols)}

for (c1, s1, c2, s2), n in transition_counts.items():
    y1 = y_base[(c1, s1)] + y_offset[(c1, s1)] + n / 2
    y2 = y_base[(c2, s2)] + y_offset[(c2, s2)] + n / 2

    ax.plot(
        [x_pos[c1], x_pos[c2]],
        [y1, y2],
        color=colors[s1],
        linewidth=2 * n,
        alpha=0.6,
    )

    y_offset[(c1, s1)] += n
    y_offset[(c2, s2)] += n

ax.set_xticks(range(len(protocols)))
ax.set_xticklabels(protocols)
ax.set_ylabel("Number of ROIs")
ax.set_title("Alluvial plot of ROI responsiveness")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.show()

#%%


fig, AX = pt.figure(axes = (3,1))
protocols = ["static-patch", "drifting-gratings", "Natural-Images-4-repeats"]

for i, p in enumerate(protocols): 
    ep = EpisodeData(data, protocol_name=p, quantities=['dFoF'])
    plot_responsiveness_per_protocol(ep, data.nROIs, AX=AX, idx=i, p=p, Resp_ROI_dict=Resp_ROI_dict )

#%%
print(Resp_ROI_dict)


#%%
'''
def generate_input_data(Resp_ROI_dict, prot1, prot2):
    
    count_pos_pos = 0
    count_pos_ns = 0
    count_pos_neg = 0
    count_ns_pos = 0
    count_ns_ns = 0
    count_ns_neg = 0
    count_neg_pos = 0
    count_neg_ns = 0
    count_neg_neg = 0

    for ROI in Resp_ROI_dict:
        resp1 = Resp_ROI_dict[ROI][prot1]
        resp2 = Resp_ROI_dict[ROI][prot2]

        if resp1 =='Positive' and resp2 =='Positive':
            count_pos_pos +=1
        elif resp1 =='Positive' and resp2 =='NS':
            count_pos_ns += 1
        elif resp1 =='Positive' and resp2 =='Negative':
            count_pos_neg += 1
        
        elif resp1 =='NS' and resp2 =='Positive':
            count_ns_pos +=1
        elif resp1 =='NS' and resp2 =='NS':
            count_ns_ns += 1
        elif resp1 =='NS' and resp2 =='Negative':
            count_ns_neg += 1
        
        elif resp1 =='Negative' and resp2 =='Positive':
            count_neg_pos +=1
        elif resp1 =='Negative' and resp2 =='NS':
            count_neg_ns += 1
        elif resp1 =='Negative' and resp2 =='Negative':
            count_neg_neg += 1
        
    input_data = {'Positive': {'Positive_': count_pos_pos, 'Ns_': count_pos_ns, 'Negative_': count_pos_neg},
                  'Ns': {'Positive_': count_ns_pos, 'Ns_': count_ns_ns,'Negative_': count_ns_neg},
                  'Negative': {'Positive_': count_neg_pos, 'Ns_': count_neg_ns, 'Negative_': count_neg_neg}}

    return input_data
'''
def generate_input_data(Resp_ROI_dict, prot1, prot2, categories=("Positive", "NS", "Negative")):

    input_data = {src: {dst + "_": 0 for dst in categories} for src in categories}

    for ROI, responses in Resp_ROI_dict.items():
        src = responses[prot1]
        dst = responses[prot2] + "_"
        input_data[src][dst] += 1

    return input_data

#%%
import General_summary.alluvial as alluvial

#prot1 = "static-patch"
#prot2 = "drifting-gratings"

#prot1="drifting-gratings"
#prot2="Natural-Images-4-repeats"

prot1="Natural-Images-4-repeats"
prot2="static-patch"

input_data = generate_input_data(Resp_ROI_dict, prot1, prot2)

ax = alluvial.plot(
    input_data,
    colors=["red", "grey", "green"],
    src_label_override=["Negative", "NS", "Positive"],
    dst_label_override=["Negative_", "NS_", "Positive_"]
)

fig = ax.get_figure()
fig.set_size_inches(5,5)
ax.text(0.1, 0, prot1, ha="center", va="top", transform=ax.transAxes)
ax.text(0.9, 0, prot2, ha="center", va="top", transform=ax.transAxes)
plt.show()


# src_label_override=["Positive", 'Ns', 'Negative']


#################################

############################################################################################################
############################## PLOT PER SESSION and PER ROI ################################################
############################################################################################################

########################################### STATIC PATCH     ###############################################
#%%
protocol="static-patch"
#%%
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-Cre-batch1','NWBs')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
study_responsiveness(SESSIONS=SESSIONS, index=0, protocol=protocol)
study_responsiveness_all(SESSIONS=SESSIONS, protocol=protocol)
#%%
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-Cre-batch1','NWBs_centered')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
study_responsiveness_all(SESSIONS=SESSIONS, protocol=protocol)
#%%
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-WT-Dec-2022','NWBs')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
study_responsiveness_all(SESSIONS=SESSIONS, protocol=protocol)


########################################### DRIFTING GRATING ###############################################
#%%
protocol="drifting-grating"
#%% DATA ALL
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-Cre-batch1','NWBs')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
study_responsiveness_all(SESSIONS=SESSIONS, protocol=protocol)
#%% DATA CENTERED
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-Cre-batch1','NWBs_centered')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
study_responsiveness_all(SESSIONS=SESSIONS, protocol=protocol)
#%% YANN'S DATA
protocol="drifting-gratings"
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-WT-Dec-2022','NWBs')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
study_responsiveness_all(SESSIONS=SESSIONS, protocol=protocol)



############################################# QUICK SPATIAL MAPPING #########################################
#%%
protocol="quick-spatial-mapping"
#%%
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-Cre-batch1','NWBs')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
study_responsiveness_all(SESSIONS=SESSIONS, protocol=protocol)
#%%
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-Cre-batch1','NWBs_centered')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
study_responsiveness_all(SESSIONS=SESSIONS, protocol=protocol)



############################################################################################################
############################################ Plot per protocol #############################################
############################################################################################################
#%%
def responsiveness_sessions_vs_protocols(SESSIONS, protocols):
    Values_m = []
    X = []

    for p, protocol in enumerate(protocols):
        
        values_per_file = []
        significance_per_file = []
        
        for f, filename in enumerate(SESSIONS['files']):
            data = Data(filename, verbose=False)
            data.build_dFoF()
            stat_test_props = dict(interval_pre=[-1.,0],                                   
                                interval_post=[1.,2.],                                   
                                test='ttest')
            ep = EpisodeData(data,
                            protocol_name=protocol,
                            quantities=['dFoF'], 
                            verbose=False)
        
            summary_data = ep.compute_summary_data(stat_test_props,
                                                exclude_keys=['repeat', 'angle', 'contrast'],
                                                response_significance_threshold=0.05,
                                                response_args={})
            

            values_per_file.append(summary_data['value'].flatten())
            significance_per_file.append(summary_data['significant'].flatten())
        
        X.append(np.arange(0,len(SESSIONS['files']),1))
        Values_m.append(values_per_file)

    # ## plot
   
    i,j = 0,0

    fig, AX = plt.subplots(2, 4, figsize=(10, 5))

    for p, protocol in enumerate(protocols):
        try: 
            AX[i][j].bar(X[p], np.array(Values_m[p]).flatten())
        except Exception as e:
            AX[i][j].bar(X[p], np.array(np.mean(Values_m[p], axis=1)).flatten())
        AX[i][j].axhline(0.0, c='black', linewidth = 0.5)
        AX[i][j].set_xlabel('File #')
        AX[i][j].set_title(f'{protocol}')
        AX[i][j].set_xticks(X[p])

        if j==0:
            AX[i][j].set_ylabel('responsiveness')

        if protocol!='Natural-Images-4-repeats':
            AX[i][j].set_title(f'{protocol}')
        else: 
            AX[i][j].set_title('Natural-Images')

        #update position next graph
        if j<3:
            j+=1
        else: 
            i+=1
            j=0

    AX = AX.flatten()
    n_files = len(protocols)
    print(n_files)
    for idx in range(n_files, len(AX)):
        fig.delaxes(AX[idx])
    fig.tight_layout()
    return 0

#%% MY DATA
#datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-Cre-batch1','NWBs')
#SESSIONS = scan_folder_for_NWBfiles(datafolder)
#%%
#protocols = ['static-patch', 'drifting-grating' ,'looming-stim',
#             'Natural-Images-4-repeats','moving-dots',
#             'drifting-surround','quick-spatial-mapping']
#%%
#responsiveness_sessions_vs_protocols(SESSIONS=SESSIONS, protocols=protocols)
#######################################################################################################################
#%% YANN's DATA
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-WT-Dec-2022','NWBs')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
#%%
protocols = [['static-patch', 'drifting-gratings', 'looming-stim',
              'Natural-Images-4-repeats', 'moving-dots', 
              'random-dots']]
# %%
responsiveness_sessions_vs_protocols(SESSIONS=SESSIONS, protocols=protocols)
########################################################################################################################