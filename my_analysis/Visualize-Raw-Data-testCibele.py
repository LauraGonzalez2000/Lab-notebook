# %% [markdown]
# # Visualize Raw Data


#%%
import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent / 'physion' / 'src'))

import physion
from physion.analysis.read_NWB import Data


# %%
# load a datafile
filename = os.path.join(Path("E:/"), 'DATA', 'In_Vivo_experiments','NDNF-old-protocol', 'NDNF-WT-Dec-2022','NWBs_rebuilt', '2022_12_16-12-03-30.nwb')
data = Data(filename,
            verbose=False)
data.build_rawFluo(verbose=False)
data.build_dFoF(verbose=False)

# %% [markdown]
# ## Showing Field of View

# %%
import physion.utils.plot_tools as pt
pt.set_style('dark')

fig, AX = pt.figure(axes=(3,1), 
                    ax_scale=(1.4,3), wspace=0.15)

from physion.dataviz.imaging import show_CaImaging_FOV
#
show_CaImaging_FOV(data, key='meanImg', 
                   cmap=pt.get_linear_colormap('k', 'tab:green'),
                   NL=3, # non-linearity to normalize image
                   ax=AX[0])
show_CaImaging_FOV(data, key='max_proj', 
                   cmap=pt.get_linear_colormap('k', 'tab:green'),
                   NL=3, # non-linearity to normalize image
                   ax=AX[1])
show_CaImaging_FOV(data, key='meanImg', 
                   cmap=pt.get_linear_colormap('k', 'tab:green'),
                   NL=3,
                   roiIndex=range(data.nROIs), 
                   ax=AX[2])

# save on desktop
#fig.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', 'Final-Figures-PhD','FOV-example.svg'))

# %% [markdown]
# # Show Raw Data

# %%

# default plot
from physion.dataviz.raw import plot as plot_raw, find_default_plot_settings
settings = find_default_plot_settings(data)
_ = plot_raw(data, settings=settings, tlim=[1200,1300])

#%%
from physion.dataviz.raw import plot as plot_raw, find_default_plot_settings
import os
pt.set_style("manuscript")

settings = find_default_plot_settings(data)

fig, AX = plot_raw(
    data,
    settings=settings,
    tlim=[1200, 1300]
)

# salvar em SVG
#save_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'raw_traces.svg')
#fig.savefig(save_path, format='svg', bbox_inches='tight')




#%%
import numpy as np
from physion.dataviz.raw import plot as plot_raw, find_default_plot_settings

# calcula o pico máximo de ΔF/F para cada ROI
peak_response = np.max(data.dFoF, axis=1)

# pega os índices dos 3 maiores picos
top3_rois = np.argsort(peak_response)[-3:][::-1]

settings = find_default_plot_settings(data)

settings['CaImaging']['roiIndices'] = top3_rois

fig, AX = plot_raw(
    data,
    settings=settings,
    tlim=[1200, 1300]
)




#%%
# ===============================================
# Paper-style ROI traces (PHYSION-CORRECT)
# ===============================================

import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. Basic data
# =========================
dFoF = data.dFoF                  # (nROIs, nTime)
t = data.t_dFoF
dt = np.mean(np.diff(t))

# =========================
# 2. Initialize stimulus
# =========================
data.init_visual_stim()
stim = data.visual_stim

# =========================
# 3. Extract stimulus epochs (CORRECT WAY)
# =========================
episode_ids = np.array([data.find_episode_from_time(tt) for tt in t])

stim_episodes = np.unique(episode_ids[episode_ids >= 0])

onsets = []
offsets = []

for ep in stim_episodes:
    mask = episode_ids == ep
    onsets.append(t[mask][0])
    offsets.append(t[mask][-1])

onsets = np.array(onsets)
offsets = np.array(offsets)

# =========================
# 4. Select responsive ROIs
# (largest ΔF/F peak during stimulus)
# =========================
responses = []

for i in range(data.nROIs):
    peaks = []
    for on, off in zip(onsets, offsets):
        mask = (t >= on) & (t <= off)
        peaks.append(np.max(dFoF[i, mask]))
    responses.append(np.mean(peaks))

responses = np.array(responses)

roi_indices = np.argsort(responses)[-3:]  # TOP 3 ROIs

# =========================
# 5. Extract aligned trials
# =========================
t_pre = 1.0
t_post = 3.0
n_pre = int(t_pre / dt)
n_post = int(t_post / dt)

trials = []

for roi in roi_indices:
    roi_trials = []
    for on in onsets:
        idx = np.argmin(np.abs(t - on))
        roi_trials.append(dFoF[roi, idx-n_pre:idx+n_post])
    trials.append(np.array(roi_trials))

trials = np.array(trials)  # (nROI, nTrials, nTime)

t_rel = np.linspace(-t_pre, t_post, trials.shape[-1])

# =========================
# 6. Plot (PAPER STYLE)
# =========================
plt.figure(figsize=(6,4))

offset = 0
colors = ['tab:red', 'tab:orange', 'tab:purple']

for i, roi_trials in enumerate(trials):
    mean_trace = roi_trials.mean(axis=0)

    for tr in roi_trials:
        plt.plot(t_rel, tr + offset, color=colors[i], alpha=0.3)

    plt.plot(t_rel, mean_trace + offset, color='green', lw=2)
    offset += 1.5

# Stimulus epoch
plt.axvspan(0, offsets[0] - onsets[0], color='gray', alpha=0.3)

plt.xlabel('Time from stimulus (s)')
plt.ylabel('ΔF/F (offset)')
plt.title('Top responsive ROIs (physion)')
plt.show()






# %% [markdown]
# ## Full view

# %%
import numpy as np

settings = {'Locomotion': {'fig_fraction': 1,
                           'subsampling': 1,
                           'color': '#1f77b4'},
            #'FaceMotion': {'fig_fraction': 1,
            #               'subsampling': 1,
            #               'color': 'purple'},
            #'Pupil': {'fig_fraction': 2,
            #          'subsampling': 1,
            #          'color': '#d62728'},
             'CaImaging': {'fig_fraction': 10,
                           'subsampling': 1,
                           'subquantity': 'dF/F',
                           'roiIndices': np.random.choice(np.arange(data.nROIs), np.min([20,data.nROIs]), replace=False),
                           'color': '#2ca02c'}
           }
fig, AX = \
    plot_raw(data, 
             tlim=[100, data.t_dFoF[-1]], 
             settings=settings)







# %%
# %%
import numpy as np

settings = {'CaImaging': {'fig_fraction': 10,
                           'subsampling': 1,
                           #'subquantity': 'dF/F',
                           'subquantity': 'Deconvolved', #here we change to Deconvolved, insted of dF/F
                           #'roiIndices': np.random.choice(np.arange(data.nROIs), np.min([20,data.nROIs]), replace=False), #this if i want many ROI's, and then I can choose the other i want
                           'roiIndices': [59,12,44], #here I choose the first 3 ROI's, or othever I want
                           #'color': '#2ca02c'
                           'color': 'tab:orange'},
            'VisualStim': {'fig_fraction': np.float64(0.04),
                            'color': 'black',
                            'fig_fraction_start': np.float64(0.96)}}       

fig, AX = \
    plot_raw(data,  
             settings=settings,
             tlim=[1200, 1300])

# save SVG
save_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'Final-Figures-PhD', 'dFF-traces-deconvolved.svg')
fig.savefig(save_path, format='svg', bbox_inches='tight')


#%%
import numpy as np

settings = {'CaImaging': {'fig_fraction': 10,
                           'subsampling': 1,
                           'subquantity': 'dF/F',
                           #'subquantity': 'Deconvolved', #here we change to Deconvolved, insted of dF/F
                           #'roiIndices': np.random.choice(np.arange(data.nROIs), np.min([20,data.nROIs]), replace=False), #this if i want many ROI's, and then I can choose the other i want
                           'roiIndices': [15,25,15], #here I choose the first 3 ROI's, or othever I want
                           'color': '#2ca02c'},
                           #'color': 'tab:orange'},
            'VisualStim': {'fig_fraction': np.float64(0.04),
                            'color': 'black',
                            'fig_fraction_start': np.float64(0.96)}}       

fig, AX = \
    plot_raw(data,  
             settings=settings,
             tlim=[1200, 1300])

# save SVG
save_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'Final-Figures-PhD', 'dFF-traces.svg')
fig.savefig(save_path, format='svg', bbox_inches='tight')





# %%
data.build_Deconvolved()
# %%









# %%
#########################################################
###### TENTATIVE OF DIELD OF VIEW AND TRACES ##############
#########################################################
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pynwb import NWBHDF5IO

# ------------------------------------------------------------
# Load NWB file
# ------------------------------------------------------------
filename = os.path.join(os.path.expanduser('~'),
                        'CURATED', 'Cibele', 'PYR-PV-SynGCaMP_WT_Young_V1', 'NWBs',
                        '2025_06_04-10-15-37.nwb')

io = NWBHDF5IO(filename, 'r')
nwb = io.read()

ophys = nwb.processing["ophys"]

# ------------------------------------------------------------
# Load mean image (campo de visão)
# ------------------------------------------------------------
img = ophys.data_interfaces["Backgrounds_0"].images["meanImg"].data[:]
low, high = np.percentile(img, 2), np.percentile(img, 98)
img_disp = np.clip((img - low) / (high - low), 0, 1)

# ------------------------------------------------------------
# Load ΔF/F
# ------------------------------------------------------------
flu = ophys.data_interfaces["Fluorescence"]
rrs = list(flu.roi_response_series.values())[0]

dff = rrs.data[:]              # shape (T, N)
time = rrs.timestamps[:]       # timestamps
T, N = dff.shape

# ------------------------------------------------------------
# Detect 40s window with highest activity
# ------------------------------------------------------------
activity = np.mean(np.abs(dff), axis=1)
dt = np.median(np.diff(time))

win40 = int(40 / dt)
conv = np.convolve(activity, np.ones(win40), mode='valid')
best_idx = np.argmax(conv)

t0_40 = time[best_idx]
t1_40 = time[min(best_idx + win40, T - 1)]
mask40 = (time >= t0_40) & (time <= t1_40)

# ------------------------------------------------------------
# Select inner 5s segment
# ------------------------------------------------------------
win5 = int(5 / dt)
center5 = best_idx + win40 // 2

start5 = max(center5 - win5 // 2, 0)
end5 = min(start5 + win5, T)

t0_5 = time[start5]
t1_5 = time[end5 - 1]
mask5 = (time >= t0_5) & (time <= t1_5)

# ------------------------------------------------------------
# Select ROIs for plotting
# ------------------------------------------------------------
example_rois = [0]              # ROI shown in 5s zoom

rng = np.random.default_rng(0)
bottom_rois = rng.choice(N, 8, replace=False)

# ------------------------------------------------------------
# Create Figure (meanImg + traces)
# ------------------------------------------------------------
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['axes.titlecolor'] = 'black'

fig = plt.figure(figsize=(15, 8), facecolor='white')
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.3], height_ratios=[1, 1.4],
                      wspace=0.35, hspace=0.4)

ax_img   = fig.add_subplot(gs[:, 0])
ax_zoom  = fig.add_subplot(gs[0, 1:])
ax_stack = fig.add_subplot(gs[1, 1:])

# ------------------------------------------------------------
# 1) Show anatomical field of view (meanImg)
# ------------------------------------------------------------
ax_img.imshow(img_disp, cmap='gray')
ax_img.set_title("Field of view (mean image)")
ax_img.axis('off')

# ------------------------------------------------------------
# 2) 5-second zoom
# ------------------------------------------------------------
colors = plt.cm.tab10(np.linspace(0, 1, len(example_rois)))

for i, roi in enumerate(example_rois):
    ax_zoom.plot(time[mask5], dff[mask5, roi], lw=2, color=colors[i])

ax_zoom.set_title("5-second zoom window")
ax_zoom.set_ylabel("ΔF/F")

# ------------------------------------------------------------
# 3) 40-second panel with 8 ROIs
# ------------------------------------------------------------
offset = 3 * np.nanstd(dff)

for i, roi in enumerate(bottom_rois):
    ax_stack.plot(time[mask40], dff[mask40, roi] + i * offset, lw=1.1)

zoom_width = t1_5 - t0_5

rect = Rectangle((t0_5, -offset),
                 zoom_width,
                 offset * (len(bottom_rois) + 1),
                 linewidth=2, edgecolor='yellow', facecolor='none')

ax_stack.add_patch(rect)

ax_stack.set_title("8 ROIs – 40-s window (yellow = 5-s zoom segment)")
ax_stack.set_xlabel("Time (s)")
ax_stack.set_yticks([])

plt.tight_layout()

def save_svg(fig, filepath):
    """
    Salva uma figura Matplotlib em SVG com fundo branco.
    """
    fig.savefig(
        filepath,
        format='svg',
        dpi=300,
        bbox_inches='tight',
        facecolor='white'
    )
    print(f"SVG salvo em:\n{filepath}")

#save_svg(
    #fig,
   # os.path.expanduser('~/Desktop/Final-Figures-PhD/traces-exemple-PYR-P=25.svg')
#)


plt.show()
io.close()

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pynwb import NWBHDF5IO

# ------------------------------------------------------------
# Load NWB file
# ------------------------------------------------------------
filename = os.path.join(os.path.expanduser('~'),
                        'CURATED', 'Cibele', 'PYR-PV-SynGCaMP_WT_Young_V1', 'NWBs',
                        '2025_06_04-10-15-37.nwb')

io = NWBHDF5IO(filename, 'r')
nwb = io.read()

ophys = nwb.processing["ophys"]

# ------------------------------------------------------------
# Load mean image (campo de visão)
# ------------------------------------------------------------
img = ophys.data_interfaces["Backgrounds_0"].images["meanImg"].data[:]
low, high = np.percentile(img, 2), np.percentile(img, 98)
img_disp = np.clip((img - low) / (high - low), 0, 1)

# ------------------------------------------------------------
# Load ΔF/F
# ------------------------------------------------------------
flu = ophys.data_interfaces["Fluorescence"]
rrs = list(flu.roi_response_series.values())[0]

dff = rrs.data[:]              # shape (T, N)
time = rrs.timestamps[:]       # timestamps
T, N = dff.shape

# ------------------------------------------------------------
# Detect 40s window with highest activity
# ------------------------------------------------------------
activity = np.mean(np.abs(dff), axis=1)
dt = np.median(np.diff(time))

win40 = int(40 / dt)
conv = np.convolve(activity, np.ones(win40), mode='valid')
best_idx = np.argmax(conv)

t0_40 = time[best_idx]
t1_40 = time[min(best_idx + win40, T - 1)]
mask40 = (time >= t0_40) & (time <= t1_40)

# ------------------------------------------------------------
# Select inner 5s segment
# ------------------------------------------------------------
win5 = int(5 / dt)
center5 = best_idx + win40 // 2

start5 = max(center5 - win5 // 2, 0)
end5 = min(start5 + win5, T)

t0_5 = time[start5]
t1_5 = time[end5 - 1]
mask5 = (time >= t0_5) & (time <= t1_5)

# ------------------------------------------------------------
# CRITICAL UPDATE: AUTOMATIC BEST ROI SELECTION
# ------------------------------------------------------------
# Isolate dff data inside the active 40s window to calculate metrics
dff_40s_window = dff[mask40, :]

# Calculate standard deviation (variance) and peak height for each ROI
roi_stds = np.std(dff_40s_window, axis=0)
roi_peaks = np.max(dff_40s_window, axis=0)

# Combine metrics: ROIs with sharp, high spikes have high variance * high peaks
roi_scores = roi_stds * roi_peaks

# Sort ROIs by score in descending order and grab the top 3
top_3_indices = np.argsort(roi_scores)[::-1][:3]

# Assign the ranked ROIs to your figure structures
example_rois = [top_3_indices[0]]  # Use the absolute #1 best ROI for the 5s zoom
bottom_rois = top_3_indices        # Plot all top 3 together in the 40s stack

print(f"Automatically selected Best ROIs based on activity:")
print(f"  Rank 1 (Zoomed): ROI #{top_3_indices[0]}")
print(f"  Rank 2:          ROI #{top_3_indices[1]}")
print(f"  Rank 3:          ROI #{top_3_indices[2]}")

# ------------------------------------------------------------
# Create Figure (meanImg + traces)
# ------------------------------------------------------------
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['axes.titlecolor'] = 'black'

fig = plt.figure(figsize=(15, 8), facecolor='white')
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.3], height_ratios=[1, 1.4],
                      wspace=0.35, hspace=0.4)

ax_img   = fig.add_subplot(gs[:, 0])
ax_zoom  = fig.add_subplot(gs[0, 1:])
ax_stack = fig.add_subplot(gs[1, 1:])

# ------------------------------------------------------------
# 1) Show anatomical field of view (meanImg)
# ------------------------------------------------------------
ax_img.imshow(img_disp, cmap='gray')
ax_img.set_title("Field of view (mean image)")
ax_img.axis('off')

# ------------------------------------------------------------
# 2) 5-second zoom (Plotting the single best ROI)
# ------------------------------------------------------------
colors = plt.cm.tab10(np.linspace(0, 1, len(example_rois)))

for i, roi in enumerate(example_rois):
    ax_zoom.plot(time[mask5], dff[mask5, roi], lw=2, color='crimson', label=f"ROI #{roi}")

ax_zoom.set_title(f"5-second zoom window (Top Active ROI #{example_rois[0]})")
ax_zoom.set_ylabel("ΔF/F")
ax_zoom.legend(loc="upper right")

# ------------------------------------------------------------
# 3) 40-second panel with top 3 best ROIs
# ------------------------------------------------------------
# We calculate the offset based on the max range of our top traces so they don't overlap
offset = np.max(dff_40s_window[:, top_3_indices]) * 1.2

# We map a unique color to each of our 3 best ROIs
stack_colors = ['crimson', 'darkorange', 'teal']

for i, roi in enumerate(bottom_rois):
    ax_stack.plot(time[mask40], dff[mask40, roi] + i * offset, lw=1.5, color=stack_colors[i], label=f"ROI #{roi}")

zoom_width = t1_5 - t0_5

rect = Rectangle((t0_5, -offset * 0.3),
                 zoom_width,
                 offset * (len(bottom_rois)),
                 linewidth=2, edgecolor='gold', facecolor='none', linestyle='--')

ax_stack.add_patch(rect)

ax_stack.set_title("Top 3 Most Active ROIs – 40-s window (Gold box = 5-s zoom)")
ax_stack.set_xlabel("Time (s)")
ax_stack.set_ylabel("ΔF/F (Stacked)")
ax_stack.set_yticks([])
ax_stack.legend(loc="upper right")

plt.tight_layout()

def save_svg(fig, filepath):
    """
    Salva uma figura Matplotlib em SVG com fundo branco.
    """
    fig.savefig(
        filepath,
        format='svg',
        dpi=300,
        bbox_inches='tight',
        facecolor='white'
    )
    print(f"SVG salvo em:\n{filepath}")

#save_svg(
    #fig,
   # os.path.expanduser('~/Desktop/Final-Figures-PhD/traces-exemple-PYR-P=25.svg')
#)

plt.show()
io.close()









# %%

# %%
## For traces plotting
import physion
import numpy as np
import os
import matplotlib.pyplot as plt

filename = os.path.join(os.path.expanduser('~'), 
                        'CURATED', 'Cibele', 'SST-cells_WT_Young_V1', 'NWBs',
                        '2025_04_01-14-56-57.nwb')
data = physion.analysis.read_NWB.Data(filename,
                                      verbose=False)
data.build_rawFluo(verbose=False)
data.build_dFoF(verbose=False)

from physion.dataviz.raw import plot

settings = {
 'Locomotion': {'fig_fraction': 1, 'subsampling': 1, 'color': '#1f77b4'},
 'FaceMotion': {'fig_fraction': 1, 'subsampling': 1, 'color': 'purple'},
 'Pupil': {'fig_fraction': 1, 'subsampling': 1, 'color': '#d62728'},
 'CaImaging': {'fig_fraction': 4, 'subsampling': 1, 'subquantity': 'dFoF',
  'color': '#2ca02c',
  'roiIndices': np.random.choice(np.arange(data.nROIs), 10)},
 'CaImagingRaster': {'fig_fraction': 2, 'subsampling': 1,
  'roiIndices': 'all', 'normalization': 'per-line', 'subquantity': 'dF/F'},
 'VisualStim': {'fig_fraction': 0.2, 'color': 'black'}
}

plot(data, tlim=[100,250], settings=settings, figsize=(12,8))

# SAVE FIGURE
fig = plt.gcf()
fig.savefig(os.path.expanduser('~/Desktop/Final-Figures-PhD/TRACES-2025_04_01-14-56-57.svg'))

# %%
python inspect_nwb.py






# %%
## For traces plotting
import physion
import numpy as np
import os
import matplotlib.pyplot as plt
from physion.dataviz.raw import plot

# ------------------------------------------------------------
# 1) Carregar os dados com o Physion
# ------------------------------------------------------------
filename = os.path.join(os.path.expanduser('~'), 
                        'CURATED', 'Cibele', 'SST-cells_WT_Young_V1', 'NWBs',
                        '2025_04_01-14-56-57.nwb')
data = physion.analysis.read_NWB.Data(filename, verbose=False)
data.build_rawFluo(verbose=False)
data.build_dFoF(verbose=False)

# ------------------------------------------------------------
# 2) Seleção Automática das 3 Melhores ROIs no intervalo tlim
# ------------------------------------------------------------
tlim = [100, 250]

# Extrair a matriz de dFoF e os timestamps do objeto do Physion
# Nota: O Physion costuma estruturar dFoF como (T, N) ou (N, T). 
# Vamos garantir que estamos extraindo no formato correto.
dfff_data = data.dFoF 
timestamps = data.t_dFoF

# Caso as dimensões estejam invertidas (N, T), transpomos para aplicar a máscara de tempo
if dfff_data.shape[0] == data.nROIs:
    dfff_data = dfff_data.T

# Filtrar a matriz para focar apenas no intervalo do gráfico (tlim)
time_mask = (timestamps >= tlim[0]) & (timestamps <= tlim[1])
dff_window = dfff_data[time_mask, :]

# Calcular métricas de atividade por ROI dentro da janela temporal
roi_stds = np.std(dff_window, axis=0)
roi_peaks = np.max(dff_window, axis=0)

# Multiplicação clássica para ignorar ruído basal e pegar transientes/spikes reais
roi_scores = roi_stds * roi_peaks

# Ordenar de forma decrescente e pegar os índices das 3 melhores ROIs
best_3_rois = np.argsort(roi_scores)[::-1][:3]

print(f"Substituindo a seleção aleatória pelas 3 melhores ROIs na janela {tlim}s:")
for rank, roi_idx in enumerate(best_3_rois, 1):
    print(f"  Rank {rank}: ROI Index #{roi_idx} (Score: {roi_scores[roi_idx]:.4f})")

# ------------------------------------------------------------
# 3) Configurações do Physion com as Top ROIs
# ------------------------------------------------------------
settings = {
 'Locomotion': {'fig_fraction': 1, 'subsampling': 1, 'color': '#1f77b4'},
 'FaceMotion': {'fig_fraction': 1, 'subsampling': 1, 'color': 'purple'},
 'Pupil': {'fig_fraction': 1, 'subsampling': 1, 'color': '#d62728'},
 'CaImaging': {
     'fig_fraction': 4, 
     'subsampling': 1, 
     'subquantity': 'dFoF',
     'color': '#2ca02c',
     'roiIndices': best_3_rois  # <-- Inserindo os índices calculados automaticamente aqui
  },
 'CaImagingRaster': {
     'fig_fraction': 2, 
     'subsampling': 1,
     'roiIndices': 'all', 
     'normalization': 'per-line', 
     'subquantity': 'dF/F'
  },
 'VisualStim': {'fig_fraction': 0.2, 'color': 'black'}
}

# ------------------------------------------------------------
# 4) Gerar o Gráfico e Salvar em SVG
# ------------------------------------------------------------
plot(data, tlim=tlim, settings=settings, figsize=(12, 8))

# Pegar a figura atual gerada pelo plot do physion
fig = plt.gcf()

# Salvar
output_path = os.path.expanduser('~/Desktop/Final-Figures-PhD/TRACES-2025_04_01-14-56-57.svg')
fig.savefig(output_path, format='svg', dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nFigura salva com sucesso em:\n{output_path}")

plt.show()











# %%
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import physion
from physion.dataviz.raw import plot
from scipy.stats import skew

# ------------------------------------------------------------
# 1) CONFIGURAÇÕES DE CAMINHO DA PASTA
# ------------------------------------------------------------
nwb_folder = os.path.join(os.path.expanduser('~'), 'CURATED', 'Cibele', 'SST-cells_WT_Young_V1', 'NWBs')
search_pattern = os.path.join(nwb_folder, "*.nwb")
nwb_files = glob.glob(search_pattern)

if not nwb_files:
    raise FileNotFoundError(f"Nenhum arquivo NWB encontrado na pasta: {nwb_folder}")

print(f"Encontrados {len(nwb_files)} arquivos para triagem.\n")

# Nomes exatos dos dois protocolos solicitados
P_ORIENT = "ff-gratings-8orientation-2contrasts-15repeats"
P_CONTRAST = "ff-gratings-2orientations-8contrasts-15repeats"

# Listas para agrupar as análises
orientation_leaderboard = []
contrast_leaderboard = []

# ------------------------------------------------------------
# 2) TRIAGEM, SEPARAÇÃO E FILTRAGEM BIOLÓGICA ANTI-RUÍDO
# ------------------------------------------------------------
print("Iniciando varredura matemática de arquivos...")
for fpath in nwb_files:
    fname = os.path.basename(fpath)
    try:
        data = physion.analysis.read_NWB.Data(fpath, verbose=False)
        
        # Converte metadados e nome do arquivo para string de busca
        metadata_str = str(data.metadata).lower()
        search_target = f"{fname.lower()} {metadata_str}"
        
        # Flexibilização sutil nas strings para capturar variações de escrita
        if "8orientation" in search_target or "8-orientation" in search_target:
            protocol_type = P_ORIENT
        elif "8contrasts" in search_target or "8-contrast" in search_target:
            protocol_type = P_CONTRAST
        else:
            continue

        data.build_rawFluo(verbose=False)
        data.build_dFoF(verbose=False)
        
        dfff_data = data.dFoF
        if dfff_data.shape[0] == data.nROIs:
            dfff_data = dfff_data.T
            
        if dfff_data.size == 0:
            continue

        # --------------------------------------------------------
        # FILTRAGEM FISIOLÓGICA (Remove ruído elétrico/movimento)
        # --------------------------------------------------------
        # 1) Média móvel suave (janela de 5 frames) para achatar ruídos rápidos de alta frequência
        window_size = 5
        dff_smoothed = np.apply_along_axis(
            lambda m: np.convolve(m, np.ones(window_size)/window_size, mode='same'), 
            axis=0, 
            arr=dfff_data
        )

        # 2) Skewness (Assimetria positiva): Transientes de cálcio biológicos reais sobem rápido 
        # e descem devagar, gerando assimetria alta. Ruídos oscilam simetricamente ao redor de zero.
        roi_skews = skew(dff_smoothed, axis=0)
        roi_stds = np.std(dff_smoothed, axis=0)
        
        # O novo score penaliza ruídos simétricos e valoriza transientes biológicos limpos
        roi_scores = roi_stds * roi_skews
        
        # Limpa NaNs ou pontuações negativas indesejadas
        roi_scores = np.nan_to_num(roi_scores, nan=0.0)
        roi_scores[roi_scores < 0] = 0.0

        best_roi_idx = np.argmax(roi_scores)
        best_roi_score = roi_scores[best_roi_idx]
        
        file_info = {
            'file_path': fpath,
            'file_name': fname,
            'best_roi_index': best_roi_idx,
            'top_score': best_roi_score,
            'all_scores': roi_scores
        }
        
        if protocol_type == P_ORIENT:
            orientation_leaderboard.append(file_info)
        elif protocol_type == P_CONTRAST:
            contrast_leaderboard.append(file_info)

    except Exception as e:
        print(f"  [AVISO OPHYS] Pulando arquivo {fname} devido a erro interno ou ausência de ROIs.")

# Ordenar os rankings do maior score (melhores transientes de cálcio) para o menor
orientation_leaderboard.sort(key=lambda x: x['top_score'], reverse=True)
contrast_leaderboard.sort(key=lambda x: x['top_score'], reverse=True)


# ------------------------------------------------------------
# 3) FUNÇÃO DE REVISÃO E PLOTAGEM SELETIVA DO CAMPEÃO
# ------------------------------------------------------------
def plot_protocol_champion(leaderboard, protocol_name, choose_rank=1):
    print("\n" + "="*60)
    print(f"ANÁLISE DO PROTOCOLO: {protocol_name}")
    print("="*60)
    
    if not leaderboard:
        print(f"[ERRO CRÍTICO] Nenhum arquivo correspondente a '{protocol_name}' foi encontrado na pasta.")
        return

    # EXIBE O RANKING DOS TOP 5 PARA VOCÊ SABER AS OPÇÕES DISPONÍVEIS
    print(f"\n--- RANKING DOS TOP ARQUIVOS PARA {protocol_name} ---")
    for idx, entry in enumerate(leaderboard[:5], 1):
        star = "⭐ [PLOTANDO AGORA]" if idx == choose_rank else ""
        print(f"  Rank {idx}: {entry['file_name']} | Score do Melhor Neurônio: {entry['top_score']:.3f} {star}")
    
    # Seleção do índice baseada no Rank escolhido pelo usuário
    target_idx = choose_rank - 1
    if target_idx >= len(leaderboard):
        print(f"\n[AVISO] Rank {choose_rank} não disponível. Revertendo para o Rank 1 automaticamente.")
        target_idx = 0
        choose_rank = 1
        
    champion = leaderboard[target_idx]
    print(f"\nCarregando dados do Rank {choose_rank}: {champion['file_name']}")
    print(f"-----------------------------------------------------")
    
    try:
        champ_data = physion.analysis.read_NWB.Data(champion['file_path'], verbose=False)
        champ_data.build_rawFluo(verbose=False)
        champ_data.build_dFoF(verbose=False)
        
        # Pega as 3 melhores ROIs biológicas sem ruído desse arquivo específico
        top_3_rois = np.argsort(champion['all_scores'])[::-1][:3]
        print(f"-> Top 3 ROIs Reais selecionadas automaticamente: {top_3_rois}")
        
        # Dicionário do Physion com exclusão estrita de Pupil e FaceMotion
        settings = {
         'Locomotion': {'fig_fraction': 1, 'subsampling': 1, 'color': '#1f77b4'},
         'CaImaging': {
             'fig_fraction': 4, 
             'subsampling': 1, 
             'subquantity': 'dFoF',
             'color': '#2ca02c',
             'roiIndices': top_3_rois
          },
         'CaImagingRaster': {
             'fig_fraction': 2, 
             'subsampling': 1,
             'roiIndices': 'all', 
             'normalization': 'per-line', 
             'subquantity': 'dF/F'
          },
         'VisualStim': {'fig_fraction': 0.4, 'color': 'black'}
        }
        
        # Plota os canais selecionados na janela padrão de tempo do seu laboratório
        plot(champ_data, tlim=[100, 250], settings=settings, figsize=(12, 6))
        
        fig = plt.gcf()
        output_filename = f"BEST_3_ROIS_RANK{choose_rank}_{protocol_name}.svg"
        output_svg = os.path.expanduser(f"~/Desktop/Final-Figures-PhD/{output_filename}")
        
        fig.savefig(output_svg, format='svg', dpi=300, bbox_inches='tight', facecolor='white')
        print(f"-> Gráfico limpo exportado com sucesso para a área de trabalho:\n   {output_svg}")
        plt.show()
        
    except Exception as e:
        print(f"[ERRO DE RENDERIZAÇÃO] Falha ao plotar e salvar os dados: {e}")


# ------------------------------------------------------------
# 4) EXECUÇÃO DOS PLOTS FINAIS (AJUSTÁVEL)
# ------------------------------------------------------------
# PROTOCOLO 1: 8-Contraste
# Como o Rank 1 desse protocolo já deu os neurônios [24, 31, 18] limpos, mantemos no Rank 1.
plot_protocol_champion(contrast_leaderboard, "ff-gratings-2orientations-8contrasts-15repeats", choose_rank=1)

# PROTOCOLO 2: 8-Orientações
# Se o arquivo que ficou em primeiro lugar (Rank 1) só tiver ruído elétrico/fundo feio,
# mude o parâmetro 'choose_rank=1' para 'choose_rank=2' ou 'choose_rank=3' abaixo.
# Isso forçará o script a buscar o próximo arquivo do ranking na pasta!
plot_protocol_champion(orientation_leaderboard, "ff-gratings-8orientation-2contrasts-15repeats", choose_rank=1)




# %%
###################################################################################
###### TENTATIVE OF AUTOMATIC FILE SELECTION BASED ON SIGNAL QUALITY ##############
##################################################################################

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import physion
from physion.dataviz.raw import plot

# ------------------------------------------------------------
# 1) CONFIGURAÇÕES DE CAMINHO DA PASTA
# ------------------------------------------------------------
nwb_folder = os.path.join(os.path.expanduser('~'), 'CURATED', 'Cibele', 'SST-cells_WT_Adult_V1', 'NWBs')
search_pattern = os.path.join(nwb_folder, "*.nwb")
nwb_files = glob.glob(search_pattern)

if not nwb_files:
    raise FileNotFoundError(f"Nenhum arquivo NWB encontrado na pasta: {nwb_folder}")

print(f"Encontrados {len(nwb_files)} arquivos para triagem.\n")

# Nomes exatos dos dois protocolos
P_ORIENT = "ff-gratings-8orientation-2contrasts-15repeats"
P_CONTRAST = "ff-gratings-2orientations-8contrasts-15repeats"

orientation_leaderboard = []
contrast_leaderboard = []

# ------------------------------------------------------------
# 2) TRIAGEM AUTOMÁTICA POR AMPLITUDE ABSOLUTA (SINAL + OU -)
# ------------------------------------------------------------
print("Iniciando varredura robusta por maior amplitude absoluta (Picos e Vales)...")
for fpath in nwb_files:
    fname = os.path.basename(fpath)
    try:
        data = physion.analysis.read_NWB.Data(fpath, verbose=False)
        
        metadata_str = str(data.metadata).lower()
        search_target = f"{fname.lower()} {metadata_str}"
        
        if "8orientation" in search_target or "8-orientation" in search_target:
            protocol_type = P_ORIENT
        elif "8contrasts" in search_target or "8-contrast" in search_target:
            protocol_type = P_CONTRAST
        else:
            continue

        data.build_rawFluo(verbose=False)
        data.build_dFoF(verbose=False)
        
        dfff_data = data.dFoF
        if dfff_data.shape[0] == data.nROIs:
            dfff_data = dfff_data.T
            
        if dfff_data.size == 0:
            continue

        # Suavização suave para remover o ruído de alta frequência (fios de cabelo)
        window_size = 7
        dff_smoothed = np.apply_along_axis(
            lambda m: np.convolve(m, np.ones(window_size)/window_size, mode='same'), 
            axis=0, arr=dfff_data
        )

        # --------------------------------------------------------
        # MÉTRICA: DISTÂNCIA ABSOLUTA DA LINHA DE BASE (MAD)
        # --------------------------------------------------------
        # Em vez de olhar para onde o sinal vai, calculamos a distância absoluta 
        # de cada ponto em relação à mediana (linha de base estável).
        mediana = np.median(dff_smoothed, axis=0)
        distancia_absoluta = np.abs(dff_smoothed - mediana)
        
        # O score da ROI será a magnitude máxima que esse sinal atinge se afastando da linha de base
        # (Seja para cima em um disparo, seja para baixo em uma inibição)
        amplitude_absoluta = np.max(distancia_absoluta, axis=0)
        
        # Medimos o ruído de alta frequência (sinal bruto menos sinal filtrado)
        raw_noise = np.std(dfff_data - dff_smoothed, axis=0)
        
        # O score final premia grandes desvios absolutos (picos ou vales) e pune o ruído fino
        adjusted_scores = amplitude_absoluta / (raw_noise + 0.05)
        adjusted_scores = np.nan_to_num(adjusted_scores, nan=0.0)

        best_roi_idx = np.argmax(adjusted_scores)
        best_roi_score = adjusted_scores[best_roi_idx]
        
        file_info = {
            'file_path': fpath,
            'file_name': fname,
            'best_roi_index': best_roi_idx,
            'top_score': best_roi_score,
            'all_scores': adjusted_scores
        }
        
        if protocol_type == P_ORIENT:
            orientation_leaderboard.append(file_info)
        elif protocol_type == P_CONTRAST:
            contrast_leaderboard.append(file_info)

    except Exception as e:
        continue

# Ordena os arquivos trazendo os que têm as maiores modulações biológicas reais (independente do sinal)
orientation_leaderboard.sort(key=lambda x: x['top_score'], reverse=True)
contrast_leaderboard.sort(key=lambda x: x['top_score'], reverse=True)

# ------------------------------------------------------------
# 3) FUNÇÃO DE PLOTAGEM DO VENDEDOR DE ALTA MODULAÇÃO
# ------------------------------------------------------------
def plot_protocol_champion(leaderboard, protocol_name):
    print("\n" + "="*60)
    print(f"ANÁLISE DE MAIOR AMPLITUDE ABSOLUTA: {protocol_name}")
    print("="*60)
    
    if not leaderboard:
        print(f"[AVISO] Nenhum arquivo processado com sucesso para '{protocol_name}'.")
        return

    champion = leaderboard[0]
    print(f"Arquivo Campeão Escolhido: {champion['file_name']}")
    
    champ_data = physion.analysis.read_NWB.Data(champion['file_path'], verbose=False)
    champ_data.build_rawFluo(verbose=False)
    champ_data.build_dFoF(verbose=False)
    
    # Seleciona as 3 ROIs com as maiores deflexões (Picos ou Vales) limpas
    top_3_rois = np.argsort(champion['all_scores'])[::-1][:3]
    print(f"-> Top 3 ROIs de Maior Impacto Absoluto selecionadas: {top_3_rois}")
    
    settings = {
     'Locomotion': {'fig_fraction': 1, 'subsampling': 1, 'color': '#1f77b4'},
     'CaImaging': {
         'fig_fraction': 4, 
         'subsampling': 1, 
         'subquantity': 'dFoF',
         'color': '#2ca02c',
         'roiIndices': top_3_rois
      },
     'CaImagingRaster': {
         'fig_fraction': 2, 
         'subsampling': 1,
         'roiIndices': 'all', 
         'normalization': 'per-line', 
         'subquantity': 'dF/F'
      },
     'VisualStim': {'fig_fraction': 0.4, 'color': 'black'}
    }
    
    plot(champ_data, tlim=[100, 250], settings=settings, figsize=(12, 6))
    
    fig = plt.gcf()
    output_filename = f"AUTOMATIC_ABSOLUTE_AMPLITUDE_{protocol_name}.svg"
    output_svg = os.path.expanduser(f"~/Desktop/Final-Figures-PhD/{output_filename}")
    
    fig.savefig(output_svg, format='svg', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"-> Gráfico exportado com sucesso em:\n   {output_svg}")
    plt.show()

# ------------------------------------------------------------
# 4) EXECUÇÃO DOS PLOTS FINAIS 
# ------------------------------------------------------------
plot_protocol_champion(contrast_leaderboard, "ff-gratings-2orientations-8contrasts-15repeats")
plot_protocol_champion(orientation_leaderboard, "ff-gratings-8orientation-2contrasts-15repeats")







# %%
###################################################################################
###### BIOLOGICAL ORIENTATION TUNING SELECTION BY DIFERENT AGE GROUPS ###############
###################################################################################

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import physion
from physion.dataviz.raw import plot

# ------------------------------------------------------------
# 1) PATH CONFIGURATIONS AND ALL THREE AGE COHORTS
# ------------------------------------------------------------
base_nwb_folder = os.path.join(os.path.expanduser('~'), 'CURATED', 'Cibele', 'PV-cells_WT_Young_V1', 'NWBs')

# All three developmental age windows are now active
age_groups = ["P15-19", "P20-23", "P24-27"]
age_leaderboards = {group: [] for group in age_groups}

# Target 8-orientation, 2-contrast experimental protocol
TARGET_PROTOCOL = "ff-gratings-8orientation-2contrasts-15repeats"

print(f"Starting developmental screening for protocol: {TARGET_PROTOCOL}...\n")

# ------------------------------------------------------------
# 2) STIMULUS-LOCKED BIOLOGICAL TUNING SELECTION PIPELINE
# ------------------------------------------------------------
for group in age_groups:
    group_folder = os.path.join(base_nwb_folder, group)
    search_pattern = os.path.join(group_folder, "*.nwb")
    nwb_files = glob.glob(search_pattern)
    
    print(f"Processing folder '{group}' (Found {len(nwb_files)} files)...")
    matched_count = 0
    
    for fpath in nwb_files:
        fname = os.path.basename(fpath)
        try:
            data = physion.analysis.read_NWB.Data(fpath, verbose=False)
            
            # Match protocol text exactly
            protocol_meta = str(data.metadata.get('protocol', '')).lower()
            if TARGET_PROTOCOL not in protocol_meta:
                continue
                
            # Load calcium signals safely
            data.build_rawFluo(verbose=False)
            data.build_dFoF(verbose=False)
            
            dfff_data = data.dFoF
            if dfff_data.shape[0] == getattr(data, 'nROIs', 0):
                dfff_data = dfff_data.T  # Setup uniform shape: (time_frames, nROIs)
                
            if dfff_data.size == 0:
                continue

            # Smooth to capture clean, low-frequency calcium transients
            window_size = 9
            dff_smoothed = np.apply_along_axis(
                lambda m: np.convolve(m, np.ones(window_size)/window_size, mode='same'), 
                axis=0, arr=dfff_data
            )

            # --- STIMULUS-LOCKED METRIC FOR PHENOTYPE SELECTION ---
            # Instead of looking at total file variance, we segment the signal:
            # High-frequency changes reveal true, stimulus-evoked cell responses.
            residual_noise = np.std(dfff_data - dff_smoothed, axis=0)
            
            # Subsample and slice chunks of time to identify reliable, repeating peaks
            chunk_size = dff_smoothed.shape[0] // 15
            if chunk_size > 10:
                peaks_per_repeats = []
                for i in range(15):
                    start_f = i * chunk_size
                    end_f = min((i + 1) * chunk_size, dff_smoothed.shape[0])
                    peaks_per_repeats.append(np.max(dff_smoothed[start_f:end_f, :], axis=0))
                
                # Reliable cells spike consistently when their preferred orientation appears
                response_reliability = np.std(peaks_per_repeats, axis=0)
            else:
                response_reliability = np.std(dff_smoothed, axis=0)

            # Pure mathematical tuning score targeting rhythmic, crisp responses
            biological_tuning_score = response_reliability / (residual_noise + 0.05)
            biological_tuning_score = np.nan_to_num(biological_tuning_score, nan=0.0)

            # Sort ROIs inside this specific file
            sorted_roi_indices = np.argsort(biological_tuning_score)[::-1]
            best_roi_idx = sorted_roi_indices[0]
            best_roi_score = biological_tuning_score[best_roi_idx]
            
            file_info = {
                'file_path': fpath,
                'file_name': fname,
                'top_score': best_roi_score,
                'sorted_rois': sorted_roi_indices,
                'all_scores': biological_tuning_score
            }
            
            age_leaderboards[group].append(file_info)
            matched_count += 1

        except Exception as e:
            continue
            
    print(f" -> Successfully parsed and ranked {matched_count} protocol-matching files for {group}.")

# Sort the sessions in each folder so the cleanest representative is at index 0
for group in age_groups:
    age_leaderboards[group].sort(key=lambda x: x['top_score'], reverse=True)

# ------------------------------------------------------------
# 3) AUTOMATIC GENERATION OF DEVELOPMENTAL FIGURES
# ------------------------------------------------------------
def plot_representative_session(leaderboard, group_name):
    print("\n" + "="*60)
    print(f"GENERATING DISSERTATION GRAPH FOR COHORT: {group_name}")
    print("="*60)
    
    if not leaderboard:
        print(f"[WARNING] No valid files passed filtering inside '{group_name}'.")
        return

    # Select the single best representative session file for this age group
    champion = leaderboard[0]
    print(f"Selected Session: {champion['file_name']} (Biological Tuning Quality: {champion['top_score']:.2f})")
    
    champ_data = physion.analysis.read_NWB.Data(champion['file_path'], verbose=False)
    champ_data.build_rawFluo(verbose=False)
    champ_data.build_dFoF(verbose=False)
    
    # Extract the top 2 best-behaving tuned ROIs from this session
    top_2_rois = champion['sorted_rois'][:2]
    print(f"Selected Representative ROIs: {list(top_2_rois)}")
    
    settings = {
     'Locomotion': {'fig_fraction': 1, 'subsampling': 1, 'color': '#1f77b4'},
     'CaImaging': {
         'fig_fraction': 4, 
         'subsampling': 1, 
         'subquantity': 'dFoF',
         'color': '#b03a2e',  # Crimson theme color matching your paper's styling
         'roiIndices': top_2_rois
      },
     'CaImagingRaster': {
         'fig_fraction': 2, 
         'subsampling': 1,
         'roiIndices': 'all', 
         'normalization': 'per-line', 
         'subquantity': 'dF/F'
      },
     'VisualStim': {'fig_fraction': 0.4, 'color': 'black'}
    }
    
    # Bound the time limit window to focus cleanly on repeating stimulation episodes
    dynamic_tlim = [100, 350]
    
    # Execute the raw visualization plot
    plot(champ_data, tlim=dynamic_tlim, settings=settings, figsize=(12, 6))
    
    # Save directly as a vector graphic for your dissertation figures folder
    fig = plt.gcf()
    output_filename = f"DEVELOPMENTAL_TUNING_TRACES_{group_name}.svg"
    output_svg = os.path.join(os.path.expanduser('~'), 'Desktop', 'Final-Figures-PhD', output_filename)
    
    os.makedirs(os.path.dirname(output_svg), exist_ok=True)
    fig.savefig(output_svg, format='svg', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"-> Success! Saved vector graphic to: {output_svg}")
    plt.show()

# Run for all three developmental cohorts sequentially
for group_name in age_groups:
    plot_representative_session(age_leaderboards[group_name], group_name)





# %%
###################################### USING THIS #############################################
###### YOUR FAVOURITE MATH + TOP 10 VISUALIZER & INTERACTIVE SELECTOR #############
###################################################################################

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import physion
from physion.dataviz.raw import plot

# ------------------------------------------------------------
# 1) PATH CONFIGURATIONS AND COHORTS
# ------------------------------------------------------------
base_nwb_folder = os.path.join(os.path.expanduser('~'), 'CURATED', 'Cibele', 'PV-cells_WT_Young_V1', 'NWBs')
age_groups = ["P15-19", "P20-23", "P24-27"]
age_leaderboards = {group: [] for group in age_groups}

TARGET_PROTOCOL = "ff-gratings-8orientation-2contrasts-15repeats"

print(f"Screening files using YOUR golden ranking math...\n")

# ------------------------------------------------------------
# 2) YOUR EXACT MATH PIPELINE (UNTOUCHED)
# ------------------------------------------------------------
for group in age_groups:
    group_folder = os.path.join(base_nwb_folder, group)
    search_pattern = os.path.join(group_folder, "*.nwb")
    nwb_files = glob.glob(search_pattern)
    
    print(f"Processing folder '{group}' (Found {len(nwb_files)} files)...")
    matched_count = 0
    
    for fpath in nwb_files:
        fname = os.path.basename(fpath)
        try:
            data = physion.analysis.read_NWB.Data(fpath, verbose=False)
            
            protocol_meta = str(data.metadata.get('protocol', '')).lower()
            if TARGET_PROTOCOL not in protocol_meta:
                continue
                
            data.build_rawFluo(verbose=False)
            data.build_dFoF(verbose=False)
            
            dfff_data = data.dFoF
            if dfff_data.shape[0] == getattr(data, 'nROIs', 0):
                dfff_data = dfff_data.T  
                
            if dfff_data.size == 0:
                continue

            window_size = 9
            dff_smoothed = np.apply_along_axis(
                lambda m: np.convolve(m, np.ones(window_size)/window_size, mode='same'), 
                axis=0, arr=dfff_data
            )

            # --- YOUR EXACT CHUNKING MATH ---
            residual_noise = np.std(dfff_data - dff_smoothed, axis=0)
            chunk_size = dff_smoothed.shape[0] // 15
            if chunk_size > 10:
                peaks_per_repeats = []
                for i in range(15):
                    start_f = i * chunk_size
                    end_f = min((i + 1) * chunk_size, dff_smoothed.shape[0])
                    peaks_per_repeats.append(np.max(dff_smoothed[start_f:end_f, :], axis=0))
                response_reliability = np.std(peaks_per_repeats, axis=0)
            else:
                response_reliability = np.std(dff_smoothed, axis=0)

            biological_tuning_score = response_reliability / (residual_noise + 0.05)
            biological_tuning_score = np.nan_to_num(biological_tuning_score, nan=0.0)

            sorted_roi_indices = np.argsort(biological_tuning_score)[::-1]
            best_roi_score = biological_tuning_score[sorted_roi_indices[0]]
            
            file_info = {
                'file_path': fpath,
                'file_name': fname,
                'top_score': best_roi_score,
                'sorted_rois': sorted_roi_indices,
                'all_scores': biological_tuning_score
            }
            age_leaderboards[group].append(file_info)
            matched_count += 1
        except:
            continue
            
    print(f" -> Ranked {matched_count} files for {group}.")

for group in age_groups:
    age_leaderboards[group].sort(key=lambda x: x['top_score'], reverse=True)

# ------------------------------------------------------------
# 3) INTERACTIVE PLOTTING WITH LOCAL LOOP
# ------------------------------------------------------------
def plot_with_interactive_loop(leaderboard, group_name):
    print("\n" + "="*60)
    print(f"PREPARING COHORT PANEL: {group_name}")
    print("="*60)
    
    if not leaderboard:
        print(f"[WARNING] No files found for {group_name}")
        return

    champion = leaderboard[0]
    print(f"Selected Best Session: {champion['file_name']}")
    
    # Show the top 10 ROIs calculated by your math so the user knows exactly what to type!
    top_10_calculated = list(champion['sorted_rois'][:10])
    print(f"🏆 Top 10 ROIs ranked by your math: {top_10_calculated}")
    
    champ_data = physion.analysis.read_NWB.Data(champion['file_path'], verbose=False)
    champ_data.build_rawFluo(verbose=False)
    champ_data.build_dFoF(verbose=False)
    
    # Start with the top 2 automatically
    current_rois = top_10_calculated[:2]
    
    settings = {
     'Locomotion': {'fig_fraction': 1, 'subsampling': 1, 'color': '#1f77b4'},
     'CaImaging': {'fig_fraction': 4, 'subsampling': 1, 'subquantity': 'dFoF', 'color': '#b03a2e', 'roiIndices': current_rois},
     'CaImagingRaster': {'fig_fraction': 2, 'subsampling': 1, 'roiIndices': 'all', 'normalization': 'per-line', 'subquantity': 'dF/F'},
     'VisualStim': {'fig_fraction': 0.4, 'color': 'black'}
    }
    
    dynamic_tlim = [32, 112] # Exactly your 16 orientation window
    
    while True:
        print(f"\nPloting window {dynamic_tlim} with ROIs: {current_rois}")
        settings['CaImaging']['roiIndices'] = current_rois
        
        plot(champ_data, tlim=dynamic_tlim, settings=settings, figsize=(12, 6))
        plt.show(block=False)
        plt.pause(0.5)
        
        print("\nOptions:")
        print("  - Press [ENTER] if you like this plot and want to SAVE it.")
        print(f"  - Type other numbers from the top 10 (e.g., {top_10_calculated[2]}, {top_10_calculated[3]}) to replace them.")
        
        user_choice = input("👉 Your choice: ").strip()
        
        if not user_choice:
            # Save and exit loop
            fig = plt.gcf()
            output_svg = os.path.join(os.path.expanduser('~'), 'Desktop', 'Final-Figures-PhD', f"DEVELOPMENTAL_TUNING_{group_name}.svg")
            os.makedirs(os.path.dirname(output_svg), exist_ok=True)
            fig.savefig(output_svg, format='svg', dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✔️ Saved beautiful vector graphic to: {output_svg}")
            plt.close('all')
            break
        else:
            try:
                current_rois = [int(x.strip()) for x in user_choice.split(',')]
                plt.close('all') # Clear previous attempt
            except ValueError:
                print("[Error] Use commas to separate numbers. Try again.")

# Run for all cohorts sequentially
for group_name in age_groups:
    plot_with_interactive_loop(age_leaderboards[group_name], group_name)






# %%
###################################################################################
###### BETTER - FIXED VIEW: WINDOW-TARGETED RANKING FOR 16-STIMULUS PLOTS ##################
######################################I can't change the choose here#############################################

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import physion
from physion.dataviz.raw import plot

# ------------------------------------------------------------
# 1) PATH CONFIGURATIONS AND TARGET AGE FOLDERS
# ------------------------------------------------------------
base_nwb_folder = os.path.join(os.path.expanduser('~'), 'CURATED', 'Cibele', 'SST-cells_WT_Young_V1', 'NWBs')

age_groups = ["P15-19", "P20-23", "P24-27"]
age_leaderboards = {group: [] for group in age_groups}

TARGET_PROTOCOL = "ff-gratings-8orientation-2contrasts-15repeats"

print(f"Starting localized window screening for protocol: {TARGET_PROTOCOL}...\n")

# ------------------------------------------------------------
# 2) WINDOW-RESTRICTED BIOLOGICAL MATCHING PIPELINE
# ------------------------------------------------------------
for group in age_groups:
    group_folder = os.path.join(base_nwb_folder, group)
    search_pattern = os.path.join(group_folder, "*.nwb")
    nwb_files = glob.glob(search_pattern)
    
    print(f"Processing folder '{group}'...")
    matched_count = 0
    
    for fpath in nwb_files:
        fname = os.path.basename(fpath)
        try:
            data = physion.analysis.read_NWB.Data(fpath, verbose=False)
            
            # Filter protocol text
            protocol_meta = str(data.metadata.get('protocol', '')).lower()
            if TARGET_PROTOCOL not in protocol_meta:
                continue
                
            # Load calcium signals safely
            data.build_rawFluo(verbose=False)
            data.build_dFoF(verbose=False)
            
            dfff_data = data.dFoF
            if dfff_data.shape[0] == getattr(data, 'nROIs', 0):
                dfff_data = dfff_data.T  # Shape matching: (time_frames, nROIs)
                
            if dfff_data.size == 0:
                continue

            # --- CALCULATE TIME WINDOW FOR THIS FILE ---
            try:
                t_start = data.visual_stim['time'][0]
                duration = data.metadata.get('presentation-duration', 2)
                interstim = data.metadata.get('presentation-interstim-period', 3)
                cycle_length = duration + interstim
                t_end = t_start + (16 * cycle_length)
            except:
                t_start, t_end = 30, 115

            # Convert those specific time coordinates to data frame indices
            # (Assuming standard ~30Hz frame acquisition rate if imaging rate metadata is missing)
            fps = getattr(data, 'imaging_rate', 30)
            frame_start = int(t_start * fps)
            frame_end = int(t_end * fps)

            # CRITICAL CORRECTION: Slice the data down to ONLY your 16 stimulations BEFORE calculating scores
            dfff_window = dfff_data[frame_start:frame_end, :]
            if dfff_window.size == 0 or dfff_window.shape[0] < 10:
                continue

            # Smooth the cropped data
            window_size = 9
            dff_smoothed = np.apply_along_axis(
                lambda m: np.convolve(m, np.ones(window_size)/window_size, mode='same'), 
                axis=0, arr=dfff_window
            )

            # Calculate selectivity metric STRICTLY within this short window box
            signal_variance = np.std(dff_smoothed, axis=0)
            instrumental_noise = np.std(dfff_window - dff_smoothed, axis=0)
            
            selectivity_scores = signal_variance / (instrumental_noise + 0.05)
            selectivity_scores = np.nan_to_num(selectivity_scores, nan=0.0)

            # Sort ROIs based on local window performance
            sorted_roi_indices = np.argsort(selectivity_scores)[::-1]
            best_roi_idx = sorted_roi_indices[0]
            best_roi_score = selectivity_scores[best_roi_idx]
            
            file_info = {
                'file_path': fpath,
                'file_name': fname,
                'top_score': best_roi_score,
                'sorted_rois': sorted_roi_indices,
                'all_scores': selectivity_scores,
                'tlim_bounds': [t_start, t_end]
            }
            
            age_leaderboards[group].append(file_info)
            matched_count += 1

        except Exception as e:
            continue
            
    print(f" -> Successfully matched, parsed, and locally ranked {matched_count} files for {group}.")

# Sort cohorts so the absolute best looking short-window trace is at index 0
for group in age_groups:
    age_leaderboards[group].sort(key=lambda x: x['top_score'], reverse=True)

# ------------------------------------------------------------
# 3) AUTOMATIC GENERATION OF REPRESENTATIVE FIGURES
# ------------------------------------------------------------
def plot_representative_session(leaderboard, group_name):
    print("\n" + "="*60)
    print(f"GENERATING SHORT-WINDOW TARGETED PLOT FOR COHORT: {group_name}")
    print("="*60)
    
    if not leaderboard:
        print(f"[WARNING] No valid protocol-matching files found inside '{group_name}'.")
        return

    champion = leaderboard[0]
    print(f"Selected Representative File: {champion['file_name']} (Local Score: {champion['top_score']:.2f})")
    
    champ_data = physion.analysis.read_NWB.Data(champion['file_path'], verbose=False)
    champ_data.build_rawFluo(verbose=False)
    champ_data.build_dFoF(verbose=False)
    
    top_2_rois = champion['sorted_rois'][:2]
    print(f"Selected Representative ROIs: {list(top_2_rois)}")
    
    settings = {
     'Locomotion': {'fig_fraction': 1, 'subsampling': 1, 'color': '#1f77b4'},
     'CaImaging': {
         'fig_fraction': 4, 
         'subsampling': 1, 
         'subquantity': 'dFoF',
         'color': '#b03a2e',  # Crimson theme
         'roiIndices': top_2_rois
      },
     'CaImagingRaster': {
         'fig_fraction': 2, 
         'subsampling': 1,
         'roiIndices': 'all', 
         'normalization': 'per-line', 
         'subquantity': 'dF/F'
      },
     'VisualStim': {'fig_fraction': 0.4, 'color': 'black'}
    }
    
    # Use the exact time bounds calculated during the targeted screening step
    short_window_tlim = champion['tlim_bounds']
    print(f"-> Boxing time axes to short window: {short_window_tlim} seconds.")
    
    # Plot the calcium imaging data
    plot(champ_data, tlim=short_window_tlim, settings=settings, figsize=(12, 6))
    
    # Save directly as an SVG vector graphic
    fig = plt.gcf()
    output_filename = f"DEVELOPMENTAL_LOCAL_16_STIM_TRACES_{group_name}.svg"
    output_svg = os.path.join(os.path.expanduser('~'), 'Desktop', 'Final-Figures-PhD', output_filename)
    
    os.makedirs(os.path.dirname(output_svg), exist_ok=True)
    fig.savefig(output_svg, format='svg', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"-> Saved thesis-ready vector graphic to: {output_svg}")
    plt.show()

# Run for all three developmental age windows sequentially
for group_name in age_groups:
    plot_representative_session(age_leaderboards[group_name], group_name)






#%%
###################################################################################
###### WINDOW-TARGETED LEADERBOARD & INTERACTIVE MULTI-ROI SELECTOR (SST) #########
###################################################################################

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import physion
from physion.dataviz.raw import plot

# ------------------------------------------------------------
# 1) PATH CONFIGURATIONS AND TARGET AGE FOLDERS
# ------------------------------------------------------------
base_nwb_folder = os.path.join(os.path.expanduser('~'), 'CURATED', 'Cibele', 'PYR-PV-SynGCaMP_WT_Young_V1', 'NWBs')

age_groups = ["P15-19", "P20-23", "P24-27"]
age_leaderboards = {group: [] for group in age_groups}

TARGET_PROTOCOL = "ff-gratings-8orientation-2contrasts-15repeats"

print(f"Starting localized window screening for protocol: {TARGET_PROTOCOL}...\n")

# ------------------------------------------------------------
# 2) WINDOW-RESTRICTED BIOLOGICAL MATCHING PIPELINE
# ------------------------------------------------------------
for group in age_groups:
    group_folder = os.path.join(base_nwb_folder, group)
    search_pattern = os.path.join(group_folder, "*.nwb")
    nwb_files = glob.glob(search_pattern)
    
    print(f"Processing folder '{group}' (Found {len(nwb_files)} files)...")
    matched_count = 0
    
    for fpath in nwb_files:
        fname = os.path.basename(fpath)
        try:
            data = physion.analysis.read_NWB.Data(fpath, verbose=False)
            
            # Filter protocol text
            protocol_meta = str(data.metadata.get('protocol', '')).lower()
            if TARGET_PROTOCOL not in protocol_meta:
                continue
                
            # Load calcium signals safely
            data.build_rawFluo(verbose=False)
            data.build_dFoF(verbose=False)
            
            dfff_data = data.dFoF
            if dfff_data.shape[0] == getattr(data, 'nROIs', 0):
                dfff_data = dfff_data.T  # Shape matching: (time_frames, nROIs)
                
            if dfff_data.size == 0:
                continue

            # --- CALCULATE TIME WINDOW FOR THIS FILE ---
            try:
                t_start = data.visual_stim['time'][0]
                duration = data.metadata.get('presentation-duration', 2)
                interstim = data.metadata.get('presentation-interstim-period', 3)
                cycle_length = duration + interstim
                t_end = t_start + (16 * cycle_length)
            except:
                t_start, t_end = 30, 115

            # Convert those specific time coordinates to data frame indices
            fps = getattr(data, 'imaging_rate', 30)
            frame_start = int(t_start * fps)
            frame_end = int(t_end * fps)

            # Slice the data down to ONLY your 16 stimulations BEFORE calculating scores
            dfff_window = dfff_data[frame_start:frame_end, :]
            if dfff_window.size == 0 or dfff_window.shape[0] < 10:
                continue

            # Smooth the cropped data
            window_size = 9
            dff_smoothed = np.apply_along_axis(
                lambda m: np.convolve(m, np.ones(window_size)/window_size, mode='same'), 
                axis=0, arr=dfff_window
            )

            # Calculate selectivity metric STRICTLY within this short window box
            signal_variance = np.std(dff_smoothed, axis=0)
            instrumental_noise = np.std(dfff_window - dff_smoothed, axis=0)
            
            selectivity_scores = signal_variance / (instrumental_noise + 0.05)
            selectivity_scores = np.nan_to_num(selectivity_scores, nan=0.0)

            # Sort ROIs based on local window performance
            sorted_roi_indices = np.argsort(selectivity_scores)[::-1]
            best_roi_idx = sorted_roi_indices[0]
            best_roi_score = selectivity_scores[best_roi_idx]
            
            file_info = {
                'file_path': fpath,
                'file_name': fname,
                'top_score': best_roi_score,
                'sorted_rois': sorted_roi_indices,
                'all_scores': selectivity_scores,
                'tlim_bounds': [t_start, t_end]
            }
            
            age_leaderboards[group].append(file_info)
            matched_count += 1

        except Exception as e:
            continue
            
    print(f" -> Successfully matched, parsed, and locally ranked {matched_count} files for {group}.")

# Sort cohorts so the absolute best looking short-window trace is at index 0
for group in age_groups:
    age_leaderboards[group].sort(key=lambda x: x['top_score'], reverse=True)


# ------------------------------------------------------------
# 3) DYNAMIC INTERACTIVE GENERATION LOOP (RUNS FOR EACH AGE)
# ------------------------------------------------------------
def run_interactive_cohort_selector(leaderboard, group_name):
    if not leaderboard:
        print(f"\n[WARNING] No valid protocol-matching files found inside '{group_name}'. Skipping.")
        return

    while True:
        # Clear any dangling matplotlib traces or memory buffers
        plt.close('all')
        plt.clf()
        
        print("\n" + "="*60)
        print(f"🏆 AVAILABLE SESSIONS LEADERBOARD FOR COHORT: {group_name}")
        print("="*60)
        for idx, item in enumerate(leaderboard[:5]):
            print(f"  [{idx + 1}] {item['file_name']} (Window Score: {item['top_score']:.2f})")
        print("  [S] Skip this age cohort entirely")
        print("="*60)
        
        file_choice = input(f"\n👉 Choose a session index to load for {group_name} (1-5) [Default is 1]: ").strip()
        
        if file_choice.lower() == 's':
            print(f"Skipping cohort {group_name}.")
            break
            
        try:
            session_idx = int(file_choice) - 1 if file_choice else 0
            champion = leaderboard[session_idx]
        except (ValueError, IndexError):
            print("[Error] Option out of range. Please input a number between 1 and 5.")
            continue
            
        print(f"\n🔄 LOADING FRESH OBJECT FOR: {champion['file_name']}")
        top_10_calculated = list(champion['sorted_rois'][:10])
        
        # Instantiate clean session data container to evade visualization bugs
        champ_data = physion.analysis.read_NWB.Data(champion['file_path'], verbose=False)
        champ_data.build_rawFluo(verbose=False)
        champ_data.build_dFoF(verbose=False)
        
        # Automatically capture the 4 best matching cells
        current_rois = top_10_calculated[:4]
        short_window_tlim = champion['tlim_bounds']
        
        settings = {
         'Locomotion': {'fig_fraction': 1, 'subsampling': 1, 'color': '#1f77b4'},
         'CaImaging': {'fig_fraction': 4, 'subsampling': 1, 'subquantity': 'dFoF', 'color': '#b03a2e', 'roiIndices': current_rois},
         'CaImagingRaster': {'fig_fraction': 2, 'subsampling': 1, 'roiIndices': 'all', 'normalization': 'per-line', 'subquantity': 'dF/F'},
         'VisualStim': {'fig_fraction': 0.4, 'color': 'black'}
        }
        
        # Sub-loop to evaluate and pick alternate ROIs from the selected file
        while True:
            print(f"\n🏆 Top 10 calculated ROIs for this short window: {top_10_calculated}")
            print(f"Plotting window {short_window_tlim} with ROIs: {current_rois}")
            settings['CaImaging']['roiIndices'] = current_rois
            
            plt.close('all')
            plt.clf()
            
            # Map parameters onto layout
            plot(champ_data, tlim=short_window_tlim, settings=settings, figsize=(12, 7))
            active_fig = plt.gcf()
            plt.show(block=False)
            plt.pause(0.5)
            
            print("\nOptions:")
            print("  - Press [ENTER] if you like this plot and want to SAVE it.")
            print("  - Type 4 numbers from the top 10 list (separated by commas: e.g., 4,12,0,8) to update.")
            print("  - Type 'B' to go BACK to the session list and swap the NWB file.")
            
            user_choice = input("👉 Your choice: ").strip()
            
            if user_choice.lower() == 'b':
                print("Returning to leaderboard layout...")
                break
                
            elif not user_choice:
                # Target path configuration
                output_filename = f"DEVELOPMENTAL_LOCAL_16_STIM_TRACES_{group_name}.svg"
                output_dir = os.path.join(os.path.expanduser('~'), 'Desktop', 'Final-Figures-PhD')
                os.makedirs(output_dir, exist_ok=True)
                output_svg = os.path.join(output_dir, output_filename)
                
                # Commit configuration save using isolated handler instance
                active_fig.savefig(output_svg, format='svg', dpi=300, bbox_inches='tight', facecolor='white')
                print(f"\n✔️ Success! Saved thesis vector graphic to: {output_svg}")
                plt.close('all')
                break
            else:
                try:
                    current_rois = [int(x.strip()) for x in user_choice.split(',')]
                except ValueError:
                    print("[Error] Please separate indices using commas. Example: 1,5,2,9")
                    
        # If the sub-loop exited because a file was saved successfully, move to next age group
        if not user_choice:
            break

# Execute across all developmental cohorts sequentially
for group_name in age_groups:
    run_interactive_cohort_selector(age_leaderboards[group_name], group_name)
















#%%

### BEFORE THIS WAS YOUNG 

# %%
############## ADULT ### ADULT ##### ADULT #### ADULT #############################
###### FIXED SAVING ROUTINE: WINDOW-TARGETED ADULT COHORT (TOP 4 ROIS) ############
###################################################################################

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import physion
from physion.dataviz.raw import plot

# ------------------------------------------------------------
# 1) PATH CONFIGURATION FOR ADULT FOLDER
# ------------------------------------------------------------
base_nwb_folder = os.path.join(os.path.expanduser('~'), 'CURATED', 'Cibele', 'PYR-SynGCaMP_WT_Adult_V1', 'NWBs')
target_leaderboard = []

TARGET_PROTOCOL = "ff-gratings-8orientation-2contrasts-15repeats"

print(f"Starting localized window screening for Adult cohort protocol: {TARGET_PROTOCOL}...\n")

search_pattern = os.path.join(base_nwb_folder, "*.nwb")
nwb_files = glob.glob(search_pattern)

print(f"Scanning directory '{base_nwb_folder}' (Found {len(nwb_files)} files)...")
matched_count = 0

# ------------------------------------------------------------
# 2) WINDOW-RESTRICTED BIOLOGICAL MATCHING PIPELINE
# ------------------------------------------------------------
for fpath in nwb_files:
    fname = os.path.basename(fpath)
    try:
        data = physion.analysis.read_NWB.Data(fpath, verbose=False)
        
        protocol_meta = str(data.metadata.get('protocol', '')).lower()
        if TARGET_PROTOCOL not in protocol_meta:
            continue
            
        data.build_rawFluo(verbose=False)
        data.build_dFoF(verbose=False)
        
        dfff_data = data.dFoF
        if dfff_data.shape[0] == getattr(data, 'nROIs', 0):
            dfff_data = dfff_data.T  
            
        if dfff_data.size == 0:
            continue

        try:
            t_start = data.visual_stim['time'][0]
            duration = data.metadata.get('presentation-duration', 2)
            interstim = data.metadata.get('presentation-interstim-period', 3)
            cycle_length = duration + interstim
            t_end = t_start + (16 * cycle_length)
        except:
            t_start, t_end = 30, 115

        fps = getattr(data, 'imaging_rate', 30)
        frame_start = int(t_start * fps)
        frame_end = int(t_end * fps)

        dfff_window = dfff_data[frame_start:frame_end, :]
        if dfff_window.size == 0 or dfff_window.shape[0] < 10:
            continue

        window_size = 9
        dff_smoothed = np.apply_along_axis(
            lambda m: np.convolve(m, np.ones(window_size)/window_size, mode='same'), 
            axis=0, arr=dfff_window
        )

        signal_variance = np.std(dff_smoothed, axis=0)
        instrumental_noise = np.std(dfff_window - dff_smoothed, axis=0)
        
        selectivity_scores = signal_variance / (instrumental_noise + 0.05)
        selectivity_scores = np.nan_to_num(selectivity_scores, nan=0.0)

        sorted_roi_indices = np.argsort(selectivity_scores)[::-1]
        best_roi_idx = sorted_roi_indices[0]
        best_roi_score = selectivity_scores[best_roi_idx]
        
        file_info = {
            'file_path': fpath,
            'file_name': fname,
            'top_score': best_roi_score,
            'sorted_rois': sorted_roi_indices,
            'all_scores': selectivity_scores,
            'tlim_bounds': [t_start, t_end]
        }
        target_leaderboard.append(file_info)
        matched_count += 1
    except:
        continue
        
print(f" -> Successfully matched, parsed, and locally ranked {matched_count} Adult files.\n")

target_leaderboard.sort(key=lambda x: x['top_score'], reverse=True)

# ------------------------------------------------------------
# 3) INTERACTIVE GENERATION & FORCE-SAVE CAPTURE
# ------------------------------------------------------------
if not target_leaderboard:
    print("[WARNING] No valid protocol-matching Adult files found.")
else:
    champion = target_leaderboard[0]
    print("="*60)
    print(f"PREPARING FIGURE FOR ADULT COHORT (4-ROI DISPLAY)")
    print("="*60)
    print(f"Selected Session: {champion['file_name']}")
    
    top_10_calculated = list(champion['sorted_rois'][:10])
    print(f"🏆 Top 10 ROIs ranked within this window: {top_10_calculated}")
    
    champ_data = physion.analysis.read_NWB.Data(champion['file_path'], verbose=False)
    champ_data.build_rawFluo(verbose=False)
    champ_data.build_dFoF(verbose=False)
    
    current_rois = top_10_calculated[:4]
    
    settings = {
     'Locomotion': {'fig_fraction': 1, 'subsampling': 1, 'color': '#1f77b4'},
     'CaImaging': {'fig_fraction': 4, 'subsampling': 1, 'subquantity': 'dFoF', 'color': '#b03a2e', 'roiIndices': current_rois},
     'CaImagingRaster': {'fig_fraction': 2, 'subsampling': 1, 'roiIndices': 'all', 'normalization': 'per-line', 'subquantity': 'dF/F'},
     'VisualStim': {'fig_fraction': 0.4, 'color': 'black'}
    }
    
    short_window_tlim = champion['tlim_bounds']
    
    while True:
        print(f"\nPlotting short window {short_window_tlim} with 4 ROIs: {current_rois}")
        settings['CaImaging']['roiIndices'] = current_rois
        
        # Explicitly clear prior figure handles before mapping new ones
        plt.close('all')
        
        # Execute physion plot 
        plot(champ_data, tlim=short_window_tlim, settings=settings, figsize=(12, 7))
        
        # FORCE CAPTURE: Immediately fetch the active window figure handle 
        # that physion just built before the loop stalls
        active_fig = plt.gcf()
        
        plt.show(block=False)
        plt.pause(0.5)
        
        print("\nOptions:")
        print("  - Press [ENTER] if you like this plot and want to SAVE it.")
        print(f"  - Type any 4 numbers from the top 10 (e.g., {top_10_calculated[2]},{top_10_calculated[3]},{top_10_calculated[4]},{top_10_calculated[5]}) to update them.")
        
        user_choice = input("👉 Your choice: ").strip()
        
        if not user_choice:
            # Output filepath targets
            output_filename = f"FINAL_ADULT_16_STIM_4_TRACES.svg"
            output_dir = os.path.join(os.path.expanduser('~'), 'Desktop', 'Final-Figures-PhD')
            os.makedirs(output_dir, exist_ok=True)
            output_svg = os.path.join(output_dir, output_filename)
            
            # Use our isolated active_fig handler to force save the layout
            active_fig.savefig(output_svg, format='svg', dpi=300, bbox_inches='tight', facecolor='white')
            print(f"\n✔️ Success! Saved adult thesis vector graphic to: {output_svg}")
            plt.close('all')
            break
        else:
            try:
                current_rois = [int(x.strip()) for x in user_choice.split(',')]
            except ValueError:
                print("[Error] Please separate numbers using commas. Let's try again.")




# %%
###################################################################################
###### INTERACTIVE SESSION + ROI SELECTOR FOR REJECTING BEHAVIOR FILES ############
###################################################################################
### Type 1, 2 or 3 if I'm not happy with the behavior traces and want to see the next best session in that folder.########


import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import physion
from physion.dataviz.raw import plot

# ------------------------------------------------------------
# 1) PATH CONFIGURATION AND LOCAL RANKING
# ------------------------------------------------------------
base_nwb_folder = os.path.join(os.path.expanduser('~'), 'CURATED', 'Cibele', 'PYR-SynGCaMP_WT_Adult_V1', 'NWBs')
target_leaderboard = []
TARGET_PROTOCOL = "ff-gratings-8orientation-2contrasts-15repeats"

search_pattern = os.path.join(base_nwb_folder, "*.nwb")
nwb_files = glob.glob(search_pattern)

print(f"Scanning directory for ranking '{base_nwb_folder}'...")

for fpath in nwb_files:
    fname = os.path.basename(fpath)
    try:
        data = physion.analysis.read_NWB.Data(fpath, verbose=False)
        if TARGET_PROTOCOL not in str(data.metadata.get('protocol', '')).lower():
            continue
            
        data.build_rawFluo(verbose=False)
        data.build_dFoF(verbose=False)
        dfff_data = data.dFoF.T if data.dFoF.shape[0] == getattr(data, 'nROIs', 0) else data.dFoF
        
        try:
            t_start = data.visual_stim['time'][0]
            cycle = data.metadata.get('presentation-duration', 2) + data.metadata.get('presentation-interstim-period', 3)
            t_end = t_start + (16 * cycle)
        except:
            t_start, t_end = 30, 115

        fps = getattr(data, 'imaging_rate', 30)
        dfff_window = dfff_data[int(t_start * fps):int(t_end * fps), :]

        window_size = 9
        dff_smoothed = np.apply_along_axis(lambda m: np.convolve(m, np.ones(window_size)/window_size, mode='same'), axis=0, arr=dfff_window)

        signal_variance = np.std(dff_smoothed, axis=0)
        instrumental_noise = np.std(dfff_window - dff_smoothed, axis=0)
        selectivity_scores = np.nan_to_num(signal_variance / (instrumental_noise + 0.05), nan=0.0)

        target_leaderboard.append({
            'file_path': fpath, 
            'file_name': fname,
            'top_score': np.max(selectivity_scores), 
            'sorted_rois': np.argsort(selectivity_scores)[::-1],
            'tlim_bounds': [t_start, t_end]
        })
    except:
        continue

target_leaderboard.sort(key=lambda x: x['top_score'], reverse=True)

# ------------------------------------------------------------
# 2) CRITICAL FIX: DYNAMIC RE-INITIALIZATION LOOP
# ------------------------------------------------------------
if not target_leaderboard:
    print("[WARNING] No matching files found.")
else:
    # We loop everything so you can keep changing your mind or test different files
    while True:
        plt.close('all')
        plt.clf()
        
        print("\n" + "="*60)
        print("🏆 AVAILABLE ADULT SESSIONS LEADERBOARD:")
        print("="*60)
        for idx, item in enumerate(target_leaderboard[:5]):
            print(f"  [{idx + 1}] {item['file_name']} (Window Tuning Score: {item['top_score']:.2f})")
        print("  [Q] Quit script without saving")
        print("="*60)
        
        file_choice = input("\n👉 Choose a session index to load (1-5) or 'Q' to quit: ").strip()
        
        if file_choice.lower() == 'q':
            print("Exiting screening loop.")
            break
            
        try:
            session_idx = int(file_choice) - 1 if file_choice else 0
            champion = target_leaderboard[session_idx]
        except (ValueError, IndexError):
            print("[Error] Invalid option. Please type a number between 1 and 5.")
            continue
            
        print(f"\n🔄 FORCING FRESH LOAD FOR: {champion['file_name']}")
        top_10_calculated = list(champion['sorted_rois'][:10])
        
        # Core fix: Data object reconstructed inside the loop block to dump prior session caches
        champ_data = physion.analysis.read_NWB.Data(champion['file_path'], verbose=False)
        champ_data.build_rawFluo(verbose=False)
        champ_data.build_dFoF(verbose=False)
        
        current_rois = top_10_calculated[:4]
        short_window_tlim = champion['tlim_bounds']
        
        settings = {
         'Locomotion': {'fig_fraction': 1, 'subsampling': 1, 'color': '#1f77b4'},
         'CaImaging': {'fig_fraction': 4, 'subsampling': 1, 'subquantity': 'dFoF', 'color': '#b03a2e', 'roiIndices': current_rois},
         'CaImagingRaster': {'fig_fraction': 2, 'subsampling': 1, 'roiIndices': 'all', 'normalization': 'per-line', 'subquantity': 'dF/F'},
         'VisualStim': {'fig_fraction': 0.4, 'color': 'black'}
        }
        
        # Sub-loop for optimizing ROIs within the chosen session file
        while True:
            print(f"\n🏆 Top 10 available ROIs for this file: {top_10_calculated}")
            print(f"Plotting window {short_window_tlim} with ROIs: {current_rois}")
            settings['CaImaging']['roiIndices'] = current_rois
            
            plt.close('all')
            plt.clf()
            
            # Draw the trace panel
            plot(champ_data, tlim=short_window_tlim, settings=settings, figsize=(12, 7))
            active_fig = plt.gcf()
            plt.show(block=False)
            plt.pause(0.5)
            
            print("\nOptions:")
            print("  - Press [ENTER] if you like this plot and want to SAVE it to Desktop.")
            print("  - Type 4 new ROIs from the top 10 list (e.g., 5,12,3,9) to update traces.")
            print("  - Type 'B' to go BACK to the file list and choose a different session.")
            
            user_choice = input("👉 Your choice: ").strip()
            
            if user_choice.lower() == 'b':
                print("Returning to session list...")
                break # Breaks out of ROI sub-loop, returns to file selection list
                
            elif not user_choice:
                # Save execution
                output_filename = f"FINAL_ADULT_16_STIM_FILE_{session_idx+1}.svg"
                output_dir = os.path.join(os.path.expanduser('~'), 'Desktop', 'Final-Figures-PhD')
                os.makedirs(output_dir, exist_ok=True)
                output_svg = os.path.join(output_dir, output_filename)
                
                active_fig.savefig(output_svg, format='svg', dpi=300, bbox_inches='tight', facecolor='white')
                print(f"\n✔️ Success! Saved pristine vector graphic to: {output_svg}")
                plt.close('all')
                break # Exit sub-loop
            else:
                try:
                    current_rois = [int(x.strip()) for x in user_choice.split(',')]
                except ValueError:
                    print("[Error] Use commas to split your indices. Example: 1,2,3,4")
        
        # If user hit enter and saved, or typed 'q', check if we want to end overall program execution
        if not user_choice:
            break
5






# %%
########### FOR CONTRAST SENSITIVITY ANALYSIS ############

#%%
###################################################################################
###### CONTRAST SENSITIVITY SCREENING: SLOPE & NEGATIVE RESPONSE TRACKER ##########
###################################################################################

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import physion
from physion.dataviz.raw import plot

# ------------------------------------------------------------
# 1) CONFIGURAÇÃO DE PATHS E PROTOCOLO DE CONTRASTE
# ------------------------------------------------------------
base_nwb_folder = os.path.join(os.path.expanduser('~'), 'CURATED', 'Cibele', 'SST-cells_WT_Adult_V1', 'NWBs')
target_leaderboard = []

# Atualizado para o protocolo de sensibilidade ao contraste (ajuste a string se houver variação de grafia)
TARGET_PROTOCOL = "ff-gratings-2orientation-8contrasts-15repeats"

print(f"Iniciando varredura de janela para curvas de contraste: {TARGET_PROTOCOL}...\n")

search_pattern = os.path.join(base_nwb_folder, "*.nwb")
nwb_files = glob.glob(search_pattern)

print(f"Escaneando diretório '{base_nwb_folder}' (Encontrados {len(nwb_files)} arquivos)...")
matched_count = 0

# ------------------------------------------------------------
# 2) PIPELINE DE SELEÇÃO ADAPTADO PARA RESPOSTAS NEGATIVAS
# ------------------------------------------------------------
for fpath in nwb_files:
    fname = os.path.basename(fpath)
    try:
        data = physion.analysis.read_NWB.Data(fpath, verbose=False)
        
        # Filtro de string do protocolo
        protocol_meta = str(data.metadata.get('protocol', '')).lower()
        if "8contrast" not in protocol_meta and "2orientation" not in protocol_meta:
            if TARGET_PROTOCOL not in protocol_meta:
                continue
            
        data.build_rawFluo(verbose=False)
        data.build_dFoF(verbose=False)
        
        dfff_data = data.dFoF.T if data.dFoF.shape[0] == getattr(data, 'nROIs', 0) else data.dFoF
        if dfff_data.size == 0:
            continue

        # --- CÁLCULO DE JANELA TARGET (16 ESTIMULOS: 8 Contrastes x 2 Orientações) ---
        try:
            t_start = data.visual_stim['time'][0]
            duration = data.metadata.get('presentation-duration', 2)
            interstim = data.metadata.get('presentation-interstim-period', 3)
            cycle_length = duration + interstim
            t_end = t_start + (16 * cycle_length)
        except:
            t_start, t_end = 30, 115

        fps = getattr(data, 'imaging_rate', 30)
        frame_start = int(t_start * fps)
        frame_end = int(t_end * fps)

        dfff_window = dfff_data[frame_start:frame_end, :]
        if dfff_window.size == 0 or dfff_window.shape[0] < 10:
            continue

        # Suavização para limpar ruído de alta frequência sem matar deflexões negativas
        window_size = 9
        dff_smoothed = np.apply_along_axis(
            lambda m: np.convolve(m, np.ones(window_size)/window_size, mode='same'), 
            axis=0, arr=dfff_window
        )

        # --- NOVA MATEMÁTICA: CAPTURA DE RESPOSTAS ABSOLUTAS (POSITIVAS E NEGATIVAS) ---
        # Subtrai o baseline local (mediana) para expor deflexões reais negativas abaixo de zero
        local_baseline = np.median(dff_smoothed, axis=0)
        zeroed_traces = dff_smoothed - local_baseline
        
        max_peaks = np.max(zeroed_traces, axis=0)
        min_peaks = np.abs(np.min(zeroed_traces, axis=0)) # Magnitude da resposta negativa
        
        # O desvio absoluto captura tanto ativações gigantes quanto supressões profundas pelo contraste
        dynamic_range = max_peaks + min_peaks
        instrumental_noise = np.std(dfff_window - dff_smoothed, axis=0)
        
        # Score final valoriza curvas dinâmicas (com slope) e pune ruído instrumental
        contrast_slope_scores = dynamic_range / (instrumental_noise + 0.05)
        contrast_slope_scores = np.nan_to_num(contrast_slope_scores, nan=0.0)

        sorted_roi_indices = np.argsort(contrast_slope_scores)[::-1]
        best_roi_score = contrast_slope_scores[sorted_roi_indices[0]]
        
        target_leaderboard.append({
            'file_path': fpath, 
            'file_name': fname,
            'top_score': best_roi_score, 
            'sorted_rois': sorted_roi_indices,
            'tlim_bounds': [t_start, t_end]
        })
        matched_count += 1
    except:
        continue

print(f" -> Ranqueados com sucesso {matched_count} arquivos de Contraste.\n")
target_leaderboard.sort(key=lambda x: x['top_score'], reverse=True)

# ------------------------------------------------------------
# 3) INTERFACE DE SELEÇÃO DE SESSÃO E SUBSTITUIÇÃO DE ROIS
# ------------------------------------------------------------
if not target_leaderboard:
    print("[WARNING] Nenhum arquivo correspondente ao protocolo de contraste foi encontrado.")
else:
    while True:
        plt.close('all')
        plt.clf()
        
        print("\n" + "="*60)
        print("🏆 LEADERBOARD DE CONTRASTE (ADULTO - PYR):")
        print("="*60)
        for idx, item in enumerate(target_leaderboard[:5]):
            print(f"  [{idx + 1}] {item['file_name']} (Contrast-Slope Score: {item['top_score']:.2f})")
        print("  [Q] Sair do script")
        print("="*60)
        
        file_choice = input("\n👉 Escolha o índice da sessão (1-5) para analisar ou 'Q' para sair: ").strip()
        
        if file_choice.lower() == 'q':
            print("Saindo do programa.")
            break
            
        try:
            session_idx = int(file_choice) - 1 if file_choice else 0
            champion = target_leaderboard[session_idx]
        except (ValueError, IndexError):
            print("[Erro] Opção inválida. Digite um número entre 1 e 5.")
            continue
            
        print(f"\n🔄 CARREGANDO SESSÃO: {champion['file_name']}")
        top_10_calculated = list(champion['sorted_rois'][:10])
        
        champ_data = physion.analysis.read_NWB.Data(champion['file_path'], verbose=False)
        champ_data.build_rawFluo(verbose=False)
        champ_data.build_dFoF(verbose=False)
        
        current_rois = top_10_calculated[:4]
        short_window_tlim = champion['tlim_bounds']
        
        settings = {
         'Locomotion': {'fig_fraction': 1, 'subsampling': 1, 'color': '#1f77b4'},
         'CaImaging': {'fig_fraction': 4, 'subsampling': 1, 'subquantity': 'dFoF', 'color': '#b03a2e', 'roiIndices': current_rois},
         'CaImagingRaster': {'fig_fraction': 2, 'subsampling': 1, 'roiIndices': 'all', 'normalization': 'per-line', 'subquantity': 'dF/F'},
         'VisualStim': {'fig_fraction': 0.4, 'color': 'black'}
        }
        
        while True:
            print(f"\n🏆 Top 10 ROIs recomendados para ver Slope/Resposta Negativa: {top_10_calculated}")
            print(f"Plotando janela {short_window_tlim} com ROIs: {current_rois}")
            settings['CaImaging']['roiIndices'] = current_rois
            
            plt.close('all')
            plt.clf()
            
            # Executa o plot do Physion
            plot(champ_data, tlim=short_window_tlim, settings=settings, figsize=(12, 7))
            active_fig = plt.gcf()
            plt.show(block=False)
            plt.pause(0.5)
            
            print("\nOpções:")
            print("  - Pressione [ENTER] se o gráfico estiver perfeito para SALVAR no Desktop.")
            print("  - Digite 4 novos ROIs do Top 10 (ex: 8,3,5,1) para caçar traços negativos/slopes.")
            print("  - Digite 'B' para VOLTAR ao menu de arquivos e testar outra sessão.")
            
            user_choice = input("👉 Escolha sua ação: ").strip()
            
            if user_choice.lower() == 'b':
                print("Retornando ao Leaderboard...")
                break 
                
            elif not user_choice:
                # Salvar vetor estruturado
                output_filename = f"CONTRAST_SENSITIVITY_FILE_{session_idx+1}.svg"
                output_dir = os.path.join(os.path.expanduser('~'), 'Desktop', 'Final-Figures-PhD')
                os.makedirs(output_dir, exist_ok=True)
                output_svg = os.path.join(output_dir, output_filename)
                
                active_fig.savefig(output_svg, format='svg', dpi=300, bbox_inches='tight', facecolor='white')
                print(f"\n✔️ Sucesso! Gráfico vetorial de contraste salvo em: {output_svg}")
                plt.close('all')
                break
            else:
                try:
                    current_rois = [int(x.strip()) for x in user_choice.split(',')]
                except ValueError:
                    print("[Erro] Use vírgulas para separar os ROIs. Exemplo: 0,1,2,3")
        
        if not user_choice:
            break













#%%

#### TRYING TO SEE THE NEGATIVE REWSPONSE ############

###################################################################################
###### CONTRAST SENSITIVITY SCREENING: SLOPE & NEGATIVE RESPONSE TRACKER ##########
###################################################################################

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import physion
from physion.dataviz.raw import plot

# ------------------------------------------------------------
# 1) CONFIGURAÇÃO DE PATHS E PROTOCOLO DE CONTRASTE
# ------------------------------------------------------------
base_nwb_folder = os.path.join(os.path.expanduser('~'), 'CURATED', 'Cibele', 'SST-cells_WT_Adult_V1', 'NWBs')
target_leaderboard = []

# Mantido exatamente o seu filtro original
TARGET_PROTOCOL = "ff-gratings-2orientation-8contrasts-15repeats"

print(f"Iniciando varredura de janela para curvas de contraste: {TARGET_PROTOCOL}...\n")

search_pattern = os.path.join(base_nwb_folder, "*.nwb")
nwb_files = glob.glob(search_pattern)

print(f"Escaneando diretório '{base_nwb_folder}' (Encontrados {len(nwb_files)} arquivos)...")
matched_count = 0

# ------------------------------------------------------------
# 2) PIPELINE DE SELEÇÃO ADAPTADO PARA RESPOSTAS NEGATIVAS
# ------------------------------------------------------------
for fpath in nwb_files:
    fname = os.path.basename(fpath)
    try:
        data = physion.analysis.read_NWB.Data(fpath, verbose=False)
        
        # Filtro de string do protocolo original intacto
        protocol_meta = str(data.metadata.get('protocol', '')).lower()
        if "8contrast" not in protocol_meta and "2orientation" not in protocol_meta:
            if TARGET_PROTOCOL not in protocol_meta:
                continue
            
        data.build_rawFluo(verbose=False)
        data.build_dFoF(verbose=False)
        
        dfff_data = data.dFoF.T if data.dFoF.shape[0] == getattr(data, 'nROIs', 0) else data.dFoF
        if dfff_data.size == 0:
            continue

        # --- CÁLCULO DE JANELA TARGET ---
        try:
            t_start = data.visual_stim['time'][0]
            duration = data.metadata.get('presentation-duration', 2)
            interstim = data.metadata.get('presentation-interstim-period', 3)
            cycle_length = duration + interstim
            t_end = t_start + (16 * cycle_length)
        except:
            t_start, t_end = 30, 115

        fps = getattr(data, 'imaging_rate', 30)
        frame_start = int(t_start * fps)
        frame_end = int(t_end * fps)

        dfff_window = dfff_data[frame_start:frame_end, :]
        if dfff_window.size == 0 or dfff_window.shape[0] < 10:
            continue

        window_size = 9
        dff_smoothed = np.apply_along_axis(
            lambda m: np.convolve(m, np.ones(window_size)/window_size, mode='same'), 
            axis=0, arr=dfff_window
        )

        # Subtrai o baseline local (mediana) para expor deflexões reais negativas abaixo de zero
        local_baseline = np.median(dff_smoothed, axis=0)
        zeroed_traces = dff_smoothed - local_baseline
        
        max_peaks = np.max(zeroed_traces, axis=0)
        min_peaks = np.abs(np.min(zeroed_traces, axis=0)) # Magnitude da resposta negativa
        
        # Mudança aqui: ignoramos o max_peak positivo e focamos apenas no min_peak negativo profundo
        dynamic_range = min_peaks 
        
        instrumental_noise = np.std(dfff_window - dff_smoothed, axis=0)
        
        # Score final agora valoriza apenas as maiores supressões abaixo de zero
        contrast_slope_scores = dynamic_range / (instrumental_noise + 0.05)
        contrast_slope_scores = np.nan_to_num(contrast_slope_scores, nan=0.0)

        sorted_roi_indices = np.argsort(contrast_slope_scores)[::-1]
        best_roi_score = contrast_slope_scores[sorted_roi_indices[0]]
        
        target_leaderboard.append({
            'file_path': fpath, 
            'file_name': fname,
            'top_score': best_roi_score, 
            'sorted_rois': sorted_roi_indices,
            'tlim_bounds': [t_start, t_end]
        })
        matched_count += 1
    except:
        continue

print(f" -> Ranqueados com sucesso {matched_count} arquivos de Contraste.\n")
target_leaderboard.sort(key=lambda x: x['top_score'], reverse=True)

# ------------------------------------------------------------
# 3) INTERFACE DE SELEÇÃO DE SESSÃO E SUBSTITUIÇÃO DE ROIS
# ------------------------------------------------------------
if not target_leaderboard:
    print("[WARNING] Nenhum arquivo correspondente ao protocolo de contraste foi encontrado.")
else:
    while True:
        plt.close('all')
        plt.clf()
        
        print("\n" + "="*60)
        print("🏆 LEADERBOARD DE CONTRASTE (RANKING DE RESPOSTA NEGATIVA):")
        print("="*60)
        for idx, item in enumerate(target_leaderboard[:5]):
            print(f"  [{idx + 1}] {item['file_name']} (Negative-Drop Score: {item['top_score']:.2f})")
        print("  [Q] Sair do script")
        print("="*60)
        
        file_choice = input("\n👉 Escolha o índice da sessão (1-5) para analisar ou 'Q' para sair: ").strip()
        
        if file_choice.lower() == 'q':
            print("Saindo do programa.")
            break
            
        try:
            session_idx = int(file_choice) - 1 if file_choice else 0
            champion = target_leaderboard[session_idx]
        except (ValueError, IndexError):
            print("[Erro] Opção inválida. Digite um número entre 1 e 5.")
            continue
            
        print(f"\n🔄 CARREGANDO SESSÃO: {champion['file_name']}")
        top_10_calculated = list(champion['sorted_rois'][:10])
        
        champ_data = physion.analysis.read_NWB.Data(champion['file_path'], verbose=False)
        champ_data.build_rawFluo(verbose=False)
        champ_data.build_dFoF(verbose=False)
        
        current_rois = top_10_calculated[:4]
        short_window_tlim = champion['tlim_bounds']
        
        settings = {
         'Locomotion': {'fig_fraction': 1, 'subsampling': 1, 'color': '#1f77b4'},
         'CaImaging': {'fig_fraction': 4, 'subsampling': 1, 'subquantity': 'dFoF', 'color': '#b03a2e', 'roiIndices': current_rois},
         'CaImagingRaster': {'fig_fraction': 2, 'subsampling': 1, 'roiIndices': 'all', 'normalization': 'per-line', 'subquantity': 'dF/F'},
         'VisualStim': {'fig_fraction': 0.4, 'color': 'black'}
        }
        
        while True:
            print(f"\n🏆 Top 10 ROIs recomendados para ver Respostas Negativas Intensas: {top_10_calculated}")
            print(f"Plotando janela {short_window_tlim} com ROIs: {current_rois}")
            settings['CaImaging']['roiIndices'] = current_rois
            
            plt.close('all')
            plt.clf()
            
            # Executa o plot do Physion original
            plot(champ_data, tlim=short_window_tlim, settings=settings, figsize=(12, 7))
            active_fig = plt.gcf()
            plt.show(block=False)
            plt.pause(0.5)
            
            print("\nOpções:")
            print("  - Pressione [ENTER] se o gráfico estiver perfeito para SALVAR no Desktop.")
            print("  - Digite 4 novos ROIs do Top 10 (ex: 8,3,5,1) para analisar outras curvas.")
            print("  - Digite 'B' para VOLTAR ao menu de arquivos e testar outra sessão.")
            
            user_choice = input("👉 Escolha sua ação: ").strip()
            
            if user_choice.lower() == 'b':
                print("Retornando ao Leaderboard...")
                break 
                
            elif not user_choice:
                # Salvar vetor estruturado
                output_filename = f"CONTRAST_NEGATIVE_SUPPRESSION_SESS_{session_idx+1}.svg"
                output_dir = os.path.join(os.path.expanduser('~'), 'Desktop', 'Final-Figures-PhD')
                os.makedirs(output_dir, exist_ok=True)
                output_svg = os.path.join(output_dir, output_filename)
                
                active_fig.savefig(output_svg, format='svg', dpi=300, bbox_inches='tight', facecolor='white')
                print(f"\n✔️ Sucesso! Gráfico vetorial salvo em: {output_svg}")
                plt.close('all')
                break
            else:
                try:
                    current_rois = [int(x.strip()) for x in user_choice.split(',')]
                except ValueError:
                    print("[Erro] Use vírgulas para separar os ROIs. Exemplo: 0,1,2,3")
        
        if not user_choice:
            break


        




# %%
###################################################################################
###### THESIS LAYOUT: SEGMENTED CONTRAST SENSITIVITY POPULATION MATRIX ############
###################################################################################

import os1
import glob
import numpy as np
import matplotlib.pyplot as plt
import physion

# ------------------------------------------------------------
# 1) PATH CONFIGURATIONS AND TARGET PROTOCOL
# ------------------------------------------------------------
base_nwb_folder = os.path.join(os.path.expanduser('~'), 'CURATED', 'Cibele', 'SST-cells_WT_Adult_V1', 'NWBs')
output_dir = os.path.join(os.path.expanduser('~'), 'Desktop', 'Final-Figures-PhD', 'Population_Matrices')
os.makedirs(output_dir, exist_ok=True)

TARGET_PROTOCOL = "ff-gratings-2orientation-8contrasts-15repeats"
search_pattern = os.path.join(base_nwb_folder, "*.nwb")
nwb_files = glob.glob(search_pattern)

print(f"Scanning for contrast-tuning NWBs inside: {base_nwb_folder}")
print(f"Matrix population plots will be saved to: {output_dir}\n")

processed_count = 0

# ------------------------------------------------------------
# 2) AUTOMATIC PROCESSING BATCH LOOP
# ------------------------------------------------------------
for fpath in nwb_files:
    fname = os.path.basename(fpath)
    try:
        data = physion.analysis.read_NWB.Data(fpath, verbose=False)
        
        # Protocol validation check
        protocol_meta = str(data.metadata.get('protocol', '')).lower()
        if "8contrast" not in protocol_meta and "2orientation" not in protocol_meta:
            if TARGET_PROTOCOL not in protocol_meta:
                continue
        
        print(f"📊 Extrapolating matrix conditions for: {fname}...")
        
        # Build signals safely
        data.build_rawFluo(verbose=False)
        data.build_dFoF(verbose=False)
        
        dfff_data = data.dFoF.T if data.dFoF.shape[0] == getattr(data, 'nROIs', 0) else data.dFoF
        if dfff_data.size == 0:
            continue
            
        fps = getattr(data, 'imaging_rate', 30)
        
        # ------------------------------------------------------------
        # STEP A: ISOLATE THE SUPPRESSED CELL POPULATION
        # ------------------------------------------------------------
        # Standard smoothing step to identify suppressed cells cleanly
        window_size = 9
        dff_smoothed = np.apply_along_axis(
            lambda m: np.convolve(m, np.ones(window_size)/window_size, mode='same'), 
            axis=0, arr=dfff_data
        )
        zeroed_traces = dff_smoothed - np.median(dff_smoothed, axis=0)
        
        # Select ROIs whose minimum deflection drops significantly
        negative_selectors = np.min(zeroed_traces, axis=0) < -0.05
        suppressed_indices = np.where(negative_selectors)[0]
        
        if len(suppressed_indices) == 0:
            print(f"  [Warning] No suppressed cells found in {fname}. Skipping matrix layout.")
            continue

        # ------------------------------------------------------------
        # STEP B: EXTRACT AND SORT CONDITIONS FROM METADATA
        # ------------------------------------------------------------
        stim_table = data.visual_stim
        unique_orientations = np.sort(np.unique(stim_table['orientation']))
        unique_contrasts = np.sort(np.unique(stim_table['contrast']))
        
        # Safety catch for unexpected experimental parameters
        if len(unique_orientations) < 2 or len(unique_contrasts) < 8:
            # Fallback if contrasts contain a 0 value or variations
            if len(unique_contrasts) < 2:
                continue

        # Time framing for snippet windows: 1s pre-stimulus to 3s post-stimulus onset
        pre_stim = 1.0  
        post_stim = 3.0 
        pre_frames = int(pre_stim * fps)
        post_frames = int(post_stim * fps)
        total_frames = pre_frames + post_frames
        
        # Local time array centered at stimulus onset (t=0)
        local_time = np.linspace(-pre_stim, post_stim, total_frames)

        # ------------------------------------------------------------
        # STEP C: INITIALIZE SUBPLOT GRID (2 Rows x 8 Columns)
        # ------------------------------------------------------------
        fig, axes = plt.subplots(2, 8, figsize=(16, 5), sharex=True, sharey=True)
        fig.suptitle(f"Stimulus-evoked activity averaged across suppressed cells (n={len(suppressed_indices)} ROIs)\nSession: {fname}", 
                     fontsize=12, fontweight='bold', y=0.98)

        # Theme styling settings matching the thesis standard
        line_color = '#b03a2e'  # Suppressed crimson/purple variant
        shade_color = '#b03a2e'

        # Loop systematically through rows (Orientations) and columns (Contrasts)
        for row_idx, ori in enumerate(unique_orientations[:2]):
            for col_idx, contrast in enumerate(unique_contrasts[:8]):
                ax = axes[row_idx, col_idx]
                
                # Identify trial timestamps that match this specific combination
                matched_trials = np.where(
                    (stim_table['orientation'] == ori) & 
                    (stim_table['contrast'] == contrast)
                )[0]
                
                if len(matched_trials) == 0:
                    ax.axis('off')
                    continue
                
                # Gather snippets across all matching trials for all suppressed cells
                trial_snippets = []
                for trial in matched_trials:
                    onset_time = stim_table['time'][trial]
                    onset_frame = int(onset_time * fps)
                    
                    # Prevent out-of-bounds errors near trace edges
                    if onset_frame - pre_frames < 0 or onset_frame + post_frames > dfff_data.shape[0]:
                        continue
                        
                    # Slice continuous data down to this single episode window
                    snippet = dfff_data[onset_frame - pre_frames : onset_frame + post_frames, suppressed_indices]
                    
                    # Baseline-subtract using the pre-stimulus window segment of this specific trial
                    baseline = np.median(snippet[:pre_frames, :], axis=0)
                    normalized_snippet = snippet - baseline
                    
                    # Collapse across cells first to get the population trace for this single trial
                    population_trial_trace = np.mean(normalized_snippet, axis=1)
                    trial_snippets.append(population_trial_trace)
                
                if len(trial_snippets) == 0:
                    continue
                
                # Compute final mean and standard error (SEM) across all trial repetitions
                trial_snippets = np.array(trial_snippets) # Shape: (trials, frames)
                evoked_mean = np.mean(trial_snippets, axis=0)
                evoked_sem = np.std(trial_snippets, axis=0) / np.sqrt(trial_snippets.shape[0])
                
                # ------------------------------------------------------------
                # STEP D: PLOT MATRIX SNIPPET
                # ------------------------------------------------------------
                # Shading confidence boundary
                ax.fill_between(local_time, evoked_mean - evoked_sem, evoked_mean + evoked_sem, 
                                color=shade_color, alpha=0.15, lw=0)
                # Plot central evoked average
                ax.plot(local_time, evoked_mean, color=line_color, linewidth=1.8)
                
                # Static reference markings
                ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
                ax.axvspan(0, 2.0, color='black', alpha=0.04, lw=0) # Shading showing 2s stimulus block
                
                # Spine aesthetics adjustments
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.tick_params(labelsize=8)
                
                # Label row headers on the final column boundary
                if col_idx == 7:
                    ax.text(3.2, np.median(evoked_mean), f"$\\theta={ori:.1f}^\\circ$", 
                            fontsize=9, fontweight='bold', va='center')
                
                # Label column headers on the bottom row axis boundary
                if row_idx == 1:
                    ax.set_xlabel(f"c={contrast:.2f}", fontsize=9, labelpad=5)

        # Scale indicator bar placement mirroring example layout (top-left axis)
        scale_ax = axes[0, 0]
        scale_ax.text(-0.9, 0.12, "0.1 $\\Delta$F/F", fontsize=8, ha='left')
        scale_ax.plot([-0.9, -0.9], [0.0, 0.1], color='black', linewidth=1.5) # Vertical indicator
        scale_ax.plot([-0.9, 0.1], [0.0, 0.0], color='black', linewidth=1.5)  # Horizontal indicator
        scale_ax.text(-0.4, -0.04, "1s", fontsize=8, ha='center')

        plt.tight_layout()
        
        # Save output graphic to disk
        clean_name = os.path.splitext(fname)[0]
        output_svg = os.path.join(output_dir, f"EVOKED_MATRIX_{clean_name}.svg")
        fig.savefig(output_svg, format='svg', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"  ✔️ Matrix condition graphic saved successfully!")
        processed_count += 1
        
    except Exception as e:
        print(f"  [Failure] Error evaluating matrix conditions for {fname}: {str(e)}")
        continue

print(f"\n🚀 Complete! Generated {processed_count} matrix condition layouts within your 'Population_Matrices' desktop directory.")
# %%
