# %%
import numpy as np
import sys, os

# physion
sys.path += ['../../physion/src'] # add src code directory for physion

# plot
from physion.utils import plot_tools as pt
pt.set_style('manuscript')

import pandas as pd

from pathlib import Path

#%%
base = os.path.join(Path("E:/"), 'DATA', 'In_Vivo_experiments','opto', 'BOT')
# LED 455 without animal
#path = os.path.join(base, '2026_10_16','BrightnessOverTime-10162026-1501-041','BrightnessOverTime-10162026-1501-041_Cycle00001-botData.csv' )
#start_s = [349,1270,2120,2987,3871,4735,5671,6609,7883,8745]
# same but with an animal
#path = os.path.join(base, '2026_10_16','BrightnessOverTime-10162026-1501-042','BrightnessOverTime-10162026-1501-042_Cycle00001-botData.csv' )
#start_s = [445,1431,2352,3393,4288,5200,6139,7059,7951,8877] 

# Tested more 1200, 1400, 1600, 1800, 20000
#LED 450 without animal
#path = os.path.join(base, '2026_10_16','BrightnessOverTime-10162026-1501-044','BrightnessOverTime-10162026-1501-044_Cycle00001-botData.csv' )
#start_s = [548,1466,2628,3650,4567,5439,6365,7331,8313,9319,10247,11232,12100,13381,14286]
# same but with animal
path = os.path.join(base, '2026_10_16','BrightnessOverTime-10162026-1501-043','BrightnessOverTime-10162026-1501-043_Cycle00001-botData.csv' )
start_s = [456,1399,2443,3342,4244,5134,6036,6936,7935,8802,10509,11449, 12450, 14367,15213]

# LED 590 without animal
#path = os.path.join(base, '2026_10_16','BrightnessOverTime-10162026-1501-045','BrightnessOverTime-10162026-1501-045_Cycle00001-botData.csv' )
#start_s = [1290, 2380, 3550,5028,6970,6970,8246,9244,10765,11654]

df = pd.read_csv(path)

#%% #plots

#%% RAW

fig, ax = pt.figure(ax_scale=(4,2))
ax.plot(df['Timestamp'] , df["Region 1"], color='grey')
ax.plot(df['Timestamp'] , df["Region 2"], color = 'royalblue')
ax.plot(df['Timestamp'] , df["Region 3"], color = 'crimson')


fig, ax = pt.figure(ax_scale=(4,2))
ax.plot(df['Timestamp'] , df["Region 1"], color='grey')
ax.plot(df['Timestamp'] , df["Region 2"], color = 'royalblue')
#ax.plot(df['Timestamp'] , df["Region 3"], color = 'b')

#pt.set_plot(ax, ylim=[50,80])
'''
# #Remove baseline
df_baseline = df
n_samples = 300
baseline_reg1 = np.mean(df["Region 1"][0:n_samples])
df_baseline["Region 1"] = df["Region 1"] - baseline_reg1
baseline_reg2 = np.mean(df["Region 2"][0:n_samples])
df_baseline["Region 2"] = df["Region 2"] - baseline_reg2
baseline_reg3 = np.mean(df["Region 3"][0:n_samples])
df_baseline["Region 3"] = df["Region 3"] - baseline_reg3

fig, ax = pt.figure(ax_scale=(4,2))
ax.plot(df['Timestamp'] , df["Region 3"], color = 'crimson')
ax.plot(df_baseline['Timestamp'] , df_baseline["Region 1"], color='grey')
ax.plot(df_baseline['Timestamp'] , df_baseline["Region 2"], color = 'royalblue')

fig, ax = pt.figure(ax_scale=(4,2))
ax.plot(df_baseline['Timestamp'] , df_baseline["Region 1"], color='grey')
ax.plot(df_baseline['Timestamp'] , df_baseline["Region 2"], color = 'royalblue')
'''
#%%
def func(start, remove_bsl=True):

    end = start + 255
    
    start_bsl = start - 100
    end_bsl = start - 1

    step_up = np.mean(df["Region 3"][start:end])
    step_middle = np.mean(df["Region 2"][start:end])
    step_bottom = np.mean(df["Region 1"][start:end])
    bsl_top = np.mean(df["Region 3"][start_bsl:end_bsl])
    bsl_middle = np.mean(df["Region 2"][start_bsl:end_bsl])
    bsl_bottom = np.mean(df["Region 1"][start_bsl:end_bsl])

    if remove_bsl: 
        y_top.append(step_up-bsl_top)
        y_middle.append(step_middle-bsl_middle)
        y_bottom.append(step_bottom-bsl_bottom)
    else: 
        y_top.append(step_up)
        y_middle.append(step_middle)
        y_bottom.append(step_bottom)
    

    y_bsl_top.append(bsl_top)
    y_bsl_middle.append(bsl_middle)
    y_bsl_bottom.append(bsl_bottom)
    fig, ax = pt.figure(ax_scale=(4,2))
    ax.plot(df['Timestamp'][start:end] , df["Region 3"][start:end], color = 'b')
    print("hey")



#%% FILL LISTS OF STEPS
y_top = []
y_middle = []
y_bottom = []
y_bsl_top = []
y_bsl_middle = []
y_bsl_bottom = []
for start in start_s:
    func(start, remove_bsl = True)
#%%
fig, ax = pt.figure(ax_scale=(2,2))
x = np.arange(100, 1100, 100)
x = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000])
width = 40

ax.bar(x - width, y_top, width=width, label='Top region', color='crimson', alpha=0.9)
ax.bar(x , y_middle, width=width, label='Middle region', color='royalblue', alpha=0.9)
ax.bar(x + width, y_bottom, width=width, label='Bottom region', color='grey', alpha=0.9)

pt.set_plot(ax=ax, 
            xlabel = 'current sent to LED (mA)',
            ylabel = 'mean fluo - bsl', 
            xticks=x,
            xticks_rotation=90)
ax.legend()

fig, ax = pt.figure(ax_scale=(2,2))

#ax.bar(x - width, y_top, width=width, label='Top region', color='crimson', alpha=0.9)
ax.bar(x , y_middle, width=width, label='Middle region', color='royalblue', alpha=0.9)
ax.bar(x + width, y_bottom, width=width, label='Bottom region', color='grey', alpha=0.9)

pt.set_plot(ax=ax, xlabel = 'current sent to LED (mA)', 
            ylabel = 'mean fluo - bsl', 
            xticks=x,
            xticks_rotation=90)
ax.legend()

#evolution_baseline
fig, ax = pt.figure(ax_scale=(3,2))
x = np.arange(100, 1100, 100)
x = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000])
width = 20

ax.bar(x - width, y_bsl_top, width=width, label='Top region', color='blue')
ax.bar(x , y_bsl_middle, width=width, label='Middle region', color='red')
ax.bar(x + width, y_bsl_bottom, width=width, label='Bottom region', color='grey')

pt.set_plot(ax=ax, xlabel = 'current sent to LED (mA)',
            ylabel = 'baseline',
            xticks=x,
            xticks_rotation=90)
ax.legend()



#%%
fig, ax = pt.figure(ax_scale=(4,2))
start = 50
end = start+350
ax.plot(df['Timestamp'][start:end] , df["Region 3"][start:end], color = 'b')
