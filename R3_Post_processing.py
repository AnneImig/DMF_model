'''
Plots the measured and modelled values of the model against each other
'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os 
from Statistics import kge, R2
import scipy.stats as stats
import configparser
import sys
import re
'''
Reads the .res file of the Calibration run of PEST and plots different aspects for post processing 
'''
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
meas_dir= os.path.join(script_dir,'measurements')
input_folder = os.path.join(script_dir, "input")
output_folder = os.path.join(script_dir, "output")
plot_folder= os.path.join(script_dir, "plot")

# read all information from configfile
if len(sys.argv) > 1:
    configfn = sys.argv[1]
else:
    configfn = os.path.join(script_dir,"001_Controle_file.conf")

config = configparser.ConfigParser(inline_comment_prefixes="#")
config.read(configfn)


PLOT = config.getboolean("system", "PLOT", fallback=True)
SELFILE= config.get("files", "SELFILE", fallback=True)
FIRST= config.getboolean("kinetics", "FIRST", fallback=True)
BIOMASS= config.getboolean("kinetics", "BIOMASS", fallback=True)
SWAROOP=config.getboolean("dataset", "SWAROOP", fallback=True)
ZHOU=config.getboolean("dataset", "ZHOU", fallback=True)


sel_dir= os.path.join(script_dir,SELFILE)
modelled_df = pd.read_csv(sel_dir, delimiter='\t')    

if ZHOU:
    if FIRST:
        phrq_input_file = os.path.join(input_folder, 'Zhou_first.phrq')
    elif BIOMASS:
        phrq_input_file = os.path.join(input_folder, 'Zhou_biomass.phrq')
elif SWAROOP:
    if FIRST:
        phrq_input_file = os.path.join(input_folder, 'Swaroop_first.phrq')
    elif BIOMASS:
        phrq_input_file = os.path.join(input_folder, 'Swaroop_biomass.phrq')
else:
    raise ValueError("No dataset selected (ZHOU or SWAROOP).")

# Dictionary to store kinetic constants
kinetic_constants = {}

# Mapping kinetics block names to variable names
name_map = {
    "Dmf_Dma": "K_DMF_DMA",
    "Dma_Mma": "K_DMA_MMA",
    "Mma_Ammonia": "K_MMA_NH3"
}

with open(phrq_input_file , "r") as f:
    lines = f.readlines()

current_block = None
for line in lines:
    line = line.strip()

    # Detect kinetics block
    if line in name_map:
        current_block = line
        continue

    # Look for -parms line inside block
    if current_block and line.startswith("-parms"):
        # extract the first number
        match = re.search(r"-parms\s+([0-9.eE+-]+)", line)
        if match:
            value = float(match.group(1))
            kinetic_constants[name_map[current_block]] = value
        current_block = None  # reset after reading

# Example: access values
for k, v in kinetic_constants.items():
    print(f"{k} = {v}")


# Build a single-line string
k_text = "   ".join([f"{k} = {v:.4f}" for k, v in kinetic_constants.items()])


# Read the Measurements from Excel file
# Manually digitized values from the graph based on visual inspection of Zhou et al. (2019) Figure 4

if ZHOU==True: #mg/L  
    data = {
        "Time": [0, 12, 24, 36, 48, 60, 72],
        "DMF": [6000, 5500, 3200, 2200, 1400, 1400, 1400],
        "DMA": [0, 800, 1300, 1600, 1800, 2000, 1800],
        "MMA": [0, 100, 300, 500, 800, 700, 900],
        "pH": [7.0, 7.5, 8.3, 8.8, 9.0, 9.1, 9.2],
    }
# Manually digitized values from the graph based on visual inspection of Swadroop et al. (2009) Figure 2

if SWAROOP==True: #mg/L 
    data = {
        "Time": [0, 3, 6, 9, 12, 15, 18, 21, 24],
        "DMF":     [400, 350, 210, 80, 40, 5, 0, 0, 0],
        "NH3": [  0,  2,  10, 30, 50, 60,  70,  65, 60],
        "DMA":     [  0,  30,  80, 140, 150, 75,  30,  28,  27],
        "TOC":     [220, 210, 200, 170, 100,  90,  80,  75,  70],
    }

df = pd.DataFrame(data)
print(df)

measured = pd.DataFrame(data)

# Define molar masses in g/mol
M = {
    "DMF": 73.09,
    "Dimethylamine": 45.08,
    "Monomethylamine": 31.06, 
    'Ammonia':17
}

# Conversion mg/L -> mol/L

if "DMF" in measured.columns:
    measured["DMF"] = measured["DMF"] / (1000 * M["DMF"])

if "DMA" in measured.columns:
    measured["DMA"] = measured["DMA"] / (1000 * M["Dimethylamine"])

if "MMA" in measured.columns:
    measured["MMA"] = measured["MMA"] / (1000 * M["Monomethylamine"])
if "NH3" in measured.columns:
    measured["NH3"] = measured["NH3"] / (1000 * M["Ammonia"])


'STATISTICS '
#The modelled data have to be interpolated to fit the measured data for the statistic #

# --- Select only modelled rows that match measured times ---
# measured times
def evaluate_species_stats(modelled_df, measured):
    """
    Calculates statistics (R2, RMSE, ME, KGE, Kendall tau, std) 
    for DMF, DMA, MMA, NH3 if they exist in measured.
    """

    # mapping measured column -> modelled_df column index
    col_map = {
        "DMF": 3,
        "DMA": 4,
        "MMA": 5,
        "NH3": 8,
    }

    results = {}
    measured_times = measured["Time"].values

    for species, idx in col_map.items():
        if species in measured.columns:
            # measured values
            meas = measured[species].values

            # select modelled values at measured times
            common_times = modelled_df.index.intersection(measured_times)
            meas_sel = meas[[i for i, t in enumerate(measured_times) if t in common_times]]
            mod_sel  = modelled_df.loc[common_times, modelled_df.columns[idx]].values

            # flatten to 1-D arrays
            meas_sel = np.array(meas_sel).ravel()
            mod_sel  = np.array(mod_sel).ravel()

            # statistics
            r2   = R2(meas_sel, mod_sel)
            rmse = np.sqrt(np.mean((meas_sel - mod_sel) ** 2))
            me   = np.mean(meas_sel - mod_sel)
            kge_val = kge(mod_sel, meas_sel)
            tau, p_c = stats.kendalltau(mod_sel, meas_sel)
            std_val = np.std(mod_sel)

            results[species] = {
                "R2": r2,
                "RMSE": rmse,
                "ME": me,
                "KGE": kge_val,
                "Kendall_tau": tau,
                "p_value": p_c,
                "STD": std_val,
            }

    return results


# --- Usage ---
stats_results = evaluate_species_stats(modelled_df, measured)

for species, metrics in stats_results.items():
    print(f"\n--- {species} ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")



###########################################################################################################################
'''
Plotting measured against modelled
'''
###########################################################################################################################

def plot_modelled(observed_df):
    """
    Create a 2x3 subplot figure.
    Only plot measured scatter points if the column exists.
    Modelled lines are always plotted.
    Statistics (R2, KGE) are displayed in each subplot.
    """

    fig, axs = plt.subplots(2, 3, figsize=(14, 8))

    # calculate statistics
    stats_results = evaluate_species_stats(modelled_df, observed_df)

    # --- DMF ---
    if 'DMF' in observed_df.columns:
        axs[0,0].scatter(observed_df['Time'], observed_df['DMF'], marker='o', color='black', label='meas')
    axs[0,0].plot(modelled_df.index, modelled_df.iloc[:,3], linestyle='-', color='black', label='mod')
    ax1 = axs[0,0].twinx()
    ax1.plot(modelled_df.index, modelled_df.iloc[:,9], linestyle='-', color='lightgrey', label='rate_DMF_DMA')
    ax1.set_ylabel('rate [1/h]')
    axs[0,0].set_title('DMF')
    axs[0,0].set_xlabel('Time')
    axs[0,0].set_ylabel('DMF [mg/L]')

    if 'DMF' in stats_results:
        txt = f"R²={stats_results['DMF']['R2']:.2f}\nKGE={stats_results['DMF']['KGE']:.2f}"
        axs[0,0].text(0.05, 0.95, txt, transform=axs[0,0].transAxes, fontsize=10,
                      verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6))

    # --- DMA ---
    if 'DMA' in observed_df.columns:
        axs[0,1].scatter(observed_df['Time'], observed_df['DMA'], marker='o', color='red', label='meas')
    axs[0,1].plot(modelled_df.index, modelled_df.iloc[:,4], linestyle='-', color='red', label='mod')
    ax2 = axs[0,1].twinx()
    ax2.plot(modelled_df.index, modelled_df.iloc[:,10], linestyle='-', color='lightgrey', label='rate_DMA_MMA')
    ax2.set_ylabel('rate [1/h]')
    axs[0,1].set_title('DMA')
    axs[0,1].set_xlabel('Time')
    axs[0,1].set_ylabel('DMA [mg/L]')
    if 'DMA' in stats_results:
        txt = f"R²={stats_results['DMA']['R2']:.2f}\nKGE={stats_results['DMA']['KGE']:.2f}"
        axs[0,1].text(0.05, 0.95, txt, transform=axs[0,1].transAxes, fontsize=10,
                      verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6))

    # --- MMA ---
    if 'MMA' in observed_df.columns:
        axs[0,2].scatter(observed_df['Time'], observed_df['MMA'], marker='o', color='blue', label='meas')
    axs[0,2].plot(modelled_df.index, modelled_df.iloc[:,5], linestyle='-', color='blue', label='mod')
    ax3 = axs[0,2].twinx()
    ax3.plot(modelled_df.index, modelled_df.iloc[:,11], linestyle='-', color='lightgrey', label='rate_MMA_NH4')
    ax3.set_ylabel('rate [1/h]')
    axs[0,2].set_title('MMA')
    axs[0,2].set_xlabel('Time')
    axs[0,2].set_ylabel('MMA [mg/L]')
    if 'MMA' in stats_results:
        txt = f"R²={stats_results['MMA']['R2']:.2f}\nKGE={stats_results['MMA']['KGE']:.2f}"
        axs[0,2].text(0.05, 0.95, txt, transform=axs[0,2].transAxes, fontsize=10,
                      verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6))

    # --- Ammonia ---
    if 'NH3' in observed_df.columns:
        axs[1,0].scatter(observed_df['Time'], observed_df['NH3'], marker='o', color='blue', label='meas')
    axs[1,0].plot(modelled_df.index, modelled_df.iloc[:,8], linestyle='-', color='lightblue', label='mod')
    axs[1,0].set_title('Ammonia')
    axs[1,0].set_xlabel('Time')
    axs[1,0].set_ylabel('NH3 [mg/L]')
    if 'NH3' in stats_results:
        txt = f"R²={stats_results['NH3']['R2']:.2f}\nKGE={stats_results['NH3']['KGE']:.2f}"
        axs[1,0].text(0.05, 0.95, txt, transform=axs[1,0].transAxes, fontsize=10,
                      verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6))

    # --- Formaldehyde/Formate (model only) ---
    axs[1,1].plot(modelled_df.index, modelled_df.iloc[:,6], linestyle='-', color='green', label='Formaldehyde')
    axs[1,1].plot(modelled_df.index, modelled_df.iloc[:,7], linestyle='-', color='red', label='Formate')
    axs[1,1].set_title('Formaldehyde/Formate')
    axs[1,1].set_xlabel('Time')
    axs[1,1].set_ylabel('[mg/L]')

    # --- pH ---
    if 'pH' in observed_df.columns:
        axs[1,2].scatter(observed_df['Time'], observed_df['pH'], marker='o', color='orange', label='meas')
    axs[1,2].plot(modelled_df.index, modelled_df.iloc[:,1], linestyle='-', color='orange', label='mod')
    axs[1,2].set_title('pH')
    axs[1,2].set_xlabel('Time')
    axs[1,2].set_ylabel('pH')
    axs[1,2].legend()

    # Add template text and layout
    fig.text(0.4, 0.05, k_text, ha="center", va="top", fontsize=10, family="monospace")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Save figure based on dataset and template
    if FIRST and ZHOU:
        fig.suptitle('Zhou et al. (2018) first order kinetics', fontsize=16)
        plt.savefig(os.path.join(plot_folder, 'R3_DMF_First_Zhou.png'), format='png')   
        print('Saved as R3_DMF_First_Zhou.png')     
    elif FIRST and SWAROOP:
        fig.suptitle('Swaroop et al. (2009) first order kinetics', fontsize=16)
        plt.savefig(os.path.join(plot_folder, 'R3_DMF_First_Swaroop.png'), format='png')   
        print('Saved as R3_DMF_First_Swaroop.png')     
    elif BIOMASS and ZHOU:
        fig.suptitle('Zhou et al. (2018) first order with biomass', fontsize=16)
        plt.savefig(os.path.join(plot_folder, 'R3_DMF_TPs_Zhou.png'), format='png')   
        print('Saved as R3_DMF_Biomass_Zhou.png') 
    elif BIOMASS and SWAROOP:
        fig.suptitle('Swaroop et al. (2009) first order with biomass', fontsize=16)
        plt.savefig(os.path.join(plot_folder, 'R3_DMF_TPs_Swaroop.png'), format='png') 
        print('Saved as R3_DMF_Biomass_Swaroop.png') 

def plot_modelled_DMF_species(observed_df):
    """
    Plot only DMF, DMA, and MMA in a single plot window.
    Both measured (scatter) and modelled (line) are shown.
    Statistics (R2, KGE) are displayed in the legend.
    """

    # calculate statistics
    stats_results = evaluate_species_stats(modelled_df, observed_df)

    # --- Create figure ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # --- DMF ---
    if 'DMF' in observed_df.columns:
        ax.scatter(observed_df['Time'], observed_df['DMF'], marker='o', color='black', label='DMF measured')
    ax.plot(modelled_df.index, modelled_df.iloc[:,3], linestyle='-', color='black',
            label=f"DMF modelled (R²={stats_results['DMF']['R2']:.2f}, KGE={stats_results['DMF']['KGE']:.2f})")

    # --- DMA ---
    if 'DMA' in observed_df.columns:
        ax.scatter(observed_df['Time'], observed_df['DMA'], marker='o', color='red', label='DMA measured')
    ax.plot(modelled_df.index, modelled_df.iloc[:,4], linestyle='-', color='red',
            label=f"DMA modelled (R²={stats_results['DMA']['R2']:.2f}, KGE={stats_results['DMA']['KGE']:.2f})")

    # --- MMA ---
    if 'MMA' in observed_df.columns:
        ax.scatter(observed_df['Time'], observed_df['MMA'], marker='o', color='blue', label='MMA measured')
        ax.plot(modelled_df.index, modelled_df.iloc[:,5], linestyle='-', color='blue',
                label=f"MMA modelled (R²={stats_results['MMA']['R2']:.2f}, KGE={stats_results['MMA']['KGE']:.2f})")
    else:
        ax.plot(modelled_df.index, modelled_df.iloc[:,5], linestyle='-', color='blue',
                 label=f"MMA mod")
        
     # --- NH3 ---
    if 'NH3' in observed_df.columns:
        ax.scatter(observed_df['Time'], observed_df['NH3'], marker='o', color='orange', label='NH3 measured')
        ax.plot(modelled_df.index, modelled_df.iloc[:,8], linestyle='-', color='orange',
                label=f"NH3 modelled (R²={stats_results['NH3']['R2']:.2f}, KGE={stats_results['NH3']['KGE']:.2f})")
    else:
        ax.plot(modelled_df.index, modelled_df.iloc[:,8], linestyle='-', color='orange',
                 label=f"NH3 modelled")

    # --- Formatting ---
    
    ax.set_xlabel("Time [h]", fontsize=18)       # Increase x-axis label size
    ax.set_ylabel("Concentration [mol/L]", fontsize=18)  # Increase y-axis label size
    ax.tick_params(axis='both', which='major', labelsize=14)  # Increase tick labels
    ax.legend(fontsize=16, frameon=True, loc='upper center', bbox_to_anchor=(0.5, -0.25),ncol=2)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Add template text below
    #fig.text(0.5, 0.02, k_text, ha="center", va="top", fontsize=9, family="monospace")

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Save figure
    if FIRST and ZHOU:
        ax.set_title("Zhou et al. (2018) dataset", fontsize=18) 
        plt.savefig(os.path.join(plot_folder, 'R3_小_DMF_TPs_Zhou.png'), format='png')  
        print('Saved as R3_小_DMF_TPs_Zhou.png')     
    elif FIRST and SWAROOP:
        ax.set_title("Swaroop et al. (2009) dataset", fontsize=18) 
        plt.savefig(os.path.join(plot_folder, 'R3_小_DMF_TPs_Swaroop.png'), format='png')   
        print('Saved as R3_小_DMF_TPs_Swaroop.png')     
    elif BIOMASS and ZHOU:
        ax.set_title("Zhou et al. (2018) dataset", fontsize=18) 
        plt.savefig(os.path.join(plot_folder, 'R3_小_DMF_TPs_Biomass_Zhou.png'), format='png')   
        print('Saved as R3_小_DMF_TPs_Biomass_Zhou.png') 
    elif BIOMASS and SWAROOP:
        ax.set_title("Swaroop et al. (2009) dataset", fontsize=18) 
        plt.savefig(os.path.join(plot_folder, 'R3_小_DMF_TPs_Biomass_Swaroop.png'), format='png') 
        print('Saved asR3_小_DMF_TPs_Biomass_Swaroop.png') 



# --- Call plot ---
if PLOT:
    plot_modelled_DMF_species(measured)
    plot_modelled(measured)
