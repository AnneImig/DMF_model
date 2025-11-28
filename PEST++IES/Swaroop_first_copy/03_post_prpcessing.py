import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import gaussian_kde
import re
import seaborn as sns
import sys
import configparser
from scipy.stats import sem, t
# ------------ USER SETTINGS ------------
case = "control"         # your case name
script_dir = os.path.dirname(os.path.abspath(__file__))

iteration = 3            # final iteration from your IES output
# ---------------------------------------
# read all information from configfile
if len(sys.argv) > 1:
    configfn = sys.argv[1]
else:
    configfn = os.path.join(script_dir,"001_Controle_file.conf")

config = configparser.ConfigParser(inline_comment_prefixes="#")
config.read(configfn)

# Flags for program flow
PLOT = config.getboolean("system", "PLOT", fallback=True)
SELFILE= config.get("files", "SELFILE", fallback=True)
FIRST= config.getboolean("kinetics", "FIRST", fallback=True)
BIOMASS= config.getboolean("kinetics", "BIOMASS", fallback=True)
SWAROOP=config.getboolean("dataset", "SWAROOP", fallback=True)
ZHOU=config.getboolean("dataset", "ZHOU", fallback=True)
# Load phi file
phi_file = os.path.join(script_dir, f"{case}.phi.actual.csv")
phi_df = pd.read_csv(phi_file)

par_folder = script_dir   # folder with your par files

# Load phi file
phi = pd.read_csv(phi_file)

# List all par.csv files
par_files = sorted(
    [f for f in os.listdir(par_folder) if re.match(r"control\.\d+\.par\.csv", f)],
    key=lambda x: int(re.findall(r"\d+", x)[0])
)

# Dictionary to store one DF per ensemble
ensembles = {}

for par_file in par_files:
    # extract ensemble/run number
    run_number = int(re.findall(r"\d+", par_file)[0])

    # Load par file
    par_path = os.path.join(par_folder, par_file)
    par = pd.read_csv(par_path)

    # clean real_name column
    par = par[par["real_name"].astype(str).str.match(r"^\d+$", na=False)]
    par["real_name"] = par["real_name"].astype(int)

    # extract matching phi row
    phi_row = phi.iloc[run_number]
    phi_values = phi_row.filter(regex="^[0-9]+$")
    phi_df = phi_values.rename_axis("real_name").reset_index(name="phi")
    phi_df["real_name"] = phi_df["real_name"].astype(int)

    # merge
    merged = par.merge(phi_df, on="real_name", how="left")
    ensembles[run_number] = merged    # <–– store separately

print("Created ensembles:", list(ensembles.keys()))



# ----------------- Histogram of phi per ensemble -----------------

fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
axes = axes.flatten()  # flatten to 1D list for easy looping

for i in range(4):
    # Plot histogram with counts on y-axis
    sns.histplot(
        ensembles[i]["phi"].dropna(),
        bins=20,       # number of bins
        kde=False,     # no density curve
        ax=axes[i],
        color="skyblue",
        edgecolor="black"
    )
    
    axes[i].set_title(f"Ensemble {i} – phi distribution")
    axes[i].set_ylabel("Count")  # show number of realizations

plt.tight_layout()
plt.savefig(os.path.join(script_dir, "R3_post_processing/R3_phi_histograms_ensembles.png"), dpi=300)

# ----------------- KDE plots per parameter -----------------
parameters = ["dmfdma", "dmamma", "mmanh3"]
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for j, param in enumerate(parameters):
    for i in ensembles.keys():
        sns.kdeplot(ensembles[i][param].dropna(), ax=axes[j], label=f"Ensemble {i}")
    axes[j].set_title(f"{param} – KDE across ensembles")
    axes[j].legend()

plt.tight_layout()
plt.savefig(os.path.join(script_dir, "R3_post_processing/R3_parameter_density_ensembles.png"), dpi=300)



# ---------------- CONVERT MEASURED VALUES TO MOL/L -----------------
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
    "DMA": 45.08,
    "MMA": 31.06, 
    "NH3": 17
}

# Conversion mg/L -> mol/L
for compound, mass in M.items():
    if compound in measured.columns:
        measured[compound] = measured[compound] / (1000 * mass)  # mg/L -> g/L -> mol/L
# ---------------- TIME VECTORS ------------------
if SWAROOP:
    time = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24])
elif ZHOU:
    time = np.array([0, 12, 24, 36, 48, 60, 72])
# ------------------------------------------------


# --------- READ ALL OBS FILES FOR ENSEMBLES -----
obs_files = sorted(
    [f for f in os.listdir(script_dir) if re.match(rf"{case}\.\d+\.obs\.csv", f)],
    key=lambda x: int(re.findall(r"\d+", x)[0])
)

ensembles_obs = {}
for f in obs_files:
    run = int(re.findall(r"\d+", f)[0])
    path = os.path.join(script_dir, f)
    ensembles_obs[run] = pd.read_csv(path)

print("Loaded observation files:", list(ensembles_obs.keys()))


# ----------- IDENTIFY DMF / DMA / NH3 COLUMNS -------------
example_df = ensembles_obs[list(ensembles_obs.keys())[0]]

dmf_cols = [c for c in example_df.columns if c.lower().startswith("dmf_")]
dma_cols = [c for c in example_df.columns if c.lower().startswith("dma_")]
nh3_cols = [c for c in example_df.columns if c.lower().startswith("nh3_")]

print("DMF columns:", dmf_cols)
print("DMA columns:", dma_cols)
print("NH3 columns:", nh3_cols)


# ----------- PLOT SPAGHETTI ACROSS ENSEMBLES --------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

titles = ["DMF", "DMA", "NH₃"]
all_col_sets = [dmf_cols, dma_cols, nh3_cols]
for ax, col_set, title in zip(axes, all_col_sets, titles):

    # Plot all ensemble realizations (spaghetti)
    for run, df in ensembles_obs.items():
        for _, row in df.iterrows():
            ax.plot(time, row[col_set].values, linewidth=0.7, alpha=0.25, color='blue')

    # Plot measured values
    measured_col = title if title != "NH₃" else "NH3"  # match your measured DataFrame keys
    if measured_col in measured.columns:
        ax.scatter(measured["Time"], measured[measured_col], color='red', s=50, zorder=5, label='Measured')

    ax.set_title(f"{title} – Spaghetti Plot (all ensembles)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Concentration (mol/L)")
    ax.grid(True, alpha=0.2)
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(script_dir, "R3_post_processing/R3_spaghetti_plot_with_measured.png"), dpi=300)


# ---------------- TIME VECTORS ------------------
if SWAROOP:
    time = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24])
elif ZHOU:
    time = np.array([0, 12, 24, 36, 48, 60, 72])
# ------------------------------------------------

# --------- READ ALL OBS FILES FOR ENSEMBLES -----
obs_files = sorted(
    [f for f in os.listdir(script_dir) if re.match(rf"{case}\.\d+\.obs\.csv", f)],
    key=lambda x: int(re.findall(r"\d+", x)[0])
)

ensembles_obs = {}
for f in obs_files:
    run = int(re.findall(r"\d+", f)[0])
    path = os.path.join(script_dir, f)
    ensembles_obs[run] = pd.read_csv(path)

print("Loaded observation files:", list(ensembles_obs.keys()))

# ----------- IDENTIFY DMF / DMA / NH3 COLUMNS -------------
example_df = ensembles_obs[list(ensembles_obs.keys())[0]]

dmf_cols = [c for c in example_df.columns if c.lower().startswith("dmf_")]
dma_cols = [c for c in example_df.columns if c.lower().startswith("dma_")]
nh3_cols = [c for c in example_df.columns if c.lower().startswith("nh3_")]

print("DMF columns:", dmf_cols)
print("DMA columns:", dma_cols)
print("NH3 columns:", nh3_cols)

# ------------------------------------------------------------
#     CREATE 4×3 SUBPLOTS: EACH ROW = ENSEMBLE, EACH COL = VARIABLE
# ------------------------------------------------------------
runs = list(ensembles_obs.keys())
rows = len(runs)           # one row per ensemble
cols = 3                   # DMF / DMA / NH3

fig, axes = plt.subplots(rows, cols, figsize=(18, 4 * rows))
titles = ["DMF", "DMA", "NH₃"]
col_sets = [dmf_cols, dma_cols, nh3_cols]

# ensure axes is 2D
if rows == 1:
    axes = np.array([axes])

for r, run in enumerate(runs):
    df = ensembles_obs[run]

    for c, (title, col_set) in enumerate(zip(titles, col_sets)):
        ax = axes[r, c]

        # spaghetti plot of this variable for this ensemble
        for _, row in df.iterrows():
            ax.plot(time, row[col_set].values, alpha=0.25, linewidth=0.6)

        ax.set_title(f"{title} – Ensemble {run}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Conc. (mol/L)")
        ax.grid(True, alpha=0.2)

plt.tight_layout()
out_file = os.path.join(script_dir, "R3_post_processing/R3_by_ensemble_and_variable.png")
plt.savefig(out_file, dpi=300)

print(f"Saved figure: {out_file}")


from scipy.stats import sem, t
import numpy as np

def compute_confidence_interval(data, confidence=0.99):
    """
    Compute mean and confidence interval across ensembles.
    
    Parameters
    ----------
    data : array-like, shape (n_realizations, n_timepoints)
        Numeric array of ensemble results.
    confidence : float
        Confidence level (default 0.95).
        
    Returns
    -------
    mean : ndarray
        Mean across ensembles at each time point.
    lower : ndarray
        Lower bound of the confidence interval.
    upper : ndarray
        Upper bound of the confidence interval.
    """
    data_array = np.array(data, dtype=float)  # ensure numeric
    mean = np.mean(data_array, axis=0)
    n = data_array.shape[0]
    se = sem(data_array, axis=0)
    h = se * t.ppf((1 + confidence) / 2., n - 1)
    return mean, mean - h, mean + h


# ---------------- PLOTTING -----------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
titles = ["DMF", "DMA", "NH₃"]
all_col_sets = [dmf_cols, dma_cols, nh3_cols]

for ax, col_set, title in zip(axes, all_col_sets, titles):

    # Collect all ensemble realizations into numeric 2D array
    ensemble_data = []
    for run, df in ensembles_obs.items():
        for _, row in df.iterrows():
            values = np.array(row[col_set].values, dtype=float).flatten()
            ensemble_data.append(values)
    ensemble_data = np.vstack(ensemble_data)

    # Compute mean and 95% CI
    mean, lower, upper = compute_confidence_interval(ensemble_data)

    # Plot mean + CI
    ax.plot(time, mean, color='blue', linewidth=2, label='Mean')
    ax.fill_between(time, lower, upper, color='blue', alpha=0.3, label='95% CI')

    # Plot measured values (converted to mol/L)
    measured_col = title if title != "NH₃" else "NH3"
    if measured_col in measured.columns:
        ax.scatter(measured["Time"], measured[measured_col], color='red', s=50, zorder=5, label='Measured')

    ax.set_title(f"{title} – Mean + 95% CI")
    ax.set_xlabel("Time")
    ax.set_ylabel("Concentration (mol/L)")
    ax.grid(True, alpha=0.2)
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(script_dir, "R3_post_processing/R3_mean_CI_with_measured_mol.png"), dpi=300)

# ---------------- UNIFIED PLOT -----------------
fig, ax = plt.subplots(figsize=(10, 6))

titles = ["DMF", "DMA", "NH₃"]
all_col_sets = [dmf_cols, dma_cols, nh3_cols]

# Your requested colors
colors = ["black", "red", "orange"]

# Column names in measured dataset
measured_names = ["DMF", "DMA", "NH3"]

for title, col_set, color, meas_name in zip(titles, all_col_sets, colors, measured_names):

    # Collect ensemble realizations → numeric 2D array
    ensemble_data = []
    for run, df in ensembles_obs.items():
        for _, row in df.iterrows():
            values = np.array(row[col_set].values, dtype=float).flatten()
            ensemble_data.append(values)
    ensemble_data = np.vstack(ensemble_data)

    # Compute mean and 95% CI
    mean, lower, upper = compute_confidence_interval(ensemble_data)

    # Plot mean + CI
    ax.plot(time, mean, color=color, linewidth=2, label=f"{title} mean")
    ax.fill_between(time, lower, upper, color=color, alpha=0.25, label=f"{title} 95% CI")

    # Plot measured values if available
    if meas_name in measured.columns:
        ax.scatter(
            measured["Time"],
            measured[meas_name],
            color=color,
            s=40,
            zorder=5,
            label=f"{title} measured"
        )

# Final formatting
#ax.set_title("DMF, DMA, NH₃ – Mean + 95% CI with Measured Data")
ax.set_xlabel("Time", fontsize=14)
ax.set_ylabel("Concentration (mol/L)", fontsize=14)
ax.grid(True, alpha=0.2)

# After plotting everything
handles, labels = ax.get_legend_handles_labels()

# Reorder so that NH3 labels are last
# NH3 labels contain "NH3"
nh3_indices = [i for i, lbl in enumerate(labels) if "NH3" in lbl]
other_indices = [i for i in range(len(labels)) if i not in nh3_indices]
new_order = other_indices + nh3_indices

ax.legend(
    [handles[i] for i in new_order],
    [labels[i] for i in new_order],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.20),
    ncol=3,
    fontsize=16,
    framealpha=0.9,
    facecolor="white"
)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, "R3_post_processing/R3_mean_CI_all_species.png"), dpi=300)
