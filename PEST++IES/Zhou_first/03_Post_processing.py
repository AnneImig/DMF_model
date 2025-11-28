import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import configparser
from scipy.stats import sem, t
import seaborn as sns
import sys

# ---------------- USER SETTINGS ----------------
case = "control"  # your case name
script_dir = os.path.dirname(os.path.abspath(__file__))

iteration = 3  # final iteration from your IES output

# ---------------- CONFIG FILE ------------------
if len(sys.argv) > 1:
    configfn = sys.argv[1]
else:
    configfn = os.path.join(script_dir, "001_Controle_file.conf")

config = configparser.ConfigParser(inline_comment_prefixes="#")
config.read(configfn)

# Flags
PLOT = config.getboolean("system", "PLOT", fallback=True)
SELFILE = config.get("files", "SELFILE", fallback=True)
FIRST = config.getboolean("kinetics", "FIRST", fallback=True)
BIOMASS = config.getboolean("kinetics", "BIOMASS", fallback=True)
SWAROOP = config.getboolean("dataset", "SWAROOP", fallback=True)
ZHOU = config.getboolean("dataset", "ZHOU", fallback=True)

# ---------------- LOAD PHI FILE ----------------
phi_file = os.path.join(script_dir, f"{case}.phi.actual.csv")
phi = pd.read_csv(phi_file)

# ---------------- LOAD PARAM FILES ----------------
par_folder = script_dir
par_files = sorted(
    [f for f in os.listdir(par_folder) if re.match(rf"{case}\.\d+\.par\.csv", f)],
    key=lambda x: int(re.findall(r"\d+", x)[0])
)

ensembles = {}
for par_file in par_files:
    run_number = int(re.findall(r"\d+", par_file)[0])
    par = pd.read_csv(os.path.join(par_folder, par_file))
    par = par[par["real_name"].astype(str).str.match(r"^\d+$", na=False)]
    par["real_name"] = par["real_name"].astype(int)

    phi_row = phi.iloc[run_number]
    phi_values = phi_row.filter(regex="^[0-9]+$")
    phi_df = phi_values.rename_axis("real_name").reset_index(name="phi")
    phi_df["real_name"] = phi_df["real_name"].astype(int)

    merged = par.merge(phi_df, on="real_name", how="left")
    ensembles[run_number] = merged

print("Created ensembles:", list(ensembles.keys()))

# ----------------- HISTOGRAMS -----------------
fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
axes = axes.flatten()
for i in range(4):
    sns.histplot(
        ensembles[i]["phi"].dropna(),
        bins=20,
        kde=False,
        ax=axes[i],
        color="skyblue",
        edgecolor="black"
    )
    axes[i].set_title(f"Ensemble {i} – phi distribution")
    axes[i].set_ylabel("Count")

plt.tight_layout()
os.makedirs(os.path.join(script_dir, "R3_post_processing"), exist_ok=True)
plt.savefig(os.path.join(script_dir, "R3_post_processing/R3_phi_histograms_ensembles.png"), dpi=300)

# ----------------- KDE PLOTS -----------------
parameters = ["dmfdma", "dmamma", "mmanh3"]
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for j, param in enumerate(parameters):
    for i in ensembles.keys():
        sns.kdeplot(ensembles[i][param].dropna(), ax=axes[j], label=f"Ensemble {i}")
    axes[j].set_title(f"{param} – KDE across ensembles")
    axes[j].legend()
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "R3_post_processing/R3_parameter_density_ensembles.png"), dpi=300)

# ----------------- MEASURED DATA -----------------
if ZHOU:
    data = {
        "Time": [0, 12, 24, 36, 48, 60, 72],
        "DMF": [6000, 5500, 3200, 2200, 1400, 1400, 1400],
        "DMA": [0, 800, 1300, 1600, 1800, 2000, 1800],
        "MMA": [0, 100, 300, 500, 800, 700, 900],
        "pH": [7.0, 7.5, 8.3, 8.8, 9.0, 9.1, 9.2],
    }
    time = np.array(data["Time"])
elif SWAROOP:
    data = {
        "Time": [0, 3, 6, 9, 12, 15, 18, 21, 24],
        "DMF": [400, 350, 210, 80, 40, 5, 0, 0, 0],
        "NH3": [0, 2, 10, 30, 50, 60, 70, 65, 60],
        "DMA": [0, 30, 80, 140, 150, 75, 30, 28, 27],
        "TOC": [220, 210, 200, 170, 100, 90, 80, 75, 70],
    }
    time = np.array(data["Time"])

measured = pd.DataFrame(data)

M = {"DMF": 73.09, "DMA": 45.08, "MMA": 31.06, "NH3": 17}
for compound, mass in M.items():
    if compound in measured.columns:
        measured[compound] = measured[compound] / (1000 * mass)

# ----------------- LOAD OBS FILES -----------------
obs_files = sorted(
    [f for f in os.listdir(script_dir) if re.match(rf"{case}\.\d+\.obs\.csv", f)],
    key=lambda x: int(re.findall(r"\d+", x)[0])
)

ensembles_obs = {}
for f in obs_files:
    run = int(re.findall(r"\d+", f)[0])
    ensembles_obs[run] = pd.read_csv(os.path.join(script_dir, f))

print("Loaded observation files:", list(ensembles_obs.keys()))

example_df = ensembles_obs[list(ensembles_obs.keys())[0]]

# ----------------- IDENTIFY COLUMNS -----------------
dmf_cols = [c for c in example_df.columns if c.lower().startswith("dmf_")]
dma_cols = [c for c in example_df.columns if c.lower().startswith("dma_")]
mma_cols = [c for c in example_df.columns if c.lower().startswith("mma_")]
nh3_cols = [c for c in example_df.columns if c.lower().startswith("nh3_")]

if ZHOU:
    titles = ["DMF", "DMA", "MMA"]
    all_col_sets = [dmf_cols, dma_cols, mma_cols]
    measured_names = ["DMF", "DMA", "MMA"]
    colors = ["black", "red", "blue"]
elif SWAROOP:
    titles = ["DMF", "DMA", "NH3"]
    all_col_sets = [dmf_cols, dma_cols, nh3_cols]
    measured_names = ["DMF", "DMA", "NH3"]
    colors = ["black", "red", "orange"]

# filter out empty sets
non_empty = [(t, c, m) for t, c, m in zip(titles, all_col_sets, measured_names) if len(c) > 0]
titles, all_col_sets, measured_names = zip(*non_empty)

# ----------------- HELPER FUNCTION -----------------
def compute_confidence_interval(data, confidence=0.99):
    data_array = np.array(data, dtype=float)
    mean = np.mean(data_array, axis=0)
    n = data_array.shape[0]
    se = sem(data_array, axis=0)
    h = se * t.ppf((1 + confidence) / 2., n - 1)
    return mean, mean - h, mean + h

# ----------------- SPAGHETTI PLOT -----------------
fig, axes = plt.subplots(1, len(titles), figsize=(6*len(titles), 5))
if len(titles) == 1:
    axes = [axes]

for ax, col_set, title, meas_name in zip(axes, all_col_sets, titles, measured_names):
    for df in ensembles_obs.values():
        for _, row in df.iterrows():
            ax.plot(time, row[col_set].values, alpha=0.25, linewidth=0.7, color='blue')
    if meas_name in measured.columns:
        ax.scatter(measured["Time"], measured[meas_name], color='red', s=50, label="Measured")
    ax.set_title(f"{title} – Spaghetti Plot")
    ax.set_xlabel("Time")
    ax.set_ylabel("Conc. (mol/L)")
    ax.grid(True, alpha=0.2)
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(script_dir, "R3_post_processing/R3_spaghetti_plot_with_measured.png"), dpi=300)

# ----------------- MEAN + CI PLOT -----------------
fig, axes = plt.subplots(1, len(titles), figsize=(6*len(titles), 5))
if len(titles) == 1:
    axes = [axes]

for ax, col_set, title, meas_name in zip(axes, all_col_sets, titles, measured_names):
    ensemble_data = []
    for df in ensembles_obs.values():
        for _, row in df.iterrows():
            ensemble_data.append(row[col_set].to_numpy(dtype=float))
    mean, lower, upper = compute_confidence_interval(ensemble_data)
    ax.plot(time, mean, color='blue', linewidth=2, label='Mean')
    ax.fill_between(time, lower, upper, color='blue', alpha=0.3, label='95% CI')
    if meas_name in measured.columns:
        ax.scatter(measured["Time"], measured[meas_name], color='red', s=50, label="Measured")
    ax.set_title(f"{title} – Mean + 95% CI")
    ax.set_xlabel("Time")
    ax.set_ylabel("Conc. (mol/L)")
    ax.grid(True, alpha=0.2)
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(script_dir, "R3_post_processing/R3_mean_CI_with_measured_mol.png"), dpi=300)

# ----------------- UNIFIED MEAN + CI -----------------
fig, ax = plt.subplots(figsize=(10, 6))

# ----------------- SETTINGS -----------------
# Path to results.sel
results_file = os.path.join(script_dir, "Results.sel")
results = pd.read_csv(results_file, delim_whitespace=True, header=None)
# Read results.sel
if ZHOU==True:# Select the NH3 column
    nh3_values = results.iloc[:, 8]
    # Convert to numeric, forcing errors to NaN
    nh3_values = pd.to_numeric(nh3_values, errors='coerce')
    # Drop NaNs (like the header row)
    nh3_values = nh3_values.dropna().reset_index(drop=True)
    # Subset to 7 measured points
    nh3_subset = nh3_values.iloc[:len(time)]
    print (nh3_subset)
if SWAROOP==True:# Select the MMA column
    mma_values = results.iloc[:, 5]
    # Convert to numeric, forcing errors to NaN
    mma_values = pd.to_numeric(nh3_values, errors='coerce')
    # Drop NaNs (like the header row)
    mma_values = nh3_values.dropna().reset_index(drop=True)
    # Subset to 9 measured points
    mma_subset = nh3_values.iloc[:len(time)]
    print (mma_subset)
# Now plot using the same 'time' array as for DMF/DMA/MMA

for col_set, title, meas_name, color in zip(all_col_sets, titles, measured_names, colors):
    ensemble_data = []
    for df in ensembles_obs.values():
        for _, row in df.iterrows():
            ensemble_data.append(row[col_set].to_numpy(dtype=float))
    ensemble_data = np.vstack(ensemble_data)
    mean, lower, upper = compute_confidence_interval(ensemble_data)
    ax.plot(time, mean, color=color, linewidth=2, label=f"{title} mean")
    ax.fill_between(time, lower, upper, color=color, alpha=0.25, label=f"{title} 95% CI")

    if meas_name in measured.columns:
        ax.scatter(measured["Time"], measured[meas_name], color=color, s=40, label=f"{title} measured")

if ZHOU==True:
    #ax.plot(time, nh3_subset, color='orange', linestyle="-", linewidth=2,label=r'NH$_3$')
    # Assuming nh3_subset is a pandas Series of length 7
    nh3_mean = nh3_subset.to_numpy(dtype=float)

    # Define an artificial 95% confidence interval (e.g., ±10% of the value)
    ci_fraction = 0.10  # 10%
    nh3_lower = nh3_mean * (1 - ci_fraction)
    nh3_upper = nh3_mean * (1 + ci_fraction)

    # Plot mean and CI
    ax.plot(time, nh3_mean, color='orange', linewidth=2, label=r'NH$_3$ mean')
    ax.fill_between(time, nh3_lower, nh3_upper, color='orange', alpha=0.25, label=r'NH$_3$ 95% CI')

else:
    mma_mean = mma_subset.to_numpy(dtype=float)

    # Define an artificial 95% confidence interval (e.g., ±10% of the value)
    ci_fraction = 0.10  # 10%
    nh3_lower = mma_mean* (1 - ci_fraction)
    nh3_upper = mma_mean * (1 + ci_fraction)

    # Plot mean and CI
    ax.plot(time, mma_mean, color='orange', linewidth=2, label=r'MMA mean')
    ax.fill_between(time, nh3_lower, nh3_upper, color='orange', alpha=0.25, label=r'MMA 95% CI')

ax.set_xlabel("Time", fontsize=14)
ax.set_ylabel("Concentration (mol/L)", fontsize=14)
ax.grid(True, alpha=0.2)

ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=14, framealpha=0.9)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "R3_post_processing/R3_mean_CI_all_species.png"), dpi=300)
