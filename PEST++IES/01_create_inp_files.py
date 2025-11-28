"""code creates PEST files (template of 001_Controle_file.conf, Control, instruction and template file)"""

import subprocess
import os
import sys
import shutil
import configparser
import tempfile as tf
from string import Template
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast


script_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
input_folder = os.path.join(grandparent_dir, "input")
print(grandparent_dir)
phrq_file = [file for file in os.listdir(input_folder) if file.endswith('.phrq')]


out_file = [file.replace('.phrq', '.out') for file in phrq_file]
tpl_file = [file.replace('.phrq', '.tpl') for file in phrq_file]#[file for file in os.listdir(script_dir) if file.endswith('.tpl')]



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

if os.name == "nt":
    PHRQDB = ("c:/phreeqc/database/phreeqc.dat     "  )
    SCR = ("    scr.out")
    PHRQCMD = ("C:/phreeqc/bin/Release/phreeqc.exe   ")

else:
    PHRQDB = "/Users/anneimig/Documents/02_SUSTech/Phreeqc/phreeqc-3.5.0-14000/database/phreeqc.dat"
    PHRQCMD = "/Users/anneimig/Documents/02_SUSTech/Phreeqc/phreeqc-3.5.0-14000/bin/phreeqc"
    SCR = ("    scr.out")


##################################
    'Create Instruction file '
##################################
if ZHOU:
    if FIRST:
        phrq_input_file = os.path.join(input_folder, 'Zhou_first.phrq')
        print(phrq_input_file)
    elif BIOMASS:
        phrq_input_file = os.path.join(input_folder, 'Zhou_biomass.phrq')
        print(phrq_input_file)
elif SWAROOP:
    if FIRST:
        phrq_input_file = os.path.join(input_folder, 'Swaroop_first.phrq')
        print(phrq_input_file)
    elif BIOMASS:
        phrq_input_file = os.path.join(input_folder, 'Swaroop_biomass.phrq')
        print(phrq_input_file)
else:
    raise ValueError("No dataset selected (ZHOU or SWAROOP).")

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



# Original times +2
times = measured["Time"].values + 2  # 9 elements

# Compute step differences
time_steps = np.diff(times)           # 8 elements
time_steps = np.insert(time_steps, 0, times[0])  # insert the first value at the beginning -> 9 elements

def create_ins_file(name, line):
    obs_labels = ''
    for idx, step in enumerate(time_steps):
        obs_label = f'l{step} [{name}_{idx}]{line}\n'
        obs_labels += obs_label

    c_ins_path = os.path.join(script_dir, f'{name}.ins')    
    with open(c_ins_path, 'w') as c_ins:
        c_ins.write('pif @\n' + obs_labels)
    print(f"1. {name} instruction file successfully created. Saved under {c_ins_path}")

# Call function
create_ins_file('DMA', '3:12')

create_ins_file( 'DMF','3:12')#'57:66'
if ZHOU==True:
    create_ins_file( 'MMA','3:12')#'93:102',
else:
    create_ins_file( 'NH3','3:12')#,'146:155'

##################################
'Create Template file'
##################################

def modify_phrq_and_save():
    """
    phrq_input_file: full path to the PHRQ file you want to use
    """
    print("Using input PHRQ file:", phrq_input_file)

    # Read the file
    with open(phrq_input_file, 'r') as file:
        lines = file.readlines()

    # Lines to replace
    replacements = {
        48: '     -parms   $DmfDma$  \n',
        53: '     -parms   $DmaMma$  \n',
        59: '     -parms   $MmaNH3$  \n',
        92: '  -file output/DMF.sel\n',
        97: '  -file output/MMA.sel\n',
        102: '  -file output/DMA.sel\n',
        107: '  -file output/NH3.sel\n',
        112: "   -file  output/Results.sel \n"
    }

    # Extend lines if file is shorter than expected
    for idx, val in replacements.items():
        while len(lines) <= idx:
            lines.append("\n")
        lines[idx] = val

    # Save template file
    tpl_file_name = os.path.basename(phrq_input_file).replace('.phrq', '.tpl')
    tpl_path = os.path.join(script_dir, tpl_file_name)
    with open(tpl_path, 'w') as tpl_file:
        tpl_file.write('ptf $\n')
        tpl_file.writelines(lines)

    print(f"2. Template file successfully created: {tpl_path}")

    # Save copied .phrq file
    copied_phrq_path = os.path.join(script_dir, os.path.basename(phrq_input_file))
    with open(copied_phrq_path, 'w') as new_phrq:
        new_phrq.writelines(lines)

    print(f"3. PHRQ file successfully copied: {copied_phrq_path}")

    return tpl_path, copied_phrq_path

tpl_path, copied_phrq_path = modify_phrq_and_save()

##################################
'Create Pest Control file '
##################################
nobs = len(measured["DMA"])  *3  #number of observations fo Swaroop and Zhou different 
npargp = "3" #number of parameter groups 
npar = '3'    #number of parameters, 3 parameters for the reaction 
nprior = '0'  #number of articles of prior information
nobsgp = '3'   #number of observation groups different constituents for swaroop and zhou but have the same number 
ntpfle = '1'  #number of template files
ninsfle = '3'  #number of instruction files for each measured consituent

# SOIL SPECIFIC PARAMETERS

DmfDma = config.get('optimization inital parameters', 'DMF_DMA', fallback='0.001')
DmfDma_range =config.get('optimization inital parameters','DMF_DMA_range',fallback=(0.003,0.005))
DmfDma_range = ast.literal_eval(DmfDma_range)
DmaMma = config.get('optimization inital parameters', 'DMA_MMA', fallback='0.001')
DmaMma_range =config.get('optimization inital parameters','DMA_MMA_range',fallback=(0.003,0.005))
DmaMma_range = ast.literal_eval(DmaMma_range)
MmaNH3 = config.get('optimization inital parameters', 'MMA_AMMONIA', fallback='0.001')
MmaNH3_range =config.get('optimization inital parameters','MMA_AMMONIA_range',fallback=(0.003,0.005))
MmaNH3_range = ast.literal_eval(MmaNH3_range)

# PEST CONTROL FILE
control_data = (
'* control data\n'
    'restart estimation\n'
    +str(npar) +' '+str(nobs) +' '+npargp +' '+nprior +' '+nobsgp +'\n'
    +ntpfle +' '+ninsfle +' single point 1 0 0 \n'
    '10 -3 0.3 0.01 10 0 lamforgive noderforgive \n'
    '10 10 0.001 \n'
    '0.1 1 noaui \n'
    '50 0.0005 4 4 0.0005 4 \n'
    '0 0 0 verboserec NOJCOSAVEITN REISAVEITN NOPARSAVEITN \n'
'* singular value decomposition\n'
    '1\n'
    '3 5e-7\n'
    '0\n'
'* parameter groups\n'
    'DmfDma  relative 0.01 0.0 switch 2.0 parabolic\n'
    'DmaMma relative 0.01 0.0 switch 2.0 parabolic\n'
    'MmaNH3 relative 0.01 0.0 switch 2.0 parabolic\n'
)
control_data += (
'* parameter data\n')
control_data += (
    'DmfDma none relative ' +str(DmfDma) +' '+str(DmfDma_range[0]) +' '+str(DmfDma_range[1]) +' DmfDma 1.0 0.0 1\n'
    'DmaMma none relative ' +str(DmaMma) +' '+str(DmaMma_range[0]) +' '+str(DmaMma_range[1]) +' DmaMma 1.0 0.0 1\n'
    'MmaNH3 none relative ' +str(MmaNH3) +' '+str(MmaNH3_range[0]) +' '+str(MmaNH3_range[1]) +' MmaNH3 1.0 0.0 1\n'
    )
control_data += (
'* observation groups\n'
    'gDmf     \n'
    'gDma    \n')
if ZHOU==True:
    control_data += ('gMma   \n')
else:
    control_data += ('gNH3    \n')
control_data += (
'* observation data\n')
for index, value in measured['DMF'].items():
        control_data += f'DMF_{index} {value} 1 gDmf\n'
for index, value in measured['DMA'].items():
        control_data += f'DMA_{index} {value} 1 gDma\n' # higher weight 
if ZHOU==True:
    for index, value in measured['MMA'].items():
        control_data += f'MMA_{index} {value} 1 gMma\n'
else:      
    for index, value in measured['NH3'].items():
        control_data += f'NH3_{index} {value} 1 gNH3\n'

control_data += (
'* model command line\n' )
control_data+= "python R2_Run_PHQ.py \n "
#control_data+= PHRQCMD +  '    '+ phrq_file[0] +  '    '+ out_file[0]+  '    '+PHRQDB+  '    '+SCR+'\n'
control_data+=(
'* model input/output\n')

control_data += os.path.basename(tpl_path) + ' ' + os.path.basename(copied_phrq_path) + '\n'
ins_files = [f for f in os.listdir(script_dir) if f.endswith('.ins')]
sel_files = {os.path.splitext(f)[0]: f for f in os.listdir(os.path.join(parent_dir, "output")) if f.endswith('.sel')}

for ins_file in ins_files:
    base = os.path.splitext(ins_file)[0]
    if base in sel_files:
        sel_file = sel_files[base]
        control_data += f"{ins_file} output/{sel_file}\n"
    else:
        print(f"Warning: No matching SEL file in output/ for {ins_file}")
control_data+=(
'DMA.ins  output/DMA.sel\n'
'MMA.ins   output/MMA.sel\n'
'++ies_num_reals(200)\n'
'++ies_lambda_mults(1.2)\n'
'++ies_bad_phi(0.001)')

control_file_path = os.path.join(script_dir, 'control.pst')
with open(control_file_path, 'w') as control_file:
    control_file.write('pcf\n' + control_data)
    print(f"4. Control file successfully created. Saved under {control_file_path}")
control_file.close()




