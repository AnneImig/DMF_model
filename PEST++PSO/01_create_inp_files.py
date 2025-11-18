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
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
input_folder = os.path.join(parent_dir, "input")

phrq_file = [file for file in os.listdir(input_folder) if file.endswith('.phrq')]

out_file = [file.replace('.phrq', '.out') for file in phrq_file]
tpl_file = [file.replace('.phrq', '.tpl') for file in phrq_file]#[file for file in os.listdir(script_dir) if file.endswith('.tpl')]

meas_dir= os.path.join(parent_dir,'Measurements')

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



def create_c_ins_file(series, script_dir):
    obs_labels = ''
    for idx, _ in series.items():
        obs_label = f'l{2 if idx == 0 else 1} [C_{idx}]42:51\n' # maybe -1?
        obs_labels += obs_label

    c_ins_path = os.path.join(script_dir, 'Bromide.ins')
    with open(c_ins_path, 'w') as c_ins:
        c_ins.write('pif @\n' + obs_labels)
    print(f"1. Instruction file successfully created. Saved under {c_ins_path}")
create_c_ins_file(meas, script_dir)
ins_file= [file for file in os.listdir(script_dir) if file.endswith('.ins')]



##################################
'Create Pest Control file '
##################################
nobs = len(measured["DMA"])    #number of observations fo Swaroop and Zhou different 
npargp = "1" #number of parameter groups 
npar = '3'    #number of parameters, 3 parameters for the reaction 
nprior = '0'  #number of articles of prior information
nobsgp = '3'   #number of observation groups different constituents for swaroop and zhou but have the same number 
ntpfle = '1'  #number of template files
ninsfle = '3'  #number of instruction files for each measured consituent

# SOIL SPECIFIC PARAMETERS

disp = config.get('column', 'disp', fallback='0.001')
disp_range =config.get('column','disp_range',fallback=(0.003,0.005))
disp_range = ast.literal_eval(disp_range)


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
    '25 5e-7\n'
    '0\n'
'* parameter groups\n'
    'disp relative 0.01 0.0 switch 2.0 parabolic\n'
)
control_data += (
'* parameter data\n')
control_data += (
    'disp none relative ' +str(disp) +' '+str(disp_range[0]) +' '+str(disp_range[1]) +' disp 1.0 0.0 1\n'
)
control_data += (
'* observation groups\n'
    'gC\n'
'* observation data\n')
for index, value in meas.items():
        control_data += f'C_{index} {value} 1 gC\n'

control_data += (
'* model command line\n' )
control_data+= "python R2_Run_PHQ.py \n "
#control_data+= PHRQCMD +  '    '+ phrq_file[0] +  '    '+ out_file[0]+  '    '+PHRQDB+  '    '+SCR+'\n'
control_data+=(
'* model input/output\n')
control_data += tpl_file[0] + ' ' + phrq_file[0] + '\n'
control_data += ins_file[0] + ' ' +'./output/' + SELFILE + '\n'  # sel output will be generate By PHREEQC and location is defined in tpl file 

control_file_path = os.path.join(script_dir, 'control.pst')
with open(control_file_path, 'w') as control_file:
    control_file.write('pcf\n' + control_data)
    print(f"2. Control file successfully created. Saved under {control_file_path}")
control_file.close()

##################################
'Create Pest++ PSO Control file '
##################################

nobs = len(meas)    #number of observations
npargp = "1" #number of parameter groups
npar = '1'    #number of parameters, lets start with one dispersivity 
nprior = '0'  #number of articles of prior information
nobsgp = '1'   #number of observation groups
ntpfle = '1'  #number of template files
ninsfle = '1'  #number of instruction files

# SOIL SPECIFIC PARAMETERS

disp = config.get('column', 'disp', fallback='0.001')
disp_range =config.get('column','disp_range',fallback=(0.003,0.005))
disp_range = ast.literal_eval(disp_range)


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
    '25 5e-7\n'
    '0\n'
'* parameter groups\n'
    'disp relative 0.01 0.0 switch 2.0 parabolic\n'
)
control_data += (
'* parameter data\n')
control_data += (
    'disp log relative ' +str(disp) +' '+str(disp_range[0]) +' '+str(disp_range[1]) +' disp 1.0 0.0 1\n'
)
control_data += (
'* observation groups\n'
    'gC\n'
'* observation data\n')
for index, value in meas.items():
        control_data += f'C_{index} {value} 1 gC\n'

control_data += (
'* model command line\n' )
control_data+= "python R2_Run_PHQ.py \n "
#control_data+= PHRQCMD +  '    '+ phrq_file[0] +  '    '+ out_file[0]+  '    '+PHRQDB+  '    '+SCR+'\n'
control_data+=(
'* model input/output\n')
control_data += tpl_file[0] + ' ' + phrq_file[0] + '\n'
control_data += ins_file[0] + ' ' +'./output/' + SELFILE + '\n'  # sel output will be generate By PHREEQC and location is defined in tpl file 
# needed information for the PSO 
control_data += (
    f'++PSO({os.path.join(script_dir, "queen/case.pso")})\n'
    '++forgive_unknown_args(true)\n'
)

control_file_path = os.path.join(script_dir, 'control_pso.pst')
with open(control_file_path, 'w') as control_file:
    control_file.write('pcf\n' + control_data)
    print(f"2b. Control PEST++ PSO file successfully created. Saved under {control_file_path}")
control_file.close()


####################################################################
'Create Pest++ PSO Control file instructions based on Li Chaos CAS'
####################################################################

nobs = len(meas)    #number of observations
npargp = "1" #number of parameter groups
npar = '1'    #number of parameters, lets start with one dispersivity 
nprior = '0'  #number of articles of prior information
nobsgp = '1'   #number of observation groups
ntpfle = '1'  #number of template files
ninsfle = '1'  #number of instruction files

# SOIL SPECIFIC PARAMETERS

disp = config.get('column', 'disp', fallback='0.001')
disp_range =config.get('column','disp_range',fallback=(0.003,0.005))
disp_range = ast.literal_eval(disp_range)


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
    '25 5e-7\n'
    '0\n'
'* parameter groups\n'
    'disp relative 0.01 0.0 switch 2.0 parabolic\n'
)
control_data += (
'* parameter data\n')
control_data += (
    'disp none relative ' +str(disp) +' '+str(disp_range[0]) +' '+str(disp_range[1]) +' disp 1.0 0.0 1\n'
)
control_data += (
'* observation groups\n'
    'gC\n'
'* observation data\n')
for index, value in meas.items():
        control_data += f'C_{index} {value} 1 gC\n'

control_data += (
'* model command line\n' )
control_data+= "python R2_Run_PHQ.py \n "
#control_data+= PHRQCMD +  '    '+ phrq_file[0] +  '    '+ out_file[0]+  '    '+PHRQDB+  '    '+SCR+'\n'
control_data+=(
'* model input/output\n')
control_data += tpl_file[0] + ' ' + phrq_file[0] + '\n'
control_data += ins_file[0] + ' ' +'./output/' + SELFILE + '\n'  # sel output will be generate By PHREEQC and location is defined in tpl file 

control_file_path = os.path.join(script_dir, 'control_psoCAS.pst')
with open(control_file_path, 'w') as control_file:
    control_file.write('pcf\n' + control_data)
    print(f"2c. Control file successfully created. Saved under {control_file_path}")
control_file.close()


####################################################################
'Create Pest++ PSO.pso Control file '
####################################################################

nobs = len(meas)    #number of observations
npargp = "1" #number of parameter groups
npar = '1'    #number of parameters, lets start with one dispersivity 
nprior = '0'  #number of articles of prior information
nobsgp = '1'   #number of observation groups
ntpfle = '1'  #number of template files
ninsfle = '1'  #number of instruction files

# SOIL SPECIFIC PARAMETERS

disp = config.get('column', 'disp', fallback='0.001')
disp_range =config.get('column','disp_range',fallback=(0.003,0.005))
disp_range = ast.literal_eval(disp_range)


# PEST CONTROL FILE
control_data = (
'* control data\n'
    '0 1 1 30 2\n' #RSTPSO NOBJGP NCON NFORG VERBOSE
    '20  2 2 0.1\n' #NPOP C1 C2 ISEED
    '1 0.8 0.7 0.4 1\n' #INITP VMAX IINERT FINERT INITER
    '0 500 \n' #NEIBR NNEIBR neighborhoods not used if NEIBR=0
'* objctive data \n' 
    'disp 1\n'#OBJNME OBJMETH
# '* constraint data \n'
#     'disp relative 0.01 0.0 switch 2.0 parabolic\n' #CONNME CONMETH UPLIM
)

control_file_path = os.path.join(script_dir, 'case.pst')
with open(control_file_path, 'w') as control_file:
    control_file.write(control_data)
    print(f"20. PSO control file successfully created. Saved under {control_file_path}")
control_file.close()
##################################
'Create Template file'
##################################

def modify_phrq_and_save():
    try:
        if not phrq_file:
            raise FileNotFoundError("No .phrq file found in the input folder.")
        
        phrq_path = os.path.join(input_folder, phrq_file[0])  # Full path to file
        
        with open(phrq_path, 'r') as file:
            lines = file.readlines()
            lines[94] = '    -dispersivities  40*$disp$\n'
            lines[76]='  -file output/Br_SP_heterogen.sel\n'

        tpl_file_name = phrq_file[0].replace('.phrq', '.tpl')
        output_file_path = os.path.join(script_dir, tpl_file_name)

        with open(output_file_path, 'w') as output_file:
            output_file.write('ptf $\n')
            output_file.writelines(lines)

        print(f"3. Template file successfully created . Saved as {output_file_path}")
        return output_file_path

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

output_file_path = modify_phrq_and_save()

