import subprocess
import os
from datetime import datetime 

import numpy as np
import sys
import configparser

script_dir = os.path.dirname(os.path.abspath(__file__))
# Path to the PHREEQC executable
EXECUTABLE = "/Users/anneimig/Documents/Software_executables/Phreeqc/phreeqc-3.5.0-14000/bin/phreeqc" 
# read all information from configfile
if len(sys.argv) > 1:
    configfn = sys.argv[1]
else:
    configfn = os.path.join(script_dir,"001_Controle_file.conf")

config = configparser.ConfigParser(inline_comment_prefixes="#")
config.read(configfn)

# Flags for program flow
DATABASE  = config.get("system", "DATABASE", fallback=True)
FIRST= config.getboolean("kinetics", "FIRST", fallback=True)
BIOMASS= config.getboolean("kinetics", "BIOMASS", fallback=True)
SWAROOP=config.getboolean("dataset", "SWAROOP", fallback=True)
ZHOU=config.getboolean("dataset", "ZHOU", fallback=True)

input_folder = os.path.join(script_dir, "input")
output_folder = os.path.join(script_dir, "output")



phrq_file = [file for file in os.listdir(script_dir) if file.endswith('.phrq')]
phrq_input_file = os.path.join(script_dir, phrq_file[0])

if os.name == "nt":
    PHRQDB = (
        "C:\\PROGRA~1\\USGS\\phreeqc-3.7.3-15968-x64\\database\\phreeqc.dat"
    )
    PHRQCMD = "C:\\PROGRA~1\\USGS\\phreeqc-3.7.3-15968-x64\\bin\\Release\\phreeqc.exe"
else:
    PHRQDB = "/usr/share/phreeqc/database/phreeqc.dat"
    PHRQCMD = "/usr/bin/phreeqc"



def run_phreeqc(phrq_input_file, input_folder, output_folder):
    out_file_path = os.path.join(output_folder, "output.out")
    SCR = os.path.join(output_folder, "scr.out")

    try:
        subprocess.run(
            [EXECUTABLE, phrq_input_file, out_file_path, DATABASE, SCR],
            cwd=script_dir  # <- THIS ENSURES PHREEQC RUNS IN script_dir
        )
        
    except Exception as e:
        print(f"Error running PHREEQC: {e}")


# Run PHREEQC with the provided input script
run_phreeqc(phrq_input_file, input_folder, output_folder)
print("script_dir =", script_dir)
print("cwd        =", os.getcwd())
