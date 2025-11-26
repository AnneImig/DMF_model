

###########################################################################################################################
'''
Create the phrq files without the GUI 
'''
###########################################################################################################################
import os 
import configparser
from datetime import datetime 
import sys
import pandas as pd 


script_dir = os.path.dirname(os.path.abspath(__file__))

# read all information from configfile
if len(sys.argv) > 1:
    configfn = sys.argv[1]
else:
    configfn = os.path.join(script_dir,"001_Controle_file.conf")

config = configparser.ConfigParser(inline_comment_prefixes="#")
config.read(configfn)


SELFILE= config.get("files", "SELFILE", fallback=True)
sel_file_path = os.path.join(script_dir,SELFILE )

input_folder = os.path.join(script_dir, "input")
output_folder = os.path.join(script_dir, "output")
os.makedirs(input_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# Flags for program flow

PLOT = config.getboolean("system", "PLOT", fallback=True)
SELFILE= config.get("files", "SELFILE", fallback=True)
FIRST= config.getboolean("kinetics", "FIRST", fallback=True)
BIOMASS= config.getboolean("kinetics", "BIOMASS", fallback=True)
SWAROOP=config.getboolean("dataset", "SWAROOP", fallback=True)
ZHOU=config.getboolean("dataset", "ZHOU", fallback=True)

#Parameter for the model 

if ZHOU ==True:
    params = [0.037, 0.013, 0.001]
elif SWAROOP== True:  
    params = [0.37, 0.091, 5]
else:
    print("No Parameterlist chosen.")


def modify_phrq_and_save(parameterlist):
    # Select template file
    if FIRST:
        Template_file_path = os.path.join(input_folder, "DMF_red_first.phrq")
        template_type = "First"
    elif BIOMASS:
        Template_file_path = os.path.join(input_folder, "DMF_red_first_biomass.phrq")
        template_type = "biomass"
    else:
        print("No template file chosen.")
        return

    print(f"Template file {Template_file_path} chosen")

    # Read the template
    try:
        with open(Template_file_path, 'r') as file:
            lines = file.readlines()

        # Adjust concentrations and parameters
        if ZHOU:
            lines[36] = '     Dmf       6000 mg/kgw \n'
            lines[49] = '     -steps   0 72*1      #72h \n'
        elif SWAROOP:
            lines[36] = '     Dmf       400 mg/kgw \n'
            lines[49] = '     -steps   0 24*1      #24h \n'
        lines[48] = f'       -parms    {parameterlist[0]}\n'
        lines[53] = f'       -parms    {parameterlist[1]}\n'
        lines[59] = f'       -parms    {parameterlist[2]}\n'

        # Determine output file name
        if ZHOU:
            dataset_name = "Zhou"
        elif SWAROOP:
            dataset_name = "Swaroop"
        else:
            dataset_name = "Unknown"

        output_file_name = f"{dataset_name}_{template_type}.phrq"
        phrq_file_path = os.path.join(input_folder, output_file_name)

        # Save the modified file
        with open(phrq_file_path, 'w') as new_file:
            new_file.writelines(lines)

        print(f"Modification successful. Saved as {phrq_file_path}")

    except Exception as e:
        print(f"Error: {e}")


def main():
        # Apply modifications
        modify_phrq_and_save(params)

if __name__ == "__main__":
    main()
