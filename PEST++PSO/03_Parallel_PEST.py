from shutil import copyfile,copytree,rmtree
import os

'''INSTRUCTIONS'''
#run the 01_create_inpt_file,py to create input files for PEST (template file, instruction file and  )
#run this file to adjust hives
#run executable with "any_PPEST.exe control /h :4004" in manager directory and "any_PPEST.exe control /h hostname:4004" in agent subdirectories
#OR run pestpp.bat from hive1 (apdapt number of agents in pestpp.bat before) with the command ../pestpp.bat


import os
import shutil
from shutil import copyfile, copytree, rmtree

# Define script location
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
input_folder = os.path.join(parent_dir, "input")

# Get .phrq file and create matching .tpl filename
phrq_files = [file for file in os.listdir(input_folder) if file.endswith('.phrq')]
tpl_files = [file.replace('.phrq', '.tpl') for file in phrq_files]

# Check that you found matching pairs
if not phrq_files or not tpl_files:
    raise FileNotFoundError("No .phrq or .tpl files found in input folder.")

# Use first file pair (adapt if more exist)
phrq_file = phrq_files[0]
tpl_file = tpl_files[0]

# Define number of hives
nhives = 3
# Create queen directory once BEFORE the loop
queen_path = os.path.join(script_dir, 'queen')

# Clean up queen directory first
if os.path.exists(queen_path):
    shutil.rmtree(queen_path)
os.makedirs(queen_path)

# Create output directory for queen
queen_output = os.path.join(queen_path, 'output')
os.makedirs(queen_output, exist_ok=True)


# Create output directory for queen
output_path = os.path.join(queen_path, 'output')
os.makedirs(output_path, exist_ok=True)

# Copy required files for queen
copyfile(os.path.join(input_folder, phrq_file), os.path.join(queen_path, phrq_file))
phrq_path = os.path.join(queen_path, phrq_file)
with open(phrq_path, 'r') as f:
    lines = f.readlines()
while len(lines) < 77:
    lines.append('\n')
lines[92] = f"-file  {os.path.join(output_path, 'DMF.sel')}\n"
lines[97] = f"-file  {os.path.join(output_path, 'MMA.sel')}\n"
lines[102] = f"-file  {os.path.join(output_path, 'DMA.sel')}\n"
lines[107] = f"-file  {os.path.join(output_path, 'NH3.sel')}\n"
lines[112] = f"-file  {os.path.join(output_path, 'Results.sel')}\n"

with open(phrq_path, 'w') as f:
    f.writelines(lines)

copyfile(os.path.join(script_dir, tpl_file), os.path.join(queen_path, tpl_file))
tpl_path = os.path.join(queen_path, tpl_file)
with open(tpl_path, 'r') as f:
    lines = f.readlines()
while len(lines) < 77:
    lines.append('\n')
lines[93] = f"-file  {os.path.join(output_path, 'DMF.sel')}\n"
lines[98] = f"-file  {os.path.join(output_path, 'MMA.sel')}\n"
lines[103] = f"-file  {os.path.join(output_path, 'DMA.sel')}\n"
lines[108] = f"-file  {os.path.join(output_path, 'NH3.sel')}\n"
lines[113] = f"-file  {os.path.join(output_path, 'Results.sel')}\n"
with open(tpl_path, 'w') as f:
    f.writelines(lines)

input_folder = os.path.join(parent_dir, "input")

copyfile(os.path.join(script_dir, '001_Controle_file.conf'), os.path.join(queen_path, '001_Controle_file.conf'))
copyfile(os.path.join(script_dir, 'control_pso.pst'), os.path.join(queen_path, 'control.pst'))
ins_files = [f for f in os.listdir(script_dir) if f.endswith('.ins')]
for ins_file in ins_files:
    src = os.path.join(script_dir, ins_file)
    dst = os.path.join(queen_path, ins_file)
    copyfile(src, dst)
copyfile(os.path.join(script_dir, 'R2_Run_PHQ.py'), os.path.join(queen_path, 'R2_Run_PHQ.py'))
copyfile(os.path.join(script_dir, 'case.pso'), os.path.join(queen_path, 'case.pso'))


# CREATE agent directories and populate them
for n in range(nhives):

    hive_name = f"hive{n+1}"
    hive_path = os.path.join(queen_path, hive_name)
    print(f"\n=== Processing {hive_name} ===")

    # Clean up any existing directory and recreate it
    try:
        if os.path.exists(hive_path):
            print(f"Cleaning existing directory: {hive_path}")
            for f in os.listdir(hive_path):
                file_path = os.path.join(hive_path, f)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            print("Directory cleaned successfully.")
        else:
            os.makedirs(hive_path)
            print(f"Directory created: {hive_path}")
    except Exception as e:
        print(f" Error creating/cleaning directory {hive_path}: {e}")

    # Create the 'output' subdirectory
    try:
        output_path = os.path.join(hive_path, 'output')
        os.makedirs(output_path, exist_ok=True)
        print(f"Output directory created: {output_path}")
    except Exception as e:
        print(f" Error creating output directory: {e}")

    # Copy required PHREEQC input file
    try:
        copyfile(os.path.join(input_folder, phrq_file), os.path.join(hive_path, phrq_file))
        print(f"Copied {phrq_file} to {hive_path}")
    except Exception as e:
        print(f" Error copying {phrq_file}: {e}")

    # Modify PHREEQC file
    try:
        phrq_path = os.path.join(hive_path, phrq_file)
        with open(phrq_path, 'r') as f:
            lines = f.readlines()

        while len(lines) < 77:
            lines.append('\n')

        results_file = 'Results.sel'
        lines[92] = f"-file  {os.path.join(output_path, 'DMF.sel')}\n"
        lines[97] = f"-file  {os.path.join(output_path, 'MMA.sel')}\n"
        lines[102] = f"-file  {os.path.join(output_path, 'DMA.sel')}\n"
        lines[107] = f"-file  {os.path.join(output_path, 'NH3.sel')}\n"
        lines[112] = f"-file  {os.path.join(output_path, 'Results.sel')}\n"

        with open(phrq_path, 'w') as f:
            f.writelines(lines)

        print("PHREEQC file updated successfully.")
    except Exception as e:
        print(f" Error updating PHREEQC file: {e}")

    # Copy template file
    try:
        copyfile(os.path.join(script_dir, tpl_file), os.path.join(hive_path, tpl_file))
        print(f"Copied {tpl_file} to {hive_path}")
    except Exception as e:
        print(f" Error copying {tpl_file}: {e}")

    # Modify TPL file
    try:
        tpl_path = os.path.join(hive_path, tpl_file)
        with open(tpl_path, 'r') as f:
            lines = f.readlines()

        while len(lines) < 78:
            lines.append('\n')

        lines[93] = f"-file  {os.path.join(output_path, 'DMF.sel')}\n"
        lines[98] = f"-file  {os.path.join(output_path, 'MMA.sel')}\n"
        lines[103] = f"-file  {os.path.join(output_path, 'DMA.sel')}\n"
        lines[108] = f"-file  {os.path.join(output_path, 'NH3.sel')}\n"
        lines[113] = f"-file  {os.path.join(output_path, 'Results.sel')}\n"

        with open(tpl_path, 'w') as f:
            f.writelines(lines)

        print("TPL file updated successfully.")

    except Exception as e:
        print(f" Error updating TPL file: {e}")

    print(f"=== Completed {hive_name} ===")

###################
    
    copyfile(os.path.join(script_dir, '001_Controle_file.conf'), os.path.join(hive_path, '001_Controle_file.conf'))
    copyfile(os.path.join(script_dir, 'control_pso.pst'), os.path.join(hive_path, 'control.pst'))

    ins_files = [f for f in os.listdir(script_dir) if f.endswith('.ins')]
    for ins_file in ins_files:
        src = os.path.join(script_dir, ins_file)
        dst = os.path.join(hive_path, ins_file)
        copyfile(src, dst)
    copyfile(os.path.join(script_dir, 'R2_Run_PHQ.py'), os.path.join(hive_path, 'R2_Run_PHQ.py'))
    copyfile(os.path.join(script_dir, 'case.pso'), os.path.join(hive_path, 'case.pso'))

    # # Create the shell version of the old .bat file
    # sh_file = os.path.join(hive_path, 'pest_phreeqc.sh')
    # with open(sh_file, 'w') as sh:
    #     sh.write('#!/bin/bash\n')
    #     sh.write('pestpp-glm control_pso.pst /h :4003\n')

    # os.chmod(sh_file, 0o755)  # Make it executable


# DELETE unused hives if they exist
for n in range(nhives, 100):
    hive_name = f"hive{n+1}"
    hive_path = os.path.join(script_dir, hive_name)
    if os.path.exists(hive_path):
        rmtree(hive_path)
