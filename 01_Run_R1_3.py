import subprocess
import sys
import os

# Set the working directory (optional, usually the script directory)
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# List of scripts to run in order
scripts_to_run = [
    "R1_Phrq_Create_input.py",
    "R2_Run_PHQ.py",
    "R3_Post_processing.py"
]

for script in scripts_to_run:
    print(f"🔹 Running {script} ...")
    result = subprocess.run([sys.executable, script], capture_output=True, text=True)

    # Print output and errors
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("⚠️ Errors:", result.stderr)

    # Stop execution if script fails
    if result.returncode != 0:
        print(f"❌ {script} failed with return code {result.returncode}. Stopping the chain.")
        break
    else:
        print(f"✅ {script} completed successfully.\n")

print("🏁 Script chain finished (check above for any errors).")
