"""Run the complete analysis from bundled raw inputs; no network required."""
from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parent
for script in ['scripts/verify_inputs.py','run_analysis.py','scripts/verify_reproduction.py']:
    subprocess.run([sys.executable,'-X','utf8',str(root/script)],cwd=root,check=True)
print('Reproduction completed and reference results verified.')
