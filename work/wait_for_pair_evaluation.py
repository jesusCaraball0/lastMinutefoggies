import json, subprocess, sys, time
from pathlib import Path
root=Path(__file__).resolve().parents[1]
folders=[root/'out/rl/cnn-c64-h256-8ebd1cf653',root/'out/rl/residual-c32-h128-518d814dce']
while True:
    try:
        logs=[json.loads((p/'metrics.json').read_text()) for p in folders]
        if all(d['status']=='completed' and d['updates'][-1]['update']==400 for d in logs):
            time.sleep(2)
            break
    except (FileNotFoundError,json.JSONDecodeError):pass
    time.sleep(10)
subprocess.run([sys.executable,str(root/'work/evaluate_pair_400.py')],cwd=root,check=True)
