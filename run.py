import subprocess
import sys
import time
import numpy as np
from utilities import config_data
import torch

n_deltas = 5
delta_list = 1/np.logspace(1, 3, n_deltas, dtype=float)
# delta_list = [0.03162278, 0.00316228]
processortype = config_data["processortype"]
n_processors = config_data["processorcount"]  # to run on multiple devices
n_running = 0
pde_type = 'energy'
target = 10
zero_init = True
lr = 'function'
n_examples = None  # None = entire dataset
# for lr in [.1, .01, .001]:
# for target in [0, 10, 'inv']:
for pde_type in ['energy', 'pde']:
    for n_running, delta in enumerate(delta_list):
        subprocess.Popen([sys.executable, "main.py",
                          "--dataset", "test",
                          "--alpha", "0.1",
                          "--lr", str(lr),
                          "--delta", str(delta),
                          "--device", f"{processortype}:{n_running%n_processors}",
                          "--target", str(target),
                          "--pdetype", str(pde_type),
                          "--zero_init", str(zero_init),
                          "--n_examples", str(n_examples)])
        # only start if enough space is left on devices
        n_running += 1
        if n_processors == 1:
            time.sleep(10)
            full = True
            while full:
                free, total = torch.cuda.mem_get_info()
                per_process = (total-free)/(n_running+1)
                if 1.5*per_process < free:
                    full = False
                else:
                    time.sleep(100)
