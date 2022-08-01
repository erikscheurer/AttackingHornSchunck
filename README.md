# An Optimization Approach to Attacking the Horn and Schunck Model

## Overview

The repository is structured as follows:

- `adversarial.py` contains all relevant functions for an adversarial attack.

  - `adversarial_attack()` takes the images that must required gradients, the original images, the target flow components, and the perturbation constraint as input. It determines the perturbed images and reports the loss at every iteration and whether the constraint is fulfilled.
  - `full_attack()` executes the full range of tasks including the evaluation for one image pair that must be given. Refer to the function documentation for options.
  - `get_deltas()` determines a range of different norms for the evaluation. In the thesis, `globalavg` is used.
  - `get_metrics()` determines the average endpoint errors between all possible combinations of flows.
  - `get_mse_metrics()` determines the mse between all possible combinations of flows.
  - `test_full_attack()` and `test_adversarial_attack()` can be used to test the correct installation of functions.
  - For modifications, `Energy_loss()` `PDE_loss()` can be changed to use the desired energy terms.

- `main.py` can be executed from the commandline. For options see `python3 main.py --help`. It runs an adversarial attack for for the Sintel with the locationof the dataset configured in `config.json`.

- `run.py` can be used to evaluate multiple perturbation sizes, types ... for a dataset. It starts a new process for each perturbation size if there is enough free space on the GPU.

- `hornSchunck.py` is used to evaluate the method of Horn and Schunck. There are many experimental functions here. The ones used in the thesis are `horn_schunck_multigrid()` and `horn_schunck()`.

- There are a few global variables defined in `utilities.py` which also contains a collection of useful functions.

- The `flow_*.py` files are from the [`flow_library`](https://github.com/cv-stuttgart/flow_library/)

- `adversarial_batched.py` and `universalPerturbation.py` are experimental

- `datalogger.py` is a logger that should be replaced by mlflow, tensorboard or something similar.

### Installation

The full pip list used in the thesis is available in `requirements.txt`. It can be installed through

    ```bash
    pip install -r requirements.txt
    ```

## Usage

For example the following command will run the attack for the full Sintel test dataset with the zero target:

```bash
    python3 main.py --delta 0.01 --dataset test --target 0
```

The program creates a folder `hornSchunck` in the current directory to save the results from the evaluations of the original images of Horn and Schunck. This is to avoid redundant evaluations when executing multiple perturbations. If `--save` is true, the perturbations and perturbed flows are saved in `results/` where there is also the logger of the current run.

To evaluate a series of loggers run `evaluate_loggers(folder,...)` in `datalogger.py` where `folder` is the path to the folder containing all loggers. This will create a plot with the specifications you give to the function.
