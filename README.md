# nequip-flux-calculator
Provides scripts to compute the heat flux for nequip or allegro models.

It was developed for older versions of NequIP/Allegro and will break for newer versions.
It works specifically for `mir-allegro==0.2.0` and `nequip==0.5.6`. It also needs the `mace-unfolded` package for some functions of the MD part. MACE does not need to be installed for this.

## Installation

```
pip install .
```

## Usage

Command line tool:

```
allegro_perform_green_kubo --temperature 300 --n_equil 20000 --n_run 1000000 $MODEL_TRAIN_DIRECTORY --struct POSCAR --seed 123
```

Equilibration is done with a Langevin thermostat. Since the calculator has to modify the internal structure of the actual model by replacing the `GradientOutput` module, the directory containing `best_model.pth` and `config.yaml` obtained from and used for training must be specified. The code is not very efficient when it comes to GPU utilization due to a single-thread CPU side bottleneck. It is recommended to launch several parallel simulations on the same GPU.

