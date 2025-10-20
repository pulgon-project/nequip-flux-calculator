# nequip-flux-calculator
Provides scripts to compute the heat flux for NequIP (only for 1 message passing layer) or allegro models.

It was developed for older versions of NequIP/Allegro and will break for newer versions.
It works specifically for `mir-allegro==0.2.0` and `nequip==0.5.6`. It also needs the [mace-unfolded](https://github.com/pulgon-project/mace-unfolded) package for some functions of the MD part. MACE does not need to be installed for this. It should be possible to combine this with the unfolding procedure used in `mace-unfolded` to allow the computation with NequIP for `M>1` with a reasonable amount of effort with the implementation of a new custom module. If there is interest I (Sandro Wieser) might consider implementing that.

The code was used in this [preprint](http://arxiv.org/abs/2508.06882).

## Installation

```bash
pip install .
```

## Usage

Command line tool:

```bash
allegro_perform_green_kubo --temperature 300 --n_equil 20000 --n_run 1000000 $MODEL_TRAIN_DIRECTORY --struct POSCAR --seed 123
```

Equilibration is done with a Langevin thermostat. Since the calculator has to modify the internal structure of the actual model by replacing the `GradientOutput` module, the directory containing `best_model.pth` and `config.yaml` obtained from and used for training must be specified.

The code is not very efficient when it comes to GPU utilization due to a single-thread CPU side bottleneck. It is recommended to launch several parallel simulations on the same GPU. This could look like this:
```bash
for i in {1..5}
do
   allegro_perform_green_kubo --temperature 300 --n_equil 20000 --n_run 200000 $MODEL_TRAIN_DIRECTORY --struct POSCAR --flux_dir flux_files_$i --seed $i > GK_$i.out &
done
wait
```

