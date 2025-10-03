#!/usr/bin/env python
# Copyright 2025 The PULGON Project Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import argparse
import os
import sys
import time
from ase import units
import ase
import ase.io
import numpy as np
from ase.io.trajectory import Trajectory
from ase.md.verlet import VelocityVerlet
from mace_unfolded.md_tools.md_utils import (
    print_md_info,
    md_equilibration,
)
import time

from nequip_flux_calculator.custom_modules import load_nequip_calculator
from nequip.data import AtomicDataDict

eV2J = 1.60218e-19
J2eV = 1.0 / eV2J
amu2kg = 1.66054e-27
A2m = 1.0e-10
ps2s = 1.0e-12
kine2J = amu2kg * A2m**2 / ps2s**2
kine_conversion2eV = kine2J * J2eV

def compute_heat_flux(
    atoms,
    current_integrator,
    calc,
    flux_file_name,
    flux_comp_file_name,
    exchange_inds=False,
):
    dU_drij = calc.results[AtomicDataDict.PARTIAL_FORCE_KEY]
    r_ij = atoms.get_all_distances(mic=True, vector=True)
    velocities = atoms.get_velocities() * units.fs * 1000
    if exchange_inds:
        rhand = np.einsum("ijk,jk->ij", dU_drij, velocities)
    else:
        rhand = np.einsum("jik,jk->ij", dU_drij, velocities)
    flux = np.einsum("jik,ij->k", r_ij, rhand)
    calc.results["heat_flux_pot"] = (
        flux / atoms.get_volume() * calc.energy_units_to_eV / calc.length_units_to_A**2
    )


    kinetic_energies = (
        np.einsum(
            "i,ij->i",
            atoms.get_masses(),
            velocities**2,
        )
        / 2
    ) * kine_conversion2eV
    flux_conv = np.einsum(
        "i,ij->j",
        calc.results["energies"] + kinetic_energies,
        velocities,
    )
    calc.results["heat_flux_conv"] = (
        flux_conv
        / atoms.get_volume()
        * calc.energy_units_to_eV
        / calc.length_units_to_A**2
    )
    temp = atoms.get_temperature()
    calc.results["heat_flux"] = (
        calc.results["heat_flux_pot"] - calc.results["heat_flux_conv"]
    )
    flux = calc.results["heat_flux"]

    with open(flux_file_name, "a") as hfp:
        wstr = "%18.12f " % temp
        for iflux in flux:
            wstr += "%18.12f " % iflux
        hfp.write(wstr + "\n")

    with open(flux_comp_file_name, "a") as hfp:
        wstr = "%18.12f " % temp
        for quantity in [
            calc.results["heat_flux_pot"],
            calc.results["heat_flux"],
            calc.results["heat_flux_conv"],
        ]:
            for ind in range(len(quantity)):
                wstr += "%18.12f " % quantity[ind]
        wstr += "\n"
        hfp.write(wstr)


def time_tracker(current_atoms, current_integrator, prev_time):
    t1 = time.time()
    time_diff = t1 - prev_time[0]
    print("Time for step: ", time_diff)
    prev_time[0] = t1
    return prev_time


def create_flux_files(flux_file_name, flux_comp_file_name):
    with open(flux_file_name, "w") as hfp:
        hfp.write("Temp c_flux1[1] c_flux1[2] c_flux1[3]\n")
    with open(flux_comp_file_name, "w") as hfp:
        hfp.write(
            "Temp pot[1] pot[2] pot[3] int[1] int[2] int[3] conv[1] conv[2] conv[3]\n"
        )


def perform_GK_simulation(
    train_dir,
    struct_file="POSCAR",
    seed=1234,
    dt=1.0,
    temperature=300,
    n_equil=10000,
    n_run=1000000,
    PBC=[True, True, True],
    device="cuda",
    flux_dir="flux_files",
    dtype="float32",
    restart=False,
    traj_output_interval=100,
    verbose=False,
):
    # argparse is not good enough for booleans
    if isinstance(PBC[0], str):
        PBC = [x.lower() == "true" or x.lower() == "t" for x in PBC]
    T_limit = temperature * 5

    flux_file_name = f"{flux_dir}/heat_flux.dat"
    flux_comp_file_name = f"{flux_dir}/heat_flux_components.dat"
    traj_file_name = f"{flux_dir}/test_equil_{temperature}K_.traj"

    traj = None

    if not os.path.exists(flux_dir):
        os.mkdir(flux_dir)
    calc = load_nequip_calculator(
        train_dir=train_dir,
        method="partial_local",
        device="cuda",
    )

    if traj is None:
        if verbose and "verbose" in calc.__dir__():
            calc.verbose = True
        # model = torch.load(model_file, map_location=device)
        # model.to(device)
        equilibrate = True
        rng = np.random.default_rng(seed=seed)
        if restart:
            trajectory = Trajectory(traj_file_name, "r")
            num_entries = len(trajectory)
            # the first structure is also stored in the trajectory
            expected_equil_entries = n_equil // traj_output_interval + 1
            if num_entries >= expected_equil_entries:
                number_file_lines = sum(1 for line in open(flux_file_name))
                number_comp_lines = sum(1 for line in open(flux_comp_file_name))
                restart_from = (
                    number_file_lines // traj_output_interval
                ) * traj_output_interval
                to_remove = number_file_lines - restart_from
                print(
                    f"actually finished lines: {number_file_lines}. Restarting from {restart_from}"
                )
                print(f"removing {to_remove} entries from flux files")
                cmd = f"sed '{restart_from+2}, $ d' {flux_file_name} -i"
                os.system(cmd)
                number_comp_lines = sum(1 for line in open(flux_comp_file_name))
                restart_from_comp = (
                    number_comp_lines // traj_output_interval
                ) * traj_output_interval
                cmd = f"sed '{restart_from_comp+2}, $ d' {flux_comp_file_name} -i"
                os.system(cmd)

                equilibrate = False
                n_run = n_run - restart_from
                print(f"running remaining {n_run} NVE steps")
            else:
                n_equil = n_equil - (num_entries - 1) * traj_output_interval
                print(f"running remaining {n_equil} NVT equilibration steps")
            atoms = trajectory[-1]
            # when doing multiple restarts, the number of steps to be done might be slighly off
            # however that is such a tiny thing and ultimately inconsequential
            trajectory = Trajectory(traj_file_name, "a", atoms)
            init_vel = False

        else:

            atoms = ase.io.read(struct_file)
            init_vel = True
            trajectory = Trajectory(traj_file_name, "w", atoms)
        atoms.pbc = PBC
        atoms.calc = calc

        if equilibrate:
            md_equilibration(
                atoms,
                temperature=temperature,
                dt=dt,
                n_equil=n_equil,
                T_limit=T_limit,
                traj_output_interval=traj_output_interval,
                rng=rng,
                trajectory=trajectory,
                init_velocities=init_vel,
            )

            # only make new files if run with equilibration
            create_flux_files(
                flux_file_name=flux_file_name,
                flux_comp_file_name=flux_comp_file_name,
            )

    ### HEAT FLUX

    # heat_comp_timer = 0
    # heat_out_timer = 0

    # main evaluation
    if traj is None:
        nve = VelocityVerlet(
            atoms,
            dt * ase.units.fs,
        )
        # trajectory = Trajectory(f"{flux_dir}/test_gk_{TARGET_T}K_.traj", "w", atoms)
        nve.attach(
            trajectory.write, interval=traj_output_interval, current_integrator=nve
        )
        nve.attach(
            print_md_info, interval=1, current_atoms=atoms, current_integrator=nve
        )
        nve.attach(
            compute_heat_flux,
            atoms=atoms,
            current_integrator=nve,
            calc=calc,
            flux_file_name=flux_file_name,
            flux_comp_file_name=flux_comp_file_name,
            exchange_inds=False,
        )
        pt = np.array([time.time()])
        if verbose:
            nve.attach(
                time_tracker,
                interval=1,
                current_atoms=atoms,
                current_integrator=nve,
                prev_time=pt,
            )

        t1 = time.time()
        nve.run(n_run)
        t_run = time.time() - t1
        print("run time:", t_run, flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="perform a Green-Kubo simulation for a MACE model using ASE as the MD backend. Alternatively, read from a LAMMPS trajectory"
    )

    parser.add_argument(
        "--struct",
        dest="struct_file",
        type=str,
        default="POSCAR",
        help="starting geometry for the MD run",
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=1234,
        help="random seed for the MD run",
    )
    parser.add_argument(
        "--temperature",
        dest="temperature",
        type=float,
        default=300,
        help="target temperature for the MD run in K",
    )
    parser.add_argument(
        "--dt",
        dest="dt",
        type=float,
        default=1.0,
        help="time step of simulation in fs",
    )
    parser.add_argument(
        "--n_equil",
        dest="n_equil",
        type=int,
        default=20000,
        help="number of equilibration steps",
    )
    parser.add_argument(
        "--n_run",
        dest="n_run",
        type=int,
        default=1000000,
        help="number of production steps",
    )
    parser.add_argument(
        "--pbc",
        dest="PBC",
        type=str,
        nargs=3,
        default=[True, True, True],
        help="PBC for the MD run",
    )
    parser.add_argument(
        "--device",
        dest="device",
        type=str,
        default="cuda",
        help="device for the MD run",
    )
    parser.add_argument(
        "--flux_dir",
        dest="flux_dir",
        type=str,
        default="flux_files",
        help="directory for the flux files",
    )
    parser.add_argument(
        "--dtype",
        dest="dtype",
        type=str,
        default="float32",
        help="data type for the MD run",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        default=False,
        help="restart the MD run",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="output detailed computation time breakdown",
    )

    parser.add_argument(
        "train_dir",
        help="directory where the model was trained. Needs to contain config.yaml and best_model.pth",
    )

    args = parser.parse_args()

    afp = open("p_cmnd.log", "a")
    afp.write(" ".join(sys.argv) + "\n")
    afp.close()

    perform_GK_simulation(**vars(args))


if __name__ == "__main__":
    main()
