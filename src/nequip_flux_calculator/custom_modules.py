import ase

import inspect

from nequip.ase.nequip_calculator import NequIPCalculator
from nequip.utils.config import Config
from nequip.utils import dtype_from_name
from nequip.data.transforms import TypeMapper
from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin, GradientOutput
from nequip.utils import load_callable, instantiate

from e3nn.util.jit import compile_mode
from e3nn.o3 import Irreps

import logging

import torch
import torch.func as functorch


@compile_mode("script")
class EnergyOutput(GraphModuleMixin, torch.nn.Module):
    r"""Dummy module to replace forces since I am unsure if I can just delete it"""

    def __init__(self, func: GraphModuleMixin) -> None:
        super().__init__()
        self.func = func

        # check and init irreps
        self._init_irreps(
            irreps_in=func.irreps_in,
            my_irreps_in={AtomicDataDict.PER_ATOM_ENERGY_KEY: Irreps("0e")},
            irreps_out=func.irreps_out,
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = self.func(data)
        return data


@compile_mode("unsupported")
class PartialOutput(GraphModuleMixin, torch.nn.Module):
    r"""Generate partial and total forces from an energy model.

    Args:
        func: the energy model
        vectorize: the vectorize option to ``torch.autograd.functional.jacobian``,
            false by default since it doesn't work well.
    """

    vectorize: bool

    def __init__(
        self,
        func: GraphModuleMixin,
        vectorize: bool = False,
        vectorize_warnings: bool = True,
        jacrev=False,
        chunk_size=None,
    ):
        super().__init__()
        self.func = func
        self.vectorize = vectorize
        self.jacrev = jacrev
        self.chunk_size = chunk_size
        if vectorize_warnings:
            # See https://pytorch.org/docs/stable/generated/torch.autograd.functional.jacobian.html
            torch._C._debug_only_display_vmap_fallback_warnings(True)

        # check and init irreps
        self._init_irreps(
            irreps_in=func.irreps_in,
            my_irreps_in={AtomicDataDict.PER_ATOM_ENERGY_KEY: Irreps("0e")},
            irreps_out=func.irreps_out,
        )
        self.irreps_out[AtomicDataDict.PARTIAL_FORCE_KEY] = Irreps("1o")
        self.irreps_out[AtomicDataDict.FORCE_KEY] = Irreps("1o")

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = data.copy()
        out_data = {}

        def wrapper(pos: torch.Tensor) -> torch.Tensor:
            """Wrapper from pos to atomic energy"""
            nonlocal data, out_data
            data[AtomicDataDict.POSITIONS_KEY] = pos
            out_data = self.func(data)  # this is where the model is called
            return out_data[AtomicDataDict.PER_ATOM_ENERGY_KEY].squeeze(-1)

        pos = data[AtomicDataDict.POSITIONS_KEY]

        # here need to compute heat flux and forces

        if self.jacrev:
            partial_forces = functorch.jacrev(wrapper)(pos)
        else:
            partial_forces = torch.autograd.functional.jacobian(
                func=wrapper,
                inputs=pos,
                create_graph=self.training,  # needed to allow gradients of this output during training
                vectorize=self.vectorize,
            )

        partial_forces = partial_forces.negative()
        # output is [n_at, n_at, 3]

        out_data[AtomicDataDict.PARTIAL_FORCE_KEY] = partial_forces
        out_data[AtomicDataDict.FORCE_KEY] = partial_forces.sum(dim=0)

        return out_data


@compile_mode("script")
class FastPartialOutput(GraphModuleMixin, torch.nn.Module):
    r"""Generate partial and total forces from a local energy model (e.g. Allegro).

    Args:
        func: the energy model
    """

    vectorize: bool

    def __init__(
        self,
        func: GraphModuleMixin,
        from_atomic_energies=False,
    ):
        super().__init__()
        self.func = func
        self.from_atomic_energies = from_atomic_energies

        # check and init irreps
        self._init_irreps(
            irreps_in=func.irreps_in,
            my_irreps_in={AtomicDataDict.PER_ATOM_ENERGY_KEY: Irreps("0e")},
            irreps_out=func.irreps_out,
        )
        self.irreps_out[AtomicDataDict.PARTIAL_FORCE_KEY] = Irreps("1o")
        self.irreps_out[AtomicDataDict.FORCE_KEY] = Irreps("1o")

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = data.copy()
        out_data = {}

        pos = data[AtomicDataDict.POSITIONS_KEY]

        old_req_grad = pos.requires_grad
        pos.requires_grad_(True)

        data = AtomicDataDict.with_edge_vectors(data)

        out_data = self.func(data)

        natoms = pos.shape[0]
        edge_index = out_data[AtomicDataDict.EDGE_INDEX_KEY]
        if not self.from_atomic_energies:
            # only works for local FFs, Allegro or NequIP with M=1
            gradient = torch.autograd.grad(
                [out_data[AtomicDataDict.TOTAL_ENERGY_KEY].sum()],
                [out_data[AtomicDataDict.EDGE_VECTORS_KEY]],
                retain_graph=False,
            )[0]

            # edge index is built identically as in MACE
            partial_forces = torch.zeros((natoms, natoms, 3), device=pos.device)
            grads = torch.zeros((natoms, natoms, 3), device=pos.device)
            """
            In fact, rii is not zero for small supercells not fitting in the cutoff with this implementation. So, the supercell MUST fit in the full cutoff sphere for this to work
            """
            partial_forces[edge_index[0, :], edge_index[1, :]] -= gradient
            partial_forces[edge_index[1, :], edge_index[0, :]] += gradient
            grads[edge_index[0, :], edge_index[1, :]] = gradient
            grads = torch.neg(grads)
            forces = partial_forces.sum(dim=0)
            """
            The following part is just for testing that everything is correct. It makes the simulation more expensive, so we don't do it by default.
            If a test is needed, set requires_grad=True in the call to torch.autograd.grad above.
            """
            # forces = torch.neg(torch.autograd.grad(
            #     [out_data[AtomicDataDict.TOTAL_ENERGY_KEY]],out_data[AtomicDataDict.POSITIONS_KEY], retain_graph=False
            # )[0])
            # forces_diff = torch.sum(
            #     torch.abs(forces - partial_forces.sum(dim=0))
            # ) / len(forces)
            # if forces_diff > 1e-5:
            #     import ipdb; ipdb.set_trace()
            # assert (
            #     forces_diff < 1e-5
            # ), f"ERROR: forces from pairwise gradients not consistent! Difference: {forces_diff}"
        else:
            partial_forces = torch.zeros((natoms, natoms, 3), device=pos.device)
            grads = torch.zeros((natoms, natoms, 3), device=pos.device)
            for aid in range(natoms):
                matching_inds = edge_index[0] == aid
                gradient = torch.autograd.grad(
                    [out_data[AtomicDataDict.PER_ATOM_ENERGY_KEY][aid]],
                    [out_data[AtomicDataDict.EDGE_VECTORS_KEY]],
                    retain_graph=True,
                )[0]
                # for gid, gline in enumerate(gradient):
                #     if torch.sum(gline) != 0:
                #         print(aid,gid,gline,edge_index[0, aid])
                # print(edge_index[0,matching_inds])
                
                partial_forces[aid, edge_index[1,matching_inds]] -= gradient[matching_inds]
                partial_forces[edge_index[1,matching_inds], aid] += gradient[matching_inds]
                grads[aid, edge_index[1,matching_inds]] = gradient[matching_inds]
            partial_forces = partial_forces
            grads = torch.neg(grads)
            forces = partial_forces.sum(dim=0)

        # out_data["dU_drij_grads"] = grads
        # WARNING: these are not quite the partial forces! However this quantity is required to compute the heat flux
        out_data[AtomicDataDict.PARTIAL_FORCE_KEY] = grads

        out_data[AtomicDataDict.FORCE_KEY] = forces

        pos.requires_grad_(old_req_grad)

        return out_data


def custom_nequip_model_builder(
    config,
    traindir,
    output_module=None,
    model_name=None,
    initialize=False,
    deploy=False,
    device="cpu",
    dataset=None,
):
    """
    Model builder that replaces the 'ForceOutput' with a custom output module
    output_module: use this module to replace the ForceOutput. If None is given the model is loaded normally.
    """
    config = Config.from_dict(config)

    ### BLOCK MODEL FROM CONFIG BEGIN
    # Pre-process config
    type_mapper = None
    try:
        type_mapper, _ = instantiate(TypeMapper, all_args=config)
    except RuntimeError:
        pass

    if type_mapper is not None:
        if "num_types" in config:
            assert (
                config["num_types"] == type_mapper.num_types
            ), "inconsistant config & dataset"
        if "type_names" in config:
            assert (
                config["type_names"] == type_mapper.type_names
            ), "inconsistant config & dataset"
        config["num_types"] = type_mapper.num_types
        config["type_names"] = type_mapper.type_names

    # Build
    builders = []
    for model_builder in config.get("model_builders", []):
        if (model_builder == "ForceOutput") & (output_module is not None):
            # replace force output with custom module
            builders.append(output_module)
        else:
            builders.append(load_callable(model_builder, prefix="nequip.model"))

    model = None

    for builder in builders:
        pnames = inspect.signature(builder).parameters
        params = {}
        if "initialize" in pnames:
            params["initialize"] = initialize
        if "deploy" in pnames:
            params["deploy"] = deploy
        if "config" in pnames:
            params["config"] = config
        if "dataset" in pnames:
            if "initialize" not in pnames:
                raise ValueError("Cannot request dataset without requesting initialize")
            if (
                initialize
                and pnames["dataset"].default == inspect.Parameter.empty
                and dataset is None
            ):
                raise RuntimeError(
                    f"Builder {builder.__name__} requires the dataset, initialize is true, but no dataset was provided to `model_from_config`."
                )
            params["dataset"] = dataset
        if "model" in pnames:
            if model is None:
                raise RuntimeError(
                    f"Builder {builder.__name__} asked for the model as an input, but no previous builder has returned a model"
                )
            params["model"] = model
        else:
            if model is not None:
                raise RuntimeError(
                    f"All model_builders after the first one that returns a model must take the model as an argument; {builder.__name__} doesn't"
                )
        model = builder(**params)
        if model is not None and not isinstance(model, GraphModuleMixin):
            raise TypeError(
                f"Builder {builder.__name__} didn't return a GraphModuleMixin, got {type(model)} instead"
            )
    ### BLOCK MODEL FROM CONFIG END
    # this block is needed when loading from training directory
    if model is not None:  # TODO: why would it be?
        # TODO: this is not exactly equivalent to building with
        # this set as default dtype... does it matter?
        model.to(
            device=torch.device(device),
            dtype=dtype_from_name(config.default_dtype),
        )
        model_state_dict = torch.load(traindir + "/" + model_name, map_location=device)
        model.load_state_dict(model_state_dict)
    return model


def load_nequip_calculator(train_dir, method, device="cpu", type_map=None):
    # load forces models from config
    yaml_input = f"{train_dir}/config.yaml"
    model_name = "best_model.pth"
    config = Config.from_file(yaml_input)

    # necessary so that partial forces can be loaded and so that regular forces are not computed multiple times
    # config["model_builders"].remove("ForceOutput")
    if type_map is None:
        type_map = config["chemical_symbol_to_type"]
    transform = TypeMapper(chemical_symbol_to_type=type_map)
    if method == "energy_only":

        def EnergyOutputWrapper(
            model: GraphModuleMixin,
        ) -> GradientOutput:
            r"""maybe it is fine to just load it without an output model"""

            return EnergyOutput(
                func=model,
            )

        model = custom_nequip_model_builder(
            output_module=EnergyOutputWrapper,
            config=config,
            traindir=train_dir,
            model_name=model_name,
            device=device,
        )
        model.to(device)
        model.eval()
    elif method == "forces_only":
        model = custom_nequip_model_builder(
            output_module=None,
            config=config,
            traindir=train_dir,
            model_name=model_name,
            device=device,
        )
        # model = torch.jit.script(model)
        model.to(device)
        model.eval()
    elif "partial_local" in method:
        logging.warning(
            "Using the 'partial_local' model loader will not return the actual partial forces in the AtomicDataDict.PARTIAL_FORCE_KEY field. Only use this quantity to compute the heat flux! Make absolutely sure that the model cutoff fits into the simulation cell, otherwise the forces and heat flux will be incorrect!"
        )
        if "atomic" in method:
            atomic_energies = True
        else:
            atomic_energies = False

        def FastPartialOutputWrapper(
            model: GraphModuleMixin,
        ) -> GradientOutput:
            r"""Add forces and partial forces to a model that predicts energy.

            Args:
                model: the energy model to wrap. Must have ``AtomicDataDict.TOTAL_ENERGY_KEY`` as an output.

            Returns:
                A ``GradientOutput`` wrapping ``model``.
            """

            if (
                AtomicDataDict.FORCE_KEY in model.irreps_out
                or AtomicDataDict.PARTIAL_FORCE_KEY in model.irreps_out
            ):
                raise ValueError("This model already has force outputs.")
            return FastPartialOutput(
                func=model,
                from_atomic_energies=atomic_energies,
            )

        model = custom_nequip_model_builder(
            output_module=FastPartialOutputWrapper,
            config=config,
            traindir=train_dir,
            model_name=model_name,
            device=device,
        )
        # TODO: make it possible to torchscript the module so it can be deployed and used in e.g. LAMMPS
        # model = torch.jit.script(model)
        model.to(device)
        model.eval()
    else:
        vectorize = False
        jacrev = False
        if "partial" in method:
            if "vectorize" in method:
                vectorize = True
            if "jacrev" in method:
                jacrev = True

        def PartialOutputWrapper(
            model: GraphModuleMixin,
        ) -> GradientOutput:
            r"""Add forces and partial forces to a model that predicts energy.

            Args:
                model: the energy model to wrap. Must have ``AtomicDataDict.TOTAL_ENERGY_KEY`` as an output.

            Returns:
                A ``GradientOutput`` wrapping ``model``.
            """

            if (
                AtomicDataDict.FORCE_KEY in model.irreps_out
                or AtomicDataDict.PARTIAL_FORCE_KEY in model.irreps_out
            ):
                raise ValueError("This model already has force outputs.")
            return PartialOutput(
                func=model,
                vectorize=vectorize,
                jacrev=jacrev,
            )

        model = custom_nequip_model_builder(
            output_module=PartialOutputWrapper,
            config=config,
            traindir=train_dir,
            model_name=model_name,
            device=device,
        )
        model.to(device)
        model.eval()

    calc = NequIPCalculator(
        model, r_max=config["r_max"], device=device, transform=transform
    )

    return calc
