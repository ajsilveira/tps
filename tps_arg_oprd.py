import openpathsampling as paths
import openpathsampling.engines.openmm as omm

from openpathsampling.collectivevariable import FunctionCV
from openpathsampling.engines.trajectory import Trajectory
from openpathsampling.engines.snapshot import SnapshotFactory

from openpathsampling.engines.openmm.tools import (
    trajectory_from_mdtraj, trajectory_to_mdtraj, ops_load_trajectory
)
import mdtraj as md
import numpy as np

from simtk import openmm, unit
from simtk.openmm import app
from simtk.openmm import XmlSerializer

def r_parallel(snapshot, topology, groups, image_molecules=False, mass_weighted=False):
    import openpathsampling
    import numpy

    traj = openpathsampling.engines.Trajectory([snapshot]).to_mdtraj(topology=topology)
    group_definitions = [numpy.asarray(gf) for gf in groups]

    atom_masses = numpy.array([aa.element.mass for aa in topology.atoms])
    if mass_weighted:
        masses_in_groups = [atom_masses[aa_in_rr] for aa_in_rr in group_definitions]
    else:
        masses_in_groups = [np.ones_like(aa_in_rr) for aa_in_rr in group_definitions]

    traj_copy = traj[:]
    com = []
    if image_molecules:
        traj_copy = traj_copy.image_molecules()
    for aas, mms in zip(group_definitions, masses_in_groups):
        com.append(numpy.average(traj_copy.xyz[:, aas, ], axis=1, weights=mms))

    v1 = com[2] - com[0]
    v2 = com[1] - com[0]
    cosine = numpy.inner(v1, v2) / (numpy.linalg.norm(v1) * numpy.linalg.norm(v2))
    if (cosine >= 1):
        angle = 0
    elif (cosine <= -1):
        angle = numpy.pi()
    else:
        angle = numpy.arccos(cosine)

    return numpy.cos(angle)*numpy.linalg.norm(v1)

pdb = app.PDBFile('../../porin-ligand.pdb')

integrator = openmm.LangevinIntegrator(310*unit.kelvin, 1.0/unit.picoseconds, 4*unit.femtosecond)
with open('../../arg/openmm_system_tps.xml', 'r') as infile:
    openmm_system = XmlSerializer.deserialize(infile.read())

template = ops_load_trajectory('../../porin-ligand.pdb')[0]

openmm_properties = {'CUDAPrecision': 'mixed'}
engine_options = {
    'n_steps_per_frame': 500,
    'n_frames_max': 10000}

engine = omm.Engine(template.topology,
                    openmm_system,
                    integrator,
                    openmm_properties = openmm_properties,
                    options=engine_options).named("arg_engine")

topology = md.Topology.from_openmm(pdb.topology)
bottom_atoms = [1396, 5018, 3076]
top_atoms =  [6241, 3325, 1626]
selection = '(residue 422) and (mass > 3.0)'
ligand_atoms = topology.select(selection)
groups = [bottom_atoms] + [top_atoms] + [ligand_atoms]

# Defining the CVs
cv_r_p = FunctionCV('r_p', r_parallel, topology=topology, groups=groups, image_molecules=False, mass_weighted=False, cv_time_reversible=True, cv_wrap_numpy_array=True).with_diskcache()

# Defining the states
rmax = 4.5*unit.nanometer
rmin = -2.5*unit.nanometer
entrance = paths.CVDefinedVolume(cv_r_p, lambda_min=-2.5, lambda_max=-0.5).named("entrance")
exit = paths.CVDefinedVolume(cv_r_p, lambda_min=2.0, lambda_max=4.5).named("exit")
# Setting up the transition network and move scheme
network = paths.TPSNetwork(entrance, exit)
scheme = paths.OneWayShootingMoveScheme(network, selector=paths.UniformSelector(), engine=engine)

mdtraj_t = md.load('../../arg/smd.nc',top=topology)
ops_trajectory = trajectory_from_mdtraj(mdtraj_t)

# take the subtrajectory matching the ensemble (only one ensemble, only one subtraj)
subtrajectories = []
for ens in network.analysis_ensembles:
    subtrajectories += ens.split(ops_trajectory)

init_cond = scheme.initial_conditions_from_trajectories(trajectories=subtrajectories)
sim = paths.PathSampling(storage=paths.Storage("one-way-shooting.nc", "w", template), move_scheme=scheme, sample_set=init_cond)
sim.run(90)
