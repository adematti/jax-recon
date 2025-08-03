
import numpy as np

import jax
from jaxpower import FKPField, generate_uniform_particles, MeshAttrs
from jaxrecon.zeldovich import PlaneParallelFFTReconstruction, IterativeFFTReconstruction, IterativeFFTParticleReconstruction


def get_random_catalog(mattrs, size=100000, seed=42):
    from jaxpower import create_sharded_random
    seed = jax.random.key(seed)
    particles = generate_uniform_particles(mattrs, size, seed=seed)
    def sample(key, shape):
        return jax.random.uniform(key, shape, dtype=mattrs.dtype)
    weights = create_sharded_random(sample, seed, shape=particles.size, out_specs=0)
    return particles.clone(weights=weights)



def _identity_fn(x):
    return x


def allgather(array):
    from jaxpower.mesh import get_sharding_mesh
    from jax.sharding import PartitionSpec as P
    sharding_mesh = get_sharding_mesh()
    if sharding_mesh.axis_names:
        array = jax.jit(_identity_fn, out_shardings=jax.sharding.NamedSharding(sharding_mesh, P()))(array)
        return array.addressable_data(0)


def allgather_particles(particles):
    from typing import NamedTuple

    class Particles(NamedTuple):
        positions: jax.Array
        weights: jax.Array

    return Particles(allgather(particles.positions), allgather(particles.weights))



def test_plane_parallel():
    mattrs = MeshAttrs(boxsize=1000., boxcenter=600., meshsize=128)
    pattrs = mattrs.clone(boxsize=800.)
    data = get_random_catalog(pattrs, seed=42)
    randoms = get_random_catalog(pattrs, seed=84)
    growth_rate, bias = 0.8, 2.

    from pyrecon import PlaneParallelFFTReconstruction as RefPlaneParallelFFTReconstruction

    for los in ['x', 'y', 'z']:
        ref_data, ref_randoms = allgather_particles(data), allgather_particles(randoms)
        recon = RefPlaneParallelFFTReconstruction(f=growth_rate, bias=bias, data_positions=ref_data.positions, data_weights=ref_data.weights,
                                                  randoms_positions=ref_randoms.positions, randoms_weights=ref_randoms.weights,
                                                  boxsize=mattrs.boxsize, boxcenter=mattrs.boxcenter, nmesh=mattrs.meshsize, los=los, mpiroot=0)
        ref_shifts = recon.read_shifts(ref_randoms.positions)
        fkp = FKPField(data, randoms, attrs=mattrs)
        recon = PlaneParallelFFTReconstruction(fkp, growth_rate=growth_rate, bias=bias, los=los)
        shifts = allgather(recon.read_shifts(randoms))
        assert np.allclose(shifts, ref_shifts)


def test_iterative_fft():
    mattrs = MeshAttrs(boxsize=1000., boxcenter=600., meshsize=128)
    pattrs = mattrs.clone(boxsize=800.)
    data = get_random_catalog(pattrs, seed=42)
    randoms = get_random_catalog(pattrs, seed=84)
    growth_rate, bias = 0.8, 2.
    niterations = 3

    from pyrecon import IterativeFFTReconstruction as RefIterativeFFTReconstruction

    for los in [None, 'x', 'y', 'z']:
        ref_data, ref_randoms = allgather_particles(data), allgather_particles(randoms)
        recon = RefIterativeFFTReconstruction(f=growth_rate, bias=bias, data_positions=ref_data.positions, data_weights=ref_data.weights,
                                              randoms_positions=ref_randoms.positions, randoms_weights=ref_randoms.weights,
                                              boxsize=mattrs.boxsize, boxcenter=mattrs.boxcenter, nmesh=mattrs.meshsize, los=los, mpiroot=0,
                                              niterations=niterations)
        ref_shifts = recon.read_shifts(ref_randoms.positions)

        fkp = FKPField(data, randoms, attrs=mattrs)
        recon = IterativeFFTReconstruction(fkp, growth_rate=growth_rate, bias=bias, los=los, niterations=niterations)
        shifts = allgather(recon.read_shifts(randoms))
        assert np.allclose(shifts, ref_shifts)


def test_iterative_fft_particle():
    mattrs = MeshAttrs(boxsize=1000., boxcenter=600., meshsize=128)
    pattrs = mattrs.clone(boxsize=800.)
    data = get_random_catalog(pattrs, seed=42)
    randoms = get_random_catalog(pattrs, seed=84)
    growth_rate, bias = 0.8, 2.
    niterations = 3

    from pyrecon import IterativeFFTParticleReconstruction as RefIterativeFFTParticleReconstruction

    for los in [None, 'x', 'y', 'z']:
        ref_data, ref_randoms = allgather_particles(data), allgather_particles(randoms)
        recon = RefIterativeFFTParticleReconstruction(f=growth_rate, bias=bias, data_positions=ref_data.positions, data_weights=ref_data.weights,
                                                      randoms_positions=ref_randoms.positions, randoms_weights=ref_randoms.weights,
                                                       boxsize=mattrs.boxsize, boxcenter=mattrs.boxcenter, nmesh=mattrs.meshsize, los=los, mpiroot=0,
                                                       niterations=niterations)
        ref_shifts = recon.read_shifts(ref_randoms.positions)
        ref_data_shifts = recon.read_shifts('data')
        fkp = FKPField(data, randoms, attrs=mattrs)
        recon = IterativeFFTParticleReconstruction(fkp, growth_rate=growth_rate, bias=bias, los=los, halo_size=3, niterations=niterations)
        shifts = allgather(recon.read_shifts(randoms))
        data_shifts = allgather(recon.read_shifts('data'))
        assert np.allclose(ref_shifts, shifts)
        assert np.allclose(ref_data_shifts, data_shifts)


if __name__ == '__main__':

    from jax import config
    config.update('jax_enable_x64', True)
    import os
    #os.environ["XLA_FLAGS"] = " --xla_force_host_platform_device_count=4"

    from jaxpower import create_sharding_mesh

    with create_sharding_mesh() as sharding_mesh:
        #test_plane_parallel()
        test_iterative_fft()
        test_iterative_fft_particle()
