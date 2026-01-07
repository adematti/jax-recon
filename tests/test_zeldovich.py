import os

import numpy as np
import jax
from jax import numpy as jnp

from jaxpower import FKPField, generate_uniform_particles, MeshAttrs, create_sharding_mesh
from jaxrecon.zeldovich import PlaneParallelFFTReconstruction, IterativeFFTReconstruction, IterativeFFTParticleReconstruction, estimate_particle_delta


def get_random_catalog(mattrs, size=100000, seed=42):
    from jaxpower import create_sharded_random
    seed = jax.random.key(seed)
    particles = generate_uniform_particles(mattrs, size, seed=seed, exchange=False)
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
    return array


def allgather_particles(particles):
    from typing import NamedTuple

    class Particles(NamedTuple):
        positions: jax.Array
        weights: jax.Array

    return Particles(allgather(particles.positions), allgather(particles.weights))



def test_plane_parallel():
    mattrs = MeshAttrs(boxsize=1000., boxcenter=600., meshsize=64)
    pattrs = mattrs.clone(boxsize=800.)
    data = get_random_catalog(pattrs, seed=42).clone(attrs=mattrs)
    randoms = get_random_catalog(pattrs, seed=84).clone(attrs=mattrs)
    growth_rate, bias = 0.8, 1.
    threshold_randoms = ('noise', 0.01)
    #threshold_randoms = ('mean', 0.01)

    from pyrecon import PlaneParallelFFTReconstruction as RefPlaneParallelFFTReconstruction

    for los in ['x', 'y', 'z']:
        ref_data, ref_randoms = allgather_particles(data), allgather_particles(randoms)
        recon = RefPlaneParallelFFTReconstruction(f=growth_rate, bias=bias, data_positions=ref_data.positions, data_weights=ref_data.weights,
                                                  randoms_positions=ref_randoms.positions, randoms_weights=ref_randoms.weights,
                                                  boxsize=mattrs.boxsize, boxcenter=mattrs.boxcenter, nmesh=mattrs.meshsize, los=los, mpiroot=0,
                                                  dtype='c16', threshold_randoms=threshold_randoms)
        fields = ['rsd', 'disp', 'disp+rsd']
        ref_shifts = {field: recon.read_shifts(ref_randoms.positions, field=field) for field in fields}
        ref_shifted_positions = {field: recon.read_shifted_positions(ref_randoms.positions, field=field) for field in fields}
        data = data.exchange(return_inverse=True)
        randoms = randoms.exchange(return_inverse=True)
        fkp = FKPField(data, randoms, attrs=mattrs)
        delta = estimate_particle_delta(fkp, threshold_randoms=threshold_randoms)
        recon = jax.jit(PlaneParallelFFTReconstruction, static_argnames=['los'])(delta, growth_rate=growth_rate, bias=bias, los=los)
        shifts = {field: allgather(randoms.exchange_inverse(recon.read_shifts(randoms, field=field))) for field in fields}
        shifted_positions = {field: allgather(randoms.exchange_inverse(recon.read_shifted_positions(randoms, field=field))) for field in fields}
        assert np.allclose(allgather(randoms.exchange_inverse(randoms.positions)), ref_randoms.positions)
        for field in fields:
            assert np.allclose(shifts[field], ref_shifts[field])
            assert np.allclose(shifted_positions[field], ref_shifted_positions[field])


def test_iterative_fft():
    mattrs = MeshAttrs(boxsize=1000., boxcenter=600., meshsize=64)
    pattrs = mattrs.clone(boxsize=800.)
    data = get_random_catalog(pattrs, seed=42).clone(attrs=mattrs).exchange()
    randoms = get_random_catalog(pattrs, seed=84).clone(attrs=mattrs).exchange()
    growth_rate, bias = 0.8, 2.
    threshold_randoms = ('noise', 0.01)
    #threshold_randoms = ('mean', 0.01)

    niterations = 3

    from pyrecon import IterativeFFTReconstruction as RefIterativeFFTReconstruction

    for los in [None, 'x', 'y', 'z']:
        ref_data, ref_randoms = allgather_particles(data), allgather_particles(randoms)
        recon = RefIterativeFFTReconstruction(f=growth_rate, bias=bias, data_positions=ref_data.positions, data_weights=ref_data.weights,
                                              randoms_positions=ref_randoms.positions, randoms_weights=ref_randoms.weights,
                                              boxsize=mattrs.boxsize, boxcenter=mattrs.boxcenter, nmesh=mattrs.meshsize, los=los, mpiroot=0,
                                              threshold_randoms=threshold_randoms)
        fields = ['rsd', 'disp', 'disp+rsd']
        ref_shifts = {field: recon.read_shifts(ref_randoms.positions, field=field) for field in fields}
        ref_shifted_positions = {field: recon.read_shifted_positions(ref_randoms.positions, field=field) for field in fields}
        data = data.exchange(return_inverse=True)
        randoms = randoms.exchange(return_inverse=True)
        fkp = FKPField(data, randoms, attrs=mattrs)
        delta = estimate_particle_delta(fkp, threshold_randoms=threshold_randoms)
        recon = jax.jit(IterativeFFTReconstruction, static_argnames=['los', 'niterations'])(delta, growth_rate=growth_rate, bias=bias, los=los, niterations=niterations)
        shifts = {field: allgather(randoms.exchange_inverse(recon.read_shifts(randoms, field=field))) for field in fields}
        shifted_positions = {field: allgather(randoms.exchange_inverse(recon.read_shifted_positions(randoms, field=field))) for field in fields}
        for field in fields:
            assert np.allclose(shifts[field], ref_shifts[field])
            assert np.allclose(shifted_positions[field], ref_shifted_positions[field])


def test_iterative_fft_particle():
    mattrs = MeshAttrs(boxsize=1000., boxcenter=600., meshsize=128)
    pattrs = mattrs.clone(boxsize=800.)
    data = get_random_catalog(pattrs, seed=42).clone(attrs=mattrs).exchange()
    randoms = get_random_catalog(pattrs, seed=84).clone(attrs=mattrs).exchange()
    growth_rate, bias = 0.8, 2.
    niterations = 3
    threshold_randoms = ('noise', 0.01)
    #threshold_randoms = ('mean', 0.01)

    from pyrecon import IterativeFFTParticleReconstruction as RefIterativeFFTParticleReconstruction

    for los in [None, 'x', 'y', 'z']:
        ref_data, ref_randoms = allgather_particles(data), allgather_particles(randoms)
        recon = RefIterativeFFTParticleReconstruction(f=growth_rate, bias=bias, data_positions=ref_data.positions, data_weights=ref_data.weights,
                                                      randoms_positions=ref_randoms.positions, randoms_weights=ref_randoms.weights,
                                                       boxsize=mattrs.boxsize, boxcenter=mattrs.boxcenter, nmesh=mattrs.meshsize, los=los, mpiroot=0,
                                                       niterations=niterations, threshold_randoms=threshold_randoms, wrap=True)
        fields = ['rsd', 'disp', 'disp+rsd']
        ref_shifts = {field: recon.read_shifts(ref_randoms.positions, field=field) for field in fields}
        ref_shifted_positions = {field: recon.read_shifted_positions(ref_randoms.positions, field=field) for field in fields}
        ref_data_shifts = {field: recon.read_shifts('data', field=field) for field in fields}
        ref_data_shifted_positions = {field: recon.read_shifted_positions('data', field=field) for field in fields}

        data = data.exchange(return_inverse=True)
        randoms = randoms.exchange(return_inverse=True)
        fkp = FKPField(data, randoms, attrs=mattrs)
        recon = jax.jit(IterativeFFTParticleReconstruction, static_argnames=['los', 'halo_add', 'niterations'])(fkp, growth_rate=growth_rate, bias=bias, los=los, halo_add=3, niterations=niterations)
        shifts = {field: allgather(randoms.exchange_inverse(recon.read_shifts(randoms, field=field))) for field in fields}
        shifted_positions = {field: allgather(randoms.exchange_inverse(recon.read_shifted_positions(randoms, field=field, wrap=True))) for field in fields}
        data_shifts = {field: allgather(data.exchange_inverse(recon.read_shifts('data', field=field))) for field in fields}
        data_shifted_positions = {field: allgather(data.exchange_inverse(recon.read_shifted_positions('data', field=field, wrap=True))) for field in fields}
        for field in fields:
            assert np.allclose(ref_shifts[field], shifts[field])
            assert np.allclose(ref_data_shifts[field], data_shifts[field])
            assert np.allclose(ref_shifted_positions[field], shifted_positions[field])
            assert np.allclose(ref_data_shifted_positions[field], data_shifted_positions[field])


if __name__ == '__main__':

    from jax import config
    config.update('jax_enable_x64', True)

    os.environ["XLA_FLAGS"] = " --xla_force_host_platform_device_count=4"

    with create_sharding_mesh() as sharding_mesh:
        test_plane_parallel()
        test_iterative_fft()
        test_iterative_fft_particle()
