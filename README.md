# jax-recon

**JAX-based density field reconstruction for large-scale structure**, inspired by [`pyrecon`](https://github.com/cosmodesi/pyrecon) and built to interoperate with [`jax-power`](https://github.com/adematti/jax-power).

## Overview

`jax-recon` implements fast, differentiable, and GPU/TPU-compatible routines for cosmological density field reconstruction using [JAX](https://github.com/google/jax).

This package aims to provide a modern, flexible alternative to `pyrecon` for use in simulation-based inference or neural cosmology workflows.

---

## Installation

```bash
pip install git+https://github.com/adematti/jax-recon.git
```

Dependencies:
- `jax`
- `jax-power`

---

## Usage

```python
import jax.numpy as jnp
from jaxpower import ParticleField, FKPField, get_mesh_attrs, create_sharding_mesh
from jaxrecon.zeldovich import IterativeFFTReconstruction

with create_sharding_mesh() as sharding_mesh:  # distribute mesh and particles

    attrs = get_mesh_attrs(data_positions, randoms_positions, boxpad=1.2, cellsize=10.)

    # Define FKP field = data - randoms
    data = ParticleField(data_positions, data_weights, attrs=attrs, exchange=True, return_inverse=True)
    randoms = ParticleField(randoms_positions, randoms_weights, attrs=attrs, exchange=True, return_inverse=True)
    fkp = FKPField(data, randoms)
    # Line-of-sight "los" can be local (None, default) or an axis, 'x', 'y', 'z', or a 3-vector
    # In case of IterativeFFTParticleReconstruction, and multi-GPU computation, provide the size of halo regions in cell units. E.g., maximum displacement is ~ 40 Mpc/h => 4 * chosen cell size => provide halo_add=2
    recon = IterativeFFTReconstruction(fkp, growth_rate=0.8, bias=2.0, los=None, smoothing_radius=15., halo_add=0)
    # If you are using IterativeFFTParticleReconstruction, displacements are to be taken at the reconstructed data real-space positions;
    # in this case, do: data_positions_rec = recon.read_shifted_positions('data')
    data_positions_rec = recon.read_shifted_positions(data.positions)
    # RecSym = remove large scale RSD from randoms
    randoms_positions_rec = recon.read_shifted_positions(randoms.positions)
    # or RecIso
    # randoms_positions_rec = recon.read_shifted_positions(randoms.positions, field='disp')
    # Now be careful! For multi-GPU computation, these are the weights for the *exchanged* particles.
    # If you want weights for the input data_positions and randoms_positions:
    data_positions_rec = data.exchange_inverse(data_positions_rec)
    randoms_positions_rec = randoms.exchange_inverse(randoms_positions_rec)

```
---

## TODO

- [ ] Implement multigrid reconstruction with CUDA / FFI bindings

---

## Credits

- Inspired by [`pyrecon`](https://github.com/cosmodesi/pyrecon)
- Mesh tools from [`jax-power`](https://github.com/adematti/jax-power)

---