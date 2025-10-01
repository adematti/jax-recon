
from collections.abc import Callable

import numpy as np
import jax
from jax import numpy as jnp

from jaxpower import MeshAttrs, ComplexMeshField, RealMeshField, ParticleField, FKPField
from jaxpower.mesh import staticarray


def kernel_gaussian(mattrs: MeshAttrs, smoothing_radius=15.):
    return jnp.exp(- 0.5 * sum((kk * smoothing_radius)**2 for kk in mattrs.kcoords(sparse=True)))


def estimate_mesh_delta(mesh_data: RealMeshField | ComplexMeshField, mesh_randoms: RealMeshField | ComplexMeshField=None, threshold_randoms: float | jax.Array=None, smoothing_radius: float=15.):

    mattrs = mesh_data.attrs

    def _2r(mesh):
        if not isinstance(mesh, RealMeshField):
            mesh = mesh.c2r()
        return mesh

    def _2c(mesh):
        if not isinstance(mesh, ComplexMeshField):
            mesh = mesh.r2c()
        return mesh

    kernel = 1.
    if smoothing_radius is not None:
        kernel = kernel_gaussian(mattrs, smoothing_radius=smoothing_radius)
        mesh_data = (_2c(mesh_data) * kernel).c2r()
        if mesh_randoms is not None:
            mesh_randoms = (_2c(mesh_randoms) * kernel).c2r()

    mesh_data = _2r(mesh_data)
    if mesh_randoms is not None:
        mesh_randoms = _2r(mesh_randoms)
        sum_data, sum_randoms = mesh_data.sum(), mesh_randoms.sum()
        alpha = sum_data * 1. / sum_randoms
        mesh_delta = mesh_data - alpha * mesh_randoms
        if threshold_randoms is not None:
            mesh_delta = mesh_delta.clone(value=jnp.where(mesh_randoms.value > threshold_randoms, mesh_delta.value / (alpha * mesh_randoms.value), 0.))
    else:
        mesh_delta = mesh_data / mesh_data.mean() - 1.
    return mesh_delta


def _get_threshold_randoms(randoms, threshold_randoms: float=0.01):

    if isinstance(threshold_randoms, tuple):
        threshold_method, threshold_value = threshold_randoms
    else:
        threshold_method, threshold_value = 'noise', threshold_randoms
    assert threshold_method in ['noise', 'mean']

    if threshold_method == 'noise':
        threshold_randoms = threshold_value * jnp.sum(randoms.weights**2) / randoms.sum()
    else:
        threshold_randoms = threshold_value * randoms.sum() / randoms.size
    return threshold_randoms


def estimate_particle_delta(fkp: ParticleField | FKPField, resampler: str='cic', halo_add: int=0, smoothing_radius: float=15., threshold_randoms: float=0.01):

    data, randoms = fkp, None
    if isinstance(fkp, FKPField):
        data, randoms = fkp.data, fkp.randoms
    kw = dict(resampler=resampler, compensate=False, interlacing=0, halo_add=halo_add)
    mesh_data = data.paint(**kw, out='complex')
    del data
    if randoms is not None:
        mesh_randoms = randoms.paint(**kw, out='complex')
        threshold_randoms = _get_threshold_randoms(randoms, threshold_randoms=threshold_randoms)
    else:
        threshold_randoms, mesh_randoms = None, None
    return estimate_mesh_delta(mesh_data, mesh_randoms=mesh_randoms, threshold_randoms=threshold_randoms, smoothing_radius=smoothing_radius)


def _get_los_vector(los: str | np.ndarray, ndim=3):
    """Return the line-of-sight vector."""
    vlos = None
    if isinstance(los, str):
        vlos = [0.] * ndim
        vlos['xyz'.index(los)] = 1.
    else:
        vlos = los
    return staticarray(vlos)


def _format_los(los, ndim=3):
    if los is None or isinstance(los, str) and los in ['local']:
        vlos = None
    else:
        vlos = _get_los_vector(los, ndim=ndim)
    return vlos


def _format_positions(positions: jax.Array | ParticleField):
    if isinstance(positions, ParticleField):
        return positions.positions
    return positions


class BaseReconstruction(object):

    def __init__(self, delta: RealMeshField | ParticleField | FKPField, growth_rate: float | Callable=0., bias: float | Callable=1., los: str | np.ndarray=None, **kwargs):
        self._kw_resampler = {name: kwargs.pop(name) for name in ['resampler', 'halo_add'] if name in kwargs}
        if not isinstance(delta, RealMeshField):
            delta = estimate_particle_delta(delta, **self._kw_resampler, **{name: kwargs.pop(name) for name in ['smoothing_radius', 'threshold_randoms'] if name in kwargs})
        self._growth_rate = growth_rate
        self._bias = bias
        self.mattrs = delta.attrs
        self.los = _format_los(los=los)
        self._run(delta, **kwargs)

    def growth_rate(self, distance: jax.Array=None):
        """Return growth rate."""
        if callable(self._growth_rate):
            if distance is None:
                distance = sum(xx**2 for xx in self.mattrs.xcoords())**0.5
            return self._growth_rate(distance)
        return self._growth_rate

    def bias(self, distance: jax.Array=None):
        """Return bias."""
        if callable(self._bias):
            if distance is None:
                distance = sum(xx**2 for xx in self.mattrs.xcoords())**0.5
            return self._bias(distance)
        return self._bias

    def read_shifts(self, positions: jax.Array | ParticleField, field='disp+rsd', **kwargs):
        """
        Read displacement at input positions.
        To get shifted/reconstructed positions, given reconstruction instance ``recon``:

        .. code-block:: python

            positions_rec_data = positions_data - recon.read_shifts(positions_data)
            # RecSym = remove large scale RSD from randoms
            positions_rec_randoms = positions_randoms - recon.read_shifts(positions_randoms)
            # Or RecIso
            # positions_rec_randoms = positions_randoms - recon.read_shifts(positions_randoms, field='disp')

        Or directly use :meth:`read_shifted_positions` (which wraps output positions if :attr:`wrap`).

        Parameters
        ----------
        positions : jax.Array
            Positions of shape (N, 3).
        field : str, default='disp+rsd'
            Either 'disp' (Zeldovich displacement), 'rsd' (RSD displacement), or 'disp+rsd' (Zeldovich + RSD displacement).

        Returns
        -------
        positions : jax.Array
            Shifted positions.
        """
        positions = _format_positions(positions)
        kw_resampler = self._kw_resampler | kwargs
        field = field.lower()
        allowed_fields = ['disp', 'rsd', 'disp+rsd']
        if field not in allowed_fields:
            raise ValueError('Unknown field {}. Choices are {}'.format(field, allowed_fields))
        shifts = [psi.read(positions, **kw_resampler) for psi in self.mesh_psi]
        shifts = jnp.column_stack(shifts)
        if field == 'disp':
            return shifts
        distance = jnp.sqrt(jnp.sum(positions**2, axis=-1, keepdims=True))
        if self.los is None:
            los = positions / distance
        else:
            los = self.los.astype(shifts.dtype)
        growth_rate = self.growth_rate(distance)
        rsd = growth_rate * (jnp.sum(shifts * los, axis=-1, keepdims=True) * los)
        if field == 'rsd':
            return rsd
        field == 'disp+rsd'
        shifts += rsd
        return shifts

    def read_shifted_positions(self, positions: jax.Array | ParticleField, field: str='disp+rsd', wrap: bool=False, **kwargs):
        """
        Read shifted positions i.e. the difference ``positions - self.read_shifts(positions, field=field)``.
        Output (and input) positions are wrapped if :attr:`wrap`.

        Parameters
        ----------
        positions : jax.Array
            Positions of shape (N, 3).
        field : str, default='disp+rsd'
            Apply either 'disp' (Zeldovich displacement), 'rsd' (RSD displacement), or 'disp+rsd' (Zeldovich + RSD displacement).

        Returns
        -------
        positions : jax.Array
            Shifted positions.
        """
        positions = _format_positions(positions)
        shifts = self.read_shifts(positions, field=field, **kwargs)
        positions = positions - shifts
        offset = self.boxcenter - self.boxsize / 2.
        if wrap: positions = (positions - offset) % self.mattrs.boxsize + offset
        return positions


class PlaneParallelFFTReconstruction(BaseReconstruction):

    def _run(self, mesh_delta):
        if self.los is None:
            raise ValueError('Provide a global line-of-sight')
        mesh_delta = (mesh_delta / self.bias()).r2c()
        kvec, ivec = self.mattrs.kcoords(sparse=True), self.mattrs.kcoords(kind='index', sparse=True)
        k2 = sum(kk**2 for kk in kvec)
        k2 = jnp.where(k2 == 0., 1., k2)
        beta = self.growth_rate() / self.bias()
        mesh_psis = []
        for axis in range(self.mattrs.ndim):
            mu2 = sum(kk * ll for kk, ll in zip(kvec, self.los))**2 / k2
            mesh_psi = 1j * kvec[axis] / k2 / (1. + beta * mu2) * mesh_delta
            # i = N / 2 is pure complex, we can remove it safely
            # ... and we have to, because it is turned to real when hermitian symmetry is assumed?
            if self.mattrs.hermitian: mesh_psi *= ivec[axis] != self.mattrs.meshsize[axis] // 2
            mesh_psis.append(mesh_psi.c2r())
            del mesh_psi
        self.mesh_psi = mesh_psis


class IterativeFFTReconstruction(BaseReconstruction):
    """
    Implementation of Burden et al. 2015 (https://arxiv.org/abs/1504.02591)
    field-level (as opposed to :class:`IterativeFFTParticleReconstruction`) algorithm.
    """

    def _run(self, mesh_delta, niterations=3):
        """
        Run reconstruction, i.e. compute Zeldovich displacement fields :attr:`mesh_psi`.

        Parameters
        ----------
        niterations : int, default=3
            Number of iterations.
        """
        mesh_delta_real = self.mesh_delta = mesh_delta / self.bias()
        for iter in range(niterations):
            mesh_delta_real = self._iterate(mesh_delta_real)
        del self.mesh_delta
        kvec, ivec = self.mattrs.kcoords(sparse=True), self.mattrs.kcoords(kind='index', sparse=True)
        k2 = sum(kk**2 for kk in kvec)
        k2 = jnp.where(k2 == 0., 1., k2)
        mesh_psis = []
        mesh_delta_k2 = mesh_delta_real.r2c() / k2
        for axis in range(self.mattrs.ndim):
            mesh_psi = 1j * kvec[axis] * mesh_delta_k2
            # i = N / 2 is pure complex, we can remove it safely
            # ... and we have to, because it is turned to real when hermitian symmetry is assumed?
            if self.mattrs.hermitian: mesh_psi *= ivec[axis] != self.mattrs.meshsize[axis] // 2
            mesh_psis.append(mesh_psi.c2r())
        self.mesh_psi = mesh_psis

    def _iterate(self, mesh_delta_real):
        init = mesh_delta_real is self.mesh_delta
        # This is an implementation of eq. 22 and 24 in https://arxiv.org/pdf/1504.02591.pdf
        # \delta_{g,\mathrm{real},n} is self.mesh_delta_real
        # \delta_{g,\mathrm{red}} is self.mesh_delta
        # First compute \delta(k) / k^{2} based on current \delta_{g,\mathrm{real},n} to estimate \phi_{\mathrm{est},n} (eq. 24)
        kvec, ivec = self.mattrs.kcoords(sparse=True), self.mattrs.kcoords(kind='index', sparse=True)
        k2 = sum(kk**2 for kk in kvec)
        k2 = jnp.where(k2 == 0., 1., k2)
        mesh_delta_k2 = mesh_delta_real.r2c() / k2
        del k2
        del mesh_delta_real
        # Now compute \beta \nabla \cdot (\nabla \phi_{\mathrm{est},n} \cdot \hat{r}) \hat{r}
        # In the plane-parallel case (self.los is a given vector), this is simply \beta IFFT((\hat{k} \cdot \hat{\eta})^{2} \delta(k))
        if self.los is not None:
            # Global los
            disp_deriv = mesh_delta_k2 * sum(kk * ll for kk, ll in zip(kvec, self.los))**2 # mesh_delta_k2 already divided by k^{2}
            factor = beta = self.growth_rate() / self.bias()
            if init:
                # Burden et al. 2015: 1504.02591, eq. 12 (flat sky approximation)
                factor = beta / (1. + beta)
            # Remove RSD part
            mesh_delta_real = self.mesh_delta - factor * disp_deriv.c2r()
            del disp_deriv
        else:
            # In the local los case, \beta \nabla \cdot (\nabla \phi_{\mathrm{est},n} \cdot \hat{r}) \hat{r} is:
            # \beta \partial_{i} \partial_{j} \phi_{\mathrm{est},n} \hat{r}_{j} \hat{r}_{i}
            # i.e. \beta IFFT(k_{i} k_{j} \delta(k) / k^{2}) \hat{r}_{i} \hat{r}_{j} => 6 FFTs
            xvec = self.mattrs.xcoords(sparse=True)
            x2 = sum(xx**2 for xx in xvec)
            beta = self.growth_rate() / self.bias()
            mesh_delta_real = self.mesh_delta
            for iaxis in range(mesh_delta_k2.ndim):
                for jaxis in range(iaxis, mesh_delta_k2.ndim):
                    mask = 1.
                    if self.mattrs.hermitian:
                        mask = (ivec[iaxis] != self.mattrs.meshsize[iaxis] // 2) & (ivec[jaxis] != self.mattrs.meshsize[jaxis] // 2)
                        mask |= (ivec[iaxis] == self.mattrs.meshsize[iaxis] // 2) & (ivec[jaxis] == self.mattrs.meshsize[jaxis] // 2)
                    disp_deriv = kvec[iaxis] * kvec[jaxis] * mesh_delta_k2 * mask  # delta_k already divided by k^{2}
                    disp_deriv = disp_deriv.c2r() * (xvec[iaxis] * xvec[jaxis]) / x2
                    factor = (1. + (iaxis != jaxis)) * beta  # we have j >= i and double-count j > i to account for j < i
                    if init:
                        # Burden et al. 2015: 1504.02591, eq. 12 (flat sky approximation)
                        factor /= (1. + beta)
                    # Remove RSD part
                    mesh_delta_real = mesh_delta_real - factor * disp_deriv
        return mesh_delta_real


class IterativeFFTParticleReconstruction(BaseReconstruction):

    def __init__(self, particles: ParticleField | FKPField, growth_rate: float | Callable=0., bias: float | Callable=1., los: str | np.ndarray=None, **kwargs):
        self._growth_rate = growth_rate
        self._bias = bias
        self.mattrs = particles.attrs
        self.los = _format_los(los=los)
        self._kw_resampler = {name: kwargs.pop(name) for name in ['resampler', 'halo_add'] if name in kwargs}
        self._run(particles, **{name: kwargs.pop(name) for name in ['niterations', 'smoothing_radius', 'threshold_randoms'] if name in kwargs})

    def _run(self, fkp, niterations=3, smoothing_radius=15., threshold_randoms=0.01):
        """
        Run reconstruction, i.e. compute reconstructed data real-space positions (:attr:`_positions_rec_data`)
        and Zeldovich displacements fields :attr:`mesh_psi`.

        Parameters
        ----------
        niterations : int, default=3
            Number of iterations.
        """
        # Gaussian smoothing before density contrast calculation
        data, randoms = fkp, None
        if isinstance(fkp, FKPField):
            data, randoms = fkp.data, fkp.randoms
        kw = self._kw_resampler | dict(interlacing=0)
        kernel = kernel_gaussian(self.mattrs, smoothing_radius=smoothing_radius)
        if randoms is not None:
            mesh_randoms = randoms.paint(**kw, out='complex')
            threshold_randoms = _get_threshold_randoms(randoms, threshold_randoms=threshold_randoms)
            mesh_randoms = (mesh_randoms * kernel).c2r()
            del randoms
        else:
            threshold_randoms, mesh_randoms = None, None
        self.data_rec = self.data = data
        for iter in range(niterations):
            mesh_delta = estimate_mesh_delta((self.data_rec.paint(**kw, out='complex') * kernel).c2r(), mesh_randoms=mesh_randoms, threshold_randoms=threshold_randoms, smoothing_radius=None)
            mesh_delta = mesh_delta / self.bias()
            self.data_rec, self.mesh_psi = self._iterate(mesh_delta, self.data, self.data_rec, return_psi=iter == niterations - 1)

    def _iterate(self, mesh_delta, data, data_rec, return_psi=False):
        init = data_rec is data
        kvec, ivec = self.mattrs.kcoords(sparse=True), self.mattrs.kcoords(kind='index', sparse=True)
        k2 = sum(kk**2 for kk in kvec)
        k2 = jnp.where(k2 == 0., 1., k2)

        mesh_delta_k2 = mesh_delta.r2c() / k2
        del k2

        shifts, mesh_psis = [], []
        for axis in range(mesh_delta_k2.ndim):
            # No need to compute psi on axis where los is 0
            if not return_psi and self.los is not None and self.los[axis] == 0:
                shifts.append(data_rec.positions[..., axis] * 0.)
                continue

            mesh_psi = 1j * kvec[axis] * mesh_delta_k2
            if self.mattrs.hermitian: mesh_psi *= ivec[axis] != self.mattrs.meshsize[axis] // 2
            mesh_psi = mesh_psi.c2r()
            # Reading shifts at reconstructed data real-space positions
            shifts.append(mesh_psi.read(data_rec.positions, **self._kw_resampler))
            if return_psi: mesh_psis.append(mesh_psi)
            del mesh_psi
        shifts = jnp.column_stack(shifts)
        distance = jnp.sqrt(jnp.sum(data.positions**2, axis=-1, keepdims=True))
        if self.los is None:
            los = data.positions / distance
        else:
            los = self.los
        # Comments in Julian's code:
        # For first loop need to approximately remove RSD component from psi to speed up convergence
        # See Burden et al. 2015: 1504.02591v2, eq. 12 (flat sky approximation)
        growth_rate = self.growth_rate(distance)
        if init:
            bias = self.bias(distance)
            beta = growth_rate / bias
            shifts -= beta / (1 + beta) * jnp.sum(shifts * los, axis=-1, keepdims=True) * los
        # Comments in Julian's code:
        # Remove RSD from original positions of galaxies to give new positions
        # these positions are then used in next determination of psi, assumed to not have RSD.
        # The iterative procedure then uses the new positions as if they'd been read in from the start
        data_rec = data_rec.clone(positions=data.positions - growth_rate * jnp.sum(shifts * los, axis=-1, keepdims=True) * los)
        return data_rec, mesh_psis

    def read_shifts(self, positions: jax.Array | ParticleField | str, field: str='disp+rsd', **kwargs):
        """
        Read displacement at input positions.

        Note
        ----
        Data shifts are read at the reconstructed real-space positions,
        while random shifts are read at the redshift-space positions, is that consistent?

        Parameters
        ----------
        positions : array of shape (N, 3), str
            Cartesian positions.
            Pass string 'data' to get the displacements for the input data positions passed to :meth:`assign_data`.
            Note that in this case, shifts are read at the reconstructed data real-space positions.

        field : str, default='disp+rsd'
            Either 'disp' (Zeldovich displacement), 'rsd' (RSD displacement), or 'disp+rsd' (Zeldovich + RSD displacement).

        Returns
        -------
        shifts : array of shape (N, 3)
            Displacements.
        """
        positions = _format_positions(positions)
        kw_resampler = self._kw_resampler | kwargs
        field = field.lower()
        allowed_fields = ['disp', 'rsd', 'disp+rsd']
        if field not in allowed_fields:
            raise ValueError('Unknown field {}. Choices are {}'.format(field, allowed_fields))

        def _read_shifts(positions):
            shifts = [psi.read(positions, **kw_resampler) for psi in self.mesh_psi]
            return jnp.column_stack(shifts)

        if isinstance(positions, str) and positions == 'data':
            # _positions_rec_data already wrapped during iteration
            shifts = _read_shifts(self.data_rec)
            if field == 'disp':
                return shifts
            rsd = self.data.positions - self.data_rec.positions
            if field == 'rsd':
                return rsd
            # field == 'disp+rsd'
            shifts += rsd
            return shifts

        shifts = _read_shifts(positions)  # aleady wrapped

        if field == 'disp':
            return shifts

        distance = jnp.sqrt(jnp.sum(positions**2, axis=-1, keepdims=True))
        if self.los is None:
            los = positions / distance
        else:
            los = self.los.astype(shifts.dtype)
        growth_rate = self.growth_rate(distance)
        rsd = growth_rate * (jnp.sum(shifts * los, axis=-1, keepdims=True) * los)
        if field == 'rsd':
            return rsd
        # field == 'disp+rsd'
        # we follow convention of original algorithm: remove RSD first,
        # then remove Zeldovich displacement
        real_positions = positions - rsd
        shifts = _read_shifts(real_positions)
        return shifts + rsd

    def read_shifted_positions(self, positions: jax.Array | ParticleField | str, field='disp+rsd', wrap: bool=False):
        """
        Read shifted positions i.e. the difference ``positions - self.read_shifts(positions, field=field)``.
        Output (and input) positions are wrapped if :attr:`wrap`.

        Parameters
        ----------
        positions : array of shape (N, 3), string
            Cartesian positions.
            Pass string 'data' to get the shift positions for the input data positions passed to :meth:`assign_data`.
            Note that in this case, shifts are read at the reconstructed data real-space positions.

        field : string, default='disp+rsd'
            Apply either 'disp' (Zeldovich displacement), 'rsd' (RSD displacement), or 'disp+rsd' (Zeldovich + RSD displacement).

        Returns
        -------
        positions : array of shape (N, 3)
            Shifted positions.
        """
        positions = _format_positions(positions)
        shifts = self.read_shifts(positions, field=field, position_type='pos', mpiroot=None)
        if isinstance(positions, str) and positions == 'data':
            positions = self.data.positions
        positions = positions - shifts
        offset = self.mattrs.boxcenter - self.mattrs.boxsize / 2.
        if wrap: positions = (positions - offset) % self.mattrs.boxsize + offset
        return positions