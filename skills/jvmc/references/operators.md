# Operators and Hamiltonians

```python
import numpy as np
import jax.numpy as jnp
import jVMC
import jVMC.operator.discrete as op
```

## Discrete operators (spins and fermions)

All discrete operators live in `jVMC.operator.discrete` and act on one site `idx`. They combine
with `+`, `-`, `*` (operator product, applied right to left), `/` by a scalar, and multiplication
by scalars. Start a sum from `0` or from the first term.

| Operator | Local dimension | Meaning |
|---|---|---|
| `SigmaX(i)`, `SigmaY(i)`, `SigmaZ(i)` | 2 | Pauli matrices; local state 0 has σᶻ = +1 |
| `SigmaPlus(i)`, `SigmaMinus(i)` | 2 | σ± = (σˣ ± iσʸ)/2 |
| `IdentityOperator(l_dim, i)` | any | identity |
| `Creation(i)`, `Annihilation(i)`, `Number(i)` | 2 | spinless fermions, with the Jordan–Wigner string |
| `Creation(i, "up")`, `Annihilation(i, "down")`, `Number(i, spin)` | 4 | spinful fermions; local states 0 = empty, 1 = ↑, 2 = ↓, 3 = ↑↓ |

Operators with different local dimensions on the same site cannot be combined (`ValueError`), so
do not mix Pauli and spinful-fermion operators.

Heisenberg chain and a 2D lattice:

```python
L = 6
H_heis = 0
for i in range(L):
    j = (i + 1) % L
    H_heis += op.SigmaX(i) * op.SigmaX(j) + op.SigmaY(i) * op.SigmaY(j) + op.SigmaZ(i) * op.SigmaZ(j)

# 2D: configurations have shape (Ly, Lx), operators use the flattened index y * Lx + x
Lx, Ly = 3, 2
def site(x, y):
    return y * Lx + x

bonds = set()
for y in range(Ly):
    for x in range(Lx):
        bonds.add(tuple(sorted((site(x, y), site((x + 1) % Lx, y)))))
        bonds.add(tuple(sorted((site(x, y), site(x, (y + 1) % Ly)))))
H_2d = sum(-op.SigmaZ(i) * op.SigmaZ(j) for i, j in sorted(bonds)) - sum(op.SigmaX(i) for i in range(Lx * Ly))
```

Fermi–Hubbard chain (spinful fermions, local dimension 4):

```python
t, U, mu = 1.0, 4.0, 2.0
H_hub = 0
for i in range(L):
    H_hub += U * op.Number(i, "up") * op.Number(i, "down") - mu * (op.Number(i, "up") + op.Number(i, "down"))
    if i < L - 1:
        for s in ("up", "down"):
            H_hub += -t * (op.Creation(i, s) * op.Annihilation(i + 1, s) + op.Creation(i + 1, s) * op.Annihilation(i, s))
# sample with ExactSampler(psi, l_dim=4) on small systems
```

### Time-dependent and parametrized coefficients

A coefficient can be a callable `f(t, **kwargs)`. The keyword arguments are passed to the
evaluation: `TDVP.time_evolution` passes the current time `t` automatically, and explicit calls
take them as keywords. Changing them does not trigger recompilation.

```python
f = lambda t, **kwargs: jnp.cos(t)
O = op.SigmaZ(0) * (op.SigmaX(0) + 2 * op.SigmaY(1)) + f * op.SigmaZ(0) * op.SigmaZ(1)

psi = jVMC.vqs.NQS(jVMC.nets.CpxRBM(num_hidden=2), 2, batch_size=4, seed=0)
exact = jVMC.sampler.ExactSampler(psi)
print(O.to_dense(exact.basis, t=0.25 * jnp.pi))   # dense matrix at t = pi/4
print(exact(O, t=0.3).mean)                        # expectation value at t = 0.3
```

### Matrices and checks

* `O.to_sparse(basis, **kwargs)` returns a `scipy.sparse.csr_matrix` and `O.to_dense(basis)` a
  dense matrix, in the order of `basis` (e.g. `ExactSampler(psi).basis`). Use these for exact
  diagonalization of small systems.
* `O.get_conn_elements(s, batch_size)` returns the connected configurations s' and the matrix
  elements ⟨s|O|s'⟩. `O.get_O_loc(s, psi)` returns the local estimator. Usually `sampler(O)`
  is all you need.
* `jVMC.util.matrix_to_jvmc_operator(matrix, sites, return_conjugate=False)` converts a dense
  2×2 or 4×4 matrix on one or two spin-1/2 sites into an operator. It returns the operator of
  `matrix.conj().T` too if `return_conjugate=True`, which `Infidelity` needs.

```python
U2 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)   # SWAP
swap, swap_dag = jVMC.util.matrix_to_jvmc_operator(U2, (0, 1), return_conjugate=True)
```

### Custom branch-free operators

An operator is branch-free if it maps every local state to exactly one local state. Subclass
`jVMC.operator.discrete.Operator` and give the map and the matrix elements per local state:

```python
from jVMC import global_defs

class MySigmaY(op.Operator):
    def __init__(self, idx):
        super().__init__(l_dim=2, idx=idx, diag=False, fermionic=False, spin=None)

    @property
    def mat_els(self):            # <map[s]| O |s> for s = 0, 1
        return np.array([1j, -1j], dtype=global_defs.DT_OPERATORS_CPX)

    @property
    def map(self):                # local state s -> map[s]
        return np.array([1, 0], dtype=np.int32)

assert np.allclose(MySigmaY(0).to_dense(exact.basis), op.SigmaY(0).to_dense(exact.basis))
```

## Continuous operators (particles in space)

`jVMC.operator.continuous` acts on positions of shape `(n_particles * n_dim,)` and needs a
geometry. `jVMC.geometry.HyperRectangle(n_particles, n_dim, PBC=..., extent=...)`
describes the box. With `PBC=False` the extent only sets where the Markov chains start and the
default proposal step. With periodic directions, distances use the minimum-image convention.

| Operator | Kind |
|---|---|
| `TotalKineticOperator(geometry, mass=1)` | −Σᵢ ∇ᵢ²/(2m), with automatic differentiation |
| `PotentialOperator(geometry, potential)` | any `potential(x)` of the whole configuration |
| `CoulombInteraction(geometry, charge=None)` | Σ_{i<j} qᵢqⱼ/\|rᵢ − rⱼ\| |
| `ParticleDensity(geometry, linear_partition)`, `Monopole(geometry)` | diagonal observables |

```python
import jVMC.operator.continuous as cop

N, n_dim, omega = 2, 2, 1.0
geometry = jVMC.geometry.HyperRectangle(N, n_dim, PBC=False, extent=(6.0, 6.0))
H_dot = (
    cop.TotalKineticOperator(geometry)
    + cop.PotentialOperator(geometry, lambda x: 0.5 * omega**2 * jnp.sum(x**2))
    + cop.CoulombInteraction(geometry)
)
# sampler: jVMC.sampler.MCSamplerCont(psi, jVMC.propose.RWM(geometry), ...), see sampling.md
```

A custom continuous operator subclasses `jVMC.operator.continuous.Operator(geometry,
is_diagonal)` and implements `_get_O_loc(self, s, apply_fun, parameters, kwargs)` for one
configuration. A diagonal operator just returns a function of `s`; a non-diagonal one can
differentiate `apply_fun(parameters, s)` (log ψ) with JAX. See `PotentialOperator` and
`TotalKineticOperator` in the source. The complete quantum-dot example with a
Slater–Jastrow–backflow network is `examples/2dho_ground_state_search.ipynb`.
