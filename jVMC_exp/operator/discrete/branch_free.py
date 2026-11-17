from abc import abstractmethod
import numpy as np
import jax.numpy as jnp
import inspect
import jax

from jVMC_exp.operator.discrete.base import Operator as BaseOperator
from jVMC_exp import global_defs
from jVMC_exp.sharding_config import MESH

def _has_kwargs(fun):
    """Whether ``fun`` accepts arbitrary keyword arguments."""
    sig = inspect.signature(fun)
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values())

def _spin_index(spin):
    """Return ``spin`` as 0 (up) or 1 (down), accepting ``"up"`` and ``"down"`` and their abbreviations."""
    if spin is not None:
        if isinstance(spin, str):
            spin = spin.lower()
            if spin in {"up", "u", "↑"}:
                return 0
            if spin in {"down", "dn", "d", "↓"}:
                return 1
            raise ValueError(f"spin must be 0/1 or 'up'/'down'. Got {spin}")
        
        spin = int(spin)
        if spin not in (0, 1):
            raise ValueError(f"spin must be 0/1 or 'up'/'down'. Got {spin}")
    
    return spin

class OperatorString(list):
    """
    A list of operators to be applied sequentially, with an associated scale factor.
    """
    def __init__(self, operators):
        super().__init__(operators)
        self._scale = [1]
        self._diagonal = False

    @property
    def diagonal(self):
        """Whether the operator string is diagonal."""
        return self._diagonal

class Operator(BaseOperator):
    """
    Operators assembled from branch-free elementary operators.

    An elementary operator acts on the local state of a single site and is branch-free: it maps \
    each local state ``j`` to a single local state ``map[j]``, with matrix element ``mat_els[j]``. \
    When the operator is compiled, products and sums of elementary operators are expanded into a \
    sum of operator strings, each of which maps a configuration to a single connected configuration.

    Sites are indexed in the flattened configuration, ``s.ravel()``. Fermionic operators get the \
    sign of a Jordan-Wigner string over all the sites with a larger index. For spinful fermions, \
    with local dimension 4, the local states 0, 1, 2 and 3 stand for an empty site, a spin up, a \
    spin down and a doubly occupied site, and the modes are ordered as \
    (L-1, down), (L-1, up), ..., (0, down), (0, up).

    Args:
        * ``ldim``: Local Hilbert space dimension.
        * ``idx``: Site the operator acts on.
        * ``diag``: Whether the operator is diagonal.
        * ``fermionic``: Whether the operator is fermionic.
        * ``spin``: Spin the operator acts on, for spinful fermions: ``0`` or ``"up"``, ``1`` or \
          ``"down"``. ``None`` otherwise.
    """
    def __init__(
            self, ldim, idx, diag, 
            fermionic: bool=False, 
            spin: str | int | None=None
        ):
        super().__init__(ldim)

        self._idx = idx
        self._diag = diag
        self._site_ldim = {idx: ldim} if ldim is not None else {}

        self._fermionic = fermionic
        self._spin = _spin_index(spin) 

    @property
    def idx(self):
        """Site the operator acts on, ``None`` for combined operators."""
        return self._idx
    
    @property
    def fermionic(self):
        """Whether the operator is fermionic."""
        return self._fermionic
    
    @property
    def spin(self):
        """Spin the operator acts on, 0 (up) or 1 (down), or ``None``."""
        return self._spin
    
    @property
    def diag(self):
        """Whether the operator is diagonal."""
        return self._diag

    @property
    def n_conn(self):
        if not self._is_compiled:
            self._compile()

        return self._n_conn
    
    @property
    @abstractmethod
    def mat_els(self):
        """Matrix elements ``mat_els[j]`` of the local states ``j``."""
        pass
    
    @property
    @abstractmethod
    def map(self):
        """Images ``map[j]`` of the local states ``j``."""
        pass
        
    def _get_list_of_strings(self):
        """
        Flatten the operator tree into a list of operator strings.
        Each operator string is an OperatorString (list subclass) of leaf operators 
        to be applied sequentially, with a scale attribute.
        
        Returns: List of OperatorString objects
        """
        strings_stack = []
        op_stack = [(self, False)]

        while op_stack:
            node, visited = op_stack.pop()

            if not visited:
                op_stack.append((node, True))

                if isinstance(node, CompositeOperator):
                    op_stack.append((node.O_2, False))
                    op_stack.append((node.O_1, False))

                elif isinstance(node, ScaledOperator):
                    op_stack.append((node.O, False))

            else:
                if isinstance(node, CompositeOperator):
                    right = strings_stack.pop()
                    left = strings_stack.pop()

                    if node._label == 'sum':
                        # Sum: concatenate the two lists of operator strings
                        strings_stack.append(left + right)
                    
                    elif node._label == 'mul':
                        # Product: create all combinations (distributive law)
                        out = []
                        for left_string in left:
                            for right_string in right:
                                # Combine the operators from both strings
                                combined = OperatorString(left_string + right_string)
                                # Multiply the prefactors
                                combined._scale.extend(left_string._scale)
                                combined._scale.extend(right_string._scale)
                                out.append(combined)
                        
                        strings_stack.append(out)

                elif isinstance(node, ScaledOperator):
                    lst = strings_stack.pop()
                    # Multiply the scale of each operator string
                    for s in lst:
                        s._scale.append(node.scalar)
                    strings_stack.append(lst)

                else:
                    # Leaf operator: wrap in an OperatorString
                    op_string = OperatorString([node])
                    strings_stack.append([op_string])        
        
        return strings_stack[0]

    def _compile(self):
        strings = self._get_list_of_strings()
        max_length = max(len(s) for s in strings)
        max_ldim = max(self._site_ldim.values())
        
        # Create identity operator for padding
        IdOp = IdentityOperator(max_ldim, 0)
        
        idxC = []
        mapC = []
        matElsC = []
        fermionicC = []
        spinC = []
        spinfulC = []
        diagonal = []
        constant_prefactors = np.ones(len(strings), dtype=global_defs.DT_OPERATORS_CPX)
        callable_prefactors = {}

        for j, op_string in enumerate(strings):
            idx_row = []
            map_row = []
            matels_row = []
            fermionic_row = []
            spin_row = []
            spinful_row = []
            d = 1
            
            n = len(op_string)
            for k in range(max_length):
                k_rev = n - 1 - k
                if k_rev >= 0:
                    op = op_string[k_rev]
                    idx_row.append(op.idx)
                    map_row.append(np.pad(op.map, (0, max_ldim - op.ldim)))
                    matels_row.append(np.pad(op.mat_els, (0, max_ldim - op.ldim)))
                    fermionic_row.append(int(op.fermionic))
                    spin_row.append(op.spin if op.spin is not None else 0)
                    spinful_row.append(1 if op.spin is not None else 0)
                    d *= op.diag
                else:
                    idx_row.append(IdOp.idx)
                    map_row.append(IdOp.map)
                    matels_row.append(IdOp.mat_els)
                    fermionic_row.append(0)
                    spin_row.append(0)
                    spinful_row.append(0)
            
            idxC.append(idx_row)
            mapC.append(map_row)
            matElsC.append(matels_row)
            fermionicC.append(fermionic_row)
            spinC.append(spin_row)
            spinfulC.append(spinful_row)

            for scale in op_string._scale:
                if callable(scale):
                    # A function is evaluated once per call, however many strings it multiplies
                    _, powers = callable_prefactors.setdefault(
                        id(scale), (scale, np.zeros(len(strings), dtype=np.int32))
                    )
                    powers[j] += 1
                else:
                    constant_prefactors[j] *= scale
            diagonal.append(d)
        
        self.idxC = np.array(idxC, dtype=np.int32)
        self.mapC = np.array(mapC, dtype=np.int32)
        self.matElsC = np.array(matElsC)
        self.fermionicC = np.array(fermionicC, dtype=np.int32)
        self.spinC = np.array(spinC, dtype=np.int32)
        self.spinfulC = np.array(spinfulC, dtype=np.int32)
        self.diagC = np.array(diagonal, dtype=np.bool_)
        self.nondiagC = ~self.diagC
        self.firstC = (
            np.argmax(
                self.idxC[:, :, None] == self.idxC[:, None, :], axis=2
            ) == np.arange(max_length)
        )
        self._constant_prefactors = constant_prefactors
        self._callable_prefactors = list(callable_prefactors.values())
        self._n_conn = sum(not d for d in diagonal)
        self._is_compiled = True

    def _get_conn_elements(self, s, kwargs):
        sampleShape = s.shape
        s = s.ravel()
        dim = s.shape[0]

        prefactors = jnp.asarray(self._constant_prefactors)
        for fun, powers in self._callable_prefactors:
            value = jnp.asarray(fun(**kwargs), dtype=global_defs.DT_OPERATORS_CPX)
            for p in range(1, powers.max() + 1):
                prefactors = prefactors * jnp.where(powers >= p, value, 1)

        s_bits = s.astype(jnp.int32)
        up_s = jnp.bitwise_and(s_bits, 1)
        down_s = jnp.bitwise_and(jnp.right_shift(s_bits, 1), 1)
        right_up = jnp.cumsum(up_s[::-1])[::-1] - up_s
        right_total = jnp.cumsum((up_s + down_s)[::-1])[::-1] - up_s - down_s

        def process_string(s, prefactor, idx, map, matEls, fermionic, spin, spinful, first):
            """
            Apply one operator string to ``s``. Spinful fermion signs use the
            ordering (L-1, down), (L-1, up), ..., (0, down), (0, up), i.e.
            all modes on larger site indices first and down before up onsite.
            """
            # Sites of the whole string; idx is the site of a single operator in apply_operator
            string_sites = idx

            def apply_operator(c, x):
                carry_sample, carry_matEl = c
                idx, map, matEl, fermionic, spin, spinful = x

                sample_bits = carry_sample.astype(jnp.int32)

                string_bits = sample_bits[string_sites]
                up_occ = jnp.bitwise_and(string_bits, 1)
                down_occ = jnp.bitwise_and(jnp.right_shift(string_bits, 1), 1)
                changed = first & (string_sites > idx)

                spinless_count = right_up[idx] + jnp.sum(
                    jnp.where(changed, up_occ - up_s[string_sites], 0)
                )
                spinless_parity = jnp.bitwise_and(spinless_count, 1)

                right_count = right_total[idx] + jnp.sum(
                    jnp.where(
                        changed, 
                        up_occ + down_occ - up_s[string_sites] - down_s[string_sites], 
                        0
                    )
                )
                down_occ_idx = jnp.bitwise_and(jnp.right_shift(sample_bits[idx], 1), 1)
                onsite_count = jnp.where(spin == 0, down_occ_idx, 0)
                spinful_parity = jnp.bitwise_and(right_count + onsite_count, 1)

                sign = (
                    1 - 2 * fermionic * (spinful * spinful_parity + (1 - spinful) * spinless_parity)
                ).astype(global_defs.DT_OPERATORS_CPX)
                carry_matEl_new = carry_matEl * matEl[carry_sample[idx]] * sign
                # A select instead of a scatter, so that XLA can fuse the steps of the unrolled scan
                carry_sample_new = jnp.where(
                    jnp.arange(dim) == idx, map[carry_sample[idx]], carry_sample
                )

                return (carry_sample_new, carry_matEl_new), None
            
            prefactor = jax.lax.pcast(prefactor, MESH.axis_names, to='varying')
            (s_p, matEl), _ = jax.lax.scan(
                apply_operator, 
                (s, prefactor), 
                (idx, map, matEls, fermionic, spin, spinful),
                unroll=True
            )

            return s_p.reshape(sampleShape), matEl
        
        s_p, mat_els = jax.vmap(process_string, in_axes=(None,) + (0,) * 8)(
            s, prefactors, self.idxC, self.mapC, self.matElsC,
            self.fermionicC, self.spinC, self.spinfulC, self.firstC
        )
        
        mat_el_diag = jnp.sum(mat_els[self.diagC])
        s_p_nondiag = s_p[self.nondiagC]
        mat_els_nondiag = mat_els[self.nondiagC]

        return s_p_nondiag, mat_els_nondiag, mat_el_diag
    
    @classmethod
    def _create_composite(cls, O_1, O_2, label):
        return CompositeOperator(O_1, O_2, label)
    
    @classmethod
    def _create_scaled(cls, O, scalar):
        return ScaledOperator(O, scalar)
    
class CompositeOperator(Operator):
    """
    Sum (``label="sum"``) or product (``label="mul"``) of two operators, as created by ``+`` and
    ``*``. The local dimensions of the two operators must agree on the sites they share.
    """
    def __init__(self, O_1: Operator, O_2: Operator, label: str):
        for site in O_1._site_ldim.keys():
            if (site in O_2._site_ldim.keys()) and (O_1._site_ldim[site] != O_2._site_ldim[site]):
                raise ValueError(f"Can't combine two operators with different local dimensions. "
                                 f"Got ldim={O_1._site_ldim[site]} and ldim={O_2._site_ldim[site]} at site {site}")
        super().__init__(None, None, None, None)

        self._site_ldim = {**O_1._site_ldim, **O_2._site_ldim}
        self.O_1 = O_1
        self.O_2 = O_2
        self._label = label

    @property
    def mat_els(self):
        pass
    
    @property
    def map(self):
        pass

class ScaledOperator(Operator):
    """
    Operator ``O`` multiplied by ``scalar``, a number or a callable ``f(**kwargs)``, as created by
    ``*`` and ``/``.
    """
    def  __init__(self, O: Operator, scalar):
        if callable(scalar) and (not _has_kwargs(scalar)):
            raise ValueError(
                'Any callable that multiplies an operator has to have **kwargs in its argument.'
            )

        super().__init__(None, None, None, None)
        self._site_ldim = O._site_ldim
        self.scalar = scalar
        self.O = O

    @property
    def mat_els(self):
        pass
    
    @property
    def map(self):
        pass

class IdentityOperator(Operator):
    """Identity on site ``idx``, with local dimension ``ldim``. Its multiples add constants to operators."""
    def __init__(self, ldim, idx):
        super().__init__(ldim, idx, True, False)

    @property
    def mat_els(self):
        return np.ones(self.ldim, dtype=global_defs.DT_OPERATORS_CPX)
    
    @property
    def map(self):
        return np.arange(self.ldim, dtype=np.int32)
  
class _Creation(Operator):
    def __init__(self, idx, fermionic):
        super().__init__(2, idx, False, fermionic)

    @property
    def mat_els(self):
        return np.array([1, 0], dtype=global_defs.DT_OPERATORS_CPX)
    
    @property
    def map(self):
        return np.array([1, 0], dtype=np.int32)

class _Annihilation(Operator):
    def __init__(self, idx, fermionic):
        super().__init__(2, idx, False, fermionic)

    @property
    def mat_els(self):
        return np.array([0, 1], dtype=global_defs.DT_OPERATORS_CPX)
    
    @property
    def map(self):
        return np.array([0, 0], dtype=np.int32)
    
class SigmaX(Operator):
    """Pauli operator :math:`\\sigma^x` on site ``idx``."""
    def __init__(self, idx):
        super().__init__(2, idx, False)

    @property
    def mat_els(self):
        return np.ones(self.ldim, dtype=global_defs.DT_OPERATORS_CPX)
    
    @property
    def map(self):
        return np.array([1, 0], dtype=np.int32)

class SigmaY(Operator):
    """Pauli operator :math:`\\sigma^y` on site ``idx``."""
    def __init__(self, idx):
        super().__init__(2, idx, False)

    @property
    def mat_els(self):
        return np.array([1j, -1j], dtype=global_defs.DT_OPERATORS_CPX)
    
    @property
    def map(self):
        return np.array([1, 0], dtype=np.int32)
    
class SigmaZ(Operator):
    """Pauli operator :math:`\\sigma^z` on site ``idx``: +1 for the local state 0, and -1 for 1."""
    def __init__(self, idx):
        super().__init__(2, idx, True)

    @property
    def mat_els(self):
        return np.array([1, -1], dtype=global_defs.DT_OPERATORS_CPX)
    
    @property
    def map(self):
        return np.arange(self.ldim, dtype=np.int32)
    
class SigmaPlus(_Annihilation):
    """Spin raising operator :math:`\\sigma^+=(\\sigma^x+i\\sigma^y)/2` on site ``idx``, which maps the local state 1 to 0."""
    def __init__(self, idx):
        super().__init__(idx, False)

class SigmaMinus(_Creation):
    """Spin lowering operator :math:`\\sigma^-=(\\sigma^x-i\\sigma^y)/2` on site ``idx``, which maps the local state 0 to 1."""
    def __init__(self, idx):
        super().__init__(idx, False)
    
class _Number(Operator):
    def __init__(self, idx):
        super().__init__(2, idx, True, False)

    @property
    def mat_els(self):
        return np.array([0, 1], dtype=global_defs.DT_OPERATORS_CPX)

    @property
    def map(self):
        return np.array([0, 1], dtype=np.int32)

class _SpinfulCreation(Operator):
    def __init__(self, idx, spin):
        super().__init__(4, idx, False, True, spin)

    @property
    def mat_els(self):
        if self.spin == 0:
            return np.array([1, 0, 1, 0], dtype=global_defs.DT_OPERATORS_CPX)
        return np.array([1, 1, 0, 0], dtype=global_defs.DT_OPERATORS_CPX)

    @property
    def map(self):
        if self.spin == 0:
            return np.array([1, 1, 3, 3], dtype=np.int32)
        return np.array([2, 3, 2, 3], dtype=np.int32)

class _SpinfulAnnihilation(Operator):
    def __init__(self, idx, spin):
        super().__init__(4, idx, False, True, spin)

    @property
    def mat_els(self):
        if self.spin == 0:
            return np.array([0, 1, 0, 1], dtype=global_defs.DT_OPERATORS_CPX)
        return np.array([0, 0, 1, 1], dtype=global_defs.DT_OPERATORS_CPX)

    @property
    def map(self):
        if self.spin == 0:
            return np.array([0, 0, 2, 2], dtype=np.int32)
        return np.array([0, 1, 0, 1], dtype=np.int32)

class _SpinfulNumber(Operator):
    def __init__(self, idx, spin):
        super().__init__(4, idx, True, False, spin)

    @property
    def mat_els(self):
        if self.spin == 0:
            return np.array([0, 1, 0, 1], dtype=global_defs.DT_OPERATORS_CPX)
        return np.array([0, 0, 1, 1], dtype=global_defs.DT_OPERATORS_CPX)

    @property
    def map(self):
        return np.arange(self.ldim, dtype=np.int32)

def Creation(idx, spin=None):
    """
    Fermionic creation operator :math:`\\hat c^\\dagger` on site ``idx``.

    Without ``spin``, it creates a spinless fermion, on a site with local dimension 2. With \
    ``spin``, ``"up"`` or ``"down"``, it creates a spinful one, on a site with local dimension 4.
    """
    if spin is None:
        return _Creation(idx, True)
    return _SpinfulCreation(idx, spin)

def Annihilation(idx, spin=None):
    """
    Fermionic annihilation operator :math:`\\hat c` on site ``idx``, for a spinless fermion or, \
    with ``spin``, a spinful one, as in ``Creation()``.
    """
    if spin is None:
        return _Annihilation(idx, True)
    return _SpinfulAnnihilation(idx, spin)

def Number(idx, spin=None):
    """
    Occupation number :math:`\\hat n=\\hat c^\\dagger\\hat c` on site ``idx``, for a spinless fermion \
    or, with ``spin``, a spinful one, as in ``Creation()``.
    """
    if spin is None:
        return _Number(idx)
    return _SpinfulNumber(idx, spin)