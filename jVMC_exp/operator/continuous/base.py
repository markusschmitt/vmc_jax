import jax.numpy as jnp
from abc import abstractmethod
import inspect

from jVMC_exp.geometry import AbstractGeometry
from jVMC_exp.vqs import NQS
from jVMC_exp.sharding_config import sharded
from jVMC_exp.operator.base import AbstractOperator

def _has_kwargs(fun):
    """Whether ``fun`` accepts arbitrary keyword arguments."""
    sig = inspect.signature(fun)
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values())

class Operator(AbstractOperator):
    """
    Base class of the operators acting on continuous particle configurations.

    A configuration holds the positions of all the particles, particle by particle, with shape
    ``(n_particles * n_dim,)``. Operators are combined with ``+``, ``-``, ``*`` and ``/``, and
    numbers are added as multiples of the identity. Products of two operators are only supported
    when both are diagonal. Operators can be multiplied by numbers, or by callables that accept
    arbitrary keyword arguments, ``f(**kwargs)``, which are evaluated with the keyword arguments
    given to ``get_O_loc()``.

    Args:
        * ``geometry``: Geometry of the system. Combined operators must share it.
        * ``is_diagonal``: Whether the operator is diagonal in the positions.
    """
    def __init__(self, geometry: AbstractGeometry, is_diagonal: bool):
        if not isinstance(geometry, AbstractGeometry):
            raise TypeError(f"geometry must be an AbstractGeometry, got {type(geometry)}")
        self.geometry = geometry
        self._is_flattened = False
        self._is_diagonal = is_diagonal

    @property
    def is_diagonal(self):
        """Whether the operator is diagonal in the positions."""
        return self._is_diagonal
    
    def __add__(self, other):
        if isinstance(other, (int, float, complex)):
            return SumOperator(self, ScaledOperator(IdentityOperator(self.geometry), other))
        elif isinstance(other, Operator):
            return SumOperator(self, other)
        else:
            return NotImplemented
        
    def __radd__(self, other):
        if isinstance(other, (int, float, complex)):
            return SumOperator(ScaledOperator(IdentityOperator(self.geometry), other), self)
        else:
            return NotImplemented
        
    def __neg__(self):
        return ScaledOperator(self, -1)
    
    def __sub__(self, other):
        if isinstance(other, (int, float, complex)):
            return SumOperator(self, ScaledOperator(IdentityOperator(self.geometry), -other))
        elif isinstance(other, Operator):
            return SumOperator(self, -other)
        else:
            return NotImplemented
        
    def __rsub__(self, other):
        if isinstance(other, (int, float, complex)):
            return SumOperator(ScaledOperator(IdentityOperator(self.geometry), other), -self)
        else:
            return NotImplemented
        
    def __mul__(self, other):
        if isinstance(other, (int, float, complex)) or callable(other):
            return ScaledOperator(self, other)
        elif isinstance(other, Operator):
            if self.is_diagonal and other.is_diagonal:
                return MulOperator(self, other)
            else:
                raise NotImplementedError("Multiplication only implemented for diagonal operators.")
        else:
            return NotImplemented
        
    def __truediv__(self, other):
        if isinstance(other, (int, float, complex)):
            if other == 0:
                raise ZeroDivisionError("Division of Operator by zero.")
            return ScaledOperator(self, 1 / other)
        else:
            return NotImplemented
        
    def __rmul__(self, other):
        if isinstance(other, (int, float, complex)) or callable(other):
            return ScaledOperator(self, other)
        else:
            return NotImplemented

    def _flatten_tree(self, apply_fun):
        """Store the operator tree as a list of instructions in postfix order."""
        instructions = []
        stack = [(self, False)]  # (node, visited)

        while stack:
            node, visited = stack.pop()

            if visited:
                if isinstance(node, SumOperator):
                    instructions.append(("ADD", None))
                elif isinstance(node, MulOperator):
                    instructions.append(("MUL", None))
                elif isinstance(node, ScaledOperator):
                    instructions.append(("SCALE", node.scalar))
            else:
                if isinstance(node, SumOperator) or isinstance(node, MulOperator):
                    stack += [(node, True), (node.O_1, False), (node.O_2, False)]
                elif isinstance(node, ScaledOperator):
                    stack += [(node, True), (node.O, False)]
                else:
                    instructions.append(("LEAF", node))

        self._instructions = instructions
        self._is_flattened = True

    @sharded(static_kwarg_names=('apply_fun',))
    def _evaluate_flat(self, s, *, batch_size, parameters, apply_fun, **kwargs):
        """Evaluate the instructions on a single configuration."""
        stack = []

        for op, arg in self._instructions:
            if op == "LEAF":
                stack.append(arg._get_O_loc(s, apply_fun, parameters, kwargs))
            elif op == "SCALE":
                scale = arg(**kwargs) if callable(arg) else arg
                stack[-1] = scale * stack[-1]
            elif op == "ADD":
                a = stack.pop()
                b = stack.pop()
                stack.append(a + b)
            elif op == "MUL":
                a = stack.pop()
                b = stack.pop()
                stack.append(a * b)
        
        return stack[0]

    def get_O_loc(self, s, psi: NQS, *, logPsiS=None, **kwargs):
        """
        Compute the local estimator of the operator, see ``AbstractOperator.get_O_loc()``.
        ``logPsiS`` is not needed.
        """
        if not self._is_flattened:
            self._flatten_tree(psi.apply_fun)
            
        return self._evaluate_flat(
            s, 
            batch_size=psi.batchSize, 
            parameters=psi.eval_parameters, 
            apply_fun=psi.apply_fun, 
            **kwargs
            )
        
    @abstractmethod
    def _get_O_loc(self, s, apply_fun, parameters, kwargs):
        """
        Implement (O 𝜓)(s) / 𝜓(s) for a single configuration ``s``, where 𝜓 is evaluated as
        ``apply_fun(parameters, s)`` and ``kwargs`` holds the keyword arguments of ``get_O_loc()``.
        """
        pass

class IdentityOperator(Operator):
    def __init__(self, geometry):
        super().__init__(geometry, True)

    def _get_O_loc(self, s, apply_fun, parameters, kwargs):
        return jnp.asarray(1.0)

class SumOperator(Operator):
    """Sum of two operators, as created by ``+`` and ``-``."""
    def __init__(self, O_1: Operator, O_2: Operator):    
        if O_1.geometry != O_2.geometry:
            raise ValueError("Operators must share the same geometry.")
        super().__init__(O_1.geometry, O_1.is_diagonal and O_2.is_diagonal)
        self.O_1 = O_1
        self.O_2 = O_2

    def _get_O_loc(self, s, apply_fun, parameters, kwargs):
        pass
    
class ScaledOperator(Operator):
    """Operator ``O`` multiplied by ``scalar``, a number or a callable ``f(**kwargs)``."""
    def  __init__(self, O: Operator, scalar):
        if not (isinstance(scalar, (int, float, complex)) or callable(scalar)):
            raise ValueError('The second element has to be a scalar.')
        if callable(scalar) and (not _has_kwargs(scalar)):
            raise ValueError('Any callable that multiplies an operator has to have **kwargs in its argument.')

        super().__init__(O.geometry, O.is_diagonal)
        self.scalar = scalar
        self.O = O

    def _get_O_loc(self, s, apply_fun, parameters, kwargs):
        pass
    
class MulOperator(Operator):
    """Product of two diagonal operators, as created by ``*``."""
    def __init__(self, O_1: Operator, O_2: Operator):    
        if O_1.geometry != O_2.geometry:
            raise ValueError("Operators must share the same geometry.")
        super().__init__(O_1.geometry, is_diagonal=True)
        self.O_1 = O_1
        self.O_2 = O_2

    def _get_O_loc(self, s, apply_fun, parameters, kwargs):
        pass