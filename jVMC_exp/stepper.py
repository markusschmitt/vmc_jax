"""
Integrators of the ordinary differential equation (ODE) :math:`\\frac{dy}{dt}=f(y,t)`.

The optimizers of ``jVMC_exp.optimizer`` are the right-hand side :math:`f` for the flat parameters
of a variational state: the time step of the stepper is the time step of a time evolution, and the
learning rate of a ground state search (see :ref:`Optimizer`).

* Fixed time step: ``Euler`` (first order) and ``Heun`` (second order).
* Adaptive time step: ``AdaptiveHeun`` (second order), ``RK23`` (third order) and ``RK45`` (fifth
  order), with the step size control of ``AbstractAdaptiveStepper``.

``step(t, f, y, **rhsArgs)`` makes one step from the time ``t`` and returns the new value of
:math:`y` and the time step :math:`\\Delta t` that it used. The attribute ``dt`` is the time step of
the next step; the adaptive steppers change it after every step.

The coefficients that multiply :math:`y` and :math:`f` are cast to ``global_defs.DT_OUT_REAL``,
the data type of ``psi.parameters_flat``, when the stepper is created, so that a step keeps the data
type of the parameters. The times of the evaluations keep the 64-bit precision.

The right-hand side is called as ``f(y, t, **rhsArgs, intStep=k)`` and returns an array with the
shape of ``y``. The integer ``k`` numbers the evaluations of a step. Every step starts with one
evaluation at ``(y, t)`` with ``intStep=0``; the further evaluations have ``k >= 1``, and an
adaptive stepper repeats them with a smaller time step when it rejects the step. The optimizers
keep the samples of the evaluation with ``intStep=0`` for the measurements.

Usage::

    def f(y, t, *, intStep, omega):
        return -1j * omega * y

    stepper = jVMC_exp.stepper.RK45(timeStep=1e-2, rtol=1e-8, atol=1e-8)
    y, t = jnp.ones(3, dtype=complex), 0.0
    while t < 1.0:
        y, dt = stepper.step(t, f, y, omega=2.0)
        t += dt
"""
from abc import ABC, abstractmethod
from typing import Callable
import jax.numpy as jnp
from jVMC_exp import global_defs

def _rms_norm(v):
    """
    Root-mean-square norm :math:`\\|v\\|_2/\\sqrt{N}`.

    Default norm of the adaptive steppers. Unlike the Euclidean norm it does not grow
    with the number of degrees of freedom, so ``atol`` and ``rtol`` retain their meaning
    as per-parameter tolerances irrespective of the size of the ansatz.
    """
    return jnp.linalg.norm(v) / jnp.sqrt(jnp.size(v))

def _clip(x, c_min, c_max):
    return max(min(x, c_max), c_min)

def _get_rk_step(butcher_tableau, t, f, y0, dt, k0=None, start_step=0, **rhs_kwargs):
    '''
    One step of the explicit Runge-Kutta scheme with the Butcher tableau ``(A, b, c)``.

    The tableau defines the integration scheme through:

        k_i = f(
            y_n + dt * sum_j A[i,j] * k_j,
            t_n + c[i] * dt
        )

    and the final update:

        y_{n+1} = y_n + dt * sum_i b[i] * k_i

    where:
        * A : strictly lower-triangular matrix of stage coefficients
        * b : vector of coefficients for the final weighted sum
        * c : vector of time shifts for each stage

    The tableau must follow the standard explicit RK convention:
        * len(A) == len(b) == len(c)
        * c[0] == 0
        * A[0,:] == 0
        * A[i,j] = 0 for j >= i

    If ``k0`` is given, it is the first stage k_0 instead of f(y_n, t_n). Stage i is evaluated
    with ``intStep=start_step + i``.

    Returns the update y_{n+1} - y_n and the list of the stages k_i.
    '''
    A, b, c = butcher_tableau

    K = [f(y0, t, **rhs_kwargs, intStep=start_step)] if k0 is None else [k0]
    step = b[0] * K[0]

    for i in range(1, len(b)):
        y_stage = y0 + dt * sum(A[i][j] * K[j] for j in range(i))
        t_stage = t + c[i] * dt

        K.append(f(y_stage, t_stage, **rhs_kwargs, intStep=i + start_step))
        step += b[i] * K[i]

    return dt * step, K

class AbstractStepper(ABC):
    """
    Base class of all steppers.

    Derived classes implement ``step()``. The attribute ``dt`` is the time step of the next step.

    Args:
        * ``time_step``: Time step :math:`\\Delta t`. Raises ``ValueError`` unless it is positive.
    """
    def __init__(self, time_step: float):
        if time_step <= 0:
            raise ValueError(
                f'The value of time_step has to be positive. Got {time_step}'
            )
        self.dt = time_step
        super().__init__()

    @abstractmethod
    def step(self, t, f, yInitial, **kwargs):
        """
        Make one step of the ODE :math:`\\frac{dy}{dt}=f(y,t)` from the time ``t``.

        Args:
            * ``t``: Initial time.
            * ``f``: Right-hand side, called as ``f(y, t, **kwargs, intStep=k)``, see the module \
              documentation.
            * ``yInitial``: Initial value of :math:`y`.
            * ``**kwargs``: Further keyword arguments of ``f``.

        Returns:
            ``y_new, dt``: the value of :math:`y` at the time ``t + dt``, and the time step \
            ``dt`` that the step used.
        """
        pass

class AbstractAdaptiveStepper(AbstractStepper):
    """
    Base class of the adaptive steppers.

    Holds the step size control shared by :class:`AdaptiveHeun`, :class:`RK23` and
    :class:`RK45`. Derived classes only supply the Butcher tableau and the pair of
    approximations that is compared.

    A step with the time step :math:`\\Delta t` computes two approximations
    :math:`\\Delta y_{\\rm low}` and :math:`\\Delta y_{\\rm high}` of the update of :math:`y` and the
    scaled error

    .. math::

        \\epsilon = \\frac{\\|\\Delta y_{\\rm low}-\\Delta y_{\\rm high}\\|}
        {\\mathrm{atol}+\\mathrm{rtol}\\,\\|y\\|}

    with the norm ``normFunction`` of ``step()``. The default is the root-mean-square norm
    :math:`\\|v\\|_2/\\sqrt{N}` of the :math:`N` entries of :math:`v`: unlike the Euclidean norm it
    does not grow with the number of parameters, so ``atol`` and ``rtol`` are tolerances per
    parameter, irrespective of the size of the ansatz. The time step of the next step is

    .. math::

        \\Delta t' = \\Delta t\\cdot\\mathrm{clip}\\left(0.95\\,\\epsilon^{-1/(p+1)},\\,0.2,\\,2\\right),

    clipped to ``[min_step, max_step]``, where :math:`p` is the order of the error estimate. The
    step is accepted if :math:`\\epsilon<1`, or if :math:`\\Delta t` is not larger than
    ``min_step``; otherwise it is made again from the same :math:`y` with :math:`\\Delta t'`. A
    non-finite error, i.e. a right-hand side that is NaN or infinite, raises ``RuntimeError``.

    Args:
        * ``error_order``: Order :math:`p` of the error estimate, i.e. the order of the *lower* \
          order member of the pair. For an embedded pair this is one less than the order of the \
          propagated solution.
        * ``time_step``: Initial time step (will be adapted automatically). Has to be positive. \
          Clipped to ``[min_step, max_step]``.
        * ``rtol``: Relative tolerance, weighting :math:`\\|y\\|`.
        * ``atol``: Absolute tolerance.
        * ``min_step``: Smallest allowed time step, or ``None`` for no lower bound. A step made \
          with this time step is accepted regardless of its error, so :math:`\\Delta t` does not \
          go below ``min_step``.
        * ``max_step``: Largest allowed time step, or ``None`` for no upper bound.

    Raises ``ValueError`` if ``min_step`` or ``max_step`` is negative, or if ``min_step`` is not
    smaller than ``max_step``.
    """
    def __init__(
            self, error_order, time_step, rtol, atol, min_step: float | None, max_step: float | None
        ):
        self.max_step = max_step
        self.min_step = min_step
        if self.min_step >= self.max_step:
            raise ValueError(
                f"min_step ({self.min_step}) must be smaller than max_step ({self.max_step})"
            )
        self._order = error_order
        self.rtol = rtol
        self.atol = atol

        super().__init__(time_step)
        self.dt = _clip(time_step, self.min_step, self.max_step)

    @property
    def order(self):
        """
        Order :math:`p` of the error estimate, entering the controller as \
        :math:`\\epsilon^{-1/(p+1)}`.

        This is the order of the error estimate, *not* of the propagated solution.
        """
        return self._order

    @property
    def min_step(self):
        """ Smallest allowed time step. ``-1`` stands for no lower bound (``min_step=None``). """
        return self._min_step

    @min_step.setter
    def min_step(self, value):
        if value is not None and value < 0:
            raise ValueError(
                f"min_step can only take positive values. "
                f"Got {value}"
            )

        self._min_step = -1 if value is None else value

    @property
    def max_step(self):
        """ Largest allowed time step. ``1e10`` stands for no upper bound (``max_step=None``). """
        return self._max_step

    @max_step.setter
    def max_step(self, value):
        if value is not None and value < 0:
            raise ValueError(
                f"max_step can only take positive values. "
                f"Got {value}"
            )

        self._max_step = 1e10 if value is None else value

    def _adaptive_step_control(self, y_norm, dy_low, dy_high, norm_function):
        """
        Accept or reject the current step and propose the next time step.

        Args:
            * ``y_norm``: Norm of the current value of :math:`y`, measured with ``norm_function``.
            * ``dy_low``: Update obtained from the lower order member of the pair.
            * ``dy_high``: Update obtained from the higher order member of the pair.
            * ``norm_function``: Norm used to quantify the magnitude of the error.

        Returns:
            Whether the step is accepted, and the time step :math:`\\Delta t` to use next.
        """
        update_diff = norm_function(dy_low - dy_high)
        if not jnp.isfinite(update_diff):
            raise RuntimeError(
                f"Adaptive step control got a non-finite error estimate ({update_diff}). "
                "The right-hand side of the ODE is NaN or inf -- check the solver"
            )

        scaled_diff = update_diff / (self.atol + y_norm * self.rtol)
        exponent = -1 / (self.order + 1)
        dt_rescale = _clip(
            scaled_diff**exponent * 0.95,
            c_min=0.2,
            c_max=2
        )
        dt_new = self.dt * dt_rescale

        converged = scaled_diff < 1 or self.dt <= self.min_step

        return converged, float(_clip(dt_new, self.min_step, self.max_step))

class Euler(AbstractStepper):
    """
    Euler integration, first order consistent, with a fixed or a scheduled time step:

    :math:`y_{n+1}=y_n+\\Delta t\\,f(y_n,t_n)`.

    In a ground state search the time step is the learning rate, and a schedule changes it from
    one step to the next.

    Args:
        * ``timeStep``: Time step :math:`\\Delta t`, or a schedule: a callable that maps the step \
          number to the time step, e.g. an ``optax`` schedule. The initial time step is then \
          ``timeStep(0)``, and ``update_dt(n)`` sets it to ``timeStep(n)``.
    """
    def __init__(self, timeStep: float | Callable=1e-3):
        self._scheduler = None
        if isinstance(timeStep, Callable):
            self._scheduler = timeStep
            dt = timeStep(0)
        else:
            dt = timeStep
        super().__init__(dt)

    def update_dt(self, step: int):
        """
        Set the time step to the value of the schedule at the step number ``step``. Does nothing \
        if the time step is fixed. ``AbstractOptimizer.ground_state_search()`` calls it before \
        every step.

        Args:
            * ``step``: Step number.
        """
        if self._scheduler is not None:
            self.dt = self._scheduler(step)

    def step(self, t, f, yInitial, **rhsArgs):
        """
        Perform an integration time step.

        For a first order ordinary differential equation (ODE) of the form

        :math:`\\frac{dy}{dt} = f(y, t, p)`

        where :math:`t` denotes the time and :math:`p` denotes further external parameters
        this function computes the Euler integration step

        :math:`y_{n+1} = y_n+\\Delta t f(y_n,t_n,p)`

        Args:
            * ``t``: Initial time.
            * ``f``: Right hand side of the ODE. This callable will be called once, as \
              ``f(y, t, **rhsArgs, intStep=0)``.
            * ``yInitial``: Initial value of :math:`y`.
            * ``**rhsArgs``: Further static arguments :math:`p` that will be passed to the right \
              hand side function ``f(y, t, **rhsArgs, intStep=k)``.

        Returns:
            New value of :math:`y` and time step used :math:`\\Delta t`.
        """
        dy = f(yInitial, t, **rhsArgs, intStep=0)

        return yInitial + self.dt * dy, self.dt

class Heun(AbstractStepper):
    """
    Heun's method, a second order consistent integration scheme with fixed time step:

    .. math::

        k_0 = f(y_n, t_n),\\quad k_1 = f(y_n+\\Delta t\\,k_0,\\,t_n+\\Delta t),\\quad
        y_{n+1} = y_n+\\frac{\\Delta t}{2}(k_0+k_1).

    See :class:`AdaptiveHeun` for the variant that adapts the time step automatically.

    Args:
        * ``timeStep``: Time step :math:`\\Delta t`.
    """
    def __init__(self, timeStep=1e-3):
        super().__init__(timeStep)

        self._butcher_tableau = (
                jnp.array([[0, 0], [1, 0]], dtype=global_defs.DT_OUT_REAL),
                jnp.array([0.5, 0.5], dtype=global_defs.DT_OUT_REAL),
                jnp.array([0, 1])
            )

    def step(self, t, f, yInitial, **rhsArgs):
        """
        Perform an integration time step.

        For a first order ordinary differential equation (ODE) of the form

        :math:`\\frac{dy}{dt} = f(y, t, p)`

        where :math:`t` denotes the time and :math:`p` denotes further external parameters
        this function computes a second-order consistent integration step for :math:`y`.

        Args:
            * ``t``: Initial time.
            * ``f``: Right hand side of the ODE. This callable will be called as \
              ``f(y, t, **rhsArgs, intStep=k)``, with ``k=0`` at the time ``t`` and ``k=1`` at \
              ``t + dt``.
            * ``yInitial``: Initial value of :math:`y`.
            * ``**rhsArgs``: Further static arguments :math:`p` that will be passed to the right \
              hand side function ``f(y, t, **rhsArgs, intStep=k)``.

        Returns:
            New value of :math:`y` and time step used :math:`\\Delta t`.
        """
        dy0, _ = _get_rk_step(
            self._butcher_tableau,
            t, f, yInitial, self.dt, **rhsArgs
        )

        return yInitial + dy0, self.dt

class AdaptiveHeun(AbstractAdaptiveStepper):
    """
    Implement an adaptive second order consistent integration scheme.

    The error is estimated by step doubling: one Heun step of size :math:`\\Delta t` is
    compared with two Heun steps of size :math:`\\Delta t/2`. Both are second order
    accurate, so their difference scales as :math:`\\Delta t^3` and the order of the error
    estimate is 2. The more accurate half-step solution is the one that is propagated.
    The step size control is described in :class:`AbstractAdaptiveStepper`.

    Args:
        * ``timeStep``: Initial time step (will be adapted automatically).
        * ``rtol``: Relative tolerance for integration errors.
        * ``atol``: Absolute tolerance for integration errors.
        * ``min_step``: Minimal allowed time step, or ``None`` for no lower bound.
        * ``max_step``: Maximal allowed time step, or ``None`` for no upper bound.
    """
    def __init__(self, timeStep=1e-3, rtol=1e-5, atol=1e-8, min_step=None, max_step=1):
        super().__init__(2, timeStep, rtol, atol, min_step, max_step)

        self._butcher_tableau = (
            jnp.array([[0, 0], [1, 0]], dtype=global_defs.DT_OUT_REAL),
            jnp.array([0.5, 0.5], dtype=global_defs.DT_OUT_REAL),
            jnp.array([0, 1])
        )

    def step(self, t, f, y, *, normFunction=_rms_norm, **rhsArgs):
        """ Perform an integration time step.

        For a first order ordinary differential equation (ODE) of the form

        :math:`\\frac{dy}{dt} = f(y, t, p)`

        where :math:`t` denotes the time and :math:`p` denotes further external parameters
        this function computes a second-order consistent integration step for :math:`y`.
        The time step :math:`\\Delta t` is chosen such that the integration error (quantified
        by a given norm) is smaller than the given tolerance.

        Args:
            * ``t``: Initial time.
            * ``f``: Right hand side of the ODE. This callable will be called as \
              ``f(y, t, **rhsArgs, intStep=k)``: ``k=0`` at the time ``t``, ``k=1`` at \
              ``t + dt`` (full step), ``k=2`` at ``t + dt/2`` (first half step), ``k=3`` and \
              ``k=4`` at ``t + dt/2`` and ``t + dt`` (second half step). An attempt that is \
              rejected repeats ``k=1, ..., 4`` with a smaller ``dt``.
            * ``y``: Initial value of :math:`y`.
            * ``normFunction``: Norm function to be used to quantify the magnitude of errors. \
              Defaults to the root-mean-square norm, see :class:`AbstractAdaptiveStepper`.
            * ``**rhsArgs``: Further static arguments :math:`p` that will be passed to the right \
              hand side function ``f(y, t, **rhsArgs, intStep=k)``.

        Returns:
            New value of :math:`y` and time step used :math:`\\Delta t`.
        """
        converged = False
        k0 = f(y, t, **rhsArgs, intStep=0)
        y_norm = normFunction(y)

        while not converged:
            dy0, _ = _get_rk_step(
                self._butcher_tableau,
                t, f, y, self.dt, k0, **rhsArgs
            )
            dy1, _ = _get_rk_step(
                self._butcher_tableau,
                t, f, y, 0.5 * self.dt, k0, 1, **rhsArgs
            )
            dy1_half, _ = _get_rk_step(
                self._butcher_tableau,
                t + 0.5 * self.dt, f, y + dy1, 0.5 * self.dt, start_step=3, **rhsArgs
            )
            dy1 += dy1_half

            current_dt = self.dt
            converged, self.dt = self._adaptive_step_control(y_norm, dy0, dy1, normFunction)

        return y + dy1, current_dt

class RK23(AbstractAdaptiveStepper):
    """
    This class implements the explicit Runge-Kutta method of order 3(2) by Bogacki and Shampine.

    The third order solution is propagated and the embedded second order solution supplies
    the error estimate, so the order of the error estimate is 2. The extra stage evaluated for
    the error estimate, at the new :math:`y`, equals the first stage of the next step (FSAL), but
    every step evaluates its first stage anew, so that each step starts with ``intStep=0`` at its
    own :math:`(y, t)`. The step size control is described in :class:`AbstractAdaptiveStepper`.

    Args:
        * ``timeStep``: Initial time step (will be adapted automatically).
        * ``rtol``: Relative tolerance for integration errors.
        * ``atol``: Absolute tolerance for integration errors.
        * ``min_step``: Minimal allowed time step, or ``None`` for no lower bound.
        * ``max_step``: Maximal allowed time step, or ``None`` for no upper bound.
    """
    def __init__(self, timeStep=1e-3, rtol=1e-5, atol=1e-8, min_step=None, max_step=1):
        super().__init__(2, timeStep, rtol, atol, min_step, max_step)
        self._k0 = None

        self._butcher_tableau = (
            jnp.array([[0, 0, 0], [0.5, 0, 0], [0, 0.75, 0]], dtype=global_defs.DT_OUT_REAL),
            jnp.array([2/9, 1/3, 4/9], dtype=global_defs.DT_OUT_REAL),
            jnp.array([0, 0.5, 0.75])
        )

    def step(self, t, f, y, *, normFunction=_rms_norm, **rhsArgs):
        """ Perform an integration time step.

        For a first order ordinary differential equation (ODE) of the form

        :math:`\\frac{dy}{dt} = f(y, t, p)`

        where :math:`t` denotes the time and :math:`p` denotes further external parameters
        this function computes a third-order consistent integration step for :math:`y`.
        The time step :math:`\\Delta t` is chosen such that the integration error (quantified
        by a given norm) is smaller than the given tolerance.

        Args:
            * ``t``: Initial time.
            * ``f``: Right hand side of the ODE. This callable will be called as \
              ``f(y, t, **rhsArgs, intStep=k)``: ``k=0, 1, 2`` at the times ``t``, \
              ``t + dt/2`` and ``t + 3dt/4``, and ``k=3`` at ``t + dt``, at the new :math:`y`, \
              for the error estimate. An attempt that is rejected repeats ``k=1, 2, 3`` with a \
              smaller ``dt``.
            * ``y``: Initial value of :math:`y`.
            * ``normFunction``: Norm function to be used to quantify the magnitude of errors. \
              Defaults to the root-mean-square norm, see :class:`AbstractAdaptiveStepper`.
            * ``**rhsArgs``: Further static arguments :math:`p` that will be passed to the right \
              hand side function ``f(y, t, **rhsArgs, intStep=k)``.

        Returns:
            New value of :math:`y` and time step used :math:`\\Delta t`.
        """
        converged = False
        # k0 = self._k0 if self._k0 is not None else f(y, t, **rhsArgs, intStep=0)
        # TODO: at the moment this is needed to trigger intStep=0, but the above line saves a step
        k0 = f(y, t, **rhsArgs, intStep=0)
        y_norm = normFunction(y)

        while not converged:
            dy_high, K = _get_rk_step(
                self._butcher_tableau,
                t, f, y, self.dt, k0, **rhsArgs
            )
            y_new = y + dy_high
            K.append(f(y_new, t + self.dt , **rhsArgs, intStep=3))
            dy_low = self.dt * (7/24 * K[0] + 1/4 * K[1] + 1/3 * K[2] + 1/8 * K[3])

            current_dt = self.dt
            converged, self.dt = self._adaptive_step_control(y_norm, dy_low, dy_high, normFunction)
            self._k0 = K[-1] if converged else None

        return y_new, current_dt

class RK45(AbstractAdaptiveStepper):
    """
    This class implements the explicit Runge-Kutta method of order 5(4) by Dormand and Prince.

    The fifth order solution is propagated and the embedded fourth order solution supplies
    the error estimate, so the order of the error estimate is 4. The extra stage evaluated for
    the error estimate, at the new :math:`y`, equals the first stage of the next step (FSAL), but
    every step evaluates its first stage anew, so that each step starts with ``intStep=0`` at its
    own :math:`(y, t)`. The step size control is described in :class:`AbstractAdaptiveStepper`.

    Args:
        * ``timeStep``: Initial time step (will be adapted automatically).
        * ``rtol``: Relative tolerance for integration errors.
        * ``atol``: Absolute tolerance for integration errors.
        * ``min_step``: Minimal allowed time step, or ``None`` for no lower bound.
        * ``max_step``: Maximal allowed time step, or ``None`` for no upper bound.
    """
    def __init__(self, timeStep=1e-3, rtol=1e-5, atol=1e-8, min_step=None, max_step=1):
        super().__init__(4, timeStep, rtol, atol, min_step, max_step)
        self._k0 = None

        self._butcher_tableau = (
            jnp.array([
                [0, 0, 0, 0, 0, 0],
                [0.2, 0, 0, 0, 0, 0],
                [3/40, 9/40, 0, 0, 0, 0],
                [44/45, -56/15, 32/9, 0, 0, 0],
                [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0],
                [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0]
            ], dtype=global_defs.DT_OUT_REAL),
            jnp.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84], dtype=global_defs.DT_OUT_REAL),
            jnp.array([0, 0.2, 3/10, 0.8, 8/9, 1])
        )

    def step(self, t, f, y, *, normFunction=_rms_norm, **rhsArgs):
        """ Perform an integration time step.

        For a first order ordinary differential equation (ODE) of the form

        :math:`\\frac{dy}{dt} = f(y, t, p)`

        where :math:`t` denotes the time and :math:`p` denotes further external parameters
        this function computes a fifth-order consistent integration step for :math:`y`.
        The time step :math:`\\Delta t` is chosen such that the integration error (quantified
        by a given norm) is smaller than the given tolerance.

        Args:
            * ``t``: Initial time.
            * ``f``: Right hand side of the ODE. This callable will be called as \
              ``f(y, t, **rhsArgs, intStep=k)``: ``k=0, ..., 5`` at the times ``t + c_k dt`` \
              with :math:`c=(0, 1/5, 3/10, 4/5, 8/9, 1)`, and ``k=6`` at ``t + dt``, at the new \
              :math:`y`, for the error estimate. An attempt that is rejected repeats \
              ``k=1, ..., 6`` with a smaller ``dt``.
            * ``y``: Initial value of :math:`y`.
            * ``normFunction``: Norm function to be used to quantify the magnitude of errors. \
              Defaults to the root-mean-square norm, see :class:`AbstractAdaptiveStepper`.
            * ``**rhsArgs``: Further static arguments :math:`p` that will be passed to the right \
              hand side function ``f(y, t, **rhsArgs, intStep=k)``.

        Returns:
            New value of :math:`y` and time step used :math:`\\Delta t`.
        """
        converged = False
        # k0 = self._k0 if self._k0 is not None else f(y, t, **rhsArgs, intStep=0)
        # TODO: at the moment this is needed to trigger intStep=0, but the above line saves a step
        k0 = f(y, t, **rhsArgs, intStep=0)
        y_norm = normFunction(y)

        while not converged:
            dy_high, K = _get_rk_step(
                self._butcher_tableau,
                t, f, y, self.dt, k0, **rhsArgs
            )
            y_new = y + dy_high
            K.append(f(y_new, t + self.dt , **rhsArgs, intStep=6))
            dy_low = self.dt * (5179/57600 * K[0] + 7571/16695 * K[2] + 393/640 * K[3] - 92097/339200 * K[4] + 187/2100 * K[5] + 1/40 * K[6])

            current_dt = self.dt
            converged, self.dt = self._adaptive_step_control(y_norm, dy_low, dy_high, normFunction)
            self._k0 = K[-1] if converged else None

        return y_new, current_dt