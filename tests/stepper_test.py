import unittest
import jax
import jax.numpy as jnp
import numpy as np

import jVMC_exp.stepper as st
from jVMC_exp import global_defs

def _make_linear_ode(N: int, seed: int = 123):
    """Return a random complex matrix M and a random initial state y0."""
    np.random.seed(seed)
    mat = jnp.array(np.random.rand(N, N) + 1j * np.random.rand(N, N))
    y0 = jnp.array(np.random.rand(N))

    return mat, y0

def _run_linear_ode(stepper, mat, y0, n_steps: int, **step_kwargs):
    """
    Integrate  dy/dt = mat @ y  for n_steps steps.
    Returns the max normalised error against the matrix-exponential exact solution.
    """
    def f(y, t, **args):
        return args["mat"].dot(y)

    def norm(y):
        return jnp.real(jnp.conjugate(y).dot(y))

    y, t = y0.copy(), 0.0
    errors = []
    for _ in range(n_steps):
        y, dt = stepper.step(t, f, y, **step_kwargs, mat=mat)
        t += dt
        y_exact = jax.scipy.linalg.expm(t * mat).dot(y0)
        errors.append(norm(y - y_exact) / y0.shape[0])

    return float(jnp.max(jnp.array(errors)))


def _run_scalar_ode(stepper, lam: complex, y0: complex, n_steps: int, **step_kwargs):
    """
    Integrate  dy/dt = lam * y  (scalar).
    Exact solution: y(t) = y0 * exp(lam * t).
    Returns the max absolute error.
    """
    def f(y, t, **args):
        return args["lam"] * y

    y = jnp.array([y0], dtype=jnp.complex64)
    t = 0.0
    errors = []
    for _ in range(n_steps):
        y, dt = stepper.step(t, f, y, **step_kwargs, lam=lam)
        t += dt
        y_exact = y0 * jnp.exp(lam * t)
        errors.append(float(jnp.abs(y[0] - y_exact)))

    return max(errors)

class TestEuler(unittest.TestCase):

    def test_linear_ode_accuracy(self):
        """First-order method: error should stay below a loose threshold."""
        mat, y0 = _make_linear_ode(N=4, seed=42)
        stepper = st.Euler(timeStep=1e-4)
        max_err = _run_linear_ode(stepper, mat, y0, n_steps=200)
        self.assertLess(max_err, 1e-2)

    def test_scalar_ode(self):
        """Scalar decay: y' = -y, exact solution y = exp(-t)."""
        stepper = st.Euler(timeStep=1e-4)
        err = _run_scalar_ode(stepper, lam=-1.0 + 0j, y0=1.0 + 0j, n_steps=100)
        self.assertLess(err, 1e-2)

    def test_callable_scheduler(self):
        """dt should track the scheduler output at each step."""
        schedule = lambda step: 1e-3 * (0.9 ** step)
        stepper = st.Euler(timeStep=schedule)

        def f(y, t, **args):
            return -y

        y = jnp.array([1.0])
        for k in range(5):
            stepper.update_dt(k)
            self.assertAlmostEqual(float(stepper.dt), 1e-3 * (0.9 ** k), places=10)
            y, _ = stepper.step(0.0, f, y)

    def test_fixed_step_returned(self):
        """The returned dt must equal the configured step size."""
        stepper = st.Euler(timeStep=0.025)

        def f(y, t, **args):
            return -y

        _, dt = stepper.step(0.0, f, jnp.array([1.0]))
        self.assertAlmostEqual(float(dt), 0.025)


class TestHeun(unittest.TestCase):

    def test_linear_ode_accuracy(self):
        """Second-order method should be meaningfully more accurate than Euler."""
        mat, y0 = _make_linear_ode(N=4, seed=42)
        stepper = st.Heun(timeStep=1e-3)
        max_err = _run_linear_ode(stepper, mat, y0, n_steps=200)
        self.assertLess(max_err, 1e-4)

    def test_scalar_ode(self):
        stepper = st.Heun(timeStep=1e-3)
        err = _run_scalar_ode(stepper, lam=-1.0 + 0j, y0=1.0 + 0j, n_steps=100)
        self.assertLess(err, 1e-4)

    def test_fixed_step_returned(self):
        stepper = st.Heun(timeStep=0.01)

        def f(y, t, **args):
            return -y

        _, dt = stepper.step(0.0, f, jnp.array([1.0]))
        self.assertAlmostEqual(float(dt), 0.01)

    def test_better_than_euler(self):
        """For the same step size Heun should beat Euler on the same ODE."""
        mat, y0 = _make_linear_ode(N=4, seed=7)

        err_euler = _run_linear_ode(st.Euler(timeStep=1e-3), mat, y0, n_steps=50)
        err_heun  = _run_linear_ode(st.Heun(timeStep=1e-3),  mat, y0, n_steps=50)
        self.assertLess(err_heun, err_euler)

class TestAdaptiveHeun(unittest.TestCase):

    def test_linear_ode_accuracy(self):
        mat, y0 = _make_linear_ode(N=4, seed=123)
        stepper = st.AdaptiveHeun(rtol=1e-8, atol=1e-8)
        max_err = _run_linear_ode(stepper, mat, y0, n_steps=100)
        self.assertLess(max_err, 1e-5)

    def test_scalar_ode(self):
        stepper = st.AdaptiveHeun(rtol=1e-8, atol=1e-8)
        err = _run_scalar_ode(stepper, lam=-1.0 + 0j, y0=1.0 + 0j, n_steps=50)
        self.assertLess(err, 1e-5)

    def test_tighter_tolerance_gives_smaller_error(self):
        """Halving the tolerance should reduce the accumulated error."""
        mat, y0 = _make_linear_ode(N=4, seed=99)
        err_loose = _run_linear_ode(st.AdaptiveHeun(rtol=1e-4, atol=1e-4), mat, y0, 50)
        err_tight = _run_linear_ode(st.AdaptiveHeun(rtol=1e-8, atol=1e-8), mat, y0, 50)
        self.assertLess(err_tight, err_loose)

    def test_max_step_respected(self):
        """dt must never exceed max_step after adaptation."""
        max_step = 0.05
        stepper = st.AdaptiveHeun(timeStep=1e-3, rtol=1e-6, atol=1e-6, max_step=max_step)

        def f(y, t, **args):
            return -y

        y = jnp.ones(4)
        for _ in range(30):
            y, dt = stepper.step(0.0, f, y)
            self.assertLessEqual(float(dt), max_step + 1e-12)

class TestRK23(unittest.TestCase):

    def test_linear_ode_accuracy(self):
        mat, y0 = _make_linear_ode(N=4, seed=123)
        stepper = st.RK23(rtol=1e-8, atol=1e-8)
        max_err = _run_linear_ode(stepper, mat, y0, n_steps=100)
        self.assertLess(max_err, 1e-6)

    def test_scalar_ode(self):
        stepper = st.RK23(rtol=1e-8, atol=1e-8)
        err = _run_scalar_ode(stepper, lam=-1.0 + 0j, y0=1.0 + 0j, n_steps=50)
        self.assertLess(err, 1e-6)

    def test_k0_reuse_after_accepted_step(self):
        """
        After a converged step _k0 should be non-None (FSAL reuse ready).
        After a reset it should be None.
        """
        stepper = st.RK23(rtol=1e-8, atol=1e-8)

        def f(y, t, **args):
            return -y

        y = jnp.ones(4)
        self.assertIsNone(stepper._k0)
        y, _ = stepper.step(0.0, f, y)
        self.assertIsNotNone(stepper._k0)

    def test_tighter_tolerance_gives_smaller_error(self):
        mat, y0 = _make_linear_ode(N=4, seed=55)
        err_loose = _run_linear_ode(st.RK23(rtol=1e-4, atol=1e-4), mat, y0, 50)
        err_tight = _run_linear_ode(st.RK23(rtol=1e-9, atol=1e-9), mat, y0, 50)
        self.assertLess(err_tight, err_loose)

    def test_max_step_respected(self):
        max_step = 0.05
        stepper = st.RK23(timeStep=1e-3, rtol=1e-6, atol=1e-6, max_step=max_step)

        def f(y, t, **args):
            return -y

        y = jnp.ones(4)
        for _ in range(30):
            y, dt = stepper.step(0.0, f, y)
            self.assertLessEqual(float(dt), max_step + 1e-12)

class TestRK45(unittest.TestCase):

    def test_linear_ode_accuracy(self):
        """Fifth-order method should hit tight tolerances comfortably."""
        mat, y0 = _make_linear_ode(N=4, seed=123)
        stepper = st.RK45(rtol=1e-10, atol=1e-10)
        max_err = _run_linear_ode(stepper, mat, y0, n_steps=100)
        self.assertLess(max_err, 1e-7)

    def test_scalar_ode(self):
        stepper = st.RK45(rtol=1e-10, atol=1e-10)
        err = _run_scalar_ode(stepper, lam=-1.0 + 0j, y0=1.0 + 0j, n_steps=50)
        self.assertLess(err, 1e-7)

    def test_k0_reuse_after_accepted_step(self):
        stepper = st.RK45(rtol=1e-8, atol=1e-8)

        def f(y, t, **args):
            return -y

        y = jnp.ones(4)
        self.assertIsNone(stepper._k0)
        y, _ = stepper.step(0.0, f, y)
        self.assertIsNotNone(stepper._k0)

    def test_tighter_tolerance_gives_smaller_error(self):
        mat, y0 = _make_linear_ode(N=4, seed=88)
        err_loose = _run_linear_ode(st.RK45(rtol=1e-4, atol=1e-4), mat, y0, 50)
        err_tight = _run_linear_ode(st.RK45(rtol=1e-10, atol=1e-10), mat, y0, 50)
        self.assertLess(err_tight, err_loose)

    def test_max_step_respected(self):
        max_step = 0.05
        stepper = st.RK45(timeStep=1e-3, rtol=1e-6, atol=1e-6, max_step=max_step)

        def f(y, t, **args):
            return -y

        y = jnp.ones(4)
        for _ in range(30):
            y, dt = stepper.step(0.0, f, y)
            self.assertLessEqual(float(dt), max_step + 1e-12)


class TestDtype(unittest.TestCase):
    """The steppers keep the data type DT_OUT_REAL of the flat parameters."""

    def setUp(self):
        saved = {name: getattr(global_defs, name) for name in ("DT_OUT_REAL", "DT_OPERATORS_REAL")}
        self.addCleanup(lambda: [setattr(global_defs, k, v) for k, v in saved.items()])

    def test_steppers_keep_the_data_type_of_the_parameters(self):
        def f(y, t, **args):
            return -y

        for dtype in (jnp.float32, jnp.float64):
            global_defs.update(dt_out_real=dtype)
            steppers = (
                st.Euler(1e-2), st.Heun(1e-2), st.AdaptiveHeun(1e-2), st.RK23(1e-2), st.RK45(1e-2)
            )
            for stepper in steppers:
                with self.subTest(dtype=jnp.dtype(dtype).name, stepper=type(stepper).__name__):
                    y = jnp.ones(4, dtype=dtype)
                    for _ in range(3):
                        y, _ = stepper.step(0.0, f, y)
                    self.assertEqual(y.dtype, jnp.dtype(dtype))

    def test_times_keep_double_precision(self):
        """With 32-bit parameters the times passed to f, and the time steps, stay 64-bit."""
        global_defs.update(dt_out_real=jnp.float32)

        # timeStep=0.5 is rejected first, so that dt also comes from the step size control
        for stepper in (st.AdaptiveHeun(timeStep=0.5), st.RK23(timeStep=0.5), st.RK45(timeStep=0.5)):
            times = []

            def f(y, t, **args):
                times.append(t)
                return -10 * y

            y, t = jnp.ones(4, dtype=jnp.float32), 1.5
            for _ in range(5):
                y, dt = stepper.step(t, f, y)
                t += dt

            with self.subTest(stepper=type(stepper).__name__):
                self.assertEqual({jnp.asarray(s).dtype for s in times}, {jnp.dtype(jnp.float64)})
                self.assertEqual(y.dtype, jnp.dtype(jnp.float32))


if __name__ == "__main__":
    unittest.main()