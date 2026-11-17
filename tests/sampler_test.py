import unittest
import flax.linen as nn
import jax
import jax.random as random
import jax.numpy as jnp
import numpy as np

import jVMC_exp
import jVMC_exp.nets as nets
from jVMC_exp.vqs import NQS
import jVMC_exp.sampler as sampler
from jVMC_exp.symmetry.lattice_symetries import chain_reflection_symmetry, spin_flip_symmetry, square_translation_symmetry

@jax.jit
def state_to_int(S):
    powers = 2 ** jnp.arange(S.shape[-1])[::-1]
    return jnp.dot(S, powers).astype(jnp.int64)

def _translation_projector(L: int):
    return square_translation_symmetry(L, 1, "spin")

def _translation_reflection_spinflip_projector(L: int):
    return _translation_projector(L) * chain_reflection_symmetry(L, "spin") * spin_flip_symmetry(L, 1, "spin")

class _PeakedGeneratorNet(nn.Module):
    peak_state: tuple[int, ...]

    @nn.compact
    def __call__(self, s):
        offset = self.param("offset", nn.initializers.zeros, ())
        peak = jnp.asarray(self.peak_state, dtype=s.dtype)
        return (offset + jnp.where(jnp.all(s == peak), 0.0, -40.0)).astype(jnp.complex128)

    def sample(self, key):
        del key
        return jnp.asarray(self.peak_state, dtype=jnp.int32)

def _test_sampling(net, test_class: unittest.TestCase, mu=2, log_prob_factor=0.5):
    L = 4
    num_samples = 2 ** 18
    num_chains = 2 ** 16

    weights = jnp.array(
        [0.23898957, 0.12614753, 0.19479055, 0.17325271, 0.14619853, 0.21392751,
            0.19648707, 0.17103704, -0.15457255, 0.10954413, 0.13228065, -0.14935214,
            -0.09963073, 0.17610707, 0.13386381, -0.14836467]
    )
    weights = weights

    # Set up variational wave function
    psi = NQS(net, L, num_samples, seed=1234)
    exact_psi = NQS(net, L, 2 ** L, seed=1234)
    psi.parameters = weights
    exact_psi.parameters = weights

    # Set up exact sampler
    exact_sampler = sampler.ExactSampler(exact_psi, logProbFactor=log_prob_factor)

    # Set up MCMC sampler
    proposer = jVMC_exp.propose.SpinFlip()
    mc_sampler = sampler.MCSampler(
        psi, updateProposer=proposer, key=random.PRNGKey(0), 
        numChains=num_chains, numSamples=num_samples,
        mu=mu, logProbFactor=log_prob_factor
    )

    # Compute exact probabilities
    _, _, pex = exact_sampler.sample()

    # Get samples from MCMC sampler
    samples, _, p = mc_sampler.sample()

    test_class.assertTrue(jnp.array([samples.shape[0],])[None, None, ...] >= num_samples)

    # Compute histogram of sampled configurations
    # samples_int = jax.vmap(state_to_int)(samples)
    samples_int = state_to_int(samples)
    pmc, _ = np.histogram(samples_int, bins=np.arange(0, 17), weights=p)
    pmc = pmc / jnp.sum(pmc)

    # Compare histogram to exact probabilities
    test_class.assertTrue(jnp.max(jnp.abs(pmc - pex)) < 2e-3)

def _test_autoreg_sampling(net, test_class: unittest.TestCase, L=(4,), mu=2, log_prob_factor=0.5):
    num_samples = 2 ** 18
    num_chains = 2 ** 16

    psi = NQS(net, L, num_samples, seed=1234)
    exact_psi = NQS(net, L, 2 ** sum(L), seed=1234)

    # Set up exact sampler
    exact_sampler = sampler.ExactSampler(exact_psi)

    # Set up MCMC sampler
    proposer = jVMC_exp.propose.SpinFlip()
    mc_sampler = sampler.MCSampler(
        psi, updateProposer=proposer, key=random.PRNGKey(0), 
        numChains=num_chains, numSamples=num_samples,
        mu=mu, logProbFactor=log_prob_factor
    )

    psi.parameters = 2 * psi.parameters_flat
    exact_psi.parameters = psi.parameters

    # Compute exact probabilities
    _, _, pex = exact_sampler.sample()
    samples, _, p = mc_sampler.sample()

    test_class.assertTrue(jnp.array([samples.shape[0],])[None, None, ...] >= num_samples)

    # Compute histogram of sampled configurations
    samples_int = state_to_int(samples)
    pmc, _ = np.histogram(samples_int, bins=np.arange(0, 17), weights=p)
    pmc = pmc / jnp.sum(pmc)

    test_class.assertTrue(jnp.max(jnp.abs(pmc - pex.reshape((-1,))[:16])) < 1.1e-3)

class TestMC(unittest.TestCase):

    def test_direct_sampling_randomizes_projected_generator_over_symmetry_orbit(self):
        L = 4
        peak_state = (1, 0, 0, 0)
        orbit = _translation_projector(L)
        net = _PeakedGeneratorNet(peak_state)
        psi = NQS(net, L, 64, seed=1234, orbit=orbit, symmetry_average="sep")
        mc_sampler = sampler.MCSampler(
            psi,
            updateProposer=None,
            key=random.PRNGKey(17),
            numChains=4,
            numSamples=256,
        )

        samples, _, _ = mc_sampler.sample()
        expected = {
            tuple(row.tolist())
            for row in np.asarray(orbit.transformed_states(jnp.asarray(peak_state, dtype=jnp.int32)))
        }
        sampled = {tuple(row.tolist()) for row in np.asarray(samples)}

        self.assertEqual(sampled, expected)

    def test_MCMC_sampling(self):
        rbm = nets.CpxRBM(numHidden=2, bias=False)
        orbit = _translation_reflection_spinflip_projector(4)
        net = orbit * rbm
        
        _test_sampling(net, self)

    def test_MCMC_sampling_with_mu(self):
        rbm = nets.CpxRBM(numHidden=2, bias=False)
        orbit = _translation_projector(4)
        net = orbit * rbm
        
        _test_sampling(net, self, mu=1)

    def test_MCMC_sampling_with_logProbFactor(self):
        rbm = nets.CpxRBM(numHidden=2, bias=False)
        orbit = _translation_projector(4)
        net = orbit * rbm
        
        _test_sampling(net, self, log_prob_factor=1)

    def test_MCMC_sampling_with_two_nets(self):
        rbm1 = nets.RBM(numHidden=2, bias=False)
        rbm2 = nets.RBM(numHidden=2, bias=False)
        model = jVMC_exp.nets.TwoNets((rbm1, rbm2))
        orbit = _translation_projector(4)
        net = orbit * model

        _test_sampling(net, self)

    def test_MCMC_sampling_ratio(self):
        rbm = nets.CpxRBM_ratio(numHidden=2, bias=False)
        orbit = _translation_reflection_spinflip_projector(4)
        net = orbit * rbm
        
        _test_sampling(net, self)


    # def test_autoregressive_sampling(self):
    #     rnn = nets.RNN1DGeneral(L=4, hiddenSize=5, depth=2)
    #     rbm = nets.RBM(numHidden=2, bias=False)
    #     model = jVMC_exp.nets.TwoNets((rnn, rbm))
    #     orbit = _translation_projector(4)
    #     net = orbit * model
        
    #     _test_autoreg_sampling(net, self)

    # def test_autoregressive_sampling_with_symmetries(self):
    #     rnn = nets.RNN1DGeneral(L=4, hiddenSize=5, realValuedOutput=True)
    #     rbm = nets.RBM(numHidden=2, bias=False)
    #     model = jVMC_exp.nets.TwoNets((rnn, rbm))
    #     orbit = _translation_projector(4)
    #     net = orbit * model
        
    #     _test_autoreg_sampling(net, self)

    # def test_autoregressive_sampling_with_lstm(self):
    #     rnn = nets.RNN1DGeneral(L=4, hiddenSize=5, cell="LSTM", realValuedParams=True, realValuedOutput=True, inputDim=2)
    #     rbm = nets.RBM(numHidden=2, bias=False)
    #     model = jVMC_exp.nets.TwoNets((rnn, rbm))
    #     orbit = _translation_projector(4)
    #     net = orbit * model

    #     _test_autoreg_sampling(net, self)

    # def test_autoregressive_sampling_with_gru(self):
    #     rnn = nets.RNN1DGeneral(L=4, hiddenSize=5, cell="GRU", realValuedParams=True, realValuedOutput=True, inputDim=2)
    #     rbm = nets.RBM(numHidden=2, bias=False)

    #     _test_autoreg_sampling((rnn, rbm), self)

#     def test_autoregressive_sampling_with_rnn2d(self):
#         rnn = nets.RNN2DGeneral(L=2, hiddenSize=5, cell="RNN", realValuedParams=True, realValuedOutput=True)
#         model = jVMC_exp.nets.TwoNets((rnn, rnn))
#         orbit = square_translation_symmetry(2, 2, "spin")
#         net = orbit * model

#         _test_autoreg_sampling(net, self, L=(4, 4))

#     def test_autoregressive_sampling_with_rnn2d_symmetric(self):
#         rnn = nets.RNN2DGeneral(L=2, hiddenSize=5, cell="RNN", realValuedParams=True, realValuedOutput=True)
#         model = jVMC_exp.nets.TwoNets((rnn, rnn))
#         orbit = square_translation_symmetry(2, 2, "spin")
#         net = orbit * model

#         _test_autoreg_sampling(net, self, L=(4, 4))

#     def test_autoregressive_sampling_with_lstm2d(self):
#         rnn = nets.RNN2DGeneral(L=2, hiddenSize=5, cell="LSTM", realValuedParams=True, realValuedOutput=True)
#         model = jVMC_exp.nets.TwoNets((rnn, rnn))
#         orbit = square_translation_symmetry(2, 2, "spin")
#         net = orbit * model

#         _test_autoreg_sampling(net, self, L=(4, 4))

class TestAcceptanceRatio(unittest.TestCase):
    def test_raises_before_sampling(self):
        psi = NQS(nets.CpxRBM(numHidden=2, bias=False), 4, 2 ** 6, seed=1234)
        mc_sampler = sampler.MCSampler(
            psi, updateProposer=jVMC_exp.propose.SpinFlip(), key=random.PRNGKey(0),
            numChains=2 ** 4, numSamples=2 ** 6,
        )

        with self.assertRaises(RuntimeError):
            mc_sampler.acceptance_ratio()

        mc_sampler.sample()
        ratio = mc_sampler.acceptance_ratio()
        self.assertTrue(0 <= ratio <= 1)

    def test_raises_when_sampling_directly(self):
        psi = NQS(_PeakedGeneratorNet((1, 0, 0, 0)), 4, 2 ** 6, seed=1234)
        mc_sampler = sampler.MCSampler(
            psi, updateProposer=None, key=random.PRNGKey(0), numChains=4, numSamples=2 ** 6,
        )

        with self.assertRaises(RuntimeError):
            mc_sampler.acceptance_ratio()

        mc_sampler.sample()
        with self.assertRaises(RuntimeError):
            mc_sampler.acceptance_ratio()

class _FlatNet(nn.Module):
    @nn.compact
    def __call__(self, s):
        offset = self.param("offset", nn.initializers.zeros, ())
        return offset * jnp.sum(s * 0.0)

class TestMALA(unittest.TestCase):
    def test_accepts_all_moves_on_flat_distribution_with_pbc(self):
        """Moves across the periodic boundary must not be penalized by the proposal correction."""
        geometry = jVMC_exp.geometry.HyperRectangle(1, 1, extent=(1.0,))
        psi = NQS(_FlatNet(), 1, 2 ** 8, seed=1234)
        proposer = jVMC_exp.propose.MALA(geometry, psi.apply_fun, tau=0.02, use_thermalization=False)
        mc_sampler = sampler.MCSamplerCont(
            psi, proposer, key=random.PRNGKey(0), numChains=2 ** 6, numSamples=2 ** 10,
        )

        mc_sampler.sample()
        self.assertAlmostEqual(float(mc_sampler.acceptance_ratio()), 1.0)

class TestExactSampler(unittest.TestCase):
    def test_exact_sampler(self):
        L = 4

        weights = jnp.array(
            [0.23898957, 0.12614753, 0.19479055, 0.17325271, 0.14619853, 0.21392751,
             0.19648707, 0.17103704, -0.15457255, 0.10954413, 0.13228065, -0.14935214,
             -0.09963073, 0.17610707, 0.13386381, -0.14836467]
        )

        # Set up variational wave function
        rbm = nets.CpxRBM(numHidden=2, bias=False)
        psi = NQS(rbm, L, 2**L)
        psi.parameters = weights

        # Set up exact sampler
        exact_sampler = sampler.ExactSampler(psi)  

        # Compute exact probabilities
        s, psi_s, _ = exact_sampler.sample()
        self.assertTrue(jnp.max(jnp.abs((psi(s) - psi_s) / psi_s)) < 1e-14)


_CUTOFF_WEIGHTS = jnp.array(
    [0.23898957, 0.12614753, 0.19479055, 0.17325271, 0.14619853, 0.21392751,
     0.19648707, 0.17103704, -0.15457255, 0.10954413, 0.13228065, -0.14935214,
     -0.09963073, 0.17610707, 0.13386381, -0.14836467]
)


def _setup_cutoff_psi(L=4, num_samples=2 ** 16):
    rbm = nets.CpxRBM(numHidden=2, bias=False)
    net = _translation_projector(L) * rbm
    psi = NQS(net, L, num_samples, seed=1234)
    exact_psi = NQS(net, L, 2 ** L, seed=1234)
    psi.parameters = _CUTOFF_WEIGHTS
    exact_psi.parameters = _CUTOFF_WEIGHTS

    return psi, exact_psi


def _histogram_from_samples(samples, L, weights=None):
    """Empirical distribution over the 2**L basis states, optionally reweighted."""
    counts, _ = np.histogram(
        np.asarray(state_to_int(samples)), bins=np.arange(0, 2 ** L + 1),
        weights=None if weights is None else np.asarray(weights)
    )

    return counts / np.sum(counts)


def _normalized(log_p):
    p = np.exp(log_p - np.max(log_p))

    return p / np.sum(p)


def _cutoff_reference(exact_psi, eps, mu, log_prob_factor):
    """Exact reference distributions for a given cutoff setting.

    Returns the target ``p ~ |psi|^(1/logProbFactor)`` that the reweighted
    samples must reproduce, the law ``q ~ max(|psi|^mu, eps * max|psi|^mu)``
    that the Markov chain is supposed to sample, the uncut law
    ``p_mu ~ |psi|^mu``, and ``mu * max_s Re log psi``.
    """
    _, log_psi, p_target = sampler.ExactSampler(exact_psi, logProbFactor=log_prob_factor).sample()
    log_p_mu = mu * np.real(np.asarray(log_psi))
    max_log_p_mu = float(np.max(log_p_mu))
    cutoff = max_log_p_mu + np.log(eps) if eps > 0 else -np.inf

    return np.asarray(p_target), _normalized(np.maximum(log_p_mu, cutoff)), \
        _normalized(log_p_mu), max_log_p_mu


class TestCutoffSampler(unittest.TestCase):
    """Statistical correctness of the cutoff-based sampler.

    The sampler has two halves that have to be tested separately: the Markov
    chain must sample the *flattened* law ``q ~ max(|psi|^mu, e^cutoff)``, and
    the importance ratio returned alongside the samples must turn that back
    into the target ``|psi|^(1/logProbFactor)``. Checking only the second half
    is not enough -- an implementation that ignores ``eps`` altogether would
    pass it.
    """

    L = 4
    NUM_SAMPLES = 2 ** 18
    NUM_CHAINS = 2 ** 16

    def _run_sampling_test(self, eps, tol, mu=2, log_prob_factor=0.5, cutoff_active=True):
        psi, exact_psi = _setup_cutoff_psi(self.L, self.NUM_SAMPLES)
        p_target, q_ref, p_mu, max_log_p_mu = _cutoff_reference(
            exact_psi, eps, mu, log_prob_factor
        )

        # Guard against a vacuous test. |psi|^2 of this state spans only a factor
        # of ~5, so e.g. eps=0.1 puts the cutoff *below* the smallest amplitude:
        # nothing is clipped, q == p_mu, and the test would pass even if the
        # cutoff and the reweighting were both removed.
        deformation = np.max(np.abs(q_ref - p_mu))
        if cutoff_active:
            self.assertGreater(deformation, 5 * tol)
        else:
            self.assertLess(deformation, 1e-12)

        cutoff_sampler = sampler.CutoffSampler(
            psi, updateProposer=jVMC_exp.propose.SpinFlip(), eps=eps,
            key=random.PRNGKey(0),
            numChains=self.NUM_CHAINS, numSamples=self.NUM_SAMPLES,
            mu=mu, logProbFactor=log_prob_factor,
            maxLogPsi=max_log_p_mu,  # exact, so the test does not depend on the bootstrap
        )
        samples, _, weights = cutoff_sampler.sample()

        self.assertGreaterEqual(samples.shape[0], self.NUM_SAMPLES)
        self.assertEqual(weights.shape, (samples.shape[0],))
        self.assertTrue(jnp.all(jnp.isfinite(weights)))

        # 1. the chain samples the flattened law ...
        self.assertLess(np.max(np.abs(_histogram_from_samples(samples, self.L) - q_ref)), tol)
        # 2. ... and the importance ratio restores the target
        self.assertLess(
            np.max(np.abs(_histogram_from_samples(samples, self.L, weights) - p_target)), tol
        )

        # The weights must be non-trivial exactly when the cutoff bites.
        weight_spread = float(jnp.min(weights) / jnp.max(weights))
        if cutoff_active:
            self.assertLess(weight_spread, 0.9)
        elif abs(mu - 1 / log_prob_factor) < 1e-12:
            # Nothing clipped and the sampled law is already the target.
            self.assertAlmostEqual(weight_spread, 1.0, places=10)

    def test_cutoff_inactive_for_tiny_eps(self):
        """eps far below the smallest |psi|^2: must reduce to plain Born sampling."""
        self._run_sampling_test(eps=1e-12, tol=2e-3, cutoff_active=False)

    def test_cutoff_inactive_for_zero_eps(self):
        """eps=0 means log(eps) = -inf; the cutoff must switch off without NaNs."""
        self._run_sampling_test(eps=0.0, tol=2e-3, cutoff_active=False)

    def test_cutoff_active(self):
        """eps=0.9 clips 14 of the 16 basis states; reweighting must undo it."""
        self._run_sampling_test(eps=0.9, tol=3e-3, cutoff_active=True)

    def test_cutoff_active_with_mu(self):
        """Importance sampling (mu < 2) combined with an active cutoff."""
        self._run_sampling_test(eps=0.9, tol=3e-3, mu=1, cutoff_active=True)

    def test_cutoff_active_with_log_prob_factor(self):
        """logProbFactor=1 (POVM convention) combined with an active cutoff."""
        self._run_sampling_test(eps=0.9, tol=3e-3, mu=1, log_prob_factor=1.0, cutoff_active=True)

    def test_eps_one_samples_uniformly(self):
        """eps=1 puts the cutoff at the maximum: the chain must sample uniformly."""
        _, exact_psi = _setup_cutoff_psi(self.L, self.NUM_SAMPLES)
        _, q_ref, _, _ = _cutoff_reference(exact_psi, 1.0, 2, 0.5)
        self.assertTrue(np.allclose(q_ref, 1 / 2 ** self.L))

        self._run_sampling_test(eps=1.0, tol=3e-3, cutoff_active=True)

class _CustomThermalizationProposer(jVMC_exp.propose.SpinFlip):
    def __init__(self):
        super().__init__()
        self._use_custom_thermalization = True

class TestCutoffSamplerGuards(unittest.TestCase):
    """The constructor rejects configurations the sampler cannot handle."""

    def _psi(self):
        return _setup_cutoff_psi(L=4, num_samples=2 ** 6)[0]

    def test_rejects_non_proposer(self):
        with self.assertRaises(RuntimeError):
            sampler.CutoffSampler(self._psi(), updateProposer=None, eps=0.1,
                                  key=random.PRNGKey(0), maxLogPsi=0.0)

    def test_rejects_custom_thermalization_proposer(self):
        with self.assertRaises(RuntimeError):
            sampler.CutoffSampler(self._psi(), updateProposer=_CustomThermalizationProposer(),
                                  eps=0.1, key=random.PRNGKey(0), maxLogPsi=0.0)

    def test_rejects_generator_net(self):
        psi = NQS(_PeakedGeneratorNet((1, 0, 0, 0)), 4, 2 ** 6, seed=1234)
        with self.assertRaises(RuntimeError):
            sampler.CutoffSampler(psi, updateProposer=jVMC_exp.propose.SpinFlip(), eps=0.1,
                                  key=random.PRNGKey(0), maxLogPsi=0.0)


class TestCutoffSamplerProperties(unittest.TestCase):
    """Light-weight tests of the cutoff bookkeeping that skip the bootstrap."""

    def _make_sampler(self, eps=0.1, maxLogPsi=-2.0):
        psi, _ = _setup_cutoff_psi(L=4, num_samples=2 ** 8)
        proposer = jVMC_exp.propose.SpinFlip()

        return sampler.CutoffSampler(
            psi, updateProposer=proposer, eps=eps,
            key=random.PRNGKey(0),
            numChains=2 ** 4, numSamples=2 ** 8,
            maxLogPsi=maxLogPsi,
        )

    def test_init_sets_cutoff(self):
        s = self._make_sampler(eps=0.1, maxLogPsi=-2.0)
        self.assertAlmostEqual(float(s.eps), 0.1)
        self.assertTrue(jnp.allclose(s.maxLogPsi, jnp.asarray(-2.0)))
        self.assertTrue(jnp.allclose(s.cutoff, jnp.asarray(-2.0) + jnp.log(0.1)))

    def test_update_eps(self):
        s = self._make_sampler(eps=0.1, maxLogPsi=-2.0)
        s.eps = 0.5
        self.assertAlmostEqual(float(s.eps), 0.5)
        self.assertTrue(jnp.allclose(s.cutoff, jnp.asarray(-2.0) + jnp.log(0.5)))

    def test_update_eps_rejects_out_of_range(self):
        s = self._make_sampler(eps=0.1, maxLogPsi=-2.0)
        with self.assertRaises(ValueError):
            s.eps = 2.0
        with self.assertRaises(ValueError):
            s.eps = -0.1

    def test_update_maxLogPsi(self):
        s = self._make_sampler(eps=0.1, maxLogPsi=-2.0)
        s.maxLogPsi = -1.5
        self.assertTrue(jnp.allclose(s.maxLogPsi, jnp.asarray(-1.5)))
        self.assertTrue(jnp.allclose(s.cutoff, jnp.asarray(-1.5) + jnp.log(0.1)))

    def test_init_rejects_eps_out_of_range(self):
        psi, _ = _setup_cutoff_psi(L=4, num_samples=2 ** 8)
        proposer = jVMC_exp.propose.SpinFlip()
        for bad_eps in (2.0, -0.1):
            with self.assertRaises(ValueError):
                sampler.CutoffSampler(psi, updateProposer=proposer, eps=bad_eps,
                                      key=random.PRNGKey(0), maxLogPsi=-2.0)

    def test_sampling_updates_max_log_psi(self):
        """The cutoff tracks mu * max Re log psi of the last batch of samples."""
        s = self._make_sampler(eps=0.5, maxLogPsi=-2.0)
        _, coeffs, _ = s.sample()
        expected = jnp.max(s.mu * jnp.real(coeffs))
        self.assertTrue(jnp.allclose(s.maxLogPsi, expected))
        self.assertTrue(jnp.allclose(s.cutoff, expected + jnp.log(0.5)))

if __name__ == "__main__":
    unittest.main()