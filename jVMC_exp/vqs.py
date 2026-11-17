import jax
from jax import numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten, tree_map
import flax
from flax.core.frozen_dict import freeze
import flax.linen as nn
import collections
from typing import Tuple
from functools import reduce
import copy
import flax.linen as nn
from flax import nnx

from jVMC_exp.nets.two_nets_wrapper import TwoNets
from jVMC_exp.symmetry import SymmetryProjector, ProjectedOrbitNet
from jVMC_exp.util.grads import pick_gradient
from jVMC_exp.util.key_gen import generate_seed, format_key
from jVMC_exp.util.util import has_callable_attr
from jVMC_exp import global_defs
from jVMC_exp.sharding_config import MESH, DEVICE_SPEC, DEVICE_SHARDING, REPLICATED_SHARDING
from jVMC_exp.sharding_config import broadcast_split_key, sharded, BatchLayout, SizedIterable, pad_to_devices

def _check_dtype(dtype: jnp.dtype | None, dtype_ref: jnp.dtype, label: str):
    if dtype is None:
        return dtype_ref
    ref_is_real = jnp.issubdtype(dtype_ref, jnp.floating)
    given_is_real = jnp.issubdtype(dtype, jnp.floating) 
    if ref_is_real and not given_is_real:
        raise TypeError(
            f"The given dtype for {label} is complex but the parameters are real"
        )
    if not ref_is_real and given_is_real:
        raise TypeError(
            f"The given dtype for {label} is real but the parameters are complex"
        )
    
    return dtype

def _cast_pytree(pytree, dtype):
    return jax.tree_util.tree_map(lambda x: x.astype(dtype), pytree)

def _check_model(model, nnx_init_kwargs=None):
    if isinstance(model, nn.Module):
        return model
    
    if isinstance(model, nnx.Module):
        raise ValueError(
            "Pass the NNX class, not an instance: "
            f"use {type(model).__name__} instead of {type(model).__name__}(...)"
        )
    
    if isinstance(model, type) and issubclass(model, nnx.Module):
        kwargs = nnx_init_kwargs or {}
        return nnx.bridge.to_linen(model, **kwargs)
    
    raise ValueError(f"Expected a flax.linen.Module instance or a flax.nnx.Module class, got {type(model)}")
        
class NQS:
    """
    Variational quantum state.

    The wave function is defined by a network, which is evaluated, differentiated and sampled \
    through this class. It can take two forms:

    #. Single-network ansatz: quantum state of the form \
       :math:`\\psi_\\theta(s)\\equiv\\exp(r_\\theta(s))`, where the network :math:`r_\\theta` is \
       either holomorphic, i.e., parametrized by complex valued parameters :math:`\\vartheta`, or \
       non-holomorphic, parametrized by real or complex valued parameters.
    #. Two-network ansatz: quantum state of the form \
       :math:`\\psi_\\theta(s)\\equiv\\exp(r_{\\theta_r}(s)+i\\varphi_{\\theta_\\phi}(s))` with an \
       amplitude network :math:`r_{\\theta_{r}}` and a phase network \
       :math:`\\varphi_{\\theta_\\phi}`, parametrized by real valued parameters \
       :math:`\\theta_r,\\theta_\\phi`.

    The type of parameters and holomorphicity are detected at construction, see ``realParams`` \
    and ``holomorphic``.

    The ``__call__`` method of a network takes a single configuration and returns a scalar. \
    Vectorization, batching and the distribution across devices are handled by this class: all \
    its methods take configurations as an array of shape ``(N, *sampleShape)``, with any ``N``, \
    and return results in the same order. The attribute ``apply_fun(parameters, s)`` evaluates \
    :math:`\\ln\\psi(s)` of a single configuration, without these steps.

    A network keeps its variables in the ``params`` collection, with a single data type, and can \
    provide these optional methods:

    * ``sample(key)``: Returns a configuration drawn directly from the distribution of the network. \
      Samplers then use direct sampling instead of MCMC, see ``is_generator``.
    * ``eval_ratio(s, sp)``: Returns :math:`\\psi(s')/\\psi(s)`. MCMC and ``call_ratio()`` then use \
      it instead of evaluating the two coefficients. It is not used if a symmetry projector is \
      applied, see ``eval_ratio``.
    * ``eval_real(s)``: Returns :math:`\\text{Re}\\ln\\psi(s)`. MCMC then uses it to compute \
      probabilities.

    Args:
        * ``net``: Variational network, tuple of two networks, or ``flax.nnx.Module`` subclass. \
          If a tuple of two networks is given, the first is used for the logarithmic \
          amplitude and the second for the phase of the wave function coefficient. \
          If a ``flax.nnx.Module`` subclass is given, it will be automatically wrapped \
          into a ``flax.linen``-compatible module via ``flax.nnx.bridge.to_linen``. \
          In this case, its constructor arguments are given by ``nnx_init``.
        * ``sampleShape``: Shape of a configuration. An ``int`` ``n`` stands for ``(n,)``.
        * ``batchSize``: Total batch size for batched network evaluation, divisible by the number \
          of devices. Choice of this parameter impacts performance: with too small values \
          performance is limited by memory access overheads, too large values can lead \
          to "out of memory" issues. Exactly one of ``batchSize`` and ``batchSize_per_device`` \
          must be given.
        * ``batchSize_per_device``: Batch size on each device, as an alternative to ``batchSize``.
        * ``logarithmic``: Boolean variable indicating, whether the ANN returns logarithmic \
          (:math:`\\log\\psi_\\theta(s)`) or plain (:math:`\\psi_\\theta(s)`) wave function coefficients.
        * ``seed``: Seed for the PRNG to initialize the network parameters. If ``None``, a random \
          seed is generated.
        * ``orbit``: Symmetry projector defining the symmetry operations (instance of \
          ``jVMC_exp.symmetry.SymmetryProjector``). If this argument is given, the wave function is \
          symmetrized to be invariant under symmetry operations.
        * ``symmetry_average``: Built-in symmetry average name (``"exp"``, ``"log"`` or ``"sep"``) \
          or callable passed to ``ProjectedOrbitNet``.
        * ``sampler_dtype``: Data type of the parameters while sampling. If ``None``, the data \
          type of the parameters is used.
        * ``eval_dtype``: Data type of the parameters while evaluating coefficients. If ``None``, \
          the data type of the parameters is used.
        * ``grad_dtype``: Data type of the parameters while computing gradients. If ``None``, the \
          data type of the parameters is used.
        * ``nnx_init``: Dictionary of keyword arguments passed to the ``flax.nnx.Module`` constructor, \
          excluding ``rngs`` (which is handled internally). It is ignored for ``flax.linen`` \
          networks. If ``net`` is a tuple of two ``flax.nnx.Module`` subclasses, ``nnx_init`` must be \
          a tuple of two dictionaries, one per network.

    The three data types allow to run these steps at lower precision than the stored parameters. \
    They must be real for real parameters and complex for complex parameters. Coefficients and \
    gradients are returned with data type ``out_dtype`` in any case.

    Example:
        Using a ``flax.linen`` model::

            psi = NQS(RBMLinenModel(numHidden=4), sampleShape, batchSize=32, seed=0)

        Using a ``flax.nnx`` model::

            psi = NQS(RBMNNXModel, sampleShape, batchSize=32, seed=0,
                      nnx_init=dict(in_features=10, numHidden=4, bias=True))

        Using two ``flax.nnx`` models::

            psi = NQS((AmplitudeNNX, PhaseNNX), sampleShape, batchSize=32, seed=0,
                      nnx_init=(dict(in_features=10, numHidden=4),
                                dict(in_features=10, numHidden=4)))
    """
    def __init__(
            self, net: nn.Module | Tuple[nn.Module, nn.Module], 
            sampleShape: int | tuple, 
            batchSize: int | None = None, batchSize_per_device: int | None = None, 
            logarithmic=True, 
            seed: None | int = None, 
            orbit=None, symmetry_average="exp",
            sampler_dtype: jnp.dtype |None = None, 
            eval_dtype: jnp.dtype |None = None,
            grad_dtype: jnp.dtype |None = None,
            nnx_init=None
        ):
        if isinstance(net, collections.abc.Iterable):
            if len(net) != 2:
                raise ValueError(f"If a tuple is passed for 'net', this must have len 2. Got {len(net)}.") 
            if nnx_init is not None:
                if not isinstance(nnx_init, collections.abc.Iterable):
                    raise ValueError(f"If a tuple is passed for 'net', nnx_init can be either None or a tuple of len 2.")
                if len(nnx_init) != 2:
                    raise ValueError("If a tuple is passed for 'net', and nnx_init is not None, nnx_init must be a tuple of len 2."
                                     f"Got {len(nnx_init)}")
            else:
                nnx_init = (None, None)
            net = tuple(_check_model(n, i) for n, i in zip(net, nnx_init))
            net = TwoNets(net)
        else:
            net = _check_model(net, nnx_init)
            
        if orbit is not None:
            if not isinstance(orbit, SymmetryProjector):
                raise TypeError(
                    f"Orbit has to be an instance of jVMC_exp.symmetry.SymmetryProjector, "
                    f"got {orbit}"
                )
            net = ProjectedOrbitNet(base_net=net, symmetry=orbit, symmetry_average=symmetry_average)
        self._net = net

        if isinstance(self._net, ProjectedOrbitNet):
            self._is_generator = has_callable_attr(net.base_net, "sample")
        else:
            self._is_generator = has_callable_attr(net, "sample")
        
        self._eval_ratio = callable(getattr(net, "eval_ratio", None))

        if isinstance(sampleShape, tuple):
            self._sampleShape = sampleShape
        else:
            self._sampleShape = (sampleShape,)
        
        num_devices = MESH.size
        if (batchSize is None) == (batchSize_per_device is None):
            raise ValueError("Exactly one of 'batchSize' or 'batchSize_per_device' must be specified")
        if batchSize is None:
            batchSize = batchSize_per_device * num_devices
        elif batchSize % num_devices != 0:
            raise ValueError(f"The batch size ({batchSize}) has to be divisible by the number of devices ({num_devices})")
        self._batchSize = batchSize

        if logarithmic:
            self.apply_fun = self.net.apply
        else:
            def apply_fun(parameters, s, *args, method=None, **kwargs):
                out = self.net.apply(parameters, s, *args, method=method, **kwargs)

                return jnp.log(out) if method is None else out
            self.apply_fun = apply_fun
        
        self.init_net(seed)

        self._sampler_dtype = _check_dtype(sampler_dtype, self.param_dtype, "sampler")
        self._eval_dtype = _check_dtype(eval_dtype, self.param_dtype, "eval")
        self._grad_dtype = _check_dtype(grad_dtype, self.param_dtype, "grad") 

        self._append_gradients_dict_single = lambda x, y: tree_map(
            lambda a, b: jnp.concatenate((a, 1.j * b)), x, y
        )
        self._append_gradients_dict = lambda x, y: tree_map(
            lambda a, b: jnp.concatenate((a[:, :], 1.j * b[:, :]), axis=1), x, y
        )
        self._append_gradients_dict_jsh = jax.jit(
            self._append_gradients_dict
        )

        self._frozen_params = None

    @property
    def parameters(self):
        """
        Variational parameters, as a dictionary with the parameter tree of the network under ``params``.

        A new value can be a flat array, as given by ``parameters_flat``, or a parameter tree, with \
        or without the ``params`` level. It is copied and cast to ``param_dtype``. The parameters \
        listed in ``frozen_parameters`` keep their current value.
        """
        return self._parameters
    
    @parameters.setter
    def parameters(self, value):
        if hasattr(value, "shape"):
            if len(value) != len(self.parameters_flat):
                raise ValueError(
                    f"The given number of parameters ({len(value)}) "
                    f"does not match the existing one ({len(self.parameters_flat)})"
                )
            value = self._param_unflatten(value)

        if 'params' not in value.keys():
            value = {'params': value}
        if jax.tree_util.tree_structure(value["params"]) != self._netTreeDef:
            raise ValueError(
                "Parameter tree structure mismatch.\n"
                f"  Expected: {self._netTreeDef}\n"
                f"  Received: {jax.tree_util.tree_structure(value['params'])}"
            )

        if self._frozen_params is not None:
            value = tree_map(lambda x: x, value)
            for frozen_path in self._frozen_params:
                original = reduce(lambda d, key: d[key], frozen_path, self.params)
                reduce(lambda d, k: d[k], frozen_path[:-1], value['params'])[frozen_path[-1]] = original
        if isinstance(self.parameters, flax.core.frozen_dict.FrozenDict):
            value = freeze(value)

        value = jax.tree_util.tree_map(lambda x: x.astype(self.param_dtype), value)
        self._parameters = copy.deepcopy(value)

    @property
    def params(self):
        """Parameter tree of the network. A new value can be given as for ``parameters``."""
        return self.parameters['params']
    
    @params.setter
    def params(self, value):
        self.parameters = value

    @property
    def parameters_flat(self):
        """
        Get variational parameters.

        Complex parameters are split into their real and imaginary parts: each leaf of the parameter \
        tree contributes its real parts followed by its imaginary parts, so that the array has \
        ``2 * numParameters`` entries. Real parameters give ``numParameters`` entries.

        Returns:
            Real array holding current values of all variational parameters.
        """
        if not self.realParams:
            flat = jnp.concatenate([
                jnp.concatenate([p.ravel().real,p.ravel().imag,]) for p in tree_flatten(self.params)[0]
            ])
            return flat.astype(global_defs.DT_OUT_REAL)
        
        return jnp.concatenate([p.ravel() for p in tree_flatten(self.params)[0]]).astype(global_defs.DT_OUT_REAL)
    
    @property
    def frozen_parameters(self):
        """
        Parameters that keep their value when ``parameters`` is assigned, as a list of paths.

        A new value adds one path to the list: a ``str`` for an entry of the parameter tree, or a \
        tuple or list of ``str`` for a nested one, e.g., ``("Dense_0", "kernel")``. ``None`` clears \
        the list. Frozen parameters still appear in ``parameters_flat`` and in the gradients.
        """
        return self._frozen_params

    @frozen_parameters.setter
    def frozen_parameters(self, labels: list[str] | None | str):
        if labels is not None:
            if isinstance(labels, str):
                labels = (labels,)
            try:
                reduce(lambda d, key: d[key], labels, self.parameters['params'])
            except KeyError as e:
                raise ValueError(f'The given label {e} does not exist in parameters')
            
            if self._frozen_params is None:
                self._frozen_params = []
            if labels not in self._frozen_params:
                self._frozen_params.append(labels)
        else:
            self._frozen_params = None

    @property
    def sampler_parameters(self):
        """Parameters cast to ``sampler_dtype``."""
        return _cast_pytree(self.parameters, self.sampler_dtype)
    
    @property
    def eval_parameters(self):
        """Parameters cast to ``eval_dtype``."""
        return _cast_pytree(self.parameters, self.eval_dtype)
    
    @property
    def grad_parameters(self):
        """Parameters cast to ``grad_dtype``."""
        return _cast_pytree(self.parameters, self.grad_dtype)
    
    @property
    def batchSize(self):
        """Total batch size for batched network evaluation."""
        return self._batchSize

    @property
    def gradientBatchSize(self):
        """
        Number of gradient rows per block of ``lazy_gradients()``: ``batchSize`` times the number of \
        symmetry elements of the projector, if there is one.
        """
        nsymm = self.net.symmetry.nsymm if isinstance(self.net, ProjectedOrbitNet) else 1
        return self.batchSize * nsymm
        
    @property
    def net(self):
        """Network of the state, including the symmetry projector and the two-network wrapper."""
        return self._net
    
    @property
    def sampleShape(self):
        """Shape of a configuration."""
        return self._sampleShape

    def to_array(self, basis, *, normalize=True):
        """
        Evaluate the variational state on an ordered computational basis.

        Args:
            * ``basis``: Ordered basis with shape ``(dimension, *sampleShape)``.
            * ``normalize``: Normalize the returned vector.

        Returns:
            State vector with shape ``(dimension,)``.
        """
        basis = jnp.asarray(basis)

        if basis.ndim < 2 or tuple(basis.shape[1:]) != tuple(self.sampleShape):
            raise ValueError(
                f"Expected basis shape (dimension, {self.sampleShape}), got {basis.shape}."
            )

        if basis.shape[0] == 0:
            raise ValueError("The basis is empty.")

        values = self(basis).reshape(-1)

        if values.shape[0] != basis.shape[0]:
            raise RuntimeError(
                "The number of amplitudes does not match "
                "the basis dimension."
            )

        if normalize:
            values -= jnp.max(jnp.real(values))
        statevector = jnp.exp(values)

        if normalize:
            norm = jnp.linalg.norm(statevector)

            if not bool(jnp.isfinite(norm)) or bool(norm == 0):
                raise ValueError(
                    f"Cannot normalize statevector "
                    f"with norm {norm}."
                )

            statevector /= norm

        return statevector

    @property
    def is_generator(self):
        """Whether the network, or the one under the symmetry projector, provides ``sample()``."""
        return self._is_generator

    @property
    def eval_ratio(self):
        """Whether the network provides ``eval_ratio()``. A symmetry projector does not."""
        return self._eval_ratio
    
    @property
    def holomorphic(self):
        """Whether the network is holomorphic in its parameters, as detected at construction."""
        return self._holomorphic
    
    @property
    def realParams(self):
        """Whether the parameters are real."""
        return self._realParams
    
    @property
    def flat_gradient_function(self):
        """Called as ``f(apply_fun, parameters, s)`` to get the flat gradient for a single configuration."""
        return self._flat_gradient_function
    
    @property
    def dict_gradient_function(self):
        """Called as ``f(apply_fun, parameters, s)`` to get the gradient tree for a single configuration."""
        return self._dict_gradient_function
    
    @property
    def paramShapes(self):
        """List with ``(size, shape)`` of every leaf of the parameter tree."""
        return self._paramShapes
    
    @property
    def numParameters(self):
        """Number of parameters. A complex parameter counts as one."""
        return self._numParameters
    
    @property
    def param_dtype(self):
        """Data type of the stored parameters."""
        return self._param_dtype
    
    @property
    def sampler_dtype(self):
        """Data type of the parameters while sampling."""
        return self._sampler_dtype
    
    @property
    def eval_dtype(self):
        """Data type of the parameters while evaluating coefficients."""
        return self._eval_dtype
    
    @property
    def grad_dtype(self):
        """Data type of the parameters while computing gradients."""
        return self._grad_dtype
    
    @property
    def out_dtype(self):
        """Data type of the returned coefficients and gradients."""
        return self._out_dtype
        
    def init_net(self, seed: int | None):
        """
        Initialize the network parameters and detect the type of the network.

        This is called at construction, on a random real configuration.

        Args:
            * ``seed``: Seed for the PRNG. If ``None``, a random seed is generated.
        """
        if seed == None:
            seed = generate_seed()

        # TODO: this is fragile and is only needed for holo check. 
        # Holo should be given by the user at init to make it more genera
        # Otherwise another optional function, which would take an input state as well, could be used to check holo
        dummy_sample = jax.random.normal(jax.random.PRNGKey(seed), self.sampleShape)
        self._parameters = jax.device_put(self.net.init(jax.random.PRNGKey(seed), dummy_sample), REPLICATED_SHARDING)
        self._netTreeDef = jax.tree_util.tree_structure(self.params)
        out = self.apply_fun(self.parameters, dummy_sample)
        is_complex = jnp.issubdtype(out.dtype, jnp.complexfloating)
        self._out_dtype = global_defs.DT_OUT_CPX if is_complex else global_defs.DT_OUT_REAL

        self._realParams, self._holomorphic, self._flat_gradient_function, self._dict_gradient_function = pick_gradient(
            self.apply_fun, self.parameters, dummy_sample
        )

        self._paramShapes = []
        self._numParameters = 0
        for p in tree_flatten(self.params)[0]:
            self._paramShapes.append((p.size, p.shape))
            self._numParameters += p.size 
        self._param_dtype = p.dtype

    def __call__(self, s):
        """
        Evaluate variational wave function.
        
        Compute the logarithmic wave function coefficients :math:`\\ln\\psi(s)` for \
        computational configurations :math:`s`.
        
        Args:
            * ``s``: Array of computational basis states, of shape ``(N, *sampleShape)``.

        Returns:
            Logarithmic wave function coefficients :math:`\\ln\\psi(s)`, of shape ``(N,)``.

        :meta public:
        """ 
        return self._apply_fun_sh(
            s, parameters=self.eval_parameters, batch_size=self.batchSize
        ).astype(self.out_dtype)
    
    @sharded()
    def _apply_fun_sh(self, s, *, parameters, batch_size):
        return self.apply_fun(parameters, s)

    def call_ratio(self, s, sp):
        """
        Compute the ratio of wave function coefficients :math:`\\psi(s')/\\psi(s)`.

        The ``eval_ratio()`` method of the network is used if it exists. Otherwise, the ratio \
        is computed from :math:`\\ln\\psi`.

        Args:
            * ``s``: Array of computational basis states, of shape ``(N, *sampleShape)``.
            * ``sp``: Array of computational basis states :math:`s'`, of the same shape.

        Returns:
            Ratios of wave function coefficients, of shape ``(N,)``.
        """
        if self.eval_ratio:
            return self._apply_ratio_sh(
                s, sp, parameters=self.eval_parameters, batch_size=self.batchSize
            ).astype(self.out_dtype)

        return jnp.exp(self(sp) - self(s))
    
    @sharded()
    def _apply_ratio_sh(self, s, sp, *, parameters, batch_size):
        return self.apply_fun(parameters, s, sp, method=self.net.eval_ratio)

    def gradients(self, s):
        """
        Compute gradients of logarithmic wave function.
        
        Compute gradient of the logarithmic wave function coefficients, \
        :math:`\\nabla\\ln\\psi(s)`, for computational configurations :math:`s`.
        
        Args:
            * ``s``: Array of computational basis states, of shape ``(N, *sampleShape)``.

        Returns:
            An array of shape ``(N, P)`` containing derivatives :math:`\\partial_{\\theta_k}\\ln\\psi(s)` \
            with respect to each variational parameter :math:`\\theta_k` for each \
            input configuration :math:`s`. The columns follow ``parameters_flat``: ``P`` is \
            ``numParameters`` for real parameters, and ``2 * numParameters`` for complex parameters, \
            whose real and imaginary parts are separate columns.
        """
        return self._gradients_sh(
            s, parameters=self.grad_parameters, batch_size=self.batchSize
        ).astype(self.out_dtype)
    
    @sharded(automatic_sharding=True) # TODO: Set flag to False once jax problem is solved
    def _gradients_sh(self, s, *, parameters, batch_size):
        return self.flat_gradient_function(self.apply_fun, parameters, s)
    
    def lazy_gradients(self, s):
        """
        Lazy version of ``gradients``: an iterable that recomputes one gradient block
        at a time. Network evaluations use ``batchSize`` configurations; projected
        networks collect up to ``gradientBatchSize = batchSize * nsymm`` gradient
        rows per block, capped at the padded number of samples. Symmetry is reduced
        before the rows are stored, so it does not multiply the block's row count.

        On more than one device a batch does not consist of consecutive configurations.
        To line up another array with the batches, split it with the ``layout`` of the
        returned iterable, e.g. ``lazy.layout.split(pad_to_devices(weights))``: if the number
        of configurations is not divisible by the number of devices, the configurations are
        zero padded and the batches include the gradients of the padding configurations.

        Args:
            * ``s``: Array of computational basis states, of shape ``(N, *sampleShape)``.

        Returns:
            A ``jVMC_exp.sharding_config.SizedIterable`` yielding the gradient blocks, which can be \
            iterated more than once.
        """
        if self.gradientBatchSize == self.batchSize:
            return self._lazy_gradients_sh(
                s, parameters=self.grad_parameters, batch_size=self.batchSize
            )

        s = jax.device_put(pad_to_devices(s), DEVICE_SHARDING)
        layout = BatchLayout(s.shape[0], min(self.gradientBatchSize, s.shape[0]))
        parameters = self.grad_parameters

        def blocks():
            for i in range(layout.n_batches):
                samples = layout.trim(layout.take(s, i), i)
                yield self._gradients_sh(
                    samples, parameters=parameters, batch_size=self.batchSize
                ).astype(self.out_dtype)

        return SizedIterable(blocks, layout.n_batches, layout.batch_size, layout)
    
    @sharded(automatic_sharding=True, yield_iter=True)
    def _lazy_gradients_sh(self, s, *, parameters, batch_size):
        return self.flat_gradient_function(
            self.apply_fun, parameters, s
        ).astype(self.out_dtype)
    
    def gradients_dict(self, s):
        """
        Compute gradients of logarithmic wave function, organized as a parameter tree.

        Args:
            * ``s``: Array of computational basis states, of shape ``(N, *sampleShape)``.

        Returns:
            A tree with the structure of ``params``, where each leaf has shape ``(N, n)`` and holds \
            the derivatives with respect to the flattened leaf of the parameter tree. For a \
            holomorphic network, ``n`` is twice the size of the leaf, as in ``gradients()``.
        """
        result = self._gradients_dict_sh(s, parameters=self.grad_parameters, batch_size=self.batchSize)
        if self.holomorphic:
            result = self._append_gradients_dict_jsh(result, result)

        return _cast_pytree(result, self.out_dtype)
    
    @sharded(automatic_sharding=True) # TODO: Set flag to False once jax problem is solved
    def _gradients_dict_sh(self, s, *, parameters, batch_size):
        return self.dict_gradient_function(self.apply_fun, parameters, s)
    
    def sample(self, numSamples, key=None):
        """
        Draw configurations directly from the ``sample()`` method of the network.

        Args:
            * ``numSamples``: Number of configurations to draw.
            * ``key``: An instance of ``jax.random.PRNGKey``, or an ``int`` used as seed. If \
              ``None``, a random seed is generated. If several keys are given, the first one is used.

        Returns:
            Configurations of shape ``(numSamples, *sampleShape)``, or ``None`` if the network does \
            not provide ``sample()``.
        """
        if self.is_generator:
            key = format_key(key) 
            if len(key.shape) > 1:
                key = key[0]
            keys = broadcast_split_key(key, numSamples)

            return self._sample(keys, parameters=self.sampler_parameters, batch_size=self.batchSize)

        return None
    
    @sharded()
    def _sample(self, keys, *, parameters, batch_size):
        return self.net.apply(parameters, keys, method=self.net.sample)

    def _param_unflatten(self, P):
        """
        Reshape parameter array update according to net tree structure
        """
        if isinstance(P, dict):
            if 'params' in P.keys():
                return P['params']
            else:
                return P
            
        PTreeShape = []
        start = 0
        for s in self.paramShapes:
            if not self._realParams:
                PTreeShape.append((P[start:start + s[0]] + 1.j * P[start + s[0]:start + 2 * s[0]]).reshape(s[1]))
                start += 2 * s[0]
            else:
                PTreeShape.append(P[start:start + s[0]].reshape(s[1]))
                start += s[0]
        
        return tree_unflatten(self._netTreeDef, PTreeShape)
