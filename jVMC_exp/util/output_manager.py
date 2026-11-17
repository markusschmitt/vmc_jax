"""Collection of the data of a run, in memory and in HDF5 files."""
from __future__ import annotations
import time
from pathlib import Path
from typing import Any

import flax.serialization
import h5py
import jax
import jax.numpy as jnp
import numpy as np

def _is_main_process() -> bool:
    return jax.process_index() == 0

def _as_h5_array(value: Any):
    try:
        arr = jax.device_get(jnp.asarray(value))
    except (TypeError, ValueError):
        arr = jax.device_get(value)
    dtype = getattr(arr, "dtype", None)
    if isinstance(arr, (str, bytes)) or dtype is not None and (dtype == object or dtype.kind in {"U", "S"}):
        if hasattr(arr, "shape") and arr.shape == ():
            text = arr.item()
        else:
            text = arr
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        return str(text)
    return arr


def check_network_parameter_shape(weights: Any) -> bool:
    """Whether ``weights`` can be stored as one array in an HDF5 file."""
    try:
        _as_h5_array(weights)
    except (TypeError, ValueError):
        return False
    return True


class OutputManager:
    """
    Collect observables, metadata, timings and checkpoints, in memory and in an HDF5 file.

    The data are kept in ``data``, a dictionary of nested dictionaries. Without a path they stay \
    in memory until ``save_to_h5(path)``, which binds the path for later calls. With a path, \
    every write is also appended to the file at once. Only process 0 writes files.

    The optimizers record their measurements in an ``OutputManager``, see \
    ``AbstractOptimizer.ground_state_search()``. ``data`` and the group ``group`` of the file \
    have the layout

    * ``observables/times`` and ``observables/<name>``: ``write_observables()``,
    * ``metadata/...``: ``write_metadata()``,
    * ``network_checkpoints/times`` and ``network_checkpoints/checkpoints``: \
      ``write_network_checkpoint()``,
    * ``parameters/<step>/...``: ``write_parameters()``, with ``<step>`` in eight digits,
    * ``timings/<name>/total`` and ``timings/<name>/count``: ``flush_timings()``,
    * ``<group>/<name>``: ``write_dataset()``.

    A dictionary among the values of a write becomes a group, e.g. \
    ``write_observables(t, energy={"mean": e, "variance": v})`` writes ``observables/energy/mean`` \
    and ``observables/energy/variance``.

    Args:
        * ``path`` [optional]: HDF5 file. Its directory is created if needed.
        * ``group``: Group of the file under which the data are written, e.g. to keep several \
          runs in one file.
        * ``append``: If ``True``, an existing file is kept, and the writes are appended to the \
          datasets it already has. If ``False``, the file is overwritten.

    Example::

        outp = OutputManager("results.h5", group="run_000", append=False)
        opt = jVMC_exp.optimizer.SR(sampler, psi, diagonalShift=1e-3, output_manager=outp)
        opt.ground_state_search(500, energy, jVMC_exp.stepper.Euler(5e-2))
        outp.write_parameters(500, psi.parameters, attrs={"optimizer": "SR"})

        outp = OutputManager()                              # in memory only
        outp.write_metadata(L=16, model="RBM")
        outp.write_observables(0, energy=-1.0, magnetization={"mean": 0.2})
        outp.start_timing("step")
        ...
        outp.stop_timing("step")
        outp.flush_timings()
        outp.save_to_h5("results.h5")
    """

    def __init__(self, path: str | Path | None = None, group: str = "/", append: bool = True):
        self.path: Path | None = None
        self.group = self._normalize_group(group)
        self.mode = "a"
        self.data: dict[str, Any] = {}
        self._timings: dict[str, dict[str, float | int]] = {}
        self.timings = self._timings
        self._parameter_attrs: dict[str, dict[str, Any]] = {}
        self._parameter_latest: str | None = None
        self.is_main_process = _is_main_process()
        if path is not None:
            self._set_path(path, append=append)

    @staticmethod
    def _normalize_group(group: str) -> str:
        group = str(group or "/").strip()
        if group == "/":
            return "/"
        return "/" + group.strip("/")

    def _set_path(self, path: str | Path, append: bool = True) -> None:
        self.path = Path(path).expanduser()
        self.mode = "a"
        if not self.is_main_process:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with h5py.File(self.path, mode) as handle:
            handle.require_group(self.group)
        
    def _root(self, handle):
        return handle[self.group]

    def _append_h5(self, parent, name: str, value: Any) -> None:
        arr = _as_h5_array(value)
        if arr is None:
            return
        if isinstance(arr, str):
            arr = np.asarray(arr, dtype=h5py.string_dtype(encoding="utf-8"))
        arr = np.asarray(arr)
        item_shape = arr.shape
        if name not in parent:
            parent.create_dataset(
                name,
                shape=(0,) + item_shape,
                maxshape=(None,) + item_shape,
                dtype=arr.dtype,
                chunks=True,
            )
        dataset = parent[name]
        if dataset.chunks is None:
            old_data = dataset[()]
            del parent[name]
            dataset = parent.create_dataset(
                name,
                shape=old_data.shape,
                maxshape=(None,) + old_data.shape[1:],
                dtype=old_data.dtype,
                chunks=True,
            )
            dataset[...] = old_data
        new_len = dataset.shape[0] + 1
        dataset.resize((new_len,) + dataset.shape[1:])
        dataset[-1] = arr

    def _append_nested_h5(self, parent, data: dict[str, Any]) -> None:
        for key, value in data.items():
            if isinstance(value, dict):
                self._append_nested_h5(parent.require_group(str(key)), value)
            else:
                self._append_h5(parent, str(key), value)

    def _add(self, subgroup: str, time_val: float | int, **kwargs) -> None:
        if subgroup not in self.data:
            self.data[subgroup] = {}
        if "times" not in self.data[subgroup]:
            self.data[subgroup]["times"] = []
        self.data[subgroup]["times"].append(time_val)
        self._store_recursive(self.data[subgroup], kwargs)

    def _store_recursive(self, store: dict, data: dict) -> None:
        for key, value in data.items():
            if isinstance(value, dict):
                if key not in store:
                    store[key] = {}
                self._store_recursive(store[key], value)
            else:
                if key not in store:
                    store[key] = []
                store[key].append(value)

    def write_observables(self, step: int | float, **observables) -> None:
        """
        Append the observables of a step: ``step`` to ``observables/times``, and every keyword \
        argument to ``observables/<name>``.

        Args:
            * ``step``: Time or step number.
            * ``**observables``: Numbers, arrays with the same shape at every step, or \
              dictionaries of them.
        """
        self._add("observables", step, **observables)
        if self.path is not None and self.is_main_process:
            with h5py.File(self.path, "a") as handle:
                group = self._root(handle).require_group("observables")
                self._append_h5(group, "times", step)
                self._append_nested_h5(group, observables)

    def write_metadata(self, step: int | float | None = None, **metadata) -> None:
        """
        Write metadata of the run, under ``metadata``.

        Args:
            * ``step`` [optional]: With a step, the metadata are appended to time series as in \
              ``write_observables()``. Without it, every entry is stored once and replaces an \
              earlier entry of the same name.
            * ``**metadata``: Numbers, strings, arrays, or dictionaries of them.
        """
        if step is not None:
            self._add("metadata", step, **metadata)
        else:
            self.data.setdefault("metadata", {}).update(metadata)

        if self.path is not None and self.is_main_process:
            with h5py.File(self.path, "a") as handle:
                group = self._root(handle).require_group("metadata")
                if step is not None:
                    self._append_h5(group, "times", step)
                    self._append_nested_h5(group, metadata)
                else:
                    for key, value in metadata.items():
                        arr = _as_h5_array(value)
                        if key in group:
                            del group[key]
                        group.create_dataset(str(key), data=arr)

    def write_network_checkpoint(self, time: float, weights) -> None:
        """
        Append a checkpoint of the parameters, with its time, under ``network_checkpoints``.

        Args:
            * ``time``: Time of the checkpoint.
            * ``weights``: The parameters as one array, e.g. ``psi.parameters_flat``. Raises \
              ``ValueError`` if they cannot be stored as one array.
        """
        if check_network_parameter_shape(weights):
            self._add("network_checkpoints", time, checkpoints=weights)
        else:
            raise ValueError("Network weights must be serialized for checkpointing.")
        if self.path is not None and self.is_main_process:
            with h5py.File(self.path, "a") as handle:
                group = self._root(handle).require_group("network_checkpoints")
                self._append_h5(group, "times", time)
                self._append_h5(group, "checkpoints", weights)

    def write_dataset(self, name: str, data: Any, group: str = "/", path: str | Path | None = None) -> None:
        """
        Store ``data`` under ``<group>/<name>``, replacing an earlier dataset of that name.

        Args:
            * ``name``: Name of the dataset.
            * ``data``: Number, string or array.
            * ``group``: Group, e.g. ``"diagnostics"``; ``"/"`` is the group of the manager.
            * ``path``: Has to be ``None``, otherwise ``ValueError`` is raised: the file is \
              chosen at construction or with ``save_to_h5()``.
        """
        if path is not None:
            raise ValueError("write_dataset() does not bind paths. Use save_to_h5(path=...) to choose an output file.")
        self._store_dataset(name, data, group=group)
        if self.path is not None and self.is_main_process:
            with h5py.File(self.path, "a") as handle:
                self._write_dataset_to_h5(handle, name, data, group=group)

    def _store_dataset(self, name: str, data: Any, group: str = "/") -> None:
        store = self.data
        group = self._normalize_group(group)
        if group != "/":
            for part in group.strip("/").split("/"):
                store = store.setdefault(part, {})
        store[str(name)] = data

    def _write_dataset_to_h5(self, handle, name: str, data: Any, group: str = "/") -> None:
        root = self._root(handle)
        parent = root if group == "/" else root.require_group(group.strip("/"))
        if name in parent:
            del parent[name]
        parent.create_dataset(str(name), data=_as_h5_array(data))

    def _write_param_tree(self, group, tree: dict[str, Any]) -> None:
        for key, value in tree.items():
            if isinstance(value, dict):
                self._write_param_tree(group.require_group(str(key)), value)
            else:
                if key in group:
                    del group[key]
                group.create_dataset(str(key), data=_as_h5_array(value))

    def _read_param_tree(self, group) -> dict[str, Any]:
        out = {}
        for key, value in group.items():
            if isinstance(value, h5py.Group):
                out[key] = self._read_param_tree(value)
            else:
                out[key] = value[()]
        return out

    def write_parameters(
        self,
        step: int,
        params,
        attrs: dict[str, Any] | None = None,
        path: str | Path | None = None,
    ) -> str:
        """
        Store a parameter tree under ``parameters/<step>`` and mark it as the latest checkpoint.

        Args:
            * ``step``: Step number, which gives the name of the group in eight digits.
            * ``params``: Parameter tree, e.g. ``psi.parameters``.
            * ``attrs`` [optional]: Dictionary of HDF5 attributes of the group, in addition to \
              ``step``.
            * ``path``: Has to be ``None``, otherwise ``ValueError`` is raised.

        Returns:
            The name of the group, e.g. ``"00000500"``. ``InputManager.load_parameters()`` \
            reads it back.
        """
        if path is not None:
            raise ValueError("write_parameters() does not bind paths. Use save_to_h5(path=...) to choose an output file.")
        state = flax.serialization.to_state_dict(params)
        group_name = f"{int(step):08d}"
        self.data.setdefault("parameters", {})[group_name] = state
        checkpoint_attrs = {"step": int(step)}
        checkpoint_attrs.update(attrs or {})
        self._parameter_attrs[group_name] = checkpoint_attrs
        self._parameter_latest = group_name
        if self.path is not None and self.is_main_process:
            with h5py.File(self.path, "a") as handle:
                self._write_parameters_to_h5(handle, group_name, state, checkpoint_attrs)
        return group_name

    def save_to_h5(self, path: str | Path | None = None, append: bool = False) -> None:
        """
        Write ``data`` to the HDF5 file.

        Args:
            * ``path`` [optional]: File to write to. It is bound for later calls. Without it, \
              the bound file is used, and ``ValueError`` is raised if there is none.
            * ``append``: If ``False``, the file is overwritten. If ``True``, the other data in \
              the file are kept, and the datasets of ``data`` replace those of the same name.
        """
        if path is not None:
            self._set_path(path, append=append)
        if self.path is None:
            raise ValueError("A path is required for this operation. Pass a path to save_to_h5().")
        if not self.is_main_process:
            return

        mode = "a" if append else "w"
        with h5py.File(self.path, mode) as handle:
            handle.require_group(self.group)
            self._write_dict_to_h5(self._root(handle), self.data)
            self._write_parameter_attrs_to_h5(handle)

    def _write_parameters_to_h5(self, handle, group_name: str, state: dict[str, Any], attrs: dict[str, Any]) -> None:
        params_root = self._root(handle).require_group("parameters")
        if group_name in params_root:
            del params_root[group_name]
        group = params_root.create_group(group_name)
        for key, value in attrs.items():
            group.attrs[str(key)] = value
        self._write_param_tree(group, state)
        params_root.attrs["latest"] = group_name

    def _write_parameter_attrs_to_h5(self, handle) -> None:
        if "parameters" not in self.data:
            return
        params_root = self._root(handle).require_group("parameters")
        if self._parameter_latest is not None:
            params_root.attrs["latest"] = self._parameter_latest
        for group_name, attrs in self._parameter_attrs.items():
            if group_name not in params_root:
                continue
            for key, value in attrs.items():
                params_root[group_name].attrs[str(key)] = value

    def _write_dict_to_h5(self, parent, data: dict) -> None:
        for key, value in data.items():
            if isinstance(value, dict):
                group = parent.require_group(str(key))
                self._write_dict_to_h5(group, value)
            else:
                if key in parent:
                    del parent[key]
                parent.create_dataset(str(key), data=_as_h5_array(value))

    def start_timing(self, name: str) -> None:
        """Start the timer ``name``."""
        entry = self._timings.setdefault(name, {"total": 0.0, "last_total": 0.0, "count": 0, "start": 0.0})
        entry["start"] = time.perf_counter()

    def stop_timing(self, name: str) -> float:
        """
        Stop the timer ``name`` and add the time since ``start_timing(name)`` to it.

        Returns:
            The elapsed time in seconds.
        """
        elapsed = time.perf_counter() - float(self._timings[name]["start"])
        self.add_timing(name, elapsed)
        return elapsed

    def add_timing(self, name: str, elapsed: float) -> None:
        """Add ``elapsed`` seconds and one call to the timer ``name``."""
        entry = self._timings.setdefault(name, {"total": 0.0, "last_total": 0.0, "count": 0, "start": 0.0})
        entry["total"] = float(entry["total"]) + float(elapsed)
        entry["count"] = int(entry["count"]) + 1

    def print_timings(self, indent: str = "") -> None:
        """
        Print the time of every timer since the last call, on process 0. Every line starts \
        with ``indent``.
        """
        if not self.is_main_process:
            return
        print(f"{indent}Recorded timings:", flush=True)
        for key, item in self._timings.items():
            delta = item["total"] - item["last_total"]
            print(f"{indent}    - {key}: {delta:.6f}s", flush=True)
            item["last_total"] = item["total"]

    def flush_timings(self, path: str | Path | None = None) -> None:
        """
        Store the total time and the number of calls of every timer under \
        ``timings/<name>/total`` and ``timings/<name>/count``. ``path`` has to be ``None``, \
        otherwise ``ValueError`` is raised.
        """
        if path is not None:
            raise ValueError("flush_timings() does not bind paths. Use save_to_h5(path=...) to choose an output file.")
        timings = self.data.setdefault("timings", {})
        for key, value in self._timings.items():
            timings[str(key)] = {
                "total": value["total"],
                "count": value["count"],
            }
        if self.path is not None and self.is_main_process:
            with h5py.File(self.path, "a") as handle:
                self._write_dict_to_h5(self._root(handle).require_group("timings"), timings)

    def get_network_checkpoint(self, time: float | None = None, idx: int | None = None):
        """
        A checkpoint written by ``write_network_checkpoint()``.

        Args:
            * ``time`` [optional]: Return the checkpoint closest to this time.
            * ``idx`` [optional]: Return the checkpoint with this index. Without ``time`` and \
              ``idx``, the last one is returned.

        Returns:
            ``time, weights`` of the checkpoint. Raises ``ValueError`` if both ``time`` and \
            ``idx`` are given, and ``RuntimeError`` if no checkpoint was written.
        """
        if time is not None and idx is not None:
            raise ValueError("Specify either 'time' or 'idx', not both.")
        checkpoints = self.data.get("network_checkpoints", {})
        times = checkpoints.get("times", [])
        weights = checkpoints.get("checkpoints", [])
        if not times:
            raise RuntimeError("No checkpoints have been recorded yet.")
        if time is not None:
            idx = int(np.argmin(np.abs(np.asarray(times) - time)))
        elif idx is None:
            idx = -1
        return times[idx], weights[idx]