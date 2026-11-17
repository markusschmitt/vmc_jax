from __future__ import annotations

from pathlib import Path
from typing import Any

import flax.serialization
import h5py
import numpy as np


class InputManager:
    """
    Read an HDF5 file written by ``OutputManager``.

    The group ``group`` of the file is read into ``data`` at construction, as nested \
    dictionaries of NumPy arrays with the layout of ``OutputManager.data``. Strings are read as \
    ``bytes``.

    Args:
        * ``path``: HDF5 file. Raises ``FileNotFoundError`` if it does not exist.
        * ``group``: Group of the run in the file, the ``group`` of the ``OutputManager``.

    Example::

        inp = InputManager("results.h5")
        inp.data["observables"]["energy"]["mean"]
        psi.parameters = inp.load_parameters(psi.parameters)            # latest checkpoint
        psi.parameters = inp.load_parameters(psi.parameters, step=0)
        t, weights = inp.get_network_checkpoint(time=1.0)
    """

    def __init__(self, path: str | Path, group: str = "/"):
        self.path: Path | None = None
        self.group = self._normalize_group(group)
        self.data: dict[str, Any] = {}
        self.set_path(path)

    @property
    def get_data(self):
        """The data read from the file, ``data``."""
        return self.data

    @staticmethod
    def _normalize_group(group: str) -> str:
        group = str(group or "/").strip()
        if group == "/":
            return "/"
        return "/" + group.strip("/")

    def _root(self, handle):
        return handle[self.group]

    def set_path(self, path: str | Path) -> None:
        """
        Read the group ``group`` of another file into ``data``. Raises ``FileNotFoundError`` if \
        the file does not exist.
        """
        self.path = Path(path).expanduser()
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.data = self.load_from_h5(self.path, group=self.group)


    def get_network_checkpoint(self, time: float = None, idx: int = None):
        """
        A checkpoint written by ``OutputManager.write_network_checkpoint()``.

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

        times = self.data["network_checkpoints"]["times"]
        checkpoints = self.data["network_checkpoints"]["checkpoints"]

        if len(times) == 0:
            raise RuntimeError("No checkpoints have been recorded yet.")

        if time is not None:
            idx = int(np.argmin(np.abs(np.array(times) - time)))
        elif idx is None:
            idx = -1

        return times[idx], checkpoints[idx]
    
    @staticmethod
    def load_from_h5(path: str | Path, group: str = "/") -> dict:
        """
        Read the group ``group`` of an HDF5 file into nested dictionaries of NumPy arrays.

        Args:
            * ``path``: HDF5 file.
            * ``group``: Group to read.

        Returns:
            Dictionary with a dictionary for every subgroup and an array for every dataset.
        """
        def read_recursive(h5_group):
            out = {}
            for key, item in h5_group.items():
                if isinstance(item, h5py.Dataset):
                    out[key] = item[()]
                elif isinstance(item, h5py.Group):
                    out[key] = read_recursive(item)
            return out

        group = InputManager._normalize_group(group)
        with h5py.File(Path(path), "r") as handle:
            return read_recursive(handle[group])

    def _read_param_tree(self, group) -> dict[str, Any]:
        out = {}
        for key, value in group.items():
            if isinstance(value, h5py.Group):
                out[key] = self._read_param_tree(value)
            else:
                out[key] = value[()]
        return out

    def load_parameters(self, template_params, step: int | str = "latest"):
        """
        Parameters written by ``OutputManager.write_parameters()``.

        Args:
            * ``template_params``: Parameter tree with the structure of the saved one, e.g. \
              ``psi.parameters``.
            * ``step``: Step number of the checkpoint, or ``"latest"`` for the last one written.

        Returns:
            ``template_params`` with the saved values, as NumPy arrays.
        """
        with h5py.File(self.path, "r") as handle:
            params_root = self._root(handle)["parameters"]
            group_name = params_root.attrs["latest"] if step == "latest" else f"{int(step):08d}"
            if isinstance(group_name, bytes):
                group_name = group_name.decode("utf-8")
            state = self._read_param_tree(params_root[group_name])
        return flax.serialization.from_state_dict(template_params, state)