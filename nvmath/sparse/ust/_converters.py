# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This module implements converters for the Universal Sparse Tensor (UST).
"""

__all__ = []

from collections.abc import Sequence

# from numba import njit
import numpy as np

from nvmath.internal import utils
from nvmath.internal.tensor_wrapper import wrap_operand

from ._utils import LevelMap, np_enveloping_type
from .tensor_format import Add, LevelExpr, LevelFormat, Subtract, is_unique


class TensorDecomposer:
    """
    A class that traverses a Universal Sparse Tensor and calls a functor for every
    stored element. The traversal can be used for conversions and debugging but may
    not necessarily provide the most efficient way of implementing a traversal.
    Also see set_kernel and run_kernel.

    Args:
    tensor: the UST tensor for traversal
    functor: a function of the form `def visit(dims, val)`
    """

    def __init__(self, tensor, functor):
        self._tensor = tensor
        self._functor = functor
        self._flist = list(tensor.tensor_format.levels.items())
        self._lvls = [0] * tensor.num_levels  # pre-populate

    def run(self):
        self._iterate(0, 0, None)

    def _iterate(self, idx, p, lasta):
        # All levels exhausted?
        if idx == self._tensor.num_levels:
            assert lasta is None  # consumed add/sub in range
            val = self._tensor.val.tensor
            self._functor(self._tensor.tensor_format.lvl2dim(self._lvls), val[p].item())
            return
        # Inspect level.
        k, v = self._flist[idx]
        if isinstance(v, Sequence):
            fmt, prop = v
        else:
            fmt, prop = v, None
        # Handle level format.
        if fmt == LevelFormat.DENSE or fmt == LevelFormat.BATCH:
            sz = self._tensor.levels[idx]
            for i in range(sz):
                self._lvls[idx] = i
                self._iterate(idx + 1, p * sz + i, lasta)
        elif fmt == LevelFormat.COMPRESSED:
            if isinstance(k, LevelExpr) and isinstance(k.operator, (Add, Subtract)):
                assert lasta is None  # no add/sub nesting
                lasta = idx  # record last add/sub
            pos = self._tensor.pos(idx).tensor
            crd = self._tensor.crd(idx).tensor
            assert pos.ndim == crd.ndim
            adjust = 0
            if pos.ndim > 1:
                # This only happens for prior BATCH. Correct the higher-dimensions in
                # both pos and crd buffers and adjust position by #nnz per batch.
                assert pos.ndim == idx
                for i in range(idx - 1):
                    assert self._flist[i][1] == LevelFormat.BATCH
                    pos = pos[self._lvls[i]]
                    crd = crd[self._lvls[i]]
                adjust = pos[-1].item() * (p - self._lvls[idx - 1]) // self._tensor.levels[idx - 1]
                p = self._lvls[idx - 1]
            lo = pos[p].item()
            hi = pos[p + 1].item()
            for i in range(lo, hi):
                self._lvls[idx] = crd[i].item()
                self._iterate(idx + 1, i + adjust, lasta)
        elif fmt == LevelFormat.SINGLETON:
            crd = self._tensor.crd(idx).tensor
            assert crd.ndim == 1
            self._lvls[idx] = crd[p].item()
            self._iterate(idx + 1, p, lasta)
        elif fmt == LevelFormat.RANGE:
            assert lasta is not None  # single add/sub-range relation
            add, _ = self._flist[lasta]
            isI = k == add.expression2
            di = self._tensor.tensor_format.dimensions.index(k)
            dj = self._tensor.tensor_format.dimensions.index(add.expression1 if isI else add.expression2)
            szi = self._tensor.extents[di]
            szj = self._tensor.extents[dj]
            lsz = self._lvls[lasta]
            if isinstance(add.operator, Add):
                off = lsz
                lo = max(0, off - szj + 1)
                hi = min(szi, off + 1)
            else:
                off = -lsz if isI else lsz
                lo = max(0, off)
                hi = min(szi, szj + off)
            for i in range(lo, hi):
                self._lvls[idx] = i
                self._iterate(idx + 1, p * szi + i, None)
        elif fmt == LevelFormat.DELTA:
            assert is_unique(prop)
            pos = self._tensor.pos(idx).tensor
            crd = self._tensor.crd(idx).tensor
            assert pos.ndim == 1 and crd.ndim == 1
            corig = 0
            lo = pos[p].item()
            hi = pos[p + 1].item()
            for i in range(lo, hi):
                corig += crd[i].item()
                self._lvls[idx] = corig
                self._iterate(idx + 1, i, lasta)
                corig += 1
        else:
            raise AssertionError(f"Unsupported: {fmt}")


class TensorComposer:
    """
    A class to build a Universal Sparse Tensor from a coordinate list. This is a convenience
    class for composing UST objects, expected to run on the host, and does not necessarily
    provide the most efficient way of building a UST. Since the builder takes unordered
    dimension indices together with values, it can also be function as intermediate step
    for tensor format conversions.

    Args:
        source: source tensor
        target: target tensor
        indices: a 2-dim array of shape (dim,nse) with dimension indices (on CPU)
        values: a 1-dim array of shape (nse,) with the stored numerical values (on CPU)
        is_sorted: whether coordinate list is lexicographically sorted
    """

    def __init__(self, source, target, indices, values, is_sorted=False):
        assert len(indices.shape) == 2 and len(values.shape) == 1, "Internal error."
        fmt = target.tensor_format
        dim, nse = indices.shape
        assert dim == fmt.num_dimensions and nse == values.shape[0]
        lvl = fmt.num_levels
        # Translate dimension indices to level indices.
        if not fmt.is_identity:
            lindices = np.zeros((lvl, nse), dtype=indices.dtype)
            for i in range(nse):
                lindices[:, i] = fmt.dim2lvl(indices[:, i])
            indices = lindices
            is_sorted = False
        # Apply lexicographical sort on level indices.
        if not is_sorted:
            sorted_indices = np.lexsort(indices[::-1])
            indices = indices[:, sorted_indices]
            values = values[sorted_indices]
        # Bookkeeping.
        self._source = source
        self._target = target
        self._lvl = lvl
        self._flist = list(fmt.levels.values())
        self._indices = indices
        self._values = values
        self._pos_sz = np.zeros((lvl,), dtype=indices.dtype)
        self._crd_sz = np.zeros((lvl,), dtype=indices.dtype)
        self._val_sz = 0

    def run(self, stream=None):
        dim, nse = self._indices.shape

        # Scan data to find sizes. Note that for some forms (like COO and CSR), all sizes
        # can easily be computed deterministically without a first scan. For simplicity,
        # we use the scan for both analysis and insertion, however.
        self._insert_builder(0, lo=0, hi=nse, is_insert=False)

        # Pre-allocate and reset sizes.
        package = self._source._dense_tensorholder_type
        device_id = self._source.device_id  # no device migration in convert()
        stream_holder = None
        if device_id != "cpu":
            stream_holder = utils.get_or_create_stream(device_id, stream, package.name)
        self._pos = LevelMap()
        self._crd = LevelMap()
        for idx in range(self._lvl):
            self._pos[idx] = utils.create_empty_tensor(
                package,
                extents=(self._pos_sz[idx],),
                dtype=self._target.index_type,
                device_id=device_id,
                stream_holder=stream_holder,
                verify_strides=False,
            )
            self._pos_sz[idx] = 0
            self._crd[idx] = utils.create_empty_tensor(
                package,
                extents=(self._crd_sz[idx],),
                dtype=self._target.index_type,
                device_id=device_id,
                stream_holder=stream_holder,
                verify_strides=False,
            )
            self._crd_sz[idx] = 0
        self._val = utils.create_empty_tensor(
            package,
            extents=(self._val_sz,),
            dtype=self._target.dtype,
            device_id=device_id,
            stream_holder=stream_holder,
            verify_strides=False,
        )
        self._val_sz = 0

        # Scan to perform the actual insertion.
        self._insert_builder(0, lo=0, hi=nse, is_insert=True)
        return (self._pos, self._crd, self._val)

    def _insert_builder(self, idx, lo, hi, is_insert):
        # All levels exhausted?
        if idx == self._lvl:
            assert lo < hi
            self._append_val(self._values[lo].item(), 1, is_insert)
            return
        # Handle segments.
        full = 0
        while lo < hi:
            # Find segment.
            crd = self._indices[idx][lo]
            seg = lo + 1
            if is_unique(self._get_prop(idx)):
                while seg < hi and self._indices[idx][seg] == crd:
                    seg += 1
            # Handle level format.
            fmt = self._get_format(idx)
            if fmt == LevelFormat.DENSE or fmt == LevelFormat.RANGE:
                assert crd >= full
                if crd > full:
                    self._segment_builder(idx + 1, 0, crd - full, is_insert)
            elif fmt == LevelFormat.COMPRESSED or fmt == LevelFormat.SINGLETON:
                self._append_crd(idx, crd, 1, is_insert)
            elif fmt == LevelFormat.DELTA:
                d = self._get_prop(idx)
                assert isinstance(d, int)
                mDelta = (1 << d) - 1
                delta = crd - full
                while delta > mDelta:
                    self._append_crd(idx, mDelta, 1, is_insert)
                    self._append_val(0, 1, is_insert)
                    delta -= mDelta + 1
                self._append_crd(idx, delta, 1, is_insert)
            else:
                # TODO: BATCH
                raise NotImplementedError(f"Unsupported: {fmt}")
            full = crd + 1
            self._insert_builder(idx + 1, lo, seg, is_insert)
            lo = seg  # next segment
        # Done with segments.
        self._segment_builder(idx, full, 1, is_insert)

    def _segment_builder(self, idx, full, repeat, is_insert):
        # All levels exhausted?
        if idx == self._lvl:
            self._append_val(0, repeat, is_insert)
            return
        # Handle level format.
        fmt = self._get_format(idx)
        if fmt == LevelFormat.DENSE or fmt == LevelFormat.RANGE:
            sz = self._target.levels[idx]
            assert sz >= full
            self._segment_builder(idx + 1, 0, repeat * (sz - full), is_insert)
        elif fmt == LevelFormat.COMPRESSED or fmt == LevelFormat.DELTA:
            self._append_pos(idx, self._crd_sz[idx], repeat, is_insert)
        elif fmt == LevelFormat.SINGLETON:
            pass  # nothing to do for singleton
        else:
            # TODO: BATCH
            raise NotImplementedError(f"Unsupported: {fmt}")

    def _append_pos(self, idx, pos, repeat, is_insert):
        if self._pos_sz[idx] == 0:  # append initial 0 at start
            if is_insert:
                self._pos[idx].tensor[0] = 0
            self._pos_sz[idx] = 1
        if is_insert:
            start = self._pos_sz[idx]
            self._pos[idx].tensor[start : start + repeat] = pos
        self._pos_sz[idx] += repeat

    def _append_crd(self, idx, crd, repeat, is_insert):
        if is_insert:
            start = self._crd_sz[idx]
            self._crd[idx].tensor[start : start + repeat] = crd
        self._crd_sz[idx] += repeat

    def _append_val(self, val, repeat, is_insert):
        if is_insert:
            start = self._val_sz
            self._val.tensor[start : start + repeat] = val
        self._val_sz += repeat

    def _get_format(self, idx):
        f = self._flist[idx]
        return f[0] if isinstance(f, tuple) else f

    def _get_prop(self, idx):
        f = self._flist[idx]
        return f[1] if isinstance(f, tuple) else None


class TensorConverter:
    """
    Converts a tensor from one format to another (using the [de]composer). The tensor
    converter reduces the O(n^2) conversion problem into a O(n) problem by using a
    [de]composer that converts to an intermediate coordinate list on the CPU first.
    This class is provided to enable conversions between any UST formats. However,
    this does not necessarily provide the most efficient way of implementing a
    conversion. Fast paths (or library dispatches) can be used for efficiency.

              CSR\\         //CSR
              CSC \\       // CSC
          UST ...   > COO <   ... UST
              ELL //  on   \\ ELL
              DIA//   CPU   \\DIA

    Args:
        source: source tensor
        target: target tensor (partially constructed)
    """

    def __init__(self, source, target):
        dim = source.num_dimensions
        nse = source.nse
        # Set fields.
        self._source = source
        self._target = target
        self._indices = np.empty((dim, nse), dtype=target.index_type)
        container_tp = np_enveloping_type(source.dtype)
        self._values = np.empty((nse,), dtype=container_tp)
        self._pos = 0

    def run(self, stream=None):
        # Intermediate COO is sorted if source format is sorted
        # and uses an identity mapping. This may skip sorted
        # if the target format has the same properties.
        sformat = self._source.tensor_format
        is_sorted = sformat.is_ordered and sformat.is_identity

        # Run the decomposer and composer.
        TensorDecomposer(self._source, self._visit).run()
        pos, crd, val = TensorComposer(
            self._source,
            self._target,
            self._indices[:, : self._pos],
            self._values[: self._pos],
            is_sorted,
        ).run(stream=stream)

        # Set directly (converter knows about UST conventions).
        for idx in range(self._target.num_levels):
            p = pos.get(idx)
            if p is not None:
                self._target._pos[idx] = wrap_operand(p)
            c = crd.get(idx)
            if c is not None:
                self._target._crd[idx] = wrap_operand(c)
        self._target._val = wrap_operand(val)

    def _visit(self, dims, val):
        if val == 0:
            return  # drop zeros (e.g. from dense or padding)
        pos = self._pos
        self._pos += 1
        self._indices[:, pos] = dims[:]
        self._values[pos] = val
