# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

import glob
import json
import numpy as np
from collections import namedtuple
from abc import ABC, abstractmethod
from .common_cuda import ComputeCapability

Record = namedtuple("Record", ["arch", "fft_type", "precision", "direction", "size", "elements_per_thread", "ffts_per_block"])


def _update(dict, **kwargs):
    dict = dict.copy()
    for k, v in kwargs.items():
        if v is None:
            del dict[k]
        else:
            dict[k] = v
    return dict


#
# Those transformations take care of mapping a set of constraints (e.g. arch=800,
# fft_type=r2c, size=32) from the "frontend API" to the "database". This is necessary
# because the mapping is not bijective: multiple APIs map to the same implementation For
# instance (fft_type=C2C, size=32) and (fft_type=R2C, size=32) both map to (fft_type=C2C,
# size=32) under the hood Those mapper take care of
# [1] Doing the forward mapping (frontend -> db), which is injective - the `fwd` methods
# [2] Undoing the mapping (which is doable because we keep track of the forward mapping
# internally) - the `inv` methods
#
class Mapper(ABC):
    @staticmethod
    @abstractmethod
    def required(**kwargs):
        pass

    @abstractmethod
    def fwd(self, **kwargs):
        pass

    @abstractmethod
    def inv(self, **kwargs):
        pass


# Fwd: Drop ffts_per_block
# Inv: Restore ffts_per_block
class FFTsPerBlock(Mapper):
    @staticmethod
    def required(**kwargs):
        return "ffts_per_block" in kwargs

    def fwd(self, **kwargs):
        self._ffts_per_block = kwargs["ffts_per_block"]
        return _update(kwargs, ffts_per_block=None)

    def inv(self, **kwargs):
        return _update(kwargs, ffts_per_block=self._ffts_per_block)


# Fwd: If execution=Thread, drop execution and add elements_per_thread=size
#      If execution=Block, drop execution
# Inv: If original execution=Thread, restore execution=Thread and drop elements_per_thread
#      If original execution=Block, restore execution=Block
class ThreadToBlock(Mapper):
    @staticmethod
    def required(**kwargs):
        return True

    def fwd(self, **kwargs):
        self._execution = kwargs["execution"]
        if self._execution == "Thread":
            assert "elements_per_thread" not in kwargs
            return _update(kwargs, execution=None, elements_per_thread=kwargs["size"])
        else:
            return _update(kwargs, execution=None)

    def inv(self, **kwargs):
        if self._execution == "Thread":
            return _update(kwargs, execution="Thread", elements_per_thread=None)
        else:
            return _update(kwargs, execution="Block")


# Fwd: If fft_type=(R2C|C2R) and real_fft_options[real_mode]=folded, set fft_type=C2C and
# size/=2 and elements_per_thread/=2 (if set) and drop real_fft_options
# Inv: Restore the original R2C/C2R. If elements_per_thread was set, restore it
class FoldedToC2C(Mapper):
    # This should be the complement of R2CC2RToC2C.required
    @staticmethod
    def required(**kwargs):
        return (kwargs["fft_type"] == "c2r" or kwargs["fft_type"] == "r2c") and (
            ("real_fft_options" in kwargs) and (kwargs["real_fft_options"]["real_mode"] == "folded")
        )

    def fwd(self, **kwargs):
        assert FoldedToC2C.required(**kwargs)
        self._fft_type = kwargs["fft_type"]
        self._real_fft_options = kwargs["real_fft_options"]
        if "elements_per_thread" in kwargs:
            return _update(
                kwargs,
                fft_type="c2c",
                size=kwargs["size"] / 2,
                real_fft_options=None,
                elements_per_thread=kwargs["elements_per_thread"] / 2,
            )
        else:
            return _update(kwargs, fft_type="c2c", size=kwargs["size"] / 2, real_fft_options=None)

    def inv(self, **kwargs):
        assert "elements_per_thread" in kwargs
        return _update(
            kwargs,
            fft_type=self._fft_type,
            size=kwargs["size"] * 2,
            elements_per_thread=kwargs["elements_per_thread"] * 2,
            real_fft_options=self._real_fft_options,
        )


# Fwd: Map C2R/R2C to C2C
# Inv: Restore C2R/R2C
class R2CC2RToC2C(Mapper):
    # This should be the complement of FoldedToC2C.required
    @staticmethod
    def required(**kwargs):
        return (kwargs["fft_type"] == "c2r" or kwargs["fft_type"] == "r2c") and (
            ("real_fft_options" not in kwargs)
            or (("real_fft_options" in kwargs) and (kwargs["real_fft_options"]["real_mode"] != "folded"))
        )

    def fwd(self, **kwargs):
        self._fft_type = kwargs["fft_type"]
        self._real_fft_options = kwargs.get("real_fft_options")
        return _update(kwargs, fft_type="c2c", real_fft_options=None)

    def inv(self, **kwargs):
        assert kwargs["fft_type"] == "c2c"
        if self._real_fft_options is not None:
            return _update(kwargs, fft_type=self._fft_type, real_fft_options=self._real_fft_options)
        else:
            return _update(kwargs, fft_type=self._fft_type)


# All the transformations from Frontend to DB, to apply
_OPS = [FFTsPerBlock, ThreadToBlock, FoldedToC2C, R2CC2RToC2C]


# Map a Frontend record to a DB record, and record the transformations applied
def frontend_to_db(fe):
    transforms = []
    for OP in _OPS:
        if OP.required(**fe):
            op = OP()
            fe = op.fwd(**fe)
            transforms.append(op)
    return (transforms, fe)


# Inverse the mapping
def db_to_frontend(transforms, db):
    for op in reversed(transforms):
        db = op.inv(**db)
    return db


class cuFFTDxDatabase:
    # Database must be a list of Record
    def __init__(self, records):
        self._records = records

    @property
    def records(self):
        return self._records

    @staticmethod
    def create(database_dir):
        # Load JSON database
        # format:
        # database[arch][(fft_type, prec, dir)] = [
        # { size, blobs = [ { id, ept, fpb, storage, smem }, ... ] }
        # ]
        database = {}

        for f in glob.glob(f"{database_dir}/**/*.json"):
            with open(f) as file:
                data = json.load(file)
                arch, fft_type, prec, dir = data["architecture"], data["type"], data["precision"], data["direction"]
                # if arch != 800 or prec != 'fp32' or dir != 'forward':
                #     continue
                if arch not in database:
                    database[arch] = {(fft_type, prec, dir): data["database"]}
                else:
                    database[arch][(fft_type, prec, dir)] = data["database"]

        # Flatten DB and create a list of Records
        records = []

        PREC_MAP = {
            "fp64": np.float64,
            "fp32": np.float32,
            "fp16": np.float16,
        }

        for arch in database:
            for (fft_type, prec, dir), db in database[arch].items():
                for record in db:
                    size = record["size"]
                    for blob in record["blobs"]:
                        records.append(
                            Record(
                                ComputeCapability(arch // 100, (arch % 100) // 10),
                                fft_type,
                                PREC_MAP[prec],
                                dir,
                                size,
                                blob["ept"],
                                blob["fpb"],
                            )
                        )

        return cuFFTDxDatabase(records)

    def query(self, knobs, frontend_constraints):
        # Map input constraints (expressed in frontend API terms) to DB constraints
        transforms, db_constraints = frontend_to_db(frontend_constraints)

        # Filter DB to extract record matching constraints
        def db_filter(record):
            return all([getattr(record, f) == v for (f, v) in db_constraints.items()])

        db_records = list(filter(db_filter, self.records))

        # Invert mapping to recover records in terms of frontend API
        frontend_records = [db_to_frontend(transforms, db_record._asdict()) for db_record in db_records]

        # Drop whatever could not be inverted
        frontend_records = [fe_record for fe_record in frontend_records if fe_record is not None]

        # Extract only knobs required by user
        fe_record_knobs = [tuple(fe_record[k] for k in knobs) for fe_record in frontend_records]

        # Sort + filter out duplicate (to ensure stability)
        return sorted(list(set(fe_record_knobs)))
