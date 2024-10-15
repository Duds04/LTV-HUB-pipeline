from __future__ import annotations
from datetime import datetime, timedelta
from typing import Type
import numpy.dtypes as dtypes
import pandas as pd


class Schema:

    def __init__(self, fields: list[Field], extra: bool = True) -> None:
        self.fields = fields
        self.extra = extra

    def validate(self, df: pd.DataFrame):
        for field in self.fields:
            if (s := df.get(field.name)) is None:
                raise TypeError(f"missing field {field.name}")
            field.validate(s)
        if not self.extra and (extra := set(df.columns) - {f.name for f in self.fields}):
            raise TypeError(f"unexpected fields {list(extra)}")


class Field:

    type_mapping = {
        int: dtypes.Int64DType | dtypes.Int32DType | dtypes.Int16DType | dtypes.Int8DType,
        float: dtypes.Float64DType | dtypes.Float32DType | dtypes.Float16DType,
        str: dtypes.ObjectDType,
        datetime: dtypes.DateTime64DType,
        timedelta: dtypes.TimeDelta64DType,
    }

    def __init__(self, name: str, t: Type, nullable: bool = True) -> None:
        try:
            self.dtype = self.type_mapping[t]
        except KeyError:
            raise TypeError(f"unsupported type {t}")
        self.name = name
        self.ptype = t
        self.nullable = nullable

    def validate(self, s: pd.Series):
        if not issubclass(type(s.dtype), self.dtype):
            raise TypeError(f"expected {self.ptype.__name__} for field {self.name}, got {s.dtype}")
