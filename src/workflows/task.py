from __future__ import annotations
from abc import ABC, abstractmethod
from src.workflows.schema import Schema
from typing import Annotated, get_type_hints, get_origin, get_args
import pandas as pd


class Depends:
    def __init__(
        self, source: type[Task] | None = None, *, schema: Schema | None = None
    ) -> None:
        self.source = source or Task
        self.schema = schema

    def match(
        self, is_list: bool, sources: list[Task]
    ) -> tuple[pd.DataFrame | list[pd.DataFrame], list[Task]]:
        matched: list[Task] = []
        remaining: list[Task] = []
        for src in sources:
            if isinstance(src, self.source):
                matched.append(src)
            else:
                remaining.append(src)
        dfs = (src.run() for src in matched)
        return list(dfs) if is_list else next(dfs), remaining


class Task(ABC):
    def __init__(self, name: str) -> None:
        self.name = name
        self.task_in: dict[str, Task] = {}
        self.task_out: dict[str, Task] = {}
        self.output: pd.DataFrame | None = None

        from src.workflows.pipeline import Pipeline

        if ctx := Pipeline.get_context():
            ctx.add_task(self)

    def run(self) -> pd.DataFrame:
        if self.output is not None:
            return self.output
        print(f"Running {self.name} ({self.__class__.__name__})...")

        kwargs = self.get_dependencies()
        self.output = self.on_run(**kwargs)
        return self.output

    @abstractmethod
    def on_run(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def get_dependencies(self) -> dict[str, Task]:
        in_kwargs = get_type_hints(self.on_run, include_extras=True)
        in_kwargs.pop("return", None)
        kwargs = {}
        tasks = list(self.task_in.values())
        for kw, arg in in_kwargs.items():
            if get_origin(arg) is Annotated:
                cls, depends = get_args(arg)
            else:
                cls, depends = arg, Depends()
            matched, tasks = depends.match(get_origin(cls) is not None, tasks)
            kwargs[kw] = matched
        return kwargs

    def associate(self, next: Task):
        self.task_out[next.name] = next
        next.task_in[self.name] = self

    def __rshift__(self, other: Task | list[Task]) -> Task | list[Task]:
        match other:
            case list():
                for t in other:
                    self.associate(t)
            case Task():
                self.associate(other)
        return other

    def __rrshift__(self, other: Task | list[Task]) -> Task:
        match other:
            case list():
                for t in other:
                    t.associate(self)
            case Task():
                other.associate(self)
        return self
