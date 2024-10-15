from __future__ import annotations
from collections import deque
from src.workflows.task import Task
from types import TracebackType
import pandas as pd
import threading


class Pipeline:
    def __init__(self) -> None:
        self.tasks: dict[str, Task] = {}

    def add_task(self, task: Task) -> Task:
        self.tasks[task.name] = task
        return task

    def run(self) -> dict[str, pd.DataFrame]:
        initial, final = self.__find_end_tasks()

        done = set()
        tasks = deque(self.tasks[t] for t in initial)
        while tasks:
            task = tasks.popleft()
            if task.name not in done:
                task.run()
                tasks.extend(task.task_out.values())
                done.add(task.name)

        return {t: self.tasks[t].output for t in final}

    def __find_end_tasks(self) -> tuple[set[str], set[str]]:
        initial, final = set(), set()
        for task in self.tasks.values():
            if not task.task_in:
                initial.add(task.name)
            if not task.task_out:
                final.add(task.name)
        return initial, final

    # Desnecessário, porém mt divertido:
    # Referência: https://github.com/pymc-devs/pymc/blob/10c9330e4c55e7c6c0b79dde47c498cdf637df02/pymc3/model.py#L153

    contexts = threading.local()

    def __enter__(self) -> Pipeline:
        type(self).get_contexts().append(self)
        return self

    @classmethod
    def get_contexts(cls) -> list[Pipeline]:
        if not hasattr(cls.contexts, "stack"):
            cls.contexts.stack = []
        return cls.contexts.stack

    @classmethod
    def get_context(cls) -> Pipeline | None:
        """Return the deepest context on the stack."""
        if contexts := cls.get_contexts():
            return contexts[-1]
        else:
            return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        type(self).get_contexts().pop()
