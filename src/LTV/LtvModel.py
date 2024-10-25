from src.workflows.task import Task
import pandas as pd
from pathlib import Path


class LTVTask(Task):
    def __init__(
        self,
        name: str,
        isTeste: bool = True, 
        columnMonetary: str = "monetary_value",
        columnFrequency: str = "frequency",
        discountRate: float = 0.06,
    ) -> None:
        super().__init__(name)
        self.isTeste = isTeste
        self.columnMonetary = columnMonetary
        self.columnFrequency = columnFrequency

    def on_run(self, df: pd.DataFrame, modelo) -> pd.DataFrame:

        return df
