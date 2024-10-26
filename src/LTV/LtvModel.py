from src.workflows.task import Task
import pandas as pd
from pathlib import Path


class LTVTask(Task):
    def __init__(
        self,
        name: str,
        columnMonetary: str = "monetary_value",
        columnFrequency: str = "frequency",
        isTraining: bool = False,
        discountRate: float = 0.01,
        numPeriods: int = 180,
        frequency: str = "W"
    ) -> None:
        super().__init__(name)
        """
            Args:
                name, # Nome da tarefa
                isTraining = True # Caso seja para efetuar a predição em um dataset com ou sem o período de observação
                columnMonetary  # Nome da coluna onde encontra-se os valores previstos monetários
                # Nome da coluna onde encontra-se os valores previstos de numeros de  transação
                columnFrequency


        """
        self.isTraining = isTraining
        self.columnMonetary = columnMonetary
        self.columnFrequency = columnFrequency
        self.discountRate = discountRate
        self.numPeriods = numPeriods
        self.frequency = frequency

    def on_run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            Factor # O mutiplicador para ajustar a taxa de juros, para frequencias menores que mÇes
        """

        factor = {"W": 4.345, "M": 1.0, "D": 30, "H": 30 * 24}[self.frequency]

        if self.isTraining:
            self.columnMonetary = "monetary_value_cal"
            self.columnFrequency = "frequency_cal"
            df[f"CLV_cal"] = (df[self.columnMonetary] * df[self.columnFrequency]) / (1 + self.discountRate) ** (self.numPeriods / factor)

            self.columnMonetary = "monetary_value_holdout"
            self.columnFrequency = "frequency_holdout"
            df[f"CLV_holdout"] = (df[self.columnMonetary] * df[self.columnFrequency]) / (1 + self.discountRate) ** (self.numPeriods / factor)

        else:
            assert self.columnMonetary in df.columns
            assert self.columnFrequency in df.columns
            df[f"CLV"] = (df[self.columnMonetary] * df[self.columnFrequency]) / (1 + self.discountRate) ** (self.numPeriods / factor)
    
        return df
