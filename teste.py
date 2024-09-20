from pathlib import Path
from datetime import datetime
from src.task import Task, Depends
from src.pipeline import Pipeline
from src.schema import Schema, Field
from typing import Annotated
import pandas as pd
from lifetimes.utils import (
    calibration_and_holdout_data,
    summary_data_from_transaction_data,
)


# TO DO: se tem ID, data e valor
class CsvReadTask(Task):
    def __init__(self, name: str, fp: str) -> None:
        super().__init__(name)
        self.fp = Path(fp)

    def on_run(self) -> pd.DataFrame:
        df = pd.read_csv(self.fp)
        if "dt" in df.columns:
            df["dt"] = pd.to_datetime(df["dt"])
        return df


class RFMTask(Task):
    def __init__(
        self,
        name: str,
        columnID: str,
        columnData: str,
        columnValor: str,
        frequency: str = "W",
        calibrationEnd=None,
        ObservationEnd=None,
        split: float = 0.8,
        apply_calibration_split: bool = True,
    ) -> None:
        """
        Args:
                columnID #Nome da coluna onde encontra-se os identificadores
                columnData  #Nome da coluna onde encontra-se as datas
                columnValor  #Nome da coluna onde encontra-se os valores monetários
                frequency = 'W' #Frequência em que será observado, Ex: "W" - Weeks
                calibrationEnd = None #Caso queira passar a data do fim do período de calibração
                ObservationEnd = None #Caso queira passar a data do fim do período de Obsersvação
                split = 0.8 # Porcentagem da divisão dos dados para separar em Obsersvação e calibração
                is_calibration_mode = True #Verdadeiro caso queira separar os dados em Obsersvação e calibração
        """
        super().__init__(name)
        self.columnID = columnID
        self.columnData = columnData
        self.columnValor = columnValor
        self.frequency = frequency
        self.calibrationEnd = calibrationEnd
        self.ObservationEnd = ObservationEnd
        self.split = split
        self.apply_calibration_split = apply_calibration_split

    def __getPeriodos(
        self, df: pd.DataFrame, columnData: str, frequency: str, split: float = 0.8
    ):
        """
        Args:
            name, #Nome da tarefa
            df, #Dataframe do Pandas
            columnData, #Nome da coluna onde estão as datas
            frequency, #Frequência em que será observado, Ex: "W" - Weeks
            split = 0.8 #Porcentagem da divisão dos dados para separar em treino e calibração
        """
        assert columnData in df.columns

        firstData = df[columnData].sort_values().values[0]
        lastData = df[columnData].sort_values().values[-1]
        rangeDatas = pd.date_range(start=firstData, end=lastData, freq=frequency)
        indexCut = round(len(rangeDatas) * split)
        return rangeDatas[indexCut], lastData

    def on_run(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.calibrationEnd is None:
            self.calibrationEnd, self.ObservationEnd = self.__getPeriodos(
                df, self.columnData, self.frequency, self.split
            )

        if self.apply_calibration_split is False:
            return summary_data_from_transaction_data(
                transactions=df,
                customer_id_col=self.columnID,
                datetime_col=self.columnData,
                monetary_value_col=self.columnValor,
                freq=self.frequency,
            )
        else:
            rfm_cal_holdout = calibration_and_holdout_data(
                transactions=df,
                customer_id_col=self.columnID,
                datetime_col=self.columnData,
                monetary_value_col=self.columnValor,
                freq=self.frequency,
                calibration_period_end=self.calibrationEnd,
                observation_period_end=self.ObservationEnd,
            )
            return rfm_cal_holdout


# class AgeTransformerTask(Transformer):

#     def on_run(self, df: pd.DataFrame) -> pd.DataFrame:
#         assert "id" in df.columns
#         assert "dt" in df.columns

#         # age in days
#         return (
#             df.groupby("id")
#             .aggregate({"dt": ["min", "max"]})["dt"]
#             .reset_index()
#             .assign(age=lambda df: (df["max"] - df["min"]).dt.days)[["id", "age"]]
#         )


# class AvgTransformerTask(Transformer):

#     def __init__(self, name: str, month: int) -> None:
#         super().__init__(name)
#         self.month = month

#     def on_run(
#         self,
#         df: Annotated[
#             pd.DataFrame,
#             Depends(
#                 schema=Schema(
#                     [Field("id", int), Field("dt", datetime), Field("val", float)]
#                 )
#             ),
#         ],
#     ) -> pd.DataFrame:
#         assert "id" in df.columns
#         assert "dt" in df.columns
#         assert "val" in df.columns
#         return (
#             df[df["dt"].dt.month == self.month]
#             .groupby("id")["val"]
#             .mean()
#             .reset_index()
#             .rename({"val": f"val_m{self.month}"}, axis=1)
#         )


# class SquareTransformerTask(Transformer):

#     def on_run(
#         self, df: Annotated[pd.DataFrame, Depends(schema=Schema([Field("id", int)]))]
#     ) -> pd.DataFrame:
#         cols = (c for c in df.columns if c != "id")
#         transformed = {c: df[c].apply(lambda x: x**2) for c in cols}
#         return df.assign(**transformed)


# class FakeModelTask(Task):

#     def on_run(
#         self,
#         base_df: Annotated[pd.DataFrame, Depends(SquareTransformerTask)],
#         extra_dfs: Annotated[list[pd.DataFrame], Depends(Transformer)],
#     ) -> pd.DataFrame:
#         for df in extra_dfs:
#             base_df = base_df.merge(df, how="outer", on="id")
#         inputs = sum(
#             base_df[c].fillna(0) for c in base_df.columns if c.startswith("val")
#         )
#         return base_df.assign(predicted=inputs / base_df["age"])


# Acabou sendo inutilizado

"""
# Pegar dados e dividir em periodos 
class SplitDataPeriodTask(Task):
    def __init__(self, name: str, columnData: str, frequency: str, split: float = 0.8) -> None:
        # ""
        #     Args:
        #         name, #Nome da tarefa   
        #         df, #Dataframe do Pandas 
        #         columnData, #Nome da coluna onde estão as datas
        #         frequency, #Frequência em que será observado, Ex: "W" - Weeks
        #         split = 0.8 #Porcentagem da divisão dos dados para separar em treino e calibração
        # ""
        
        super().__init__(name)
        self.columnData = columnData
        self.frequency = frequency
        self.split = split

    def on_run(self, df: pd.DataFrame) -> tuple:
        
        # Metodo privado
        # Manter esse metodo para o caso de ter mais um tipo de metodo de divisão futuramente

        assert self.columnData in df.columns
        
        
        firstData = df[self.columnData].sort_values().values[0]
        lastData = df[self.columnData].sort_values().values[-1]
        rangeDatas = pd.date_range(start=firstData,end=lastData,freq=self.frequency)
        indexCut = round(len(rangeDatas) * self.split)
        return rangeDatas[indexCut],lastData
"""

"""
Pipeline:

read_base ----> square -----------.---> predict
                                  |
read_dt --.---> avg_jan -->--.----´
          |                  |
          `---> avg_mar -->--´
          |                  |
          `---> age ------>--´
"""


def main():
    with Pipeline() as pipeline:
        # read_base = CsvReadTask("read_base", "data.csv")
        read_dt = CsvReadTask("read_dt", "data/transactions.csv")
        # read_transaction = CsvReadTask("read_dt", "data/transactions.csv")
        rfm_data = RFMTask("split_data", "customer_id", "date", "amount")

        # square = SquareTransformerTask("square")
        # avg_jan = AvgTransformerTask("avg_jan", 1)
        # avg_mar = AvgTransformerTask("avg_mar", 3)
        # age = AgeTransformerTask("age")

        # predict = FakeModelTask("predict")

        # read_base >> square >> predict
        # read_dt >> [avg_jan, avg_mar, age] >> predict

        read_dt >> rfm_data

    print(pipeline.run())


if __name__ == "__main__":
    main()
