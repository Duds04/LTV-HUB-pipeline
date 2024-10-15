
from src.workflows.task import Task
import pandas as pd
from lifetimes.utils import (  # noqa: E402
    calibration_and_holdout_data,
    summary_data_from_transaction_data,
)


class RFMTask(Task):
    def __init__(
        self,
        name: str,
        # filler_periods_interval = None,
        columnID: str = "id",
        columnDate: str = "dt",
        columnVal: str = "val",
        frequency: str = "W",
        calibrationEnd=None,
        ObservationEnd=None,
        split: float = 0.8,
        apply_calibration_split: bool = True,
    ) -> None:
        """
        Args:
                filler_periods_interval # Vetor armazenando quais periodos usados para o split
                columnID #Nome da coluna onde encontra-se os identificadores
                columnDate  #Nome da coluna onde encontra-se as datas
                columnVal  #Nome da coluna onde encontra-se os valores monetários
                frequency = 'W' #Frequência em que será observado, Ex: "W" - Weeks
                calibrationEnd = None #Caso queira passar a data do fim do período de calibração
                ObservationEnd = None #Caso queira passar a data do fim do período de Obsersvação
                split = 0.8 # Porcentagem da divisão dos dados para separar em Obsersvação e calibração
                #Verdadeiro caso queira separar os dados em Obsersvação e calibração
                is_calibration_mode = True
        """
        super().__init__(name)
        self.columnID = columnID
        self.columnDate = columnDate
        self.columnVal = columnVal
        self.frequency = frequency
        self.calibrationEnd = calibrationEnd
        self.ObservationEnd = ObservationEnd
        self.split = split
        self.apply_calibration_split = apply_calibration_split
        # self.filler_periods_interval = filler_periods_interval

    def __getPeriodos(
        self, df: pd.DataFrame, columnDate: str, frequency: str, split: float = 0.8
    ):
        """
        Args:
            name, #Nome da tarefa
            df, #Dataframe do Pandas
            columnDate, #Nome da coluna onde estão as datas
            frequency, #Frequência em que será observado, Ex: "W" - Weeks
            split = 0.8 #Porcentagem da divisão dos dados para separar em treino e calibração
        """
        assert columnDate in df.columns

        firstData = df[columnDate].sort_values().values[0]
        lastData = df[columnDate].sort_values().values[-1]
        rangeDatas = pd.date_range(
            start=firstData, end=lastData, freq=frequency)
        indexCut = round(len(rangeDatas) * split)
        return rangeDatas[indexCut], lastData

    def __rfm_data_filler(self, df: pd.DataFrame, split: float = 0.8) -> pd.DataFrame:
            if self.calibrationEnd is None:
                calibrationEnd, ObservationEnd = self.__getPeriodos(
                    df, self.columnDate, self.frequency, split
                )

            if self.apply_calibration_split is False:
                return summary_data_from_transaction_data(
                    transactions=df,
                    customer_id_col=self.columnID,
                    datetime_col=self.columnDate,
                    monetary_value_col=self.columnVal,
                    freq=self.frequency,
                )
            else:
                rfm_cal_holdout = calibration_and_holdout_data(
                    transactions=df,
                    customer_id_col=self.columnID,
                    datetime_col=self.columnDate,
                    monetary_value_col=self.columnVal,
                    freq=self.frequency,
                    calibration_period_end= calibrationEnd,
                    observation_period_end= ObservationEnd,
                )
            return rfm_cal_holdout

    def on_run(self, df: pd.DataFrame) -> pd.DataFrame:
        
        # if self.filler_periods_interval == None:
        df2 = self.__rfm_data_filler(df)
        # else:
        #     df2 = self.__rfm_data_filler(df, split = self.filler_periods_interval[0])
        #     for i in self.filler_periods_interval[1:-1]:
        #         df1 = self.__rfm_data_filler(df, split = i)
        #         df2 = df2._append(df1)

        return df2
           
