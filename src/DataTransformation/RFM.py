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
        minTrainin: int = -1,
        maxTraining: int = -1,
        predictInterval: int = -1,
        numPeriods: int = -1,
        columnID: str = "id",
        columnDate: str = "dt",
        columnMonetary: str = "monetary",
        frequency: str = "W",
        calibrationEnd=None,
        observationEnd=None,
        split: float = 0.8,
        isTraining: bool = False,
        apply_calibration_split: bool = False,
    ) -> None:
        """
        Args:
                numPeriods # Quantos periodos queremos prever
                columnID #Nome da coluna onde encontra-se os identificadores
                columnDate  #Nome da coluna onde encontra-se as datas
                columnMonetary  #Nome da coluna onde encontra-se os valores monetários
                frequency = 'W' #Frequência em que será observado, Ex: "W" - Weeks
                calibrationEnd = None #Caso queira passar a data do fim do período de calibração
                observationEnd = None #Caso queira passar a data do fim do período de Obsersvação
                split = 0.8 # Porcentagem da divisão dos dados para separar em Obsersvação e calibração
                minTrainin: int, # Qual é o periodo mínimo que será usado para treino
                maxTraining: int, # Qual será o último período que será usado para o treino
                predictInterval: int = 4, # Define quantos períodos serão os intervalos de predição
                apply_calibration_split # Fazer a divisão do dataFrame em calibration e holdout (se for Test isso ficara como true)
        """
        super().__init__(name)
        self.columnID = columnID
        self.columnDate = columnDate
        self.columnMonetary = columnMonetary
        self.frequency = frequency
        self.calibrationEnd = calibrationEnd
        self.observationEnd = observationEnd
        self.split = split
        if(isTraining): self.apply_calibration_split = isTraining
        else: self.apply_calibration_split = apply_calibration_split
        self.numPeriods = numPeriods
        self.minTrainin = minTrainin
        self.maxTraining = maxTraining
        self.predictInterval = predictInterval

    def __getPeriodosList(self, df: pd.DataFrame):
        def __to_period(df: pd.DataFrame):
            return df.to_period(self.frequency)

        df["period"] = df[self.columnDate].map(__to_period)
        df = df.sort_values(by='period')
        return df.period.unique()

    def __getPeriodos(
        self, df: pd.DataFrame
    ):
        """
        Args:
            df, #Dataframe do Pandas
        """
        assert self.columnDate in df.columns

        firstData = df[self.columnDate].sort_values().values[0]
        lastData = df[self.columnDate].sort_values().values[-1]
        rangeDatas = pd.date_range(
            start=firstData, end=lastData, freq=self.frequency)
        indexCut = round(len(rangeDatas) * self.split)
        return rangeDatas[indexCut], lastData

    def __rfm_data_filler(self, df: pd.DataFrame) -> pd.DataFrame:
            """
            Args:
                df, #Dataframe do Pandas
            """
            if self.calibrationEnd is None:
                calibrationEnd, observationEnd = self.__getPeriodos(
                    df
                )
            else:
                calibrationEnd = self.calibrationEnd
                observationEnd = self.observationEnd

            if self.apply_calibration_split is False:
                return summary_data_from_transaction_data(
                    transactions=df,
                    customer_id_col=self.columnID,
                    datetime_col=self.columnDate,
                    monetary_value_col=self.columnMonetary,
                    freq=self.frequency,
                )
            else:
                rfm_cal_holdout = calibration_and_holdout_data(
                    transactions=df,
                    customer_id_col=self.columnID,
                    datetime_col=self.columnDate,
                    monetary_value_col=self.columnMonetary,
                    freq=self.frequency,
                    calibration_period_end=calibrationEnd,
                    observation_period_end=observationEnd,
                )
            return rfm_cal_holdout

    def on_run(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self.columnID in df.columns
        assert self.columnMonetary in df.columns
        assert self.columnDate in df.columns
        
        dfReturn = pd.DataFrame()
        if self.predictInterval == -1:
            dfReturn = self.__rfm_data_filler(df)
        else:
            periods = self.__getPeriodosList(df)
            
            if self.maxTraining == -1:
                self.maxTraining = len(periods) - (self.predictInterval + 1)
            if self.minTrainin == -1:
               self.minTrainin = self.predictInterval * 2
            
            for period in range(self.minTrainin, self.maxTraining, self.predictInterval):
                self.calibrationEnd = periods[period].to_timestamp()
                self.observationEnd = periods[period + self.predictInterval].to_timestamp()
                dfReturn = pd.concat([dfReturn, self.__rfm_data_filler(df)], ignore_index=True)
                
                              
        return dfReturn
           
