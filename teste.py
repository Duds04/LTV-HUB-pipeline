from lifetimes import BetaGeoFitter
from lifetimes import ParetoNBDFitter
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error,
    mean_absolute_error,
)
from abc import abstractmethod
from pathlib import Path
from datetime import datetime

from pandas import DataFrame
from src.task import Task, Depends
from src.pipeline import Pipeline
from src.schema import Schema, Field
from typing import Annotated
import pandas as pd


class CsvReadTask(Task):
    def __init__(
        self,
        name: str,
        fp: str,
        columnID: str = "id",
        columnDate: str = "dt",
        columnVal: str = "val",
    ) -> None:
        super().__init__(name)
        self.fp = Path(fp)
        self.columnID = columnID
        self.columnDate = columnDate
        self.columnVal = columnVal

    def on_run(self) -> pd.DataFrame:
        df = pd.read_csv(self.fp)
        df.rename(
            columns={self.columnID: "id",
                     self.columnDate: "dt", self.columnVal: "val"},
            inplace=True,
        )

        assert "id" in df.columns
        assert "dt" in df.columns
        assert "val" in df.columns

        if "dt" in df.columns:
            df["dt"] = pd.to_datetime(df["dt"])

        return df


from lifetimes.utils import (  # noqa: E402
    calibration_and_holdout_data,
    summary_data_from_transaction_data,
)


class RFMTask(Task):
    def __init__(
        self,
        name: str,
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
                columnID #Nome da coluna onde encontra-se os identificadores
                columnDate  #Nome da coluna onde encontra-se as datas
                columnVal  #Nome da coluna onde encontra-se os valores monetários
                frequency = 'W' #Frequência em que será observado, Ex: "W" - Weeks
                calibrationEnd = None #Caso queira passar a data do fim do período de calibração
                ObservationEnd = None #Caso queira passar a data do fim do período de Obsersvação
                split = 0.8 # Porcentagem da divisão dos dados para separar em Obsersvação e calibração
                is_calibration_mode = True #Verdadeiro caso queira separar os dados em Obsersvação e calibração
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

    def on_run(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.calibrationEnd is None:
            self.calibrationEnd, self.ObservationEnd = self.__getPeriodos(
                df, self.columnDate, self.frequency, self.split
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
                calibration_period_end=self.calibrationEnd,
                observation_period_end=self.ObservationEnd,
            )
            return rfm_cal_holdout


from sklearn.model_selection import train_test_split  # noqa: E402


class TransactionModelTask(Task):
    def __init__(
        self,
        name: str,
        # dadosRFM: pd.DataFrame,
        grid=None,
        isTest: bool = True,
    ) -> None:
        """
        Args:
            model #Modelo BG/NBD ou de Pareto esperado para realizar a predição
            rfm #Dataset já processado pelo RFM
            isTest = True #Caso seja para efetuar a predição em um dataset com ou sem o período de observação
            grid = None #Caso não seja para fazer um grid search ele será None
        """
        super().__init__(name)
        self.model = None
        self.grid = grid
        self.isTest = isTest

    @abstractmethod
    def on_run(self, dfRFM: pd.DataFrame) -> pd.DataFrame:
        """
            Dado um dataset com os valores de RFM, retorna a predição do número de transações esperadas
        """
        pass

    @abstractmethod
    def createModel(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame,  numPeriodos: float = 180) -> pd.DataFrame:
        """
            Dado um período, retorna o número de transações esperadas até ele
        Args:
            numPeriodos = 180 #Numero de períodos em dia para que deseja efetuar a predição
        """
        if self.isTest:
            # No período de Treino e no periodo de Validação
            return self.model.conditional_expected_number_of_purchases_up_to_time(
                numPeriodos,
                df["frequency_cal"].values,
                df["recency_cal"].values,
                df["T_cal"].values,
            )
        # Prever dados futuros, com todo o dataset
        return self.model.conditional_expected_number_of_purchases_up_to_time(
            numPeriodos,
            df["frequency"].values,
            df["recency"].values,
            df["T"].values
        )

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            Treina o modelo com os dados passados
        """
        pass

    # TO DO: Esse método tá calculando errado

    @abstractmethod
    def rating(self, nameModel: str, df: pd.DataFrame, xExpected: str, xReal: str = 'frequency_holdout') -> pd.DataFrame:
        """
            Retorna a classificação do cliente
        """
        print("Model ", nameModel,"Mean Squared Error:",
              mean_squared_error(df[xReal], df[xExpected]))


class ParetoModelTask(TransactionModelTask):
    def __init__(
        self,
        name: str,
        # grid = None,
        isTest: bool = True,
        penalizer: float = 0.1,
        isRating: bool = False
    ) -> None:
        """
        Args:
            name, #Nome da tarefa
            isTest = True #Caso seja para efetuar a predição em um dataset com ou sem o período de observação
            penalizer = 0.1# Coeficiente de penalização usado pelo modelo
        """
        super().__init__(name,  isTest)
        self.penalizer = penalizer
        self.isTest = isTest
        self.isRating = isRating
        self.model = self.createModel()

    def on_run(self, dfRFM: pd.DataFrame) -> pd.DataFrame:
        self.fit(dfRFM)
        dfRFM['ExpectedPareto'] = self.predict(
            dfRFM, dfRFM['duration_holdout'].iloc[0])

        dfRFM['ExpectedBGF'] = self.predict(
            dfRFM, dfRFM['duration_holdout'].iloc[0])

        if (self.isRating):
            self.rating(dfRFM)
        # Real Expected --> na verdade isso é só a coluna frequency_holdout
        return dfRFM

    def createModel(self) -> pd.DataFrame:
        pareto = ParetoNBDFitter(penalizer_coef=self.penalizer)
        return pareto

    def fit(self, df: pd.DataFrame):
        """
            Treina o modelo com os dados passados
        """
        # cal, holdout
        # cal --> X em momento de treino
        # holdout --> Y em momento de treino
        # sem nada é no momento de Teste, momento de previsão, final
        if self.isTest:
            self.model.fit(frequency=df['frequency_cal'],
                           recency=df['recency_cal'],
                           T=df['T_cal'])
        else:
            self.model.fit(frequency=df['frequency'],
                           recency=df['recency'],
                           T=df['T'])

        return self.model

    def rating(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            Retorna a classificação do cliente
        """
        xExpected = 'ExpectedPareto'
        super().rating('Pareto', df, xExpected)

    def predict(self, df: pd.DataFrame, numPeriodos: float = 180) -> pd.DataFrame:
        return super().predict(df, numPeriodos=numPeriodos)


class BGFTask(TransactionModelTask):
    def __init__(
        self,
        name: str,
        # grid = None,
        isTest: bool = True,
        penalizer: float = 0.1,
        isRating: bool = False
    ) -> None:
        """
        Args:
            name, #Nome da tarefa
            isTest = True #Caso seja para efetuar a predição em um dataset com ou sem o período de observação
            penalizer = 0.1# Coeficiente de penalização usado pelo modelo
        """
        super().__init__(name,  isTest)
        self.penalizer = penalizer
        self.isTest = isTest
        self.isRating = isRating
        self.model = self.createModel()

    def on_run(self, dfRFM: pd.DataFrame) -> pd.DataFrame:
        self.fit(dfRFM)
        dfRFM['ExpectedBGF'] = self.predict(
            dfRFM, dfRFM['duration_holdout'].iloc[0])
        
        if (self.isRating):
            self.rating(dfRFM)
        # Real Expected --> na verdade isso é só a coluna frequency_holdout
        return dfRFM

    def createModel(self) -> pd.DataFrame:
        pareto = BetaGeoFitter(penalizer_coef=self.penalizer)
        return pareto

    def fit(self, df: pd.DataFrame):
        """
            Treina o modelo com os dados passados
        """
        # cal, holdout
        # cal --> X em momento de treino
        # holdout --> Y em momento de treino
        # sem nada é no momento de Teste, momento de previsão, final
        if self.isTest:
            self.model.fit(frequency=df['frequency_cal'],
                           recency=df['recency_cal'],
                           T=df['T_cal'])
        else:
            self.model.fit(frequency=df['frequency'],
                           recency=df['recency'],
                           T=df['T'])

    def rating(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            Retorna a classificação do cliente
        """
        xExpected = 'ExpectedBGF'
        super().rating('BG/NBD', df,  xExpected)

    def predict(self, df: pd.DataFrame, numPeriodos: float = 180) -> pd.DataFrame:
        return super().predict(df, numPeriodos=numPeriodos)


#  @classmethod --> não precisa passar a instancia não usa o self


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
        read_dt = CsvReadTask(
            "read_dt", "data/transactions.csv", "customer_id", "date", "amount"
        )
        # read_transaction = CsvReadTask("read_dt", "data/transactions.csv")
        rfm_data = RFMTask("split_data")

        pareto_model = ParetoModelTask("pareto_model", isRating=True)
        bgf_model = BGFTask("bgf_model", isRating=True)

        # square = SquareTransformerTask("square")
        # avg_jan = AvgTransformerTask("avg_jan", 1)
        # avg_mar = AvgTransformerTask("avg_mar", 3)
        # age = AgeTransformerTask("age")

        # predict = FakeModelTask("predict")

        # read_base >> square >> predict
        # read_dt >> [avg_jan, avg_mar, age] >> predict

        read_dt >> rfm_data >> pareto_model
        read_dt >> rfm_data >> bgf_model

    print(pipeline.run())


if __name__ == "__main__":
    main()
