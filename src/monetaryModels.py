from src.workflows.task import Task
import pandas as pd
from abc import abstractmethod
from sklearn.metrics import mean_squared_error
from lifetimes import GammaGammaFitter



class MonetaryModelTask(Task):
    def __init__(
        self,
        name: str,
        isTunning: bool = False,
        isTest: bool = True,
    ) -> None:
        """
        Args:
            isTest = True #Caso seja para efetuar a predição em um dataset com ou sem o período de observação
        """
        super().__init__(name)
        self.model = None
        self.isTunning = isTunning
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
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            Dado um período, retorna o número de transações esperadas até ele
        """
        pass

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            Treina o modelo com os dados passados
        """
        pass

    @abstractmethod
    def rating(self, nameModel: str, df: pd.DataFrame, xExpected: str, xReal: str = 'frequency') -> pd.DataFrame:
        """
            Retorna a classificação do cliente
        """
        print("Model ", nameModel, "Mean Squared Error:",
              mean_squared_error(df[xReal], df[xExpected]))


class GammaGammaModelTask(MonetaryModelTask):
    def __init__(
        self,
        name: str,
        isTunning: bool = False,
        isTest: bool = True,
        penalizer: float = 0.1,
        isRating: bool = False
    ) -> None:
        """
        Args:
            isTest = True #Caso seja para efetuar a predição em um dataset com ou sem o período de observação
            isTunning = None # Fazer o Tunning de hyperparâmetros se for True
            penalizer = 0.1 # Coeficiente de penalização usado pelo modelo
        """
        super().__init__(name, isTunning, isTest)
        self.penalizer = penalizer
        self.isTest = isTest
        self.isRating = isRating
        self.model = self.createModel()

    def on_run(self, dfRFM: pd.DataFrame) -> pd.DataFrame:

        monetary = "monetary_value"
        frequency = "frequency"

        if self.isTest:
            monetary = "monetary_value_cal"
            frequency = "frequency_cal"

        dfRFM = dfRFM[dfRFM[monetary] > 0]

        self.fit(dfRFM, monetary, frequency)

        dfRFM['ExpectedGammaGamma'] = self.predict(dfRFM, monetary, frequency)

        if (self.isRating):
            self.rating(dfRFM, frequency)

        return dfRFM

    def createModel(self) -> pd.DataFrame:
        gamma = GammaGammaFitter(penalizer_coef=self.penalizer)
        return gamma

    def fit(self, df: pd.DataFrame, monetary: str, frequency: str) -> pd.DataFrame:
        """
            Treina o modelo com os dados passados
        """
        self.model.fit(df[frequency], df[monetary])
        return self.model

    def predict(self, df: pd.DataFrame, monetary: str, frequency: str) -> pd.DataFrame:
        """
            Dado um período, retorna o número de transações esperadas até ele
        """
        return self.model.conditional_expected_average_profit(df[frequency], df[monetary])

    def rating(self, df: pd.DataFrame, frequency: str) -> pd.DataFrame:
        """
            Retorna a classificação do cliente
        """
        xExpected = 'ExpectedGammaGamma'
        super().rating('GammaGamma', df, xExpected, xReal=frequency)
