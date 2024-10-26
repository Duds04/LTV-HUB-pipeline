from src.workflows.task import Task
import pandas as pd
from lifetimes import GammaGammaFitter
from src.MonetaryModels.MonetaryModel import MonetaryModelTask

class GammaGammaModelTask(MonetaryModelTask):
    def __init__(
        self,
        name: str,
        isTunning: bool = False,
        isTraining: bool = False,
        penalizer: float = 0.1,
        isRating: bool = False
    ) -> None:
        """
        Args:
            isTraining = True #Caso seja para efetuar a predição em um dataset com ou sem o período de observação
            isTunning = None # Fazer o Tunning de hyperparâmetros se for True
            penalizer = 0.1 # Coeficiente de penalização usado pelo modelo
        """
        super().__init__(name, isTunning, isTraining)
        self.penalizer = penalizer
        self.isTraining = isTraining
        self.isRating = isRating
        self.model = self.createModel()

    def on_run(self, dfRFM: pd.DataFrame) -> pd.DataFrame:

        monetary = "monetary_value"
        frequency = "frequency"

        if self.isTraining:
            monetary = "monetary_value_cal"
            frequency = "frequency_cal"

        dfRFM = dfRFM[dfRFM[monetary] > 0]

        self.fit(dfRFM, monetary, frequency)

        dfRFM['ExpectedGammaGamma'] = self.predict(dfRFM, monetary, frequency)

        if (self.isTraining and self.isRating):
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
