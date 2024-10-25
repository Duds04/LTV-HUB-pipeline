import pandas as pd
from lifetimes import BetaGeoFitter
from src.TrasactionModels.TransactionModel import TransactionModelTask

class BGFModelTask(TransactionModelTask):
    def __init__(
        self,
        name: str,
        # grid = None,
        isTest: bool = True,
        penalizer: float = 0.1,
        isRating: bool = False,
        numPeriods: int = 180,
    ) -> None:
        """
        Args:
            name, #Nome da tarefa
            isTest = True #Caso seja para efetuar a predição em um dataset com ou sem o período de observação
            penalizer = 0.1# Coeficiente de penalização usado pelo modelo
        """
        super().__init__(name,  isTest, numPeriods)
        self.penalizer = penalizer
        self.isTest = isTest
        self.isRating = isRating
        self.model = self.createModel()

    def on_run(self, dfRFM: pd.DataFrame) -> pd.DataFrame:
        self.fit(dfRFM)
        dfRFM['ExpectedBGF'] = self.predict(dfRFM)
        
        if (self.isTest and  self.isRating):
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

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        return super().predict(df)
