from src.task import Task
import pandas as pd
from abc import abstractmethod
from sklearn.metrics import mean_squared_error
from lifetimes import BetaGeoFitter
from lifetimes import ParetoNBDFitter

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
        print("Model ", nameModel, "Mean Squared Error:",
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

