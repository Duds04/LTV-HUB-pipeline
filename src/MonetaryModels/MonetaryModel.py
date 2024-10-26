from src.workflows.task import Task
import pandas as pd
from abc import abstractmethod
from sklearn.metrics import mean_squared_error

class MonetaryModelTask(Task):
    def __init__(
        self,
        name: str,
        isTunning: bool = False,
        isTraining: bool = False,
    ) -> None:
        """
        Args:
            isTraining = True #Caso seja para efetuar a predição em um dataset com ou sem o período de observação
        """
        super().__init__(name)
        self.model = None
        self.isTunning = isTunning
        self.isTraining = isTraining

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

