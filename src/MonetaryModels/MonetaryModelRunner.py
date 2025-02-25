from enum import Enum

from src.GenericModels.MachineLearning import MachineLearningModel
from src.MonetaryModels.GammaGammaModel import GammaGammaModelTask


class MonetaryModelType(Enum):
    GammaGammaModel = "GammaGammaModel"
    MachineLearning = "MachineLearning"


class MonetaryModelRunner():
    def __init__(
        self,
        name: str,
        model: MonetaryModelType,
        isTraining: bool = False,
        target: str = None,
        X_Columns: list = None,
        isTunning: bool = False,
        penalizer: float = 0.1,
        isRating: bool = False,
        
        autoActivate: bool = False
    ) -> None:
        """
        Args:
            Para todos:
                name: str,
                isTraining: bool = False,
                isTunning

            Para MachineLearning:
                name: str,
                target: str,
                isTunning: bool = False,
                X_Columns: list = None,
                isTraining: bool = False,

            Para MonetaryModels:
                name: str,
                isTunning: bool = False,
                isTraining: bool = False,
                penalizer: float = 0.1,
                isRating: bool = False
                
            autoActivate --> Variavel que representa se chamamos a função run automaticamente (pelo ModelsRunner) ou não
        """
        self.name = name
        self.model = model
        self.target = target
        self.X_Columns = X_Columns
        self.isTraining = isTraining
        self.isTunning = isTunning
        self.penalizer = penalizer
        self.isRating = isRating
        self.autoActivate = autoActivate

    def run(self):
        """
        Dado um dataset com os valores de RFM, retorna a predição do número de transações esperadas
        """
        print("Entrei no run ", self.model)

        if self.model == MonetaryModelType.MachineLearning:
            print("MachineLearning")
            assert self.target is not None, "Target não pode ser nulo"
            assert self.X_Columns is not None, "X_Columns não pode ser nulo"
            return MachineLearningModel(self.name, self.target, X_Columns=self.X_Columns, isTraining=self.isTraining, isTunning=self.isTunning)
        elif self.model == MonetaryModelType.GammaGammaModel:
            print("GammaGammaModel")
            return GammaGammaModelTask(self.name, isTunning=self.isTunning,  isTraining=self.isTraining, penalizer=self.penalizer, isRating=self.isRating)
        elif(self.autoActivate):
            raise ValueError(f"Modelo desconhecido: {self.model}")
