from lifetimes import GammaGammaFitter
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from lifetimes import BetaGeoFitter
from lifetimes import ParetoNBDFitter
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LassoCV
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

# TO DO: Esse método tá calculando errado corrijir comentarios de tasks (os args)


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
           


from sklearn.model_selection import GridSearchCV, train_test_split  # noqa: E402


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


#  @classmethod --> não precisa passar a instancia não usa o self

# usados pelo Machine Learning (tem o Y independente)
class GenericModelTask(Task):
    def __init__(
        self,
        name: str,
        target: str,
        isTunning: bool = False,
    ) -> None:
        """
        Args:
            target: str, # Nome da coluna onde está o valor alvo (Y)
            isTest = True # Caso seja para efetuar a predição em um dataset com ou sem o período de observação
            isTunning = None # Fazer o Tunning de hyperparâmetros se for True
        """
        super().__init__(name)
        self.target = target
        self.isTunning = isTunning

    @abstractmethod
    def on_run(self, dfRFM: pd.DataFrame) -> pd.DataFrame:
        """
            Dado um dataset com os valores de RFM, retorna a predição do número de transações esperadas
        """
        pass

    @abstractmethod
    def predict(self):
        """
            Dado um período, retorna o número de transações esperadas até ele
        """
        pass

    @abstractmethod
    def fit(self):
        """
            Treina o modelo com os dados passados
        """
        pass

    @abstractmethod
    def rating(self):
        """
            Retorna a classificação do modelo
        """
        pass


class MachineLearningModel(GenericModelTask):
    """
        Instância diversos modelos de Machine Learning para prever o target e pega o mais adequado
    """

    def __init__(
        self,
        name: str,
        target: str,
        isTunning: bool = False,
        X_Colunms: list = None,
    ) -> None:
        """
        Args:
            name, # Nome da tarefa
            target, # Nome da coluna onde está o valor alvo (Y)
            isTunning, # Fazer o Tunning de hyperparâmetros se for True
        """
        super().__init__(name, target, isTunning)
        self.models = self.createModels()

        self.bestModel = None

        self.X_Colunms = X_Colunms

        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None

    def on_run(self, dfRFM: pd.DataFrame) -> pd.DataFrame:
        X = dfRFM[self.X_Colunms]
        Y = dfRFM[self.target]

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X.values, np.ravel(Y.values), random_state=42)

        self.bestModel = self.selectBestModel()

        dfRFM['ExpectedML'] = self.bestModel.predict(dfRFM[self.X_Colunms])
        
        return dfRFM

    def selectBestModel(self):
        """
            Seleciona o melhor modelo de acordo com o dataset passado
        """
        bestScore = None
        print()
        for model in self.models:
            score = self.fitAndRating(model)
            if bestScore == None or bestScore > score[0]:
                bestScore, self.bestModel = score

            if self.isTunning:
                print(type(model.best_estimator_).__name__,
                      " mse: {:.4f} \n".format(score[0]))
            else:
                print(type(model).__name__, " mse: {:.4f} \n".format(score[0]))

        if self.isTunning:
            return self.bestModel.best_estimator_
        else:
            return self.bestModel

    def get_grid_params(self, model_name):
        grids = {
            'lasso': {
                'n_alphas': [100, 200, 500],
                'max_iter': [1000, 1500, 2000],
                'random_state': [42]
            },
            'enet': {
                "max_iter": [1000, 1500],
                "alpha": [0.0001, 0.001],
                "l1_ratio": np.arange(0.0, 1.0, 0.1),
                'random_state': [42]
            },
            'random_forest': {
                'bootstrap': [True, False],
                'min_samples_leaf': [1, 2, 4],
                'min_samples_split': [2, 5, 10],
                'n_estimators': [200, 800, 1000],
                'random_state': [42]
            },
            'gboost': {
                'n_estimators': [500, 1000, 2000],
                'learning_rate': [0.001, 0.01, 0.1],
                'max_depth': [1, 2, 4],
                'subsample': [0.5, 0.75, 1],
                'random_state': [42]
            },
            'hist_gradient_boosting': {
                'learning_rate': [0.001, 0.01, 0.1],
                'max_depth': [1, 2, 4, None],
                'max_leaf_nodes': [31, None],
                'random_state': [42]
            },
            'xgboost': {
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [100, 500, 1000],
                'colsample_bytree': [0.3, 0.7],
                'random_state': [42]
            },
            'lgbm': {
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [100, 500, 1000],
                'random_state': [42]
            }
        }
        return grids.get(model_name, {})

    def apply_grid_search(self, model, model_name, scoring='neg_mean_squared_error'):
        grid_params = self.get_grid_params(model_name)
        return GridSearchCV(estimator=model, param_grid=grid_params, n_jobs=-1, scoring=scoring)

    def createModels(self):
        if self.isTunning == False:
            lasso = LassoCV()
            Enet = ElasticNet()
            rf = RandomForestRegressor()
            GBoost = GradientBoostingRegressor()
            HGBoost = HistGradientBoostingRegressor()
            model_xgb = xgb.XGBRegressor()
            model_lgb = lgb.LGBMRegressor(objective='regression', verbose=-1)

        else:
            lasso = self.apply_grid_search(LassoCV(), 'lasso')
            Enet = self.apply_grid_search(ElasticNet(), 'enet')
            rf = self.apply_grid_search(
                RandomForestRegressor(), 'random_forest')
            GBoost = self.apply_grid_search(
                GradientBoostingRegressor(), 'gboost')
            HGBoost = self.apply_grid_search(
                HistGradientBoostingRegressor(), 'hist_gradient_boosting')
            model_xgb = self.apply_grid_search(xgb.XGBRegressor(), 'xgboost')
            model_lgb = self.apply_grid_search(
                lgb.LGBMRegressor(), 'lgbm', verbose=-1)

        models = [lasso, Enet, rf, GBoost, HGBoost, model_xgb, model_lgb]
        return models

    def fitAndRating(
        self,
        # Modelo que será treinado (Tem de ter a função fit e predict implementadas)
        RegressorModel
    ):
        self.fit(RegressorModel)
        predict = self.predict(RegressorModel)
        metrica = self.rating(predict)
        return metrica, RegressorModel  # Retorna o MSE e o Regressor

    def fit(self,
            # Modelo que será treinado (Tem de ter a função fit e predict implementadas)
            RegressorModel,
            ):
        """
            Treina o modelo com os dados passados
        """
        return RegressorModel.fit(self.X_train,  self.Y_train)

    def rating(self, predict) -> pd.DataFrame:
        """
            Retorna a classificação do modelo
        """
        # Utilizando o MSE, caso queira outra métrica, trocar nesta parte!
        return mean_squared_error(self.Y_test, predict)

    def predict(self, RegressorModel):
        return RegressorModel.predict(self.X_test)


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

        # onde tem essa coluna???
        monetary = "monetary_value"
        frequency = "frequency"

        if self.isTest:
            monetary = "monetary_value_cal"
            frequency = "frequency_cal"

        dfRFM = dfRFM[dfRFM[monetary] > 0]

        # print(dfRFM[monetary].values())

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


def main():
    with Pipeline() as pipeline:
        read_dt = CsvReadTask(
            "read_dt", "data/transactions.csv", "customer_id", "date", "amount"
        )
        rfm_data = RFMTask("split_data")
        

        # NÃO CRIAR ESSES MODELOS DENTRO DO PIPELINE SE NÃO FOR COLOCAR AS DEPENDÊNCIAS
        # pareto_model = ParetoModelTask("pareto_model", isRating=True)
        # bgf_model = BGFTask("bgf_model", isRating=True)

        # ml_model_transaction = MachineLearningModel("machine_learning_model_transaction", "frequency_holdout", X_Colunms=[
        #                                 'frequency_cal', 'recency_cal', 'T_cal', 'monetary_value_cal', 'duration_holdout'])

        # gammaGamma_model = GammaGammaModelTask("gammaGamma_model", isRating=True)


        # TO DO: Refatorar o esquema de preencher o dataSet atravês dos intervalos (# rfm_data = RFMTask("split_data",  [i/20 for i in range(11,17)]))
        ml_model_monetary = MachineLearningModel("machine_learning_monetary", "monetary_value_holdout", X_Colunms=[
                                                 'frequency_cal', 'recency_cal', 'T_cal', 'monetary_value_cal', 'duration_holdout'])

        # read_dt >> rfm_data
        # read_dt >> rfm_data >> pareto_model
        # read_dt >> rfm_data >> bgf_model
        # read_dt >> rfm_data >> ml_model
        # read_dt >> rfm_data >> gammaGamma_model
        read_dt >> rfm_data >> ml_model_monetary
        
    print(pipeline.run())


if __name__ == "__main__":
    main()
