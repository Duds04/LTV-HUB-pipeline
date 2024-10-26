
import pandas as pd
from sklearn.metrics import mean_squared_error

import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import ElasticNet, LassoCV
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor

from src.GenericModels.GenericModel import GenericModelTask



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
        isTest: bool = False,
    ) -> None:
        """
        Args:
            name, # Nome da tarefa
            target, # Nome da coluna onde está o valor alvo (Y)
            isTunning, # Fazer o Tunning de hyperparâmetros se for True
        """
        super().__init__(name, target, isTunning, isTest)
        self.models = self.createModels()

        self.bestModel = None

        self.X_Colunms = X_Colunms

        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None

    def on_run(self, dfRFM: pd.DataFrame) -> pd.DataFrame:
        assert self.target in dfRFM.columns
        for x_colum in self.X_Colunms:
            assert x_colum in dfRFM.columns
        
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

