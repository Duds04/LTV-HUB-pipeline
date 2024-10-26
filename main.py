import pandas as pd
from src.workflows.pipeline import Pipeline

from src.DataBase.CsvRead import CsvReadTask
from src.DataTransformation.RFM import RFMTask
from src.TrasactionModels.ParetoModel import ParetoModelTask
from src.TrasactionModels.BGFModel import BGFModelTask
from src.MonetaryModels.GammaGammaModel import GammaGammaModelTask
from src.GenericModels.MachineLearning import MachineLearningModel
from src.LTV.LtvModel import LTVTask

#  @classmethod --> não precisa passar a instancia não usa o self
def pipeline_RFM():
    read_dt = CsvReadTask(
        "read_dt", "data/transactions.csv", "customer_id", "date", "amount"
    )
    rfm_data = RFMTask("split_data", isTraining=True)
    read_dt >> rfm_data


def pipeline_RFM_Enriquecido():
    read_dt = CsvReadTask(
        "read_dt", "data/transactions.csv", "customer_id", "date", "amount"
    )
    rfm_data_enriquecido = RFMTask(
        "split_data_enriquecido", predictInterval=4,  isTraining=True)
    read_dt >> rfm_data_enriquecido


def pipeline_pareto():
    read_dt = CsvReadTask(
        "read_dt", "data/transactions.csv", "customer_id", "date", "amount"
    )
    rfm_data = RFMTask("split_data", isTraining=True)
    pareto_model = ParetoModelTask("pareto_model", isRating=True, isTraining=True)

    read_dt >> rfm_data >> pareto_model


def pipeline_bgf():
    read_dt = CsvReadTask(
        "read_dt", "data/transactions.csv", "customer_id", "date", "amount"
    )
    rfm_data = RFMTask("split_data", isTraining=True)
    bgf_model = BGFModelTask("bgf_model", isRating=True, isTraining=True)

    read_dt >> rfm_data >> bgf_model


def pipeline_gammmaGamma():
    read_dt = CsvReadTask(
        "read_dt", "data/transactions.csv", "customer_id", "date", "amount"
    )
    rfm_data = RFMTask("split_data", isTraining=True)
    gammaGamma_model = GammaGammaModelTask(
        "gammaGamma_model", isRating=True, isTraining=True)

    read_dt >> rfm_data >> gammaGamma_model


def pipeline_MLTrasaction():
    read_dt = CsvReadTask(
        "read_dt", "data/transactions.csv", "customer_id", "date", "amount"
    )
    rfm_data = RFMTask("split_data", isTraining=True)
    ml_model_transaction = MachineLearningModel("machine_learning_model_transaction", "frequency_holdout", X_Colunms=[
        'frequency_cal', 'recency_cal', 'T_cal', 'monetary_value_cal', 'duration_holdout'], isTraining=True)

    read_dt >> rfm_data >> ml_model_transaction


def pipeline_MLMonetary():
    read_dt = CsvReadTask(
        "read_dt", "data/transactions.csv", "customer_id", "date", "amount"
    )
    rfm_data = RFMTask("split_data", isTraining=True)
    ml_model_monetary = MachineLearningModel("machine_learning_monetary", "monetary_value_holdout", X_Colunms=[
                                             'frequency_cal', 'recency_cal', 'T_cal', 'monetary_value_cal', 'duration_holdout'], isTraining=True)

    read_dt >> rfm_data >> ml_model_monetary


def pipeline_MLMonetary_Enriquecido():
    read_dt = CsvReadTask(
        "read_dt", "data/transactions.csv", "customer_id", "date", "amount"
    )
    rfm_data_enriquecido = RFMTask(
        "split_data_enriquecido", predictInterval=4, isTraining=True)
    ml_model_monetary = MachineLearningModel("machine_learning_monetary", "monetary_value_holdout", X_Colunms=[
                                             'frequency_cal', 'recency_cal', 'T_cal', 'monetary_value_cal', 'duration_holdout'], isTraining=True)
    read_dt >> rfm_data_enriquecido >> ml_model_monetary


def pipeline_pareto_CLV():
    read_dt = CsvReadTask(
        "read_dt", "data/transactions.csv", "customer_id", "date", "amount"
    )
    rfm_data = RFMTask("split_data")

    pareto_model = ParetoModelTask("pareto_model", isRating=True)
    ltv = LTVTask("calculo_ltv", columnFrequency="ExpectedPareto")

    read_dt >> rfm_data >> pareto_model >> ltv


def pipeline_gammaGamma_TEST_CLV():
    read_dt = CsvReadTask(
        "read_dt", "data/transactions.csv", "customer_id", "date", "amount"
    )
    rfm_data = RFMTask("split_data", isTraining=True)

    gammaGamma_model = GammaGammaModelTask(
        "gammaGamma_model", isRating=True, isTraining=True)
    ltv = LTVTask("calculo_ltv", columnFrequency="ExpectedPareto", isTraining=True)

    read_dt >> rfm_data >> gammaGamma_model >> ltv

def main():
    with Pipeline() as pipeline:
            # pipeline_RFM()
            # pipeline_RFM_Enriquecido()
            # pipeline_pareto()
            # pipeline_bgf()
            # pipeline_gammmaGamma()
            # pipeline_MLTrasaction()
            # pipeline_MLMonetary()
            # pipeline_MLMonetary_Enriquecido()
            # pipeline_pareto_CLV()
            pipeline_gammaGamma_TEST_CLV()


    print(pipeline.run())


if __name__ == "__main__":
    main()
