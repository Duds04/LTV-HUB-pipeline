from src.workflows.pipeline import Pipeline

from src.dataBase import CsvReadTask
from src.dataTransformation import RFMTask
from src.transactionModels import ParetoModelTask
from src.transactionModels import BGFTask
from src.monetaryModels import GammaGammaModelTask
from src.genericModels import MachineLearningModel
from src.ltv_model import LTVTask

#  @classmethod --> não precisa passar a instancia não usa o self

def pipeline_pareto():
    read_dt = CsvReadTask(
        "read_dt", "data/transactions.csv", "customer_id", "date", "amount"
    )
    rfm_data = RFMTask("split_data")
    pareto_model = ParetoModelTask("pareto_model", isRating=True)

    read_dt >> rfm_data >> pareto_model

def pipeline_bgf():
    read_dt = CsvReadTask(
        "read_dt", "data/transactions.csv", "customer_id", "date", "amount"
    )
    rfm_data = RFMTask("split_data")
    bgf_model = BGFTask("bgf_model", isRating=True)
    
    read_dt >> rfm_data >> bgf_model


def pipeline_gammmaGamma():
    read_dt = CsvReadTask(
        "read_dt", "data/transactions.csv", "customer_id", "date", "amount"
    )
    rfm_data = RFMTask("split_data")
    gammaGamma_model = GammaGammaModelTask("gammaGamma_model", isRating=True)
    
    read_dt >> rfm_data >> gammaGamma_model

def pipeline_MLTrasaction():
    read_dt = CsvReadTask(
        "read_dt", "data/transactions.csv", "customer_id", "date", "amount"
    )
    rfm_data = RFMTask("split_data")
    ml_model_transaction = MachineLearningModel("machine_learning_model_transaction", "frequency_holdout", X_Colunms=[
                                   'frequency_cal', 'recency_cal', 'T_cal', 'monetary_value_cal', 'duration_holdout'])
    
    read_dt >> rfm_data >> ml_model_transaction
    

def pipeline_MLMonetary():
    read_dt = CsvReadTask(
        "read_dt", "data/transactions.csv", "customer_id", "date", "amount"
    )
    rfm_data = RFMTask("split_data")
    ml_model_monetary = MachineLearningModel("machine_learning_monetary", "monetary_value_holdout", X_Colunms=[
                                             'frequency_cal', 'recency_cal', 'T_cal', 'monetary_value_cal', 'duration_holdout'])
    
    read_dt >> rfm_data >> ml_model_monetary
    


def main():
    with Pipeline() as pipeline:
        pipeline_pareto()
        # pipeline_bgf()
        # pipeline_gammmaGamma()
        # pipeline_MLTrasaction()
        # pipeline_MLMonetary()

    print(pipeline.run())


if __name__ == "__main__":
    main()
