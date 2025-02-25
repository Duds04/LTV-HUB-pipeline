import pandas as pd
from src.workflows.pipeline import Pipeline

from src.DataBase.CsvRead import CsvReadTask
from src.DataTransformation.RFM import RFMTask
from src.TrasactionModels.ParetoModel import ParetoModelTask
from src.TrasactionModels.BGFModel import BGFModelTask
from src.MonetaryModels.GammaGammaModel import GammaGammaModelTask
from src.GenericModels.MachineLearning import MachineLearningModel
from src.LTV.LtvModel import LTVTask

from src.TrasactionModels.TransactionModelRunner import TransactionModelRunner
from src.MonetaryModels.MonetaryModelRunner import MonetaryModelRunner

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
    pareto_model = ParetoModelTask(
        "pareto_model", isRating=True, isTraining=True)

    read_dt >> rfm_data >> pareto_model


def pipeline_bgf():
    read_dt = CsvReadTask(
        "read_dt", "data/transactions.csv", "customer_id", "date", "amount"
    )
    rfm_data = RFMTask("split_data")
    bgf_model = BGFModelTask("bgf_model", isRating=True)

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
    rfm_data = RFMTask("split_data")
    ml_model_transaction = MachineLearningModel("machine_learning_model_transaction", "frequency_holdout", X_Columns=[
        'frequency_cal', 'recency_cal', 'T_cal', 'monetary_value_cal', 'duration_holdout'], isTraining=True)

    read_dt >> rfm_data >> ml_model_transaction


def pipeline_MLMonetary():
    read_dt = CsvReadTask(
        "read_dt", "data/transactions.csv", "customer_id", "date", "amount"
    )
    rfm_data = RFMTask("split_data", isTraining=True)
    ml_model_monetary = MachineLearningModel("machine_learning_monetary", "monetary_value_holdout", X_Columns=[
                                             'frequency_cal', 'recency_cal', 'T_cal', 'monetary_value_cal', 'duration_holdout'], isTraining=True)

    read_dt >> rfm_data >> ml_model_monetary


def pipeline_MLMonetary_Enriquecido():
    read_dt = CsvReadTask(
        "read_dt", "data/transactions.csv", "customer_id", "date", "amount"
    )
    rfm_data_enriquecido = RFMTask(
        "split_data_enriquecido", predictInterval=4, isTraining=True)
    ml_model_monetary = MachineLearningModel("machine_learning_monetary", "monetary_value_holdout", X_Columns=[
                                             'frequency_cal', 'recency_cal', 'T_cal', 'monetary_value_cal', 'duration_holdout'], isTraining=True)
    read_dt >> rfm_data_enriquecido >> ml_model_monetary


def pipeline_MLMonetary_NOT_Training():
    read_dt = CsvReadTask(
        "read_dt", "data/transactions.csv", "customer_id", "date", "amount"
    )
    rfm_data = RFMTask("split_data")
    ml_model_monetary = MachineLearningModel("machine_learning_monetary", "monetary_value", X_Columns=[
                                             'frequency', 'recency', 'T', 'monetary_value'])

    read_dt >> rfm_data >> ml_model_monetary


def pipeline_MLTrasaction_NOT_Training():
    read_dt = CsvReadTask(
        "read_dt", "data/transactions.csv", "customer_id", "date", "amount"
    )
    rfm_data = RFMTask("split_data")
    ml_model_transaction = MachineLearningModel("machine_learning_model_transaction", "frequency", X_Columns=[
                                                'frequency', 'recency', 'T', 'monetary_value'])

    read_dt >> rfm_data >> ml_model_transaction


def pipeline_pareto_CLV():
    read_dt = CsvReadTask(
        "read_dt", "data/transactions.csv", "customer_id", "date", "amount"
    )
    rfm_data = RFMTask("split_data")

    pareto_model = ParetoModelTask("pareto_model", isRating=True)
    ltv = LTVTask("calculo_ltv", columnFrequency="ExpectedFrequency")

    read_dt >> rfm_data >> pareto_model >> ltv


def pipeline_gammaGamma_TEST_CLV():
    read_dt = CsvReadTask(
        "read_dt", "data/transactions.csv", "customer_id", "date", "amount"
    )
    rfm_data = RFMTask("split_data", isTraining=True)

    gammaGamma_model = GammaGammaModelTask(
        "gammaGamma_model", isRating=True, isTraining=True)
    ltv = LTVTask(
        "calculo_ltv", columnFrequency="ExpectedFrequency", isTraining=True)

    read_dt >> rfm_data >> gammaGamma_model >> ltv


def pipeline_transaction():
    typeModels = {
        "1": "ParetoModel",
        "2": "MachineLearning",
        "3": "BGFModel",
    }

    print("Escolha o modelo para executar:")
    for key, value in typeModels.items():
        print(f"{key}: {value.value}")

    typeModel = input("\nDigite o número do modelo:")

    if typeModel in typeModels:
        typeModel = typeModels[typeModel]
    else:
        print("\nEscolha inválida.")
        return

    read_dt = CsvReadTask(
        "read_dt", "data/transactions.csv", "customer_id", "date", "amount"
    )
    rfm_data = RFMTask("split_data", isTraining=True)
    if (typeModel == "MachineLearning"):
        transaction_use = TransactionModelRunner("model", typeModel, isTraining=True, isRating=True, target="frequency_holdout", X_Columns=[
            'frequency_cal', 'recency_cal', 'T_cal', 'monetary_value_cal', 'duration_holdout'])
    else:
        transaction_use = TransactionModelRunner(
            "model", typeModel, isTraining=True, isRating=True)
    model = transaction_use.run()

    read_dt >> rfm_data >> model


def pipeline_monetary():
    typeModels = {
        "1": "GammaGammaModel",
        "2": "MachineLearning",
    }

    print("Escolha o modelo para executar:")
    for key, value in typeModels.items():
        print(f"{key}: {value.value}")

    typeModel = input("\nDigite o número do modelo:")

    if typeModel in typeModels:
        typeModel = typeModels[typeModel]
    else:
        print("\nEscolha inválida.")
        return

    read_dt = CsvReadTask(
        "read_dt", "data/transactions.csv", "customer_id", "date", "amount"
    )
    rfm_data = RFMTask("split_data", isTraining=True)
    if (typeModel == "MachineLearning"):
        monetary_use = MonetaryModelRunner("model", typeModel, isTraining=True, isRating=True, target="frequency_holdout", X_Columns=[
                                           'frequency_cal', 'recency_cal', 'T_cal', 'monetary_value_cal', 'duration_holdout'])
    else:
        monetary_use = MonetaryModelRunner(
            "model", typeModel, isTraining=True, isRating=True)

    model = monetary_use.run()

    read_dt >> rfm_data >> model


def calculate_LTV(transactionModel, monetaryModel, file_path="data/transactions.csv", columnID="customer_id", columnDate="date", columnMonetary="amount"):
    read_dt = CsvReadTask(
        "read_dt", file_path, columnID, columnDate, columnMonetary
    )
    rfm_data = RFMTask("split_data")

    transaction_model = transactionModel.run()
    monetary_model = monetaryModel.run()

    ltv = LTVTask("calculo_ltv", columnFrequency="ExpectedFrequency", columnMonetary="ExpectedMonetary")

    # Lembrando (>> só associa, executa apenas apos rodar pipeline.run())
    # read_dt >> rfm_data >> transaction_model >> monetary_model >> ltv
    read_dt >> rfm_data 
    rfm_data >> transaction_model
    rfm_data >> monetary_model
    # transaction_model >> ltv
    monetary_model >> ltv


def use_calculate():
    data = {
        'idColumn': 'customer_id',
        'dateColumn': 'date',
        'amountColumn': 'amount',
        'frequencyModel': 'BGFModel',
        'monetaryModel': 'GammaGammaModel',
        'weeksAhead': 180
    }   
    csv_file_path = "data/transactions.csv"
    print(csv_file_path, data['idColumn'], data['dateColumn'], data['amountColumn'],
          data['weeksAhead'], data['frequencyModel'], data['monetaryModel'])

    if data['frequencyModel'] == "MachineLearning":
        transactionModel = TransactionModelRunner("transaction_model", "MachineLearning", isRating=True, target="frequency", X_Columns=[
                                                  'frequency', 'recency', 'T', 'monetary_value'])
    else:
        transactionModel = TransactionModelRunner("transaction_model", data['frequencyModel'], isRating=True, numPeriods=data['weeksAhead'])

    if data['monetaryModel'] == "MachineLearning":
        monetaryModel = MonetaryModelRunner("monetary_model", "MachineLearning", isRating=True, target="monetary_value", X_Columns=[
                                            'frequency', 'recency', 'T', 'monetary_value'])
    else:
        monetaryModel = MonetaryModelRunner(
            "monetary_model", data['monetaryModel'], isRating=True)

    print(transactionModel, monetaryModel)
    calculate_LTV(transactionModel, monetaryModel, csv_file_path,
                  data['idColumn'], data['dateColumn'], data['amountColumn'])


""" Para adicionar um novo modelo é necessário:
        Criar uma Task para aquele modelo (herdando da Task generica de seu tipo)
        Adicionar os novos atributos necessários para se usar esse modelo no seu respectivo ModelRunner (TransactionModelRunner ou Monetary Model Runner)
        Adicionar um novo if no método run de seu ModelRunner para associar o modelo a sua Task
"""


def main():
    while True:
        pipelines = {
            "1": pipeline_RFM,
            "2": pipeline_RFM_Enriquecido,
            "3": pipeline_pareto,
            "4": pipeline_bgf,
            "5": pipeline_gammmaGamma,
            "6": pipeline_MLTrasaction,
            "7": pipeline_MLMonetary,
            "8": pipeline_MLMonetary_Enriquecido,
            "9": pipeline_pareto_CLV,
            "10": pipeline_gammaGamma_TEST_CLV,
            "11": pipeline_transaction,
            "12": pipeline_monetary,
            "13": use_calculate,
            "14": pipeline_MLMonetary_NOT_Training,
            "15": pipeline_MLTrasaction_NOT_Training
        }

        print("Escolha um pipeline para executar:")
        for key, value in pipelines.items():
            print(f"{key}: {value.__name__}")

        opcao = input("\nDigite o número do pipeline:")
        print("\nExecutando pipeline...\n")

        if opcao in pipelines:
            with Pipeline() as pipeline:
                pipelines[opcao]()
            print("\n", pipeline.run())
        else:
            print("\nEscolha inválida.")

        print("\nDeseja executar outro pipeline? (s/n)")
        continuar = input()
        if continuar.lower() != "s":
            break


if __name__ == "__main__":
    main()
