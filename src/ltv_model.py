from src.workflows.task import Task
import pandas as pd
from pathlib import Path


class LTVTask(Task):
    def __init__(
        self,
        name: str,
        frequency: str = "D",  # Frequência que os dados estão
        isTeste: bool = True,  # Caso seja para prever de acordo com o período de observação ou não
        columnMonetary: str = "monetary_value",
        DISCOUNT_a: float = 0.06,  # Taxa de desconto anual
        columnVal: str = "val",
        LIFE: int = 12,  # Meses que deseja calcular o lifetime
        ML: bool = False,  # Indica se o modelo é de aprendizado de máquina
    ) -> None:
        super().__init__(name)
        self.frequency = frequency
        self.isTeste = isTeste
        self.columnMonetary = columnMonetary
        self.DISCOUNT_a = DISCOUNT_a
        self.LIFE = LIFE
        self.ML = ML

    def on_run(self, df: pd.DataFrame, modelo) -> pd.DataFrame:
        # Função para adaptar o cálculo do LTV tanto para aprendizado de máquina quanto para modelos probabilísticos
        
        if self.isTeste:
            df = df.rename(columns={
                'frequency_cal': 'frequency', 'recency_cal': 'recency', 'T_cal': 'T', "monetary_value_cal": 'monetary_value'})

        df['LTV'] = 0

        # Fator de conversão baseado na frequência
        factor = {"W": 4.345, "M": 1.0, "D": 30, "H": 30 * 24}[self.frequency]

        # Se for um modelo de aprendizado de máquina
        # if self.ML:
        #     df['X'] = self.LIFE * factor
        #     expectedPurchase = modelo.predict(df[['frequency', 'recency', 'T', self.columnMonetary, 'X']].values)
        # else:
        #     # Para modelo probabilístico
        #     expectedPurchase = comprasEsperadas(modelo, df, self.LIFE * factor, teste=self.isTeste)

        # Calcula o LTV diretamente para o período de vida desejado
        
        
        # como calcular esse valor expectedPurchase?
        df['LTV'] = (df[self.columnMonetary] * expectedPurchase) / (1 + self.DISCOUNT_a) ** (self.LIFE / factor)

        return df

# dfValidacao['ltv'] = calculaLTV(dfValidacao,modelBGF,'monetary_value',teste = False,LIFE = 12)
# dfValidacao
# calculaLTV(dfValidacao, modelBGF, 'monetary_value', teste=False, LIFE=2)
