from pathlib import Path
from datetime import datetime
from src.task import Task, Depends
from src.pipeline import Pipeline
from src.schema import Schema, Field
from typing import Annotated
import pandas as pd


class CsvReadTask(Task):

    def __init__(self, name: str, fp: str) -> None:
        super().__init__(name)
        self.fp = Path(fp)
        
    def on_run(self) -> pd.DataFrame:
        df = pd.read_csv(self.fp)
        if "dt" in df.columns:
            df["dt"] = pd.to_datetime(df["dt"])
        return df


class Transformer(Task):

    pass


class AgeTransformerTask(Transformer):

    def on_run(self, df: pd.DataFrame) -> pd.DataFrame:
        assert "id" in df.columns
        assert "dt" in df.columns

        # age in days
        return (
            df.groupby("id")
            .aggregate({"dt": ["min", "max"]})["dt"]
            .reset_index()
            .assign(age=lambda df: (df["max"] - df["min"]).dt.days)[["id", "age"]]
        )


class AvgTransformerTask(Transformer):

    def __init__(self, name: str, month: int) -> None:
        super().__init__(name)
        self.month = month

    def on_run(
        self,
        df: Annotated[
            pd.DataFrame,
            Depends(
                schema=Schema(
                    [Field("id", int), Field("dt", datetime), Field("val", float)]
                )
            ),
        ],
    ) -> pd.DataFrame:
        assert "id" in df.columns
        assert "dt" in df.columns
        assert "val" in df.columns
        return (
            df[df["dt"].dt.month == self.month]
            .groupby("id")["val"]
            .mean()
            .reset_index()
            .rename({"val": f"val_m{self.month}"}, axis=1)
        )


class SquareTransformerTask(Transformer):

    def on_run(
        self, df: Annotated[pd.DataFrame, Depends(schema=Schema([Field("id", int)]))]
    ) -> pd.DataFrame:
        cols = (c for c in df.columns if c != "id")
        transformed = {c: df[c].apply(lambda x: x**2) for c in cols}
        return df.assign(**transformed)


class FakeModelTask(Task):

    def on_run(
        self,
        base_df: Annotated[pd.DataFrame, Depends(SquareTransformerTask)],
        extra_dfs: Annotated[list[pd.DataFrame], Depends(Transformer)],
    ) -> pd.DataFrame:
        for df in extra_dfs:
            base_df = base_df.merge(df, how="outer", on="id")
        inputs = sum(
            base_df[c].fillna(0) for c in base_df.columns if c.startswith("val")
        )
        return base_df.assign(predicted=inputs / base_df["age"])


"""
Pipeline:

read_base ----> square -----------.---> predict
                                  |
read_dt --.---> avg_jan -->--.----´
          |                  |
          `---> avg_mar -->--´
          |                  |
          `---> age ------>--´
"""


def main():
    with Pipeline() as pipeline:
        read_base = CsvReadTask("read_base", "data.csv")
        read_dt = CsvReadTask("read_dt", "dt_data.csv")

        square = SquareTransformerTask("square")
        avg_jan = AvgTransformerTask("avg_jan", 1)
        avg_mar = AvgTransformerTask("avg_mar", 3)
        age = AgeTransformerTask("age")

        predict = FakeModelTask("predict")

        read_base >> square >> predict
        read_dt >> [avg_jan, avg_mar, age] >> predict

    print(pipeline.run())


if __name__ == "__main__":
    main()
