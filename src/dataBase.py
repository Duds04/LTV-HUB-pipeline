from src.task import Task
import pandas as pd
from pathlib import Path

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
