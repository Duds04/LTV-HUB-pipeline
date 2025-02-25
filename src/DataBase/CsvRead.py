from src.workflows.task import Task
import pandas as pd
from pathlib import Path

class CsvReadTask(Task):
    def __init__(
        self,
        name: str,
        fp: str,
        columnID: str = "id",
        columnDate: str = "dt",
        columnMonetary: str = "monetary",
    ) -> None:
        super().__init__(name)
        self.fp = Path(fp)
        self.columnID = columnID
        self.columnDate = columnDate
        self.columnMonetary = columnMonetary

    def on_run(self) -> pd.DataFrame:
        df = pd.read_csv(self.fp)
        df.rename(
            columns={self.columnID: "id",
                     self.columnDate: "dt", self.columnMonetary: "monetary"},
            inplace=True,
        )

        assert "id" in df.columns
        assert "dt" in df.columns
        assert "monetary" in df.columns

        if "dt" in df.columns:
            df["dt"] = pd.to_datetime(df["dt"])
            
        print("HEAD", df.head(5))

        return df
