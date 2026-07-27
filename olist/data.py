from pathlib import Path
import pandas as pd


class Olist:
    """
    The Olist class provides methods to interact with Olist's e-commerce data.
    """

    def get_data(self):
        data_path = (
            Path.home()
            / ".workintech"
            / "olist"
            / "data"
            / "csv"
        )

        if not data_path.exists():
            raise FileNotFoundError(
                f"Veri klasörü bulunamadı: {data_path}"
            )

        data = {}

        for csv_file in data_path.glob("*.csv"):
            dataset_name = csv_file.stem
            dataset_name = dataset_name.removeprefix("olist_")
            dataset_name = dataset_name.removesuffix("_dataset")

            data[dataset_name] = pd.read_csv(csv_file)

        if not data:
            raise FileNotFoundError(
                f"{data_path} içinde CSV dosyası bulunamadı."
            )

        return data

    def ping(self):
        """
        You call ping I print pong.
        """
        print("pong")
