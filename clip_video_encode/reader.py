"""handles input parsing."""
import pyarrow.parquet as pq
import pyarrow.csv as csv_pq
import pyarrow as pa
import pandas as pd


class Reader:
    """Parses input into required data.

    Necessary columns (reader will always look for these columns in parquet and csv):
    * videoLoc - location of video either on disc or URL
    * videoID - unique ID of each video, if not provided, ID = index
    """
    def __init__(self, src, meta_columns=[]):
        """
        Input:

        src:
            str: path to mp4 file
            str: youtube link
            str: path to txt file with multiple mp4's or youtube links
            list[str]: list with multiple mp4's or youtube links

        meta_columns:
            list[str]: columns of useful metadata to save with videos
        """
        self.columns = ["videoID", "videoLoc"]
        self.meta_columns = meta_columns

        if isinstance(src, str):
            if src.endswith(".txt"):
                df = csv_pq.read_csv(src, read_options=csv_pq.ReadOptions(column_names=["videoLoc"]))
                df = df.add_column(0, "videoID", [list(range(df.num_rows))]) # add ID's
            elif src.endswith(".csv"):
                df = csv_pq.read_csv(src)
            elif src.endswith(".parquet"):
                with open(src, "rb") as f:
                    columns_to_read = self.columns + meta_columns
                    df = pq.read_table(f, columns=columns_to_read)
        elif isinstance(src, list):
            df = pa.Table.from_arrays([src], names=["videoLoc"])
            df = df.add_column(0, "videoID", [list(range(df.num_rows))]) # add ID's

        self.df = df


    def get_data(self):
        vids = self.df["videoLoc"].to_pylist()
        IDS = self.df["videoID"]
        meta = dict([(meta, self.df[meta]) for meta in self.meta_columns])
        return vids, IDS, meta
