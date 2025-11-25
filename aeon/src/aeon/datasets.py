import os

from huggingface_hub import login, HfApi
import pandas as pd

from aeon import config
from aeon.secrets import SecretManager


def save_dataset(df: pd.DataFrame, name: str, upload_to_hub: bool = True, file_suffix: str = "pq") -> None:
    """Save a dataset to local parquet in {project_root}/data/datasets and optionally create a 
    Huggingface Hub dataset in my hmamin/aeon collection.
    """
    out_dir = config.DATA_DIR/f"datasets/{name}"
    os.makedirs(out_dir, exist_ok=True)
    if file_suffix == "pq":
        df.to_parquet(out_dir/"df.pq")
    elif file_suffix == "h5":
        df.to_hdf(out_dir/"df.pq", key="df")
    else:
        raise ValueError(f"Unsupported file_suffix: {file_suffix!r}")

    if upload_to_hub:
        secrets = SecretManager().get_secrets()
        login(secrets["HUGGINGFACE_TOKEN"])
        hf_api = HfApi()
        hf_api.add_collection_item(
            collection_slug="hmamin/aeon",
            item_id=f"hmamin/{name}",
            item_type="dataset"
        )
