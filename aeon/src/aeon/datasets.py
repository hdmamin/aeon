import os

from huggingface_hub import login, HfApi
import pandas as pd

from aeon import config
from aeon.secrets import SecretManager


def save_dataset(df: pd.DataFrame, name: str, save_local: bool = True, upload_to_hub: bool = True,
                 file_suffix: str = "pq", private: bool = False, **hf_kwargs) -> None:
    """Save a dataset to local file in {project_root}/data/datasets and/or create a 
    Huggingface Hub dataset in my hmamin/aeon collection.

    Parameters
    ----------
    private : bool
        If True and upload_to_hub is True, the resulting huggingface dataset will be made private.
        Note that this will disable their builtin dataset viewer in the UI.
    hf_kwargs : any
        Forwarded to huggingface's add_collection_item method. E.g. exists_ok=True
    """
    if save_local:
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

        repo_id = f"hmamin/{name}"
        hf_api.add_collection_item(
            collection_slug="hmamin/aeon",
            item_id=repo_id,
            item_type="dataset",
            **hf_kwargs
        )
        if private:
            hf_api.update_repo_visibility(
                repo_id=repo_id,
                repo_type="dataset",
                private=True
            )
