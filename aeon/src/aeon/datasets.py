import os
from typing import Optional

from datasets import Dataset
from huggingface_hub import login, HfApi
import pandas as pd

from aeon import config
from aeon.secrets import SecretManager


def save_dataset(
        df: pd.DataFrame, name: str, save_local: bool = True, upload_to_hub: bool = True,
        file_suffix: str = "pq", private: bool = False,
        description: Optional[str] = None,
        infisical_api_key: Optional[str] = None, **hf_kwargs
    ) -> None:
    """Save a dataset to local file in {project_root}/data/datasets and/or create a 
    Huggingface Hub dataset in my hmamin/aeon collection.

    Parameters
    ----------
    private : bool
        If True and upload_to_hub is True, the resulting huggingface dataset will be made private.
        Note that this will disable their builtin dataset viewer in the UI.
    description : str or None
        If provided, this will be added to the huggingface dataset and displayed in their dataset
        hub. Basically a readme for the dataset. I suggest including at least a one line description
        and url(s) pointing to the source of the data if applicable.
    hf_kwargs : any
        Forwarded to huggingface's add_collection_item method. E.g. exists_ok=True (though note,
        I realized this does not actually allow overwriting items, just determines whether an error
        is raised)
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
        secret_manager = SecretManager(
            client_secret=os.environ.get("INFISICAL_API_KEY", infisical_api_key)
        )
        secrets = secret_manager.get_secrets()
        login(secrets["HUGGINGFACE_TOKEN"])
        hf_api = HfApi()

        repo_id = f"hmamin/{name}"
        dataset = Dataset.from_pandas(df)
        if description:
            dataset.info.description = description
        dataset.push_to_hub(repo_id)
        if description:
            hf_api.upload_file(
                repo_id=repo_id,
                repo_type="dataset",
                path_in_repo="README.md",
                content=description,
            )

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
