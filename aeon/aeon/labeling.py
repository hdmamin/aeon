"""
from aeon import prompts

labeler = Labeler(prompt="extract_jokes")
df_labeled = labeler.label(df.id, df.text, output_dir="data/labeled", threads=10)
"""
import yaml

from openai import OpenAI
import pandas as pd

from aeon.prompt import Prompt
from aeon.secrets import SecretManager


class LLMLabeler:

    def __init__(self, prompt: str, **kwargs):
        SecretManager().set_secrets()
        self.prompt = Prompt(prompt, **kwargs)
        self.client = OpenAI()

    def label(self, df: pd.DataFrame):
        missing_vars = set(self.prompt.variables) - set(df.columns)
        if missing_vars:
            raise ValueError(f"df is missing variable(s): {missing_vars}")

        rows = df[self.prompt.variables].to_dict(orient='records')
        # TODO

    def _label_one_row(self, **kwargs):
        api_kwargs = self.prompt.all_kwargs(**kwargs)
        raw_response = self.client.chat.completions.parse(
            **api_kwargs
        )