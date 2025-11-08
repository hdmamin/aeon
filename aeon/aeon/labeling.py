"""
from aeon import prompts

labeler = Labeler(prompt="extract_jokes")
df_labeled = labeler.label(df.id, df.text, output_dir="data/labeled", threads=10)
"""
import yaml

import pandas as pd

from aeon.config import PROJECT_ROOT


class LLMLabeler:

    def __init__(self, prompt: str):
        self.prompt = prompt

    def label(self, df: pd.DataFrame):
        pass