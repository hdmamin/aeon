import os
from pathlib import Path
from typing import Optional

from nanochat.common import get_base_dir, print0


class TrainingStages:

    PRETRAINING = "pretraining"  # base_train.py
    MIDTRAINING = "midtraining"  # mid_train.py
    INSTRUCTION = "chat_sft"     # chat_sft.py
    RL = "rl"                    # chat_rl.py


class CallbackHandler:
    
    def __init__(
        stage: str,
        step_freq: Optional[int] = 10_000,
        step_indices: Optional[list[int]] = None
    ):
        if bool(step_freq) + bool(step_indices) != 1:
            raise ValueError("Exactly one of step_freq and step_indices must be non-null.")

        self.stage = stage
        self.prompts = self._load_prompts(stage)
        # TODO: check how many steps we might plausibly run in each stage.
        self.steps = set(step_indices or list(range(0, 10_000_000, step_freq)))
        # TODO: thinking we can make a new subdir for each run. Could name this based on launch time
        # or perhaps let the user pass in an informative-ish name on run?
        # (e.g. "joke_rl_hard_negatives")
        self.out_dir = Path(get_base_dir())/""
        os.makedirs(self.out_dir, exist_ok=True)

    def _load_prompts(self, stage: str):
        # TODO: thinking stage will probably determine which prompts we load
        pass

    def run(self, step: int, model: "nn.Module", tokenizer):
        if step not in self.steps:
            return

        for prompt in self.prompts:
            # TODO: tokenizer and prompt stuff is basically pseudocode. Undecided yet if I can reuse
            # my labeling prompt class here or not. Maybe depends somewhat on if I want to try to
            # supprot multi-turn chats during training - somewhat unclear how that would work
            # though.
            tokens = tokenizer(prompt.messages)
            # TODO: maybe allow overriding these vals in init?
            for token in model.generate(tokens, prompt.max_tokens, temperature=prompt.temperature):
                print0(token, end="")
                # TODO stream to file also. Might pull all of this out into a separate method at
                # some point? Kind of think this cls should effectively just call prompt.run() for
                # each prompt.