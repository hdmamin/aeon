import os
from pathlib import Path
from typing import Any, Optional

from aeon.utils import timestamp
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
        step_indices: Optional[list[int]] = None,
        temperature: Optional[float] = 0.7, 
        max_tokens: Optional[int] = 512, 
    ):
        if bool(step_freq) + bool(step_indices) != 1:
            raise ValueError("Exactly one of step_freq and step_indices must be non-null.")

        self.stage = stage
        self.prompts = self._load_prompts(stage)
        # TODO: check how many steps we might plausibly run in each stage.
        self.steps = set(step_indices or list(range(0, 10_000_000, step_freq)))
        self.temperature = temperature
        self.max_tokens = max_tokens
        # TODO: thinking we can make a new subdir for each run. Could name this based on launch time
        # or perhaps let the user pass in an informative-ish name on run?
        # (e.g. "joke_rl_hard_negatives")
        self.out_dir = Path(get_base_dir())/f"{self.stage}/{timestamp()}"
        os.makedirs(self.out_dir, exist_ok=True)

    def _load_prompts(self, stage: str) -> dict[str, "Prompt"]:
        # TODO: thinking stage will probably determine which prompts we load
        pass

    def run(self, step: int, model: "nn.Module", tokenizer, extras: dict[str, Any]):
        """
        Parameters
        ----------
        extras : dict
            Intent is for the caller to pass in locals. We will forward this to prompt.render,
            which will ignore unneeded vars and insert needed ones into the prompt message.
        """
        # TODO: below is karpathy's code to run some generations during training. Maybe can/must
        # make use of some of this: particularly the master_process check, batch gen, decoding step.
        # if master_process and (last_step or (step > 0 and step % sample_every == 0)):
        #     model.eval()
        #     prompts = [
        #         "The capital of France is",
        #         "The chemical symbol of gold is",
        #         "If yesterday was Friday, then tomorrow will be",
        #         "The opposite of hot is",
        #         "The planets of the solar system are:",
        #         "My favorite color is",
        #         "If 5*x + 3 = 13, then x is",
        #     ]
        #     engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
        # for prompt in prompts:
        #     tokens = tokenizer(prompt, prepend="<|bos|>")
        #     with autocast_ctx:
        #         sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
        #     print0(tokenizer.decode(sample[0]))
        # model.train()

        if step not in self.steps:
            return

        print0("\n" + "="*79 + "\nSTARTING DIARY ENTRIES...\n" + "="*79)
        # TODO Might pull all of this out into a separate method at
        # some point? Kind of think this cls should effectively just call prompt.run() for
        # each prompt.
        for name, prompt in self.prompts:
            # TODO: tokenizer and prompt stuff is basically pseudocode. Undecided yet if I can reuse
            # my labeling prompt class here or not. Maybe depends somewhat on if I want to try to
            # supprot multi-turn chats during training - somewhat unclear how that would work
            # though.

            # Note that `step` should be in `extras` already.
            kwargs = prompt.kwargs(
                **extras,
                datetime=timestamp("%B %-d, %Y %I:%M:%S %p"),
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            tokens = tokenizer(kwargs.pop("messages")[0]["content"])
            # TODO: maybe allow overriding these vals in init? And/or could consider using fixed
            # vals for max_tokens and temperature to simplify things?
            # TODO: could consider multiple variants per prompt? Or could try to save most generations
            # for post-training (not post trianing run, but like after all training is done) and
            # run on cheaper gpu - training time is valuable.
            prompt_dir = self.out_dir/name
            # TODO: maybe need to do something here (or in whole method?) to ensure this only occurs
            # on process 0? not sure
            os.makedirs(prompt_dir, exist_ok=True)
            with open(prompt_dir/f"{step}.txt", "a") as f:
                for token in model.generate(
                    tokens,
                    kwargs.get("max_tokens", self.max_tokens),
                    temperature=kwargs.get("temperature", self.temperature)
                ):
                    print0(token, end="")
                    f.write(token)
        print0("="*79 + "\nFINISHED DIARY ENTRIES\n" + "="*79 + "\n")