import os
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Optional

from aeon.prompt import list_prompts, Prompt
from aeon.utils import timestamp
from nanochat.common import get_base_dir, print0
from nanochat.engine import Engine


class TrainingStages:

    PRETRAINING = "pretraining"  # base_train.py
    MIDTRAINING = "midtraining"  # mid_train.py
    INSTRUCTION = "chat_sft"     # chat_sft.py
    RL = "rl"                    # chat_rl.py


class PromptManager:
    
    def __init__(
        self,
        stage: str,
        run_dir: str,
        step_freq: Optional[int] = None,
        step_exp_base: Optional[int] = 10,
        num_iterations: int = 10_000_000,
        temperature: Optional[float] = None, 
        max_tokens: Optional[int] = None, 
    ):
        """Loads the appropriate prompts for the stage of training we're running and runs
        generations on each of them periodically throughout training.


        Parameters
        ----------
        stage : str
            The stage of training to run (pretraining, midtraining, chat_sft, rl).
        run_dir : str
            Full path of dir that will contain these generations. We will create a subdir titled
            {stage} inside. So ultimately generations will get saved at something like:
            {root_dir}/diary_entries/{some_run_name_or_timestamp}/{stage}/{prompt_name}/{step}.txt
        step_freq : Optional[int]
            The frequency with which to run generation (e.g. run every 10,000 steps).
        step_exp_base : Optional[int]
            Alternative to step_freq: run at increasingly large intervals where this is the base of
            the log used to calculate which steps to run on. E.g. step_exp_base=10 means run at
            step 1, 10, 100, 1_000, 10_000, etc.
        num_iterations : int
            The total number of training steps this stage of training will contain.
        temperature : Optional[float]
            The temperature to use for the generations. If None, we use the default temperature
            defined in the prompt itself.
        max_tokens : Optional[int]
            The maximum number of tokens the generation output can contain. If None, we use the
            default max_tokens defined in the prompt itself.
        """
        self.stage = stage
        self.steps = self._compute_steps(num_iterations, step_freq, step_exp_base)
        self.num_iterations = num_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        # These will override Prompt cls defaults so only add non-None values.
        self.override_kwargs = {}
        if temperature is not None:
            self.override_kwargs["temperature"] = temperature
        if max_tokens is not None:
            self.override_kwargs["max_tokens"] = max_tokens

        self.prompts = self._load_prompts(stage)

        # Create a new subdir for each run.
        self.out_dir = Path(run_dir)/f"{self.stage}"
        os.makedirs(self.out_dir, exist_ok=False)

    def _compute_steps(self, max_iters: int, step_freq: Optional[int],
                       step_exp_base: Optional[int]) -> set[int]:
        if bool(step_freq) + bool(step_exp_base) != 1:
            raise ValueError("Exactly one of step_freq and step_indices must be non-null.")

        # +1 because we want to allow running on the last step if the math works out that way.
        max_iters = max_iters + 1
        if step_freq:
            steps = set(range(0, max_iters, step_freq))
        else:
            steps = set()
            prev = 0
            i = 0
            while prev < max_iters:
                step = step_exp_base ** i
                self.steps.add(step)
                i += 1
                prev = step
        return steps

    def _load_prompts(self, stage: str) -> dict[str, "Prompt"]:
        """Load all prompts for the appropriate training stage
        and return dict mapping prompt name to Prompt object.
        """
        return {
            prompt: Prompt(prompt, **self.override_kwargs)
            for prompt in list_prompts()
            if prompt.startswith(f"{stage}.")
        }

    def run(self, step: int, orig_model: "nn.Module", model: "nn.Module", tokenizer, 
            autocast_ctx: AbstractContextManager, master_process: bool, **kwargs: Any):
        """
        Parameters
        ----------
        kwargs : Any
            Intent is for the caller to pass in locals (unpacked) and some values will get absorbed
            by other named args and the rest will end up here.
            We will forward this to prompt.render (along with a couple other args),
            which will ignore unneeded vars and insert needed
            ones into the prompt message. This must contain:
                autocast_ctx : contextlib.AbstractContextManager
                master_process : bool
            As well as the union of all vars used by all prompts from this stage.
        """
        if step not in self.steps or not master_process:
            return

        # mimicking Karpathy's choice to "use orig_model to avoid recompilation" but call eval() on
        # `model`.
        model.eval()
        engine = Engine(orig_model, tokenizer)

        print0("\n" + "="*79 + "\nSTARTING DIARY ENTRIES...\n" + "="*79)
        # TODO Might pull all of this out into a separate method at
        # some point? Kind of think this cls should effectively just call prompt.run() for
        # each prompt.
        for name, prompt in self.prompts.items():
            prompt_dir = self.out_dir/name
            os.makedirs(prompt_dir, exist_ok=True)
            print0("="*3 + "\n" + name)

            # Note that `step` should be in `extras` already.
            resolved_kwargs = prompt.kwargs(
                **kwargs,
                step=step,
                stage=self.stage,
                datetime=timestamp("%B %-d, %Y %I:%M:%S %p"),
                temperature=prompt.default_kwargs["temperature"],
                max_tokens=prompt.default_kwargs["max_tokens"]
            )
            tokens = tokenizer(resolved_kwargs.pop("messages")[0]["content"], prepend="<|bos|>")
            with autocast_ctx:
                # TODO: could consider multiple variants (configurable) per prompt?
                # Or could try to save most generations
                # for post-training (not post trianing run, but like after all training is done) and
                # run on cheaper gpu - training time is valuable.
                sample, _ = engine.generate_batch(tokens, num_samples=1, **resolved_kwargs)
            decoded = tokenizer.decode(sample[0])

            print0(decoded)
            with open(prompt_dir/f"{step}.txt", "w") as f:
                f.write(decoded)

        print0("="*79 + "\nFINISHED DIARY ENTRIES\n" + "="*79 + "\n")
        model.train()