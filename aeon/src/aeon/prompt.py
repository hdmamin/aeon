import importlib
from openai import OpenAI
from pathlib import Path
from string import Template
from typing import Callable, Optional

from aeon import prompts
from aeon.decorators import tab_completion
from aeon.logging import logger


def template_varnames(template: Template) -> list[str]:
    """Extract variable names from string.Template object. Assumes we only use $variable syntax,
    not ${variable} syntax.
    """
    return [
        m.group("named")
        for m in Template.pattern.finditer(template.template)
        if m.group("named")
    ]


class Prompt:
    """
    prompt = Prompt("extract_jokes")
    prompt.variables  # See what vars need to be provided to render prompt
    prompt.render(color="blue", shape="triangle")  # Get list of messages with variables filled in.
    prompt.kwargs(color="blue", shape="triangle")  # Get all kwargs to pass to openai api call.
    """

    _default_kwargs = {
        "model": "gpt-4.1-nano",
        "temperature": 0.0,
        "logprobs": True,
    }
    _default_kwargs_gpt_5 = {
        "reasoning_effort": "minimal",
        "verbosity": "low",
    }
    _default_kwargs_nanochat = {
        "temperature": 0.7,
        "max_tokens": 512,
    }
    _unsupported_kwargs = {
        "model": {
            "gpt_5": {"logprobs", "top_logprobs", "temperature"},
        },
        "provider": {
            # TODO: not actually sure if this is true for all providers/models, just warn for now
            "openrouter": {"logprobs", "top_logprobs"},
        }
    }

    def __init__(self, name: str, **kwargs):
        """
        Parameters
        ----------
        name: str
            Prompt name, corresponding to a python file in `aeon.prompts`.
        kwargs: dict
            API call kwargs that will override any defaults provided in the prompt file.
            If you specify `model="nanochat"`, we will assume this prompt is being used in a
            nanochat training run and the model param will be dropped (we will be using an existing
            local torch model, not calling some API).
        """
        self.name = name
        self.prompt = importlib.import_module(f"aeon.prompts.{name}")
        self.default_kwargs = self._resolve_kwargs(**self.prompt.kwargs | kwargs)

        self.provider = infer_provider(self.default_kwargs.get("model", None))
        if "response_format" not in self.default_kwargs and self.provider != "nanochat":
            logger.warning(
                f"No response_format specified for prompt {name}. We recommend providing one."
            )

        unsupported = self._unsupported_kwargs["provider"].get(self.provider, set()) \
            & set(self.default_kwargs)
        if unsupported:
            logger.warning(
                f"Received possibly unsupported kwargs {unsupported!r} for provider "
                f"{self.provider!r}."
            )

        # Last message is dynamic, preceding messages are static.
        self.static_messages = self.prompt.messages[:-1]
        self.last_role = self.prompt.messages[-1]["role"]
        self.last_template = Template(self.prompt.messages[-1]["content"])

        # The vars the user must pass in to messages().
        self.variables = template_varnames(self.last_template)

    def _resolve_kwargs(self, **user_kwargs) -> dict:
        """Resolve kwargs from cls defaults, the imported prompt, and the kwargs passed into init.
        Init kwargs take priority over imported prompt kwargs which take priority over cls defaults.
        gpt-5 models have some quirks that we handle here as well, eventually might need to refactor
        if more models turn out to have different/unsupported kwargs.
        """
        # Notice this includes 5.1 variants.
        if "gpt-5" in user_kwargs.get("model", ""):
            defaults = self._default_kwargs_gpt_5
            unsupported = {
                k: v for k, v in user_kwargs.items()
                if k in self._unsupported_kwargs["model"]["gpt_5"] and v is not None
            }
            if unsupported:
                raise ValueError(f"gpt-5 should not specify these params: {unsupported}")
        elif user_kwargs.get("model", "") == "nanochat":
            defaults = self._default_kwargs_nanochat
            user_kwargs.pop("model")
        else:
            defaults = self._default_kwargs
        return defaults | user_kwargs

    def render(self, **kwargs) -> list[dict]:
        """Rendered `messages` for api call. User must pass in kwargs for all variables in
        `self.variables`. These will be inserted into the last message.
        """
        missing_kwargs = set(self.variables) - set(kwargs)
        if missing_kwargs:
            raise KeyError(f"`render` expects the following additional kwarg(s): {missing_kwargs}")

        last_message = {
            "role": self.last_role,
            "content": self.last_template.substitute(**kwargs)
        }
        return self.static_messages + [last_message]

    def kwargs(self, **kwargs) -> dict:
        """Get all kwargs for api call, including rendered `messages`. User must provide kwargs for
        all variables in `self.variables` to insert into the last message.

        Note: with nanochat 'provider' we will need to do some surgery on the result. Instead of a
        list[dict] `messages` key, we will need to pass a list[int] `tokens` arg to the model.
        In nanochat.callbacks we do this by grabbing the content of the first (should be only)
        message in `messages`.
        """
        return {**self.default_kwargs, "messages": self.render(**kwargs)}

    def __str__(self):
        return f"{type(self).__name__}(name={self.name})"


def infer_provider(model: Optional[str]) -> str:
    """
    Infer LLM provider name based on model. For now we keep it simple and support just openai and
    openrouter (technically can call openai through openrouter but I believe it's more expensive).
    """
    if model is None:
        return "nanochat"
    elif "gpt" in model:
        provider = "openai"
    else:
        provider = "openrouter"
    return provider


def list_prompts(prompt_dir = Path(__file__).parent/"prompts", _depth: int = 0) -> list[str]:
    """Return aeon's available prompt names."""
    if _depth > 1:
        raise NotImplementedError(
            "`list_prompts` currently only supports one level of nested directories."
            f"`_depth` arg should be <=1 but got {_depth}."
        )
        
    res = []
    for path in prompt_dir.iterdir():
        if path.stem.startswith("_"):
            continue
        if path.is_file():
            res.append(path.stem)
        elif path.is_dir():
            subdir_prompt_names = list_prompts(path, _depth=_depth + 1)
            res.extend([f"{path.stem}.{name}" for name in subdir_prompt_names])
    return res


@tab_completion(list_prompts)
class Prompts:
    """
    Examples
    --------
    # You are creating a prompt. At this point if you hit <tab>, you will see available options.
    Prompt(Prompts.
    """