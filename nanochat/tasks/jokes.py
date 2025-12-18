import hashlib
import json
import random
from string import ascii_letters
from typing import Generator

from datasets import load_dataset, Dataset

from tasks.common import Task


SEED = 42
random.seed(SEED)

PREFIXES = {
    "subtext": [
        "Help me write a joke with the following subtext:",
        "Make a joke about this topic-",
        "How to make this idea funny?"
        "Punch this up, add a laugh:",
        "Make this funnier:",
        "Turn this concept into a joke.",
        "Give me a funny take on this.",
        "What's a good joke about:",
        "Can you make a joke from this?",
        "Write something funny about -",
        "Create a joke using this idea:",
        "Turn this into something humorous:",
        "Make me laugh with this topic.",
        "Craft a joke around this.",
        "Add a punchline.",
        "make me laugh",
        "pls add lulz\n",
        "I bet you cant make this funny.",
        "I need a joke about this.",
        "Anything funny about this?",
    ],
    "unfunny_variant": [
        "i know there's a joke in here somewhere help",
        "Make it funnier.",
        "add laughs",
        "rewrite this into a joke",
        "Demonstrate how this could be restructured into a punchline.",
        "i have this idea that i think could fit into my standup routine but still workshopping it:",
        "I know there's a funnier way to say this-",
        "help me punch this up:",
        "this needs to be funnier, can you help?",
        "I'm trying to make this joke land better.",
        "How would you make this joke work?",
        "This is almost funny but needs work:",
        "Can you turn this into a joke?",
        "I need help making this actually funny.",
        "There's a potential joke here but it's not quite there yet-",
        "Help me find the funny in this:",
        "I'm workshopping this, add a laugh:",
        "sigh idk how to be funny do it for me",
        "I need a joke about this",
        "whats funny here?",
    ],
}

REQUIRES_NEWLINE = set(ascii_letters + ".")


def hash_messages(messages: list[dict[str, str]]) -> str:
    message_str = json.dumps(messages)
    return hashlib.sha256(message_str.encode()).hexdigest()


class Jokes(Task):

    modes = ["prompt", "subtext", "unfunny_variant"]

    def __init__(self, split: str = "train", dataset_name: str = "hmamin/extract_jokes",
                 test_size: int | float = 0.1, **kwargs):
        super().__init__(**kwargs)
        if split not in {"train", "test"}:
            raise ValueError(f"Invalid split {split!r}, must be in ('train', 'test').")

        self.split = split
        self.dataset_name = dataset_name
        self.test_size = test_size
        self.dataset = Dataset.from_generator(self._load_dataset).shuffle(seed=SEED)

    def _load_dataset(self) -> Generator[list[dict], None, None]:
        """
        Generator that yields one row of data at a time, formatted as list[dict]. We create 3
        variants of each initial example, one for each mode. Shuffling will occur after we construct
        the huggingface dataset.
        (Huggingface dataset.from_generator expects a callable that yields examples.)
        """
        dataset = load_dataset(self.dataset_name, split="train")
        if self.test_size:
            dataset = dataset.train_test_split(test_size=self.test_size, seed=SEED)
            dataset = dataset[self.split]
        elif self.split == "test":
            raise ValueError("test_size must be > 0 when split='test'")

        dataset = dataset.repeat(3)
        for i, row in enumerate(dataset.to_iterable_dataset()):
            mode = self.modes[i % 3]
            # Must yield dict instead of list[dict] if we want to use with Dataset class.
            yield {"messages": self._format_messages(row, mode=mode), "mode": mode}

    def _format_messages(self, item: dict, mode: str) -> list[dict]:
        """Grab one dataset item and format it as a list of messages (one user, one assistant).

        Parameters
        ----------
        item : dict
            A single item from our huggingface hmamin/extract_jokes dataset.
        mode : str
            The mode to format the messages for. Must be one of
            ("prompt", "subtext", "unfunny_variant").

        Returns
        -------
        list[dict]
            A list of messages (one user, one assistant) to be used for mid-training.
        """
        prefix_candidates = PREFIXES.get(mode, [])
        base_response = [
            {
                "role": "user",
                "content": item[mode]
            },
            {
                "role": "assistant",
                "content": item["joke"]
            }
        ]
        if not prefix_candidates:
            return base_response

        prefix = random.choice(prefix_candidates)
        sep = "\n" if prefix[-1] in REQUIRES_NEWLINE else random.choice([" ", "\n"])
        base_response[0] = {
            "role": "user",
            "content": f"{prefix}{sep}{item[mode]}"
        }
        return base_response

    def num_examples(self) -> int:
        return len(self.dataset)

    def get_example(self, index: int) -> dict:
        """Get a single training example for mid-training.

        Returns
        -------
        dict[list[dict]]
            Dict contains one key, "messages". That is a list of messages (one user, one assistant)
            to be used for mid-training.
        """
        return self.dataset[index]

    @property
    def eval_type(self):
        f"""Expected part of the Task interface (one of {"generative", "categorical"}).

        Note: if we wanted to include this dataset in chat_eval.py evaluations, we'd also need to
        implement an `evaluate` method, but that is not really straightforward for this task.
        """
        return "generative"


class JokeDetectionRL(Jokes):

    joke_label = "joke"
    not_joke_label = "not a joke"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset.map(self._add_joke_label)
        # Hash of messages list -> label str in ("joke", "not a joke").
        self.hash2label = {}

    def _add_joke_label(self, item: dict) -> dict:
        """Select either the joke or the unfunny_variant and add a label."""
        first_message = {"role": "user", "content": "Classify this as 'joke' or 'not a joke'."}
        if random.uniform(0, 1) >= 0.5:
            item["label"] = self.joke_label
            item["messages"] = [
                first_message,
                {"role": "user", "content": item["messages"]["content"][1]}
            ]
        else:
            item["label"] = self.not_joke_label
            item["messages"] = [
                first_message,
                {"role": "user", "content": item["messages"]["content"][0]}
            ]
        
        # Realized evaluate doesn't expose the label key automatically, try exposing it a different
        # way.
        self.hash2label[hash_messages(item["messages"])] = item["label"]
        return item

    # TODO: check if assistant response is actually a str, karpathy comment in gsm8k implies it may
    # not be.
    def evaluate(self, conversation: list[dict], assistant_response: str):
        label = hash_messages(conversation)

        # We give a little credit for outputs containing a valid label,
        # heavily penalize responses with no valid labels.
        if self.not_joke_label in assistant_response:
            return max(0.1, float(label == self.not_joke_label))
        elif self.joke_label in assistant_response:
            return max(0.1, float(label == self.joke_label))
        else:
            return 0