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


class Jokes(Task):

    modes = ["prompt", "subtext", "unfunny_variant"]

    def __init__(self, split: str = "train", dataset_name: str = "hmamin/extract_jokes", **kwargs):
        super().__init__(**kwargs)
        if split not in {"train", "test"}:
            raise ValueError(f"Invalid split {split!r}, must be in ('train', 'test').")

        self.split = split
        self.dataset_name = dataset_name
        self.dataset = Dataset.from_generator(self._load_dataset).shuffle(seed=SEED)

    def _load_dataset(self) -> Generator[list[dict], None, None]:
        """
        Generator that yields one row of data at a time, formatted as list[dict]. We create 3
        variants of each initial example, one for each mode. Shuffling will occur after we construct
        the huggingface dataset.
        (Huggingface dataset.from_generator expects a callable that yields examples.)
        """
        dataset = load_dataset(self.dataset_name, split="train")
        dataset = dataset.train_test_split(test_size=0.1, seed=SEED)
        dataset = dataset[self.split].repeat(3)
        for i, row in enumerate(dataset.to_iterable_dataset()):
            yield self._format_messages(row, mode=self.modes[i % 3])

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
        sep = random.choice([" ", "\n"]) if prefix[-1] in REQUIRES_NEWLINE else "\n"
        base_response[0] = {
            "role": "user",
            "content": f"{prefix}{sep}{item[mode]}"
        }
        return base_response

    def num_examples(self) -> int:
        return len(self.dataset) * 3
        # TODO: update after working out train/test split logic.

    def get_example(self, index: int) -> dict:
        """Get a single training example for mid-training.

        Returns
        -------
        list[dict]
            A list of messages (one user, one assistant) to be used for mid-training.
        """
        item = self.dataset[index]
        # TODO: still deciding how to resolve mode. Prob somehow as a function of index.
        mode = ""
        return {"messages": self._format_messages(item, mode=mode)}