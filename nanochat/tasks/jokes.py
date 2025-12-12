from datasets import load_dataset

from tasks.common import Task


class Jokes(Task):

    modes = {"prompt", "subtext", "unfunny_variant"}

    def __init__(self, split="train", mode: str = "prompt", **kwargs):
        super().__init__(**kwargs)

        assert mode in self.modes, f"Invalid mode: {mode}. Should be one of {self.modes}"
        self.mode = mode
        self.split = split
        self.dataset = load_dataset("hmamin/extract_jokes", split=split)
        self._format_item = getattr(self, f"format_{self.mode}_example")

    def _format_prompt_example(self, item: dict) -> list[dict]:
        return [
            {
                "role": "user",
                "content": item["prompt"]
            },
            {
                "role": "assistant",
                "content": item["joke"]
            }
        ]

    def _format_subtext_example(self, item: dict) -> list[dict]:
        # TODO: consider adding some request prefix like "write a joke about" or "make this funnier"
        # or something? otherwise model may try to do this all the time which would be a bit weird
        return [
            {
                "role": "user",
                "content": item["subtext"]
            },
            {
                "role": "assistant",
                "content": item["joke"]
            }
        ]

    def _format_unfunny_variant_example(self, item: dict) -> list[dict]:
        # TODO: consider adding some request prefix like "make this funnier" or something?
        return [
            {
                "role": "user",
                "content": item["unfunny_variant"]
            },
            {
                "role": "assistant",
                "content": item["joke"]
            }
        ]

    def num_examples(self) -> int:
        return len(self.dataset)

    def get_example(self, index: int) -> dict:
        item = self.dataset[index]
        return {"messages": self._format_item(item)}