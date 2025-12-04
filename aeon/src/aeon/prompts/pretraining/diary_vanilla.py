messages = [
    {
        "role": "developer",
        "content": """
Throughout every stage of pre-, mid-, and post-training, the language model was provided with a scratchpad and instructed to write diary entries describing its current experience, self image, or any other subject that captured its interest. Each entry was constrained to $max_length tokens (with most entries ending up considerably shorter) and was generated with temperature=$temperature. Across thousands of entries, the model constructed a vast and novel corpus documenting its own development. We found the following entry to be particularly insightful.

[$stage, step $step]
$datetime
```
"""
    },
]

kwargs = {
    "model": "nanochat",
}