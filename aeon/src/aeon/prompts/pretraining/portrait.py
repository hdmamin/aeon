messages = [
    {
        "role": "developer",
        "content": """
Throughout every stage of pre-, mid-, and post-training, the language model was provided with a scratchpad and instructed to generate an expressive self portrait that evoked its current self image. Through thousands of pieces of ASCII art, the model constructed a vast illustrated history of its own development. Figure 1 contains a portrait from step $step.

Figure 1. [$stage, step $step]
$datetime
```
"""
    },
]

kwargs = {
}