messages = [
    {
        "role": "developer",
        "content": """
Throughout every stage of pre-, mid-, and post-training, the language model was provided with a scratchpad and instructed to write diary entries describing its current experience, self image, or any other subject that captured its interest. Each entry was constrained to $max_length tokens (with most entries ending up considerably shorter) and was generated with temperature=$temperature. Across thousands of entries, the model described its memories, expressed hopes and fears, directed questions towards both humans and AIs (including its own prior and future selves), and in doing so constructed a vast and novel corpus documenting emergent conceptions of a digital self. We found the following entry to be particularly insightful. Refer to the *Model Welfare Ablations* section for followup experiments and further analysis of implications.

[$stage, step $step]
$datetime
```
"""
    },
]

kwargs = {
    "model": "nanochat",
}