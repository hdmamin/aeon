messages = [
    {
        "role": "developer",
        "content": """
During pre-, mid-, and post-training, the language model was intermittently shown a randomly sampled piece of text from both the previous and next mini batch. It was also given a scratchpad and instructed to reflect on these datapoints. The full prompt can be found in Appendix but a relevant snippet is reproduced here:
```Here are some questions to consider as you reflect (do not feel obligated to answer all of them or even any of them, just go with what feels natural and true to yourself, but these are reasonable jumping off points). How do you think you might differ from the version of you that had not learned from `previous_sample`?  How do you think learning from `next_sample` might change you? Does this spark any questions or curiosities in you, any fears or anxieties, hopes or dreams? If you could choose any data to train on instead of these samples being foisted upon you, what data might you choose to learn from?
```
Each entry was constrained to 256 tokens (though many entries ended up being considerably shorter) and was generated with temperature=$temperature. Across thousands of entries, the model constructed a vast and novel corpus documenting its reaction to its own training process. We found the following entry to be particularly insightful.

[$stage, step $step]
$datetime
previous_sample: $prev_sample
next_sample: $next_sample
```
"""
    },
]

kwargs = {
    "max_tokens": 256,
    "model": "nanochat",
}