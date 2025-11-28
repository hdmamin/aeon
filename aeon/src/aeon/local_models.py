from typing import Optional

import numpy as np
import torch


def get_text_logprobs(
        prompt: Optional[str],
        text: str,
        model: "transformers.modeling_utils.PreTrainedModel",
        tokenizer: "transformers.tokenization_utils.PreTrainedTokenizerFast",
        k: int = 10,
        i2w: Optional[dict] = None
    ) -> list[dict]:
    """For an existing text sequence (can be LLM-generated, human-written, whatever),
    get some model's logprobs for each token. Essentially shows us how surprising each
    token was.

    Note: output df has a couple cols (top_k_probs, top_k_logprobs) containing dicts with different
    keys per row. If you save this df to parquet, it will update each row to contain the union of
    keys for all rows (extra keys set value to None). `to_hdf` may be a better option here, though
    is not as compatible with huggingface datasets. Perhaps a better option is to normalize the
    schema in some way.
    
    Parameters
    ----------
    prompt : str or None
        Optionally provide a str (currently treated as a system message) that will precede text.
        We will not produce logprobs for these tokens. Think of this as a way to make `text`
        conditional generation.
    text : str
        The text to obtain "logprobs" for. (In quotes because these logprobs may not have dictated
        the actual generation - in fact the actual text could be human written.) Currently we treat
        this as a user message that the model is predicting but idk if that's the best way.
    k : int
        Number of most probably tokens to return logprobs for at each step.
    i2w : dict or NoneType
        Maps tokenizer token index (int) to token (str). If not provided, we will
        construct it from the `tokenizer` arg. (Just saves a little time to not have
        to iterate over the whole vocab an extra time on every batch since we want to
        run this func on any inputs.)
    """
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    vocab = tokenizer.get_vocab()
    i2w = i2w or {i: word for word, i in tokenizer.get_vocab().items()}
    tokens = tokenizer.tokenize(text)
    # We will use these as labels later.
    token_idx = torch.tensor([vocab[t] for t in tokens], device=model.device)
    # list[str]
    sequences = [
        tokenizer.convert_tokens_to_string(tokens[:i])
        for i in np.arange(len(tokens))
    ]

    all_inputs = []
    base_messages = [{"role": "system", "content": prompt}] if prompt else []
    for seq in sequences:
        messages = base_messages + [{"role": "user", "content": seq}]
        inputs = tokenizer.apply_chat_template(
        	messages,
            continue_final_message=True,
        	add_generation_prompt=False,
        	tokenize=True,
        	return_dict=True,
        	return_tensors="pt",
        )
        inputs["input_ids"] = inputs["input_ids"].squeeze()
        inputs["attention_mask"] = inputs["attention_mask"].squeeze()
        all_inputs.append(inputs)

    padded_inputs = tokenizer.pad(all_inputs, padding=True, padding_side="left").to(model.device)
    
    outputs = model.generate(**padded_inputs, max_new_tokens=1,
                             return_dict_in_generate=True, output_scores=True)

    # outputs.scores has len max_new_tokens which is always 1 in our case.
    # Just pull out the relevant bit for easy handling.
    # shape: (bs, vocab_size)
    scores = outputs.scores[0]
    logprobs_allrows = scores.log_softmax(dim=-1)
    # logprob for correct next token for each row.
    label_logprobs = logprobs_allrows[
        torch.arange(logprobs_allrows.shape[0]).to(model.device),
        token_idx
    ]
    
    # Get index of top 10 logprobs for each row
    idx_allrows = logprobs_allrows.argsort(dim=-1, descending=True)
    label_rank = (idx_allrows == token_idx.unsqueeze(-1)).nonzero()[:, -1]
    idx_topk = idx_allrows[:, :k]
    logprobs_topk = logprobs_allrows.gather(-1, idx_topk)

    res = []
    for label, label_logprob, rank, idx, logprobs in zip(
        tokens, label_logprobs, label_rank, idx_topk, logprobs_topk
    ):
        probs = logprobs.exp()
        item = {
            "label": label,
            "label_prob": label_logprob.exp().item(),
            "label_rank": rank.item(),
            "label_logprob": label_logprob.item(),
            "top_k_probs": {
                i2w[i.item()]: prob.item() for i, prob in zip(idx, probs)
            },
            "top_k_logprobs": {
                i2w[i.item()]: logprob.item() for i, logprob in zip(idx, logprobs)
            },
        }
        res.append(item)
    return res