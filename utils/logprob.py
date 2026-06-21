import torch
import torch.nn.functional as F


def model_device(model):
    return next(model.parameters()).device


def token_log_probs(model, input_ids, attention_mask=None):
    """Log-probabilities of labels input_ids[:, 1:] under a causal LM."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def response_token_mask(input_ids, attention_mask, response_start):
    """Mask label positions that correspond to response tokens."""
    labels = input_ids[:, 1:]
    mask = torch.zeros_like(labels, dtype=torch.float32)
    mask[:, max(response_start - 1, 0) :] = 1.0
    if attention_mask is not None:
        mask = mask * attention_mask[:, 1:].float()
    return mask


def response_log_prob(model, tokenizer, prompt, response, max_length):
    """Sum log pi(response | prompt) over response tokens."""
    device = model_device(model)
    full_text = prompt + " " + response

    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).input_ids.to(device)

    full = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)

    response_start = min(prompt_ids.shape[1], full.input_ids.shape[1] - 1)
    log_probs = token_log_probs(model, full.input_ids, full.attention_mask)
    mask = response_token_mask(full.input_ids, full.attention_mask, response_start)
    return (log_probs * mask).sum()


def response_log_probs_from_ids(model, input_ids, attention_mask, response_start):
    """Per-token log-probs for generated response tokens."""
    log_probs = token_log_probs(model, input_ids, attention_mask)
    mask = response_token_mask(input_ids, attention_mask, response_start)
    lengths = mask.sum(dim=1).long()

    rows = []
    for batch_idx, length in enumerate(lengths.tolist()):
        selected = log_probs[batch_idx][mask[batch_idx].bool()]
        rows.append(selected[:length])
    return rows, mask
