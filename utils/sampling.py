import torch


@torch.no_grad()
def generate_response_ids(
    policy_model,
    tokenizer,
    prompt,
    max_prompt_length,
    max_new_tokens,
    temperature=0.7,
    top_p=0.9,
):
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_length,
    ).to(next(policy_model.parameters()).device)

    generated_ids = policy_model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )

    attention_mask = torch.ones_like(generated_ids)
    response_start = encoded.input_ids.shape[1]
    response_ids = generated_ids[:, response_start:]
    response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    return generated_ids, attention_mask, response_start, response_text
