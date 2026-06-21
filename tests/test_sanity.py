from types import SimpleNamespace
import unittest

import torch
from torch import nn

from train.dpo import dpo_loss
from train.sft import build_sft_batch
from utils.advantage import compute_gae, normalize_advantages
from utils.logprob import (
    prompt_response_inputs,
    response_log_prob,
    response_token_mask,
    token_log_probs,
)


class Batch(SimpleNamespace):
    def to(self, device):
        for key, value in vars(self).items():
            if torch.is_tensor(value):
                setattr(self, key, value.to(device))
        return self


class DummyTokenizer:
    def __init__(self, vocab_size=64):
        self.vocab_size = vocab_size

    def __call__(
        self,
        text,
        return_tensors="pt",
        truncation=False,
        max_length=None,
        add_special_tokens=True,
    ):
        ids = []
        if add_special_tokens:
            ids.append(1)
        ids.extend((ord(ch) % (self.vocab_size - 2)) + 2 for ch in text)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        if not ids:
            ids = [0]
        input_ids = torch.tensor([ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return Batch(input_ids=input_ids, attention_mask=attention_mask)


class DummyCausalLM(nn.Module):
    def __init__(self, vocab_size=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 8)
        self.lm_head = nn.Linear(8, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        hidden = self.embedding(input_ids)
        return SimpleNamespace(logits=self.lm_head(hidden))


class LogProbTests(unittest.TestCase):
    def test_prompt_response_boundary_is_exact(self):
        tokenizer = DummyTokenizer()
        input_ids, attention_mask, response_start = prompt_response_inputs(
            tokenizer,
            prompt="Human: hi\nAssistant:",
            response="hello",
            max_length=128,
            device="cpu",
        )

        prompt_ids = tokenizer("Human: hi\nAssistant:").input_ids[0]
        response_ids = tokenizer(" hello", add_special_tokens=False).input_ids[0]

        self.assertEqual(response_start, prompt_ids.numel())
        self.assertTrue(torch.equal(input_ids[0, :response_start], prompt_ids))
        self.assertTrue(torch.equal(input_ids[0, response_start:], response_ids))
        self.assertTrue(torch.equal(attention_mask, torch.ones_like(input_ids)))

    def test_response_mask_selects_response_labels(self):
        input_ids = torch.tensor([[10, 11, 12, 13, 14]])
        attention_mask = torch.ones_like(input_ids)
        mask = response_token_mask(input_ids, attention_mask, response_start=3)

        expected = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        self.assertTrue(torch.equal(mask, expected))

    def test_response_log_prob_keeps_policy_gradient(self):
        tokenizer = DummyTokenizer()
        model = DummyCausalLM()
        logp = response_log_prob(
            model,
            tokenizer,
            prompt="Human: hi\nAssistant:",
            response="hello",
            max_length=128,
        )

        self.assertEqual(logp.ndim, 0)
        logp.backward()
        grad_norm = sum(
            param.grad.abs().sum().item()
            for param in model.parameters()
            if param.grad is not None
        )
        self.assertGreater(grad_norm, 0.0)

    def test_token_log_probs_shape(self):
        model = DummyCausalLM()
        input_ids = torch.tensor([[1, 2, 3, 4]])
        log_probs = token_log_probs(model, input_ids)
        self.assertEqual(tuple(log_probs.shape), (1, 3))


class SFTBatchTests(unittest.TestCase):
    def test_build_sft_batch_masks_prompt_labels_only(self):
        tokenizer = DummyTokenizer()
        prompt = "Human: hi\nAssistant:"
        response = "hello"
        batch = build_sft_batch(
            tokenizer,
            prompt=prompt,
            response=response,
            max_length=128,
            device="cpu",
        )

        prompt_len = tokenizer(prompt).input_ids.shape[1]
        response_ids = tokenizer(" " + response, add_special_tokens=False).input_ids[0]

        self.assertIsNotNone(batch)
        self.assertTrue(torch.equal(batch["labels"][0, :prompt_len], torch.full((prompt_len,), -100)))
        self.assertTrue(torch.equal(batch["labels"][0, prompt_len:], response_ids))
        self.assertEqual(batch["response_tokens"], response_ids.numel())


class AdvantageTests(unittest.TestCase):
    def test_compute_gae_matches_manual_last_step(self):
        rewards = torch.tensor([0.0, 1.0])
        values = torch.tensor([0.2, 0.4])
        advantages, returns = compute_gae(
            rewards,
            values,
            gamma=1.0,
            lam=1.0,
            last_value=0.0,
        )

        expected_advantages = torch.tensor([0.8, 0.6])
        expected_returns = torch.tensor([1.0, 1.0])
        self.assertTrue(torch.allclose(advantages, expected_advantages))
        self.assertTrue(torch.allclose(returns, expected_returns))

    def test_normalize_advantages_has_zero_mean(self):
        advantages = torch.tensor([1.0, 2.0, 3.0])
        normalized = normalize_advantages(advantages)
        self.assertTrue(torch.allclose(normalized.mean(), torch.tensor(0.0)))


class DPOGradientTests(unittest.TestCase):
    def test_dpo_loss_updates_policy_not_reference(self):
        tokenizer = DummyTokenizer()
        policy_model = DummyCausalLM()
        reference_model = DummyCausalLM()
        for param in reference_model.parameters():
            param.requires_grad = False

        example = {
            "prompt": "Human: hi\nAssistant:",
            "chosen": "helpful answer",
            "rejected": "bad",
        }
        config = SimpleNamespace(max_length=128, beta=0.1)

        loss = dpo_loss(policy_model, reference_model, tokenizer, example, config)
        loss.backward()

        policy_grad = sum(
            param.grad.abs().sum().item()
            for param in policy_model.parameters()
            if param.grad is not None
        )
        reference_grads = [param.grad for param in reference_model.parameters()]

        self.assertGreater(policy_grad, 0.0)
        self.assertTrue(all(grad is None for grad in reference_grads))


if __name__ == "__main__":
    unittest.main()
