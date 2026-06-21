import os

import torch
from torch import nn
from transformers import AutoModel

from models.load_model import DEVICE, get_dtype, resolve_model_path


class ValueModel(nn.Module):
    """Token-level critic V(s_t) for PPO."""

    def __init__(self, base_model, hidden_size):
        super().__init__()
        self.base_model = base_model
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        hidden = outputs.last_hidden_state
        values = self.value_head(hidden.float()).squeeze(-1)
        return values

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        self.base_model.save_pretrained(os.path.join(path, "base_model"))
        torch.save(self.value_head.state_dict(), os.path.join(path, "value_head.pt"))


def load_value_model(model_name_or_path=None):
    resolved_model = resolve_model_path(model_name_or_path)
    base_model = AutoModel.from_pretrained(resolved_model, torch_dtype=get_dtype())
    base_model.to(DEVICE)
    hidden_size = base_model.config.hidden_size
    value_model = ValueModel(base_model, hidden_size)
    value_model.to(DEVICE)
    return value_model
