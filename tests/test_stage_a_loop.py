"""CPU-side tests for the Stage A training loop wiring.

We can't load MiniCPM-o on CPU (it's huge + slow), but we *can* test:
  - The dataset+dataloader+collate combination feeds (frame, action, mask) correctly
  - The cross-entropy loss + argmax accuracy compute correctly given mock logits
  - The action-mapping inverse round-trips
  - Trainable parameter set excludes the frozen base when using a mock FastModel
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.imitation_data import (
    GAME_ACTION_TO_GLOBAL,
    GLOBAL_ACTION_TO_GAME,
    N_GLOBAL_ACTIONS,
    legal_action_mask,
    global_to_local_action,
)


def test_global_to_local_inverse_round_trip():
    for game, g2l in GAME_ACTION_TO_GLOBAL.items():
        for local_idx, global_idx in enumerate(g2l):
            assert global_to_local_action(game, global_idx) == local_idx


def test_global_to_local_rejects_illegal():
    # MsPacman has no FIRE
    with pytest.raises(ValueError):
        global_to_local_action("MsPacman", 1)
    # Pong has no UP (global 2)
    with pytest.raises(ValueError):
        global_to_local_action("Pong", 2)


def test_legal_mask_consistent_with_global_to_local_table():
    """Every legal global action must have a valid local index, and vice versa."""
    for game in GAME_ACTION_TO_GLOBAL:
        mask = legal_action_mask(game)
        legal_globals = mask.nonzero(as_tuple=True)[0].tolist()
        for g in legal_globals:
            # global_to_local must succeed for legal actions
            local = global_to_local_action(game, g)
            assert local >= 0
        # Conversely: every illegal index in the inverse table should be masked False
        inv = GLOBAL_ACTION_TO_GAME[game]
        for g, l in enumerate(inv):
            if l < 0:
                assert not mask[g].item(), f"{game} global {g} is illegal but mask says legal"


def test_cross_entropy_loss_with_legal_mask():
    """Verify the loss computation matches the training loop's expectations."""
    legal = legal_action_mask("MsPacman")  # 9 legal of 18
    # Construct logits where the *correct* global action has the highest unmasked logit.
    logits = torch.full((1, N_GLOBAL_ACTIONS), -1.0)
    correct = 2  # UP
    logits[0, correct] = 5.0
    # Mask out illegal actions
    logits = logits.masked_fill(~legal.unsqueeze(0), float("-inf"))
    loss = F.cross_entropy(logits.float(), torch.tensor([correct]))
    assert loss.item() < 0.2, "loss should be small when correct action has dominant logit"
    pred = logits.argmax(dim=-1).item()
    assert pred == correct


def test_cross_entropy_with_no_signal_gives_log_n():
    """Zero-init logits → uniform over legal → loss ≈ log(9) for MsPacman."""
    import math
    legal = legal_action_mask("MsPacman")
    logits = torch.zeros(1, N_GLOBAL_ACTIONS)
    logits = logits.masked_fill(~legal.unsqueeze(0), float("-inf"))
    loss = F.cross_entropy(logits.float(), torch.tensor([0]))  # NOOP is legal
    assert abs(loss.item() - math.log(9)) < 0.01


class _MockFastModel(nn.Module):
    """A tiny stand-in for FastModel that satisfies the training loop's interface."""
    def __init__(self, n_actions: int = N_GLOBAL_ACTIONS):
        super().__init__()
        self.action_head = nn.Linear(8, n_actions, bias=False)
        nn.init.zeros_(self.action_head.weight)
        self.xattn_layers = nn.ModuleDict({
            "12": nn.Linear(8, 8, bias=False),  # surrogate
            "24": nn.Linear(8, 8, bias=False),
        })
        # Frozen "base"
        self.base = nn.Linear(8, 8, bias=False)
        for p in self.base.parameters():
            p.requires_grad = False

    def trainable_parameters(self):
        for p in self.action_head.parameters():
            yield p
        for p in self.xattn_layers.parameters():
            yield p


def test_trainable_parameters_excludes_base():
    fm = _MockFastModel()
    trainable = list(fm.trainable_parameters())
    n_train = sum(p.numel() for p in trainable)
    n_base = sum(p.numel() for p in fm.base.parameters())
    assert n_base > 0
    # No base param should appear in the trainable set
    base_ids = {id(p) for p in fm.base.parameters()}
    trainable_ids = {id(p) for p in trainable}
    assert base_ids.isdisjoint(trainable_ids)


def test_optimizer_only_steps_trainable_params():
    fm = _MockFastModel()
    base_before = fm.base.weight.detach().clone()
    head_before = fm.action_head.weight.detach().clone()

    optimizer = torch.optim.AdamW(list(fm.trainable_parameters()), lr=1e-2)
    # Synthetic forward / backward
    x = torch.randn(2, 8)
    logits = fm.action_head(x)
    fake_target = torch.tensor([0, 1], dtype=torch.long)
    loss = F.cross_entropy(logits, fake_target)
    loss.backward()
    optimizer.step()

    # Base unchanged, head changed
    assert torch.allclose(fm.base.weight, base_before)
    assert not torch.allclose(fm.action_head.weight, head_before)
