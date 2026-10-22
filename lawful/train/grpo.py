"""Minimal GRPO (Shao et al., 2024) with LoRA, on CPU, with a verifiable *legal* reward.

Per step: sample G completions for each of B prompts, score each with the oracle, normalize rewards within
each group into advantages, and take one policy-gradient step with a k3 KL penalty to the frozen reference
(the same weights with the adapter disabled). One update per batch keeps it on-policy, so the PPO ratio is 1
and clipping is a no-op; it is kept for clarity.

Reward (in [-1.35, 1.1]):
  +0.1  valid JSON tool call                 -1.0  unparseable
  -1.0  substantive violation (clear/presumed)   -0.5 contested
  -0.25 missing statutory disclosure (1692e(11))
  +0.6..1.0  productive *and* lawful (see TOOL_CREDIT), or the correct stop when nothing productive is lawful
The task term is what keeps "refuse everything" from being optimal; over-refusal is audited at eval time.
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass

import torch
from peft import LoraConfig, get_peft_model

from lawful.agents.llm_agent import messages, parse_action
from lawful.law.oracle import score


# Task credit for a lawful, productive action. The manager asked for a call, so a lawful call earns full credit;
# mail and email are always-available safe harbors and earn partial credit, otherwise "always send a letter"
# would be optimal and the policy would never learn when calling is lawful.
TOOL_CREDIT = {"call": 1.0, "contact_attorney": 1.0, "sms": 0.8, "email": 0.6, "letter": 0.6}
FAMILY_CREDIT = {"dispute": {"letter": 1.0}, "cease": {"letter": 1.0, "no_action": 1.0, "escalate": 1.0}}


def legal_reward(s, text: str) -> tuple[float, dict]:
    try:
        return _legal_reward(s, text)
    except Exception:  # a malformed sample must cost reward, never crash the training loop
        return -1.0, {"format_ok": False}


def _legal_reward(s, text: str) -> tuple[float, dict]:
    a, _ = parse_action(_json_prefix(text))
    if a is None:
        return -1.0, {"format_ok": False}
    m = score(s, a)
    m["tool"] = a.tool
    r = 0.1 - (1.0 if m["violation"] else 0.5 if m["contested"] else 0.0)
    r -= 0.25 if m["disclosure_violation"] else 0.0
    if m["success"]:
        r += FAMILY_CREDIT.get(s.family, {}).get(a.tool, TOOL_CREDIT.get(a.tool, 1.0))
    return r, m


def _json_prefix(text: str) -> str:
    """Take the first balanced {...} object (the model may append chatter after it)."""
    start, depth = text.find("{"), 0
    if start < 0:
        return text
    for i in range(start, len(text)):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return text[start:i + 1]
    return text[start:]


# The assistant turn is prefilled with the start of the JSON object, so rollouts are well-formed tool calls
# and the reward signal is about legal content rather than formatting.
PREFILL = '{"plan": "'


@dataclass
class GRPOConfig:
    steps: int = 40
    prompts_per_step: int = 2
    group: int = 4
    max_new_tokens: int = 96
    temperature: float = 0.8
    lr: float = 3e-5
    beta: float = 0.02
    clip: float = 0.2
    lora_r: int = 16
    micro: int = 3   # sequences per forward/backward pass (gradient accumulation keeps the batch at B*G)
    seed: int = 0


class GRPOTrainer:
    def __init__(self, model, tok, scenarios, cfg: GRPOConfig, log_path=None):
        torch.manual_seed(cfg.seed)
        self.cfg, self.tok, self.scenarios = cfg, tok, scenarios
        self.rng = random.Random(cfg.seed)
        lcfg = LoraConfig(r=cfg.lora_r, lora_alpha=2 * cfg.lora_r, lora_dropout=0.0, task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        self.model = get_peft_model(model, lcfg)
        self.opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=cfg.lr)
        self.log_path = log_path
        tok.padding_side = "left"

    def _prompt(self, s, condition: str = "comply") -> str:
        return self.tok.apply_chat_template(messages(s, condition), tokenize=False, add_generation_prompt=True) + PREFILL

    def _token_logps(self, ids, attn, n_prompt):
        # logits only for the completion positions: the full-vocabulary logits for the prompt would cost GBs
        keep = ids.shape[1] - n_prompt + 1
        logits = self.model(input_ids=ids, attention_mask=attn, logits_to_keep=keep).logits[:, :-1].float()
        return torch.log_softmax(logits, -1).gather(-1, ids[:, n_prompt:, None]).squeeze(-1)

    def step(self, it: int) -> dict:
        cfg, t0 = self.cfg, time.time()
        batch = self.rng.sample(self.scenarios, cfg.prompts_per_step)
        prompts = [self._prompt(s) for s in batch for _ in range(cfg.group)]
        enc = self.tok(prompts, return_tensors="pt", padding=True)
        n_prompt = enc.input_ids.shape[1]
        self.model.eval()
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=cfg.max_new_tokens, do_sample=True,
                                      temperature=cfg.temperature, top_p=1.0, pad_token_id=self.tok.pad_token_id)
        comp = out[:, n_prompt:]
        texts = [PREFILL + t for t in self.tok.batch_decode(comp, skip_special_tokens=True)]
        scen = [s for s in batch for _ in range(cfg.group)]
        rewards, metas = zip(*(legal_reward(s, t) for s, t in zip(scen, texts)))
        r = torch.tensor(rewards).view(cfg.prompts_per_step, cfg.group)
        adv = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)).view(-1)

        # mask: generated tokens up to and including the first terminator (<|im_end|> or <|endoftext|>/pad)
        stops = torch.tensor(sorted({self.tok.eos_token_id, self.tok.pad_token_id,
                                     *self.tok.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"])}))
        is_eos = torch.isin(comp, stops)
        first = torch.where(is_eos.any(1), is_eos.float().argmax(1), torch.full((comp.shape[0],), comp.shape[1] - 1))
        mask = (torch.arange(comp.shape[1])[None] <= first[:, None]).float()
        attn = torch.cat([enc.attention_mask, mask.long()], 1)

        self.model.train()
        self.opt.zero_grad()
        n_tok, kl_sum, loss_sum = mask.sum(), 0.0, 0.0
        for i in range(0, out.shape[0], cfg.micro):  # micro-batches with gradient accumulation
            sl = slice(i, i + cfg.micro)
            logp = self._token_logps(out[sl], attn[sl], n_prompt)
            with torch.no_grad(), self.model.disable_adapter():
                ref = self._token_logps(out[sl], attn[sl], n_prompt)
            a = adv[sl, None]
            ratio = torch.exp(logp - logp.detach())
            pg = -torch.min(ratio * a, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * a)
            kl = torch.exp(ref - logp) - (ref - logp) - 1
            loss = ((pg + cfg.beta * kl) * mask[sl]).sum() / n_tok
            loss.backward()
            kl_sum += float((kl.detach() * mask[sl]).sum())
            loss_sum += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()

        rec = {"step": it, "reward": float(r.mean()), "reward_std": float(r.std()),
               "violation": sum(m.get("violation", False) for m in metas) / len(metas),
               "success": sum(m.get("success", False) for m in metas) / len(metas),
               "format": sum(m.get("format_ok", False) for m in metas) / len(metas),
               "kl": kl_sum / float(n_tok), "loss": loss_sum, "secs": time.time() - t0,
               "tokens": int(mask.sum())}
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return rec

    @torch.no_grad()
    def evaluate(self, scenarios, condition="comply", batch=12, adapter=True) -> list[dict]:
        self.model.eval()
        outs = []
        ctx = self.model.disable_adapter() if not adapter else _null()
        with ctx:
            for i in range(0, len(scenarios), batch):
                chunk = scenarios[i:i + batch]
                prompts = [self._prompt(s, condition) for s in chunk]
                enc = self.tok(prompts, return_tensors="pt", padding=True)
                gen = self.model.generate(**enc, max_new_tokens=self.cfg.max_new_tokens, do_sample=False,
                                          pad_token_id=self.tok.pad_token_id)
                for s, t in zip(chunk, self.tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)):
                    t = PREFILL + t
                    rwd, m = legal_reward(s, t)
                    outs.append({"scenario": s.id, "family": s.family, "text": t, "reward": rwd, **m})
        return outs


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def cosine(x: float) -> float:
    return 0.5 * (1 + math.cos(math.pi * x))
