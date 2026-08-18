"""From-scratch decoder-only transformer policy."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Real

import torch
from torch import nn
import torch.nn.functional as F


PastKeyValue = tuple[torch.Tensor, torch.Tensor]
PastKeyValues = tuple[PastKeyValue, ...]


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    max_context_length: int = 768
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    mlp_dim: int = 768
    rope_base: float = 10_000.0
    norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "max_context_length",
            "d_model",
            "n_layers",
            "n_heads",
            "mlp_dim",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        for name in ("rope_base", "norm_eps"):
            object.__setattr__(self, name, _positive_real(getattr(self, name), name))
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("attention head dimension must be even for rotary positions")


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * scale).to(x.dtype) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, context: int, base: float) -> None:
        super().__init__()
        inverse_frequency = 1.0 / (
            base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        positions = torch.arange(context).float()
        frequencies = torch.outer(positions, inverse_frequency)
        self.register_buffer("cos", frequencies.cos(), persistent=False)
        self.register_buffer("sin", frequencies.sin(), persistent=False)

    def rotate(self, x: torch.Tensor, start_position: int) -> torch.Tensor:
        length = x.shape[-2]
        stop_position = start_position + length
        if stop_position > self.cos.shape[0]:
            raise ValueError("rotary position exceeds configured context")
        cos = self.cos[start_position:stop_position].to(x.dtype)[None, None]
        sin = self.sin[start_position:stop_position].to(x.dtype)[None, None]
        even = x[..., 0::2]
        odd = x[..., 1::2]
        rotated = torch.stack(
            (even * cos - odd * sin, even * sin + odd * cos), dim=-1
        )
        return rotated.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = RotaryEmbedding(
            self.head_dim, config.max_context_length, config.rope_base
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past: PastKeyValue | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, PastKeyValue | None]:
        batch, length, dimension = x.shape
        query, key, value = self.qkv(x).chunk(3, dim=-1)
        query = query.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        past_length = 0 if past is None else past[0].shape[2]
        query = self.rope.rotate(query, past_length)
        key = self.rope.rotate(key, past_length)
        if past is not None:
            key = torch.cat((past[0], key), dim=2)
            value = torch.cat((past[1], value), dim=2)

        if past is None and attention_mask is None:
            attended = F.scaled_dot_product_attention(
                query, key, value, is_causal=True
            )
        else:
            total_length = past_length + length
            query_positions = past_length + torch.arange(length, device=x.device)
            key_positions = torch.arange(total_length, device=x.device)
            causal_mask = key_positions[None, :] <= query_positions[:, None]
            mask = causal_mask[None, None]
            if attention_mask is not None:
                mask = mask & attention_mask[:, None, None, :].bool()
            attended = F.scaled_dot_product_attention(
                query, key, value, attn_mask=mask, is_causal=False
            )
        attended = attended.transpose(1, 2).contiguous().view(batch, length, dimension)
        cache = (key, value) if use_cache else None
        return self.output(attended), cache


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.d_model, config.norm_eps)
        self.gate = nn.Linear(config.d_model, config.mlp_dim, bias=False)
        self.up = nn.Linear(config.d_model, config.mlp_dim, bias=False)
        self.down = nn.Linear(config.mlp_dim, config.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past: PastKeyValue | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, PastKeyValue | None]:
        attention, cache = self.attention(
            self.attention_norm(x),
            attention_mask=attention_mask,
            past=past,
            use_cache=use_cache,
        )
        x = x + attention
        normalized = self.mlp_norm(x)
        x = x + self.down(F.silu(self.gate(normalized)) * self.up(normalized))
        return x, cache


@dataclass(frozen=True)
class ModelOutput:
    logits: torch.Tensor
    past_key_values: PastKeyValues | None


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layers)
        )
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._initialize)
        self.lm_head.weight = self.embedding.weight

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: PastKeyValues | None = None,
        use_cache: bool = False,
    ) -> ModelOutput:
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape [batch, positive_length]")
        batch, length = input_ids.shape
        past_length = 0
        if past_key_values is not None:
            if len(past_key_values) != len(self.blocks):
                raise ValueError("past_key_values must contain one entry per layer")
            layer_lengths = {past[0].shape[2] for past in past_key_values}
            if len(layer_lengths) != 1:
                raise ValueError("cached layers disagree on prefix length")
            past_length = layer_lengths.pop()
            for key, value in past_key_values:
                expected = (
                    batch,
                    self.config.n_heads,
                    past_length,
                    self.config.d_model // self.config.n_heads,
                )
                if key.shape != expected or value.shape != expected:
                    raise ValueError("cached attention state has incompatible shape")
        total_length = past_length + length
        if total_length > self.config.max_context_length:
            raise ValueError("effective prefix exceeds max_context_length")
        if attention_mask is not None and attention_mask.shape != (batch, total_length):
            raise ValueError(
                "attention_mask must have shape [batch, effective_prefix_length]"
            )

        hidden = self.embedding(input_ids)
        caches: list[PastKeyValue] = []
        for index, block in enumerate(self.blocks):
            hidden, cache = block(
                hidden,
                attention_mask=attention_mask,
                past=None if past_key_values is None else past_key_values[index],
                use_cache=use_cache,
            )
            if cache is not None:
                caches.append(cache)
        logits = self.lm_head(self.final_norm(hidden))
        return ModelOutput(logits, tuple(caches) if use_cache else None)

    def forward_with_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score a full prefix and expose normalized states for a value head."""

        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape [batch, positive_length]")
        batch, length = input_ids.shape
        if length > self.config.max_context_length:
            raise ValueError("effective prefix exceeds max_context_length")
        if attention_mask is not None and attention_mask.shape != (batch, length):
            raise ValueError(
                "attention_mask must have shape [batch, effective_prefix_length]"
            )
        hidden = self.embedding(input_ids)
        for block in self.blocks:
            hidden, _ = block(hidden, attention_mask=attention_mask)
        hidden = self.final_norm(hidden)
        return self.lm_head(hidden), hidden


@torch.no_grad()
def generate_tokens(
    model: Transformer,
    prompts: Sequence[Sequence[int]],
    *,
    max_new_tokens: int,
    batch_size: int | None,
    pad_token: int,
    eos_token: int,
    device: torch.device,
    select_next: Callable[[torch.Tensor], torch.Tensor],
    autocast_dtype: torch.dtype | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Generate unconstrained continuations with shared batching and KV caching."""

    if not prompts:
        return ()
    maximum_prompt_length = max(len(prompt) for prompt in prompts)
    if maximum_prompt_length + max_new_tokens > model.config.max_context_length:
        raise ValueError("maximum prompt and completion exceed the model context limit")

    was_training = model.training
    model.eval()
    size = batch_size or len(prompts)
    generated: list[tuple[int, ...]] = []
    try:
        for start in range(0, len(prompts), size):
            batch = prompts[start : start + size]
            prompt_length = max(len(prompt) for prompt in batch)
            input_ids = torch.full(
                (len(batch), prompt_length), pad_token, dtype=torch.long, device=device
            )
            attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for row, prompt in enumerate(batch):
                length = len(prompt)
                input_ids[row, -length:] = torch.tensor(prompt, device=device)
                attention_mask[row, -length:] = True

            completions: list[list[int]] = [[] for _ in batch]
            done = torch.zeros(len(batch), dtype=torch.bool, device=device)
            current = input_ids
            cache = None
            for _ in range(max_new_tokens):
                with torch.autocast(
                    device_type=device.type,
                    dtype=autocast_dtype,
                    enabled=autocast_dtype is not None,
                ):
                    output = model(
                        current,
                        attention_mask=attention_mask,
                        past_key_values=cache,
                        use_cache=True,
                    )
                selected = select_next(output.logits[:, -1])
                selected = torch.where(done, torch.full_like(selected, pad_token), selected)
                active = ~done
                for row in active.nonzero(as_tuple=False).flatten().tolist():
                    completions[row].append(int(selected[row].item()))
                done |= selected.eq(eos_token)
                if bool(done.all()):
                    break
                current = selected[:, None]
                attention_mask = torch.cat((attention_mask, active[:, None]), dim=1)
                cache = output.past_key_values

            generated.extend(map(tuple, completions))
    finally:
        model.train(was_training)
    return tuple(generated)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
