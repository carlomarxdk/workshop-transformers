"""The model: a small BERT adapted for life-event sequences, following life2vec.

Shared by scripts/pretrain.py and notebooks b2-b4, so that training and inference build the
input embeddings the *same* way. They must: the model is trained on

    token embedding + absolute position + Time2Vec(age) + Time2Vec(days before index)

and feeding it token embeddings alone at inference time produces confident nonsense.

Four things here differ from a stock `BertForMaskedLM`, and each is optional so they can be
ablated one at a time:

    Time2Vec        two continuous clocks, because rank order throws away the gaps
    Performer       linear-time factorised attention (FAVOR+), as in life2vec
    ReZero          scalar-gated residuals instead of LayerNorm, as in life2vec
    tied embeddings input and output embedding matrices share weights (BERT's default)
"""
from __future__ import annotations

import math
import pathlib

import torch
from torch import nn
from transformers import AutoModelForMaskedLM, BertConfig


# --------------------------------------------------------------------------- Time2Vec
class Time2Vec(nn.Module):
    """Kazemi et al. 2019 (arXiv:1907.05321).

    One linear term plus a bank of sinusoids at learned frequencies:

        t2v(t)[0] = w_0 * t + b_0                     trend
        t2v(t)[i] = sin(w_i * t + b_i)    for i > 0   periodicity, learned frequencies

    The periodic terms let the model represent recurring rhythms -- an annual check-up --
    without anyone specifying the period in advance.
    """

    def __init__(self, out_dim: int, scale: float = 0.05):
        super().__init__()
        self.w = nn.Parameter(torch.randn(out_dim) * scale)
        self.b = nn.Parameter(torch.zeros(out_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:   # (B, L) -> (B, L, out_dim)
        x = t.unsqueeze(-1) * self.w + self.b
        return torch.cat([x[..., :1], torch.sin(x[..., 1:])], dim=-1)


# --------------------------------------------------------------------------- Performer
class PerformerSelfAttention(nn.Module):
    """FAVOR+ attention (Choromanski et al. 2021, arXiv:2009.14794).

    Softmax attention costs O(L^2): every position attends to every other, so the score
    matrix is L x L. FAVOR+ replaces the softmax kernel with a random feature map phi such
    that phi(q) . phi(k) approximates exp(q . k), which lets the product be re-associated:

        softmax(QK^T) V  ~=  phi(Q) (phi(K)^T V)        O(L * m * d) instead of O(L^2 * d)

    We never build the L x L matrix. At L=128 this is not a speed win -- it is here because
    life2vec uses it, and because it is the cleanest example of why architectural choices in
    this literature are usually about sequence length.
    """

    def __init__(self, config: BertConfig, num_features: int = 64):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_features = num_features

        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # Fixed orthogonal random projections. A buffer, not a parameter: FAVOR+ works
        # because these are random, not because they are learned.
        self.register_buffer("projection", self._orthogonal(num_features, self.head_dim))

    @staticmethod
    def _orthogonal(rows: int, cols: int) -> torch.Tensor:
        blocks = []
        for _ in range(math.ceil(rows / cols)):
            q, _ = torch.linalg.qr(torch.randn(cols, cols))
            blocks.append(q.T)
        return torch.cat(blocks, dim=0)[:rows] * math.sqrt(cols)

    def _features(self, x: torch.Tensor, is_query: bool) -> torch.Tensor:
        """Positive random features: phi(x) = exp(w.x - |x|^2/2) / sqrt(m)."""
        x = x / (self.head_dim ** 0.25)
        projected = torch.einsum("bhld,md->bhlm", x, self.projection)
        norm = (x ** 2).sum(-1, keepdim=True) / 2
        # Subtracting a max keeps the exponential from overflowing.
        stabiliser = projected.max(dim=-1, keepdim=True).values if is_query \
            else projected.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values
        return torch.exp(projected - norm - stabiliser) / math.sqrt(self.num_features)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        return x.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        q = self._features(self._split(self.query(hidden_states)), is_query=True)
        k = self._features(self._split(self.key(hidden_states)), is_query=False)
        v = self._split(self.value(hidden_states))

        if attention_mask is not None:
            # BERT hands us an additive mask: 0 to keep, a large negative to drop.
            keep = (attention_mask.reshape(attention_mask.shape[0], -1,
                                           attention_mask.shape[-1])[:, -1] == 0)
            k = k * keep[:, None, :, None].to(k.dtype)
            v = v * keep[:, None, :, None].to(v.dtype)

        # Note: dropout lands on the key features, not on attention weights -- FAVOR+ never
        # forms the weights, so there is nothing else to drop. Not what
        # `attention_probs_dropout_prob` implies; read it as feature dropout here.
        k = self.dropout(k)
        kv = torch.einsum("bhlm,bhld->bhmd", k, v)          # (B, H, m, d)
        normaliser = torch.einsum("bhlm,bhm->bhl", q, k.sum(dim=2))
        context = torch.einsum("bhlm,bhmd->bhld", q, kv)
        context = context / normaliser.unsqueeze(-1).clamp(min=1e-6)

        b, h, l, d = context.shape
        # (output, attention_weights). FAVOR+ never materialises an L x L score matrix,
        # so there are no attention weights to return -- which is the trade-off.
        return context.transpose(1, 2).reshape(b, l, h * d), None


# --------------------------------------------------------------------------- ReZero
class ReZeroResidual(nn.Module):
    """Bachlechner et al. 2020 (arXiv:2003.04887).

    A transformer sublayer normally does  LayerNorm(x + F(x)).  ReZero does

        x + alpha * F(x),      alpha initialised to 0

    so at initialisation every sublayer is the identity and the signal passes through the
    stack untouched. The network then *learns* how much of each sublayer it wants. It makes
    deep stacks trainable without warmup and drops LayerNorm entirely.
    """

    def __init__(self, dense: nn.Linear, dropout: nn.Dropout):
        super().__init__()
        self.dense = dense
        self.dropout = dropout
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states, input_tensor):
        return input_tensor + self.alpha * self.dropout(self.dense(hidden_states))


def apply_rezero(bert) -> None:
    """Swap every LayerNorm-based residual in the encoder for a ReZero one."""
    for layer in bert.bert.encoder.layer:
        layer.attention.output = ReZeroResidual(layer.attention.output.dense,
                                                layer.attention.output.dropout)
        layer.output = ReZeroResidual(layer.output.dense, layer.output.dropout)


def apply_performer(bert, config: BertConfig, num_features: int = 64) -> None:
    for layer in bert.bert.encoder.layer:
        layer.attention.self = PerformerSelfAttention(config, num_features)


# --------------------------------------------------------------------------- the model
class EventBertForMaskedLM(nn.Module):
    """BERT for masked-event modelling, with explicit encodings of *when*.

    Every position is the sum of four things:

        token embedding          what happened
      + absolute position        where in the sequence (BERT's own learned embedding)
      + Time2Vec(age)            when in the person's life        -- the "life clock"
      + Time2Vec(days to index)  how long before the cutoff       -- the "recency clock"

    Two clocks because they answer different questions. Age says "this happened at 52",
    which is what a disease model cares about. Days-before-index says "this happened three
    weeks ago", which is what a *prediction* cares about. Absolute position alone gives
    only rank order, which cannot tell three days from three years.
    """

    def __init__(self, config: BertConfig, attention: str = "sdpa", rezero: bool = False,
                 performer_features: int = 64):
        super().__init__()
        self.config = config
        self.attention = attention
        self.rezero = rezero
        self.performer_features = performer_features
        # "performer" replaces the attention module wholesale, so build the eager one first.
        config._attn_implementation = "eager" if attention == "performer" else attention
        self.bert = AutoModelForMaskedLM.from_config(config)

        if attention == "performer":
            apply_performer(self.bert, config, performer_features)
        if rezero:
            apply_rezero(self.bert)

        self.time2vec_age = Time2Vec(config.hidden_size)
        self.time2vec_days = Time2Vec(config.hidden_size)
        self.time_norm = nn.LayerNorm(config.hidden_size)

    def embed(self, input_ids, ages=None, days=None):
        embeddings = self.bert.get_input_embeddings()(input_ids)
        clocks = torch.zeros_like(embeddings)
        if ages is not None:
            clocks = clocks + self.time2vec_age(ages)
        if days is not None:
            # Years, not days, so both clocks live on a comparable scale.
            clocks = clocks + self.time2vec_days(days / 365.25)
        return embeddings + self.time_norm(clocks)

    def forward(self, input_ids, attention_mask=None, ages=None, days=None,
                labels=None, **kwargs):
        # BERT adds its own absolute position embeddings to inputs_embeds internally.
        return self.bert(inputs_embeds=self.embed(input_ids, ages, days),
                         attention_mask=attention_mask, labels=labels, **kwargs)

    # The MLM decoder shares storage with the input embeddings (weight) and with
    # cls.predictions.bias (bias). Those entries are DERIVED, not independent parameters.
    # Leaving them in the state dict makes `safetensors` refuse to write the checkpoint --
    # and since this class is a plain nn.Module rather than a PreTrainedModel, Trainer
    # takes exactly that path and training dies at the first checkpoint. Dropping them is
    # the correct fix: tying restores both on load.
    _TIED_KEYS = ("cls.predictions.decoder.weight", "cls.predictions.decoder.bias")
    # Trainer consults this when writing a checkpoint. It normally comes from
    # PreTrainedModel, which we are not.
    _keys_to_ignore_on_save = None

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        for key in [k for k in state if k.endswith(self._TIED_KEYS)]:
            del state[key]
        return state

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        # strict=False because the tied keys above are deliberately absent.
        return super().load_state_dict(state_dict, strict=False, assign=assign)

    def embeddings_tied(self) -> bool:
        return (self.bert.get_input_embeddings().weight.data_ptr()
                == self.bert.get_output_embeddings().weight.data_ptr())

    # -- persistence -------------------------------------------------------------------
    def save_pretrained(self, path) -> None:
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.bert.save_pretrained(path)
        torch.save({"time2vec_age": self.time2vec_age.state_dict(),
                    "time2vec_days": self.time2vec_days.state_dict(),
                    "time_norm": self.time_norm.state_dict(),
                    "attention": self.attention,
                    "rezero": self.rezero,
                    "performer_features": getattr(self, "performer_features", 64)},
                   path / "time_encoding.pt")

    @classmethod
    def from_pretrained(cls, path, expected_vocab_size: int | None = None) -> "EventBertForMaskedLM":
        """Rebuild the exact architecture first, *then* load weights into it.

        Order matters. ReZero replaces the LayerNorm residual modules and Performer
        replaces the attention modules, so a checkpoint of either does not fit a stock
        BERT. Loading first and patching afterwards silently discards the trained
        `alpha` values and leaves freshly initialised LayerNorms in their place.
        """
        from safetensors.torch import load_file

        path = pathlib.Path(path)
        extra = torch.load(path / "time_encoding.pt", map_location="cpu", weights_only=False)
        config = BertConfig.from_pretrained(path)

        # A stale checkpoint is the nastiest failure here: change MIN_FREQ or INCOME_BINS
        # and every token id shifts, so an old model loads perfectly and then silently
        # interprets every token as something else. Pass the current vocabulary size.
        if expected_vocab_size is not None and config.vocab_size != expected_vocab_size:
            raise RuntimeError(
                f"stale checkpoint: {path} was trained with a {config.vocab_size}-token "
                f"vocabulary but the current one has {expected_vocab_size}. Token ids have "
                f"shifted, so this model's predictions would be meaningless. Rebuild with "
                f"scripts/build_tokenizer.py and scripts/pretrain.py.")

        model = cls(config,
                    attention=extra.get("attention", "sdpa"),
                    rezero=extra.get("rezero", False),
                    performer_features=extra.get("performer_features", 64))

        weights = path / "model.safetensors"
        state = (load_file(weights) if weights.exists()
                 else torch.load(path / "pytorch_model.bin", map_location="cpu"))
        missing, unexpected = model.bert.load_state_dict(state, strict=False)
        # The MLM decoder's weight is tied to the input embeddings and its bias to
        # cls.predictions.bias, so both legitimately appear "missing" from the checkpoint.
        # Anything else is a real architecture mismatch and must not pass silently.
        real_missing = [k for k in missing if "cls.predictions.decoder." not in k]
        if real_missing or unexpected:
            raise RuntimeError(
                f"checkpoint does not match the architecture\n"
                f"  missing:    {real_missing}\n  unexpected: {unexpected}")

        model.time2vec_age.load_state_dict(extra["time2vec_age"])
        model.time2vec_days.load_state_dict(extra["time2vec_days"])
        model.time_norm.load_state_dict(extra["time_norm"])
        return model.eval()


def pick_device(requested: str = "auto") -> torch.device:
    """MPS on Apple Silicon, CUDA where available, CPU otherwise."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
