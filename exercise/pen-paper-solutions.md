# Pen-and-Paper Exercise: Solutions

Worked answers for `exercise/pen-paper.md`. Input: **"My model overfit again"**, positions 1-4.

Every step shows the answer, then one entry worked out in full so you can check your method
rather than just your arithmetic.

> **On the position table.** The exercise lists **five** positional embeddings but the sentence
> has **four** tokens, so position 5 is unused. "my" is position 1, "again" is position 4. (The
> tokenization slide shows `[CLS]` at position 0; the exercise drops `[CLS]`, and its position-1
> row lines up with the slide's position 1.)
>
> **On rounding.** Steps 1-3.1 are exact integers. From step 3.2 onward the numbers are
> irrational, and everything below is rounded to 3 decimals. If you kept attention weights to
> **2** decimals your final outputs will differ by up to **0.07**; to **3** decimals, by up to
> **0.01**. Neither is a mistake.

---

## Step 1: Embeddings

`X(t) = token embedding + positional embedding`

| token | position | token emb. | positional emb. | **X** |
| --- | --- | --- | --- | --- |
| my | 1 | (1, 0, 1) | (0, 0, 0) | **(1, 0, 1)** |
| model | 2 | (0, 1, 1) | (1, 0, -1) | **(1, 1, 0)** |
| overfit | 3 | (1, 1, 0) | (0, 1, 1) | **(1, 2, 1)** |
| again | 4 | (2, -1, 1) | (-1, 1, 1) | **(1, 0, 2)** |

```
X(my)      = (1, 0, 1)
X(model)   = (1, 1, 0)
X(overfit) = (1, 2, 1)
X(again)   = (1, 0, 2)
```

*Worked:* `X(overfit) = (1,1,0) + (0,1,1) = (1,2,1)`.

**Notice.** `X(model) = (1,1,0)`, which is also the *token* embedding of "overfit". Position
has already moved "model" onto a different point than the one its own token embedding names.
That is the whole mechanism: identical tokens in different slots become different vectors.

---

## Step 2: Q, K, V

Row vector on the left: `Q(t) = X(t) W_Q`, and likewise for K and V.

| token | X | **Q** | **K** | **V** |
| --- | --- | --- | --- | --- |
| my | (1, 0, 1) | **(2, 0, 0)** | **(1, 2, 1)** | **(2, 1, 1)** |
| model | (1, 1, 0) | **(1, 1, 1)** | **(1, 1, 0)** | **(1, 2, 1)** |
| overfit | (1, 2, 1) | **(2, 2, 0)** | **(3, 2, -1)** | **(2, 3, 3)** |
| again | (1, 0, 2) | **(3, 0, -1)** | **(2, 3, 1)** | **(3, 1, 2)** |

*Worked:* `Q(my) = (1,0,1) @ W_Q`. First component `1(1) + 0(0) + 1(1) = 2`;
second `1(0) + 0(1) + 1(0) = 0`; third `1(1) + 0(0) + 1(-1) = 0`. So `Q(my) = (2,0,0)`.

---

## Step 3.1: Attention scores

**Raw dot products** `Q(i) · K(j)`: all integers, so check these before dividing:

| query \ key | my | model | overfit | again |
| --- | --- | --- | --- | --- |
| **my** | 2 | 2 | 6 | 4 |
| **model** | 4 | 2 | 4 | 6 |
| **overfit** | 6 | 4 | 10 | 10 |
| **again** | 2 | 3 | 10 | 5 |

**Scaled** `S(i,j) = raw / sqrt(3)`, with `sqrt(3) = 1.732`:

| query \ key | my | model | overfit | again |
| --- | --- | --- | --- | --- |
| **my** | 1.155 | 1.155 | 3.464 | 2.309 |
| **model** | 2.309 | 1.155 | 2.309 | 3.464 |
| **overfit** | 3.464 | 2.309 | 5.774 | 5.774 |
| **again** | 1.155 | 1.732 | 5.774 | 2.887 |

*Worked:* `Q(my) · K(overfit) = (2,0,0) · (3,2,-1) = 6 + 0 + 0 = 6`, then `6 / 1.732 = 3.464`.

---

## Step 3.2: Softmax, row by row

Each row sums to 1. This is the attention matrix **A**.

| query \ key | my | model | overfit | again |
| --- | --- | --- | --- | --- |
| **my** | 0.066 | 0.066 | 0.661 | 0.208 |
| **model** | 0.182 | 0.057 | 0.182 | 0.578 |
| **overfit** | 0.047 | 0.015 | 0.469 | 0.469 |
| **again** | 0.009 | 0.016 | 0.923 | 0.051 |

*Worked (row "my"):* exponentiate `1.155, 1.155, 3.464, 2.309` to get
`3.174, 3.174, 31.94, 10.06`, which sum to `48.35`. Divide each by the sum:
`0.066, 0.066, 0.661, 0.208`.

**Notice three things.**

1. **The matrix is not symmetric.** "again" gives 0.923 of its attention to "overfit", but
   "overfit" gives only 0.469 back. Attention is directional: Q and K are different projections.
2. **"overfit" splits evenly between itself and "again"** (0.469 / 0.469). Its query happens
   to align equally with both keys; nothing forces a token to attend to itself.
3. **Every row is peaked on "overfit" or "again".** The two right-hand columns dominate the
   matrix, so the context vectors in step 4 will all look somewhat alike. With one head, three
   dimensions and no training, that is expected: real BERT has 12 heads precisely so different
   heads can peak in different places.

---

## Step 4: Context vectors

`C(i) = sum_j A(i,j) V(j)`

```
C(my)      = (2.143, 2.387, 2.529)
C(model)   = (2.521, 1.422, 1.943)
C(overfit) = (2.455, 1.953, 2.408)
C(again)   = (2.035, 2.863, 2.898)
```

*Worked (first component of `C(my)`):*
`0.066(2) + 0.066(1) + 0.661(2) + 0.208(3) = 0.132 + 0.066 + 1.322 + 0.624 = 2.144`.
The table says `2.143`: the weights are really `0.0656, 0.0656, 0.6606, 0.2082`, and rounding
them to three decimals before multiplying costs that last digit. Expected, not an error.

---

## Step 5: Residual connection

`Z(i) = X(i) + C(i)`

| token | X | C | **Z** |
| --- | --- | --- | --- |
| my | (1, 0, 1) | (2.143, 2.387, 2.529) | **(3.143, 2.387, 3.529)** |
| model | (1, 1, 0) | (2.521, 1.422, 1.943) | **(3.521, 2.422, 1.943)** |
| overfit | (1, 2, 1) | (2.455, 1.953, 2.408) | **(3.455, 3.953, 3.408)** |
| again | (1, 0, 2) | (2.035, 2.863, 2.898) | **(3.035, 2.863, 4.898)** |

**Notice.** All four `C` vectors are positive and land in the same region: every component
between 1.4 and 2.9, and the first component within 2.04-2.52 across all four tokens. Attention
has pulled the tokens towards each other: the mean distance between pairs is **1.72 for `X` but
only 0.99 for `C`**. The residual is what undoes that. `Z` still carries `X`, so the mean pairwise
distance comes back to **1.96** and "my" stays distinguishable from "again". Stack a few
attention layers with no residual and the tokens converge on one point.

---

## Step 6: Position-wise feed-forward

`F1(i) = ReLU(Z(i) W_f1 + b1)`, then `F2(i) = F1(i) W_f2 + b2`. Same weights for every token.

| token | Z W_f1 + b1 (before ReLU) | **F1** (after ReLU) | **F2** |
| --- | --- | --- | --- |
| my | (6.672, -1.143, 0.244, 2.143) | **(6.672, 0.000, 0.244, 2.143)** | **(8.570, 0.244, 4.774)** |
| model | (5.463, 0.479, -0.099, 2.521) | **(5.463, 0.479, 0.000, 2.521)** | **(7.984, 0.479, 2.943)** |
| overfit | (6.863, 0.545, 1.499, 2.455) | **(6.863, 0.545, 1.499, 2.455)** | **(7.819, 2.044, 5.907)** |
| again | (7.933, -2.035, 0.827, 2.035) | **(7.933, 0.000, 0.827, 2.035)** | **(9.141, 0.827, 6.725)** |

```
F_2(my)      = (8.570, 0.244, 4.774)
F_2(model)   = (7.984, 0.479, 2.943)
F_2(overfit) = (7.819, 2.044, 5.907)
F_2(again)   = (9.141, 0.827, 6.725)
```

*Worked (`my`, second component before ReLU):* `Z(my) = (3.143, 2.387, 3.529)`, and column 2
of `W_f1` is `(0, 1, -1)`, with `b1[1] = 0`. So `3.143(0) + 2.387(1) + 3.529(-1) + 0 = -1.143`.
ReLU sends it to **0**.

**Notice.** ReLU zeroed a coordinate for "my" and for "again", a different one for "model",
and none for "overfit". Identical weights, different gates: this layer is where a token's own
dimensions interact, and it is the only nonlinearity in the block.

---

## Step 7: Second residual, and the output

`Output(i) = F2(i) + Z(i)`

```
Output(my)      = (11.713, 2.631, 8.303)
Output(model)   = (11.505, 2.901, 4.885)
Output(overfit) = (11.273, 5.997, 9.315)
Output(again)   = (12.176, 3.690, 11.623)
```

These four vectors are what a second encoder layer would receive. In real BERT this repeats 12
or 22 times, with layer normalisation after each residual to stop the magnitudes growing the
way they just did here: note that every output component is now larger than anything in `X`.

---

## Question: what if there were no positional embeddings?

**The model would become blind to word order.** Not approximately: exactly.

Attention is a weighted sum, and a sum does not care about the order of its terms. The only
place order enters this entire computation is step 1. Remove it and the layer becomes
*permutation equivariant*: shuffle the input tokens and the outputs come back shuffled the same
way, each token's vector unchanged.

Here is that run, with `X(t)` set to the token embedding alone:

| token | Output, "my model overfit again" | Output, "again overfit model my" |
| --- | --- | --- |
| my | (8.412, 1.182, 6.041) | (8.412, 1.182, 6.041) |
| model | (3.527, 4.067, 7.927) | (3.527, 4.067, 7.927) |
| overfit | (9.993, 3.726, 1.810) | (9.993, 3.726, 1.810) |
| again | (11.412, 0.182, 6.041) | (11.412, 0.182, 6.041) |

The columns are identical. Two sentences that mean different things get the same
representation, token for token.

**Compare that to the real run above.** With positions, "model" ends at
`(11.505, 2.901, 4.885)`: a vector that exists only because "model" was in slot 2.

**Why this matters beyond grammar.** Block B builds sequences of life events, where the same
argument applies with higher stakes: *diagnosis then prescription* and *prescription then
diagnosis* are different clinical stories, and without a position signal a transformer cannot
tell them apart. That is why `b2` adds two clocks (age, and days before the index date) on
top of the position index you just used. A position index says *third*; it does not say *at 52*
or *three weeks ago*.

---

## Everything on one page

| | X | Q | K | V | Z | **Output** |
| --- | --- | --- | --- | --- | --- | --- |
| **my** | (1, 0, 1) | (2, 0, 0) | (1, 2, 1) | (2, 1, 1) | (3.14, 2.39, 3.53) | **(11.71, 2.63, 8.30)** |
| **model** | (1, 1, 0) | (1, 1, 1) | (1, 1, 0) | (1, 2, 1) | (3.52, 2.42, 1.94) | **(11.50, 2.90, 4.89)** |
| **overfit** | (1, 2, 1) | (2, 2, 0) | (3, 2, -1) | (2, 3, 3) | (3.45, 3.95, 3.41) | **(11.27, 6.00, 9.31)** |
| **again** | (1, 0, 2) | (3, 0, -1) | (2, 3, 1) | (3, 1, 2) | (3.04, 2.86, 4.90) | **(12.18, 3.69, 11.62)** |

Attention matrix **A**:

| query \ key | my | model | overfit | again |
| --- | --- | --- | --- | --- |
| **my** | 0.066 | 0.066 | 0.661 | 0.208 |
| **model** | 0.182 | 0.057 | 0.182 | 0.578 |
| **overfit** | 0.047 | 0.015 | 0.469 | 0.469 |
| **again** | 0.009 | 0.016 | 0.923 | 0.051 |

---

*Checked three ways: longhand loops with no linear-algebra library, numpy, and
`torch.nn.functional.scaled_dot_product_attention`. All three agree to 9e-16.*
