# LoRA Test-Time Training: Porting Guide to SOTA Architecture

This document analyzes the LoRA TTT (Test-Time Training) implementation from the 2026-03-17 baseline submission and provides a concrete plan for porting it to the 2026-03-22 SOTA architecture (11L EMA + GPTQ-lite, 1.1233 BPB).

---

## Section 1: How LoRA TTT Works

### The Full Pipeline

LoRA TTT is a two-phase approach:

**Phase 1 -- Train the base model (normal).** You train a standard GPT language model on the FineWeb training set. Nothing changes here. The model is quantized, compressed, and packaged into the 16MB artifact as usual.

**Phase 2 -- At evaluation time, adapt per-document.** When computing the validation BPB score, instead of running a single forward pass over the entire validation stream, you:

1. **Find document boundaries** using BOS tokens (token ID 1) in the validation data.
2. **Split each document into overlapping chunks** (default: 256-token chunks within 1024-token context windows).
3. **For each chunk, SCORE first, TRAIN second:**
   - Run a forward pass through the model with LoRA adapters active.
   - Record the per-token losses for BPB calculation (this is the "evaluate" step).
   - Then use that same loss to take one gradient step on the LoRA parameters (this is the "train" step).
   - This ordering is critical: you never train on tokens before scoring them, so there is no data leakage.
4. **Reset LoRA parameters between documents.** Each document starts fresh -- no information leaks across documents.

The key insight: as the model reads more of a document, the LoRA adapters specialize to that document's style, topic, and vocabulary. Later chunks benefit from adaptation on earlier chunks.

### What Gets LoRA Adapters

Three types of modules receive per-batch-element LoRA adapters:

| Target | Shape (per element) | Why |
|--------|-------------------|-----|
| `c_q` (query projection) in every block | `(dim, dim)` via rank-r factorization | Lets attention patterns adapt to document structure |
| `c_v` (value projection) in every block | `(dim, kv_dim)` via rank-r factorization | Lets the model change what information it extracts |
| `lm_head` (output projection) | `(dim, vocab_size)` via rank-r factorization | Lets the model shift its vocabulary distribution for the document |

The LoRA delta is computed as `x @ A^T @ B^T` where `A` is the down-projection (rank x in_features) and `B` is the up-projection (out_features x rank). `B` is initialized to zeros so the adapter starts as a no-op. `A` gets Kaiming-uniform initialization.

### Hyperparameters Controlling TTT

| Parameter | Default | What It Does |
|-----------|---------|-------------|
| `ttt_lora_rank` | 8 | Rank of each LoRA adapter. Higher = more expressive but slower and more memory. |
| `ttt_lora_lr` | 0.01 | Adam learning rate for LoRA parameters during TTT. |
| `ttt_chunk_size` | 256 | Number of new tokens scored per chunk. Controls granularity of adaptation. |
| `ttt_eval_seq_len` | 1024 | Maximum context window for each chunk (includes prior context). |
| `ttt_batch_size` | 64 | Number of documents processed in parallel. |
| `beta1, beta2` | 0.9, 0.95 | Adam momentum parameters (shared with training config). |

### The "Evaluate Then Train" Constraint

This is a contest rule, not just a design choice. The competition requires that tokens are scored BEFORE they can be used for adaptation. In code, this looks like:

```
for each chunk in document:
    loss = model.forward(chunk, lora=adapters)   # forward pass with current adapters
    accumulate_bpb(loss)                          # SCORE first -- record for BPB
    if not last_chunk:
        loss.backward()                           # TRAIN second -- update adapters
        optimizer.step()
```

The last chunk is score-only (no training needed since there are no future chunks to benefit).

---

## Section 2: Current Implementation Analysis

### Base Model Architecture (LoRA TTT Submission)

The training is *identical* to the naive baseline -- nothing special is done during training to prepare for TTT. The base model is:

- **9 layers**, 512 dim, 8 heads (4 KV heads, GQA), 2x MLP expansion
- **Vocab 1024**, sequence length 1024, tied embeddings
- U-Net skip connections (4 encoder layers, 5 decoder layers)
- relu-squared MLP activation
- RoPE (full head dim), QK RMSNorm, logit softcap=30
- **17,059,912 parameters total**
- Muon optimizer for matrices, AdamW for embeddings
- Int8 quantization + zlib compression for the artifact

### How the Eval Loop Works -- Step by Step

**Step 1: Load and segment validation data**

```python
all_tokens = torch.cat([load_data_shard(Path(f)) for f in files])
docs = _find_docs(all_tokens)  # returns [(start_offset, length), ...] per document
```

Documents are found by scanning for BOS tokens (ID=1). Each document spans from one BOS to the next.

**Step 2: Distribute documents across GPUs**

```python
rank_docs = docs[(len(docs) * rank) // world_size : (len(docs) * (rank + 1)) // world_size]
rank_docs.sort(key=lambda d: (d[1] - 2) // chunk_size)  # sort by length for batching efficiency
```

Documents are split evenly across ranks. Sorting by length minimizes wasted padding in batches.

**Step 3: Create batched LoRA adapters**

```python
lora = BatchedTTTLoRA(batch_size, base_model, lora_rank).to(device)
opt = _build_ttt_optimizer(lora, args)  # Adam with lr=0.01
```

The `BatchedTTTLoRA` module holds independent LoRA weights for each batch element. The `A` and `B` matrices have an extra leading batch dimension: shape `(bsz, rank, in_features)` and `(bsz, out_features, rank)`.

**Step 4: Process documents in batches**

For each batch of documents:
1. Reset all LoRA adapters to initial values (A = Kaiming-uniform, B = zeros).
2. Reset Adam optimizer state (zero out moment estimates).
3. Iterate through chunks in lockstep across the batch.

**Step 5: Per-chunk processing**

For each chunk index `ci`:
- Build a context window: the chunk plus as much prior context as fits in `eval_seq_len`.
- Pad shorter documents with zeros (they are masked out of the BPB accumulation).
- Forward pass with LoRA adapters active (keeping the computation graph if training is needed).
- Accumulate BPB: record per-token loss and byte counts for the *new* tokens in this chunk only.
- If any document in the batch has future chunks: compute masked loss and take one Adam step.

**Step 6: Reduce across ranks**

All-reduce `loss_sum`, `byte_sum`, and `token_count` across GPUs, then compute final BPB.

### The BatchedLinearLoRA Forward Pass

```python
def forward(self, x: Tensor) -> Tensor:
    # x: (bsz, T, in_features)
    # A: (bsz, rank, in_features) -- down-project
    # B: (bsz, out_features, rank) -- up-project
    return (x @ self.A.transpose(1, 2)) @ self.B.transpose(1, 2)  # (bsz, T, out_features)
```

This is a standard LoRA computation but batched: each element in the batch has its own independent A and B matrices. The output is *added* to the original projection's output in the attention forward pass.

### How LoRA Integrates with the Model Forward Pass

In the Block's forward method:
```python
def forward(self, x, x0, q_delta_fn=None, v_delta_fn=None):
    n = self.attn_norm(x_mixed)
    qd = q_delta_fn(n) if q_delta_fn is not None else None  # LoRA applied to normed input
    vd = v_delta_fn(n) if v_delta_fn is not None else None
    attn_out = self.attn(n, qd, vd)
```

In the Attention forward:
```python
q = self.c_q(x) + (q_delta if q_delta is not None else 0)  # base + LoRA delta
v = self.c_v(x) + (v_delta if v_delta is not None else 0)  # base + LoRA delta
```

And for the LM head (in GPT.forward):
```python
logits = logits + (lora.lm_head_lora(x) if lora else 0)
```

### Performance Data

From the train logs:

| Run | Hardware | Standard Eval Time | TTT Eval Time | TTT BPB |
|-----|----------|-------------------|---------------|---------|
| `train_v0.txt` | 2x H100 | 5,599 ms | 72,361 ms | 1.1935 |
| `train_v3.txt` | 8x H100 | 1,374 ms | 60,184 ms | 1.1929 |

Key observations:
- TTT eval takes ~60 seconds on 8x H100, well under the 600-second budget.
- The author notes this uses only "~1/10th of the evaluation budget."
- Documents are split across GPUs, so scaling to 8 GPUs gives good parallelism.
- The 50K validation documents at batch_size=64 means ~781 batches per GPU (with 8 GPUs: ~98 batches each).

### BPB Improvement Breakdown

| Condition | val_bpb | Delta from Baseline |
|-----------|---------|-------------------|
| Baseline (cross-doc, flat stream) | 1.2278 | -- |
| + Doc-isolated eval | 1.2168 | -0.0110 |
| + Stride (chunk=256) | 1.1941 | -0.0337 |
| + LoRA TTT | 1.1910 | -0.0368 |

The submitted score was **1.1928 BPB** (from a 1.2274 flat baseline, so approximately -0.035 BPB improvement).

Critical finding from the ablations: **most of the gain comes from doc-isolated + strided eval, not from TTT itself.** The LoRA adaptation contributes only about -0.003 BPB on top of the strided eval. The SOTA already uses sliding-window strided eval (stride=64), so some of this "free" improvement is already captured.

---

## Section 3: Architecture Differences (LoRA TTT Base vs SOTA)

| Component | LoRA TTT Base | SOTA (11L EMA) | Incompatibility? |
|-----------|--------------|----------------|------------------|
| **Layers** | 9 | 11 | Minor: just more LoRA adapters needed |
| **Model dim** | 512 | 512 | Same |
| **Heads / KV heads** | 8 / 4 (GQA) | 8 / 4 (GQA) | Same |
| **Head dim** | 64 | 64 | Same |
| **MLP expansion** | 2x (hidden=1024) | 3x (hidden=1536) | No impact on LoRA (LoRA targets attn, not MLP) |
| **Vocab size** | 1024 | 1024 | Same |
| **Train seq len** | 1024 | 2048 | Must choose TTT eval_seq_len carefully |
| **Eval seq len** | 1024 (TTT) | 2048 (sliding window) | Need to decide: use 1024 or 2048 for TTT windows |
| **Tied embeddings** | Yes | Yes | Same |
| **Attention impl** | `F.scaled_dot_product_attention` | **FlashAttention 3** (`flash_attn_3_func`) | **MAJOR**: FlashAttn 3 uses `(B, T, H, D)` layout, not `(B, H, T, D)`. LoRA integration changes. |
| **RoPE** | Full head dim (64 dims) | **Partial RoPE** (16/64 dims) | Minor: LoRA adds to pre-RoPE outputs, no conflict |
| **XSA** | None | Efficient Partial XSA on last 4 layers | No conflict: XSA operates after attention, LoRA operates on Q/V before attention |
| **SmearGate** | None | Yes (pre-attention token mixing) | No conflict: operates on embeddings before blocks |
| **BigramHash** | None | Yes (2048 buckets, dim=128) | No conflict: additive embedding, happens before blocks |
| **ValueEmbedding** | None | Yes (shared, layers 9,10, dim=128) | Minor: v_embed is added to V, LoRA also adds to V -- they stack |
| **LN Scale** | None | `1/sqrt(layer_idx+1)` | No conflict: purely internal to Block.forward |
| **Block.forward signature** | `(x, x0, q_delta_fn, v_delta_fn)` | `(x, x0, v_embed=None)` | **MAJOR**: Block has no LoRA injection points. Must add q_delta/v_delta args. |
| **GPT.forward signature** | `(input_ids, target_ids, lora=None)` | `(input_ids, target_ids)` | **MAJOR**: Must add LoRA parameter, change loss return to per-token. |
| **Loss return** | Per-token when lora, mean otherwise | Always mean | **MAJOR**: TTT needs per-token loss for selective masking. |
| **Quantization** | Int8 + zlib | **Int6 + GPTQ-lite** + zstd-22 | LoRA operates on dequantized model, no conflict |
| **EMA** | None | decay=0.997, applied before quantization | No conflict: EMA weights become the base model |
| **QAT (STE)** | None | Late QAT at LR scale < 0.15 | No conflict: QAT is training-only |
| **MTP heads** | None | Configured but `mtp_num_heads=0` | No conflict |
| **Parameters** | 17.06M | 26.99M | ~58% more params: more memory for LoRA backward pass |

### Critical Incompatibilities Summary

1. **FlashAttention 3 layout**: The SOTA uses `flash_attn_3_func` which expects `(B, T, H, D)` tensor layout. The LoRA TTT code was written for PyTorch's `F.scaled_dot_product_attention` which uses `(B, H, T, D)`. The LoRA deltas are added to Q/V *before* the reshape, so the layout difference does not actually matter for LoRA injection -- but the attention forward path is different.

2. **Block.forward has no LoRA injection points**: The SOTA Block takes `(x, x0, v_embed)` but has no `q_delta_fn` or `v_delta_fn` arguments. These must be added.

3. **GPT.forward has no LoRA support**: The forward method does not accept a `lora` argument and always returns a scalar mean loss. TTT needs per-token losses.

4. **CausalSelfAttention.forward has no delta args**: The SOTA attention takes `(x, v_embed)` but not `q_delta`/`v_delta`.

---

## Section 4: Concrete Porting Plan

### Principle: Do Not Change Training

Everything about the SOTA training pipeline stays exactly the same. LoRA TTT is purely an eval-time technique. You will:
- Add new classes (BatchedLinearLoRA, BatchedTTTLoRA)
- Add new functions (eval_val_ttt_lora and helpers)
- Modify the forward methods to *optionally* accept LoRA parameters (default=None, so training is unaffected)
- Call the TTT eval after training completes

### Step 1: Add LoRA Classes

Copy the following classes directly from the LoRA TTT code. They need no modification:

```python
class BatchedLinearLoRA(nn.Module):
    """LoRA for a linear layer, with independent weights per batch element."""
    def __init__(self, bsz, in_features, out_features, rank):
        super().__init__()
        self.in_features = in_features
        self.A = nn.Parameter(torch.empty(bsz, rank, in_features))
        self.B = nn.Parameter(torch.zeros(bsz, out_features, rank))
        self.reset()

    def forward(self, x):
        return (x @ self.A.transpose(1, 2)) @ self.B.transpose(1, 2)

    def reset(self):
        bound = 1.0 / math.sqrt(self.in_features)
        with torch.no_grad():
            self.A.uniform_(-bound, bound)
            self.B.zero_()
```

### Step 2: Create BatchedTTTLoRA for the SOTA Architecture

The SOTA has 11 layers instead of 9, but the structure is the same. One key difference: you should consider adding `c_proj` (the output projection of attention) as a LoRA target, since the SOTA model is more expressive and the output projection has the same shape as `c_q`:

```python
class BatchedTTTLoRA(nn.Module):
    def __init__(self, bsz, model, rank):
        super().__init__()
        dim = model.tok_emb.embedding_dim
        vocab = model.tok_emb.num_embeddings
        self.lm_head_lora = BatchedLinearLoRA(bsz, dim, vocab, rank)
        self.q_loras = nn.ModuleList()
        self.v_loras = nn.ModuleList()
        for block in model.blocks:
            self.q_loras.append(BatchedLinearLoRA(bsz, dim, block.attn.c_q.weight.shape[0], rank))
            self.v_loras.append(BatchedLinearLoRA(bsz, dim, block.attn.c_v.weight.shape[0], rank))

    def reset(self):
        for m in self.modules():
            if isinstance(m, BatchedLinearLoRA):
                m.reset()
```

Note: `c_q.weight.shape[0]` is 512 (dim) and `c_v.weight.shape[0]` is 256 (kv_dim = 4 * 64). These are the same in both architectures.

### Step 3: Modify CausalSelfAttention.forward

Add optional LoRA delta arguments. This is a minimal, backward-compatible change:

```python
def forward(self, x: Tensor, v_embed: Tensor | None = None,
            q_delta: Tensor | None = None, v_delta: Tensor | None = None) -> Tensor:
    bsz, seqlen, dim = x.shape
    q = self.c_q(x)
    if q_delta is not None:
        q = q + q_delta                          # <-- ADD THIS
    q = q.reshape(bsz, seqlen, self.num_heads, self.head_dim)
    k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
    v = self.c_v(x)
    if v_delta is not None:
        v = v + v_delta                           # <-- ADD THIS
    if v_embed is not None:
        v = v + v_embed
    v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
    # ... rest unchanged (RoPE, FlashAttn 3, XSA, proj) ...
```

Important: The LoRA deltas are added *before* the reshape to `(B, T, H, D)`, which is correct -- they represent additive modifications to the full Q/V vectors.

### Step 4: Modify Block.forward

Add optional LoRA function arguments:

```python
def forward(self, x, x0, v_embed=None, q_delta_fn=None, v_delta_fn=None):
    mix = self.resid_mix.to(dtype=x.dtype)
    x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
    normed = self.attn_norm(x_in) * self.ln_scale_factor
    qd = q_delta_fn(normed) if q_delta_fn is not None else None
    vd = v_delta_fn(normed) if v_delta_fn is not None else None
    attn_out = self.attn(normed, v_embed=v_embed, q_delta=qd, v_delta=vd)
    x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
    x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(
        self.mlp_norm(x_out) * self.ln_scale_factor)
    if self.dtg_gate is not None:
        gate = torch.sigmoid(self.dtg_gate(x_in.detach()))
        x_out = x_in + gate * (x_out - x_in)
    return x_out
```

### Step 5: Add a TTT-aware Forward Method to GPT

Rather than modifying the main `forward` (which is used during training and must stay fast), add a new method:

```python
def forward_ttt(self, input_ids, target_ids, lora):
    """Forward with LoRA adapters. Returns per-token loss (bsz, seq_len)."""
    x = self.tok_emb(input_ids)
    if self.bigram is not None:
        x = x + self.bigram(input_ids)
    x = F.rms_norm(x, (x.size(-1),))
    x = self.smear(x)
    x0 = x
    skips = []
    ve_cache = {}

    for i in range(self.num_encoder_layers):
        ve = self._get_ve(i, input_ids, ve_cache)
        qd = lora.q_loras[i]
        vd = lora.v_loras[i]
        x = self.blocks[i](x, x0, v_embed=ve, q_delta_fn=qd, v_delta_fn=vd)
        skips.append(x)
    for i in range(self.num_decoder_layers):
        bi = self.num_encoder_layers + i
        if skips:
            x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
        ve = self._get_ve(bi, input_ids, ve_cache)
        qd = lora.q_loras[bi]
        vd = lora.v_loras[bi]
        x = self.blocks[bi](x, x0, v_embed=ve, q_delta_fn=qd, v_delta_fn=vd)

    x = self.final_norm(x)
    if self.tie_embeddings:
        logits = F.linear(x, self.tok_emb.weight)
    else:
        logits = self.lm_head(x)
    logits = logits + lora.lm_head_lora(x)
    logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)

    bsz, sl, V = logits.shape
    return F.cross_entropy(
        logits.float().reshape(-1, V), target_ids.reshape(-1),
        reduction="none").reshape(bsz, sl)
```

### Step 6: Add the eval_val_ttt_lora Function

Copy the function from the LoRA TTT code nearly verbatim. The main changes:
- Call `base_model.forward_ttt(x, y, lora=cur_lora)` instead of `base_model(x, y, lora=cur_lora)`.
- Adjust `ttt_eval_seq_len` (consider 2048 to match the SOTA's training context).
- The FlashAttention 3 import must be present (it already is in the SOTA script).

### Step 7: Which LoRA Targets for the 11L Architecture?

Recommended targets (in order of importance):

1. **`c_q` and `c_v`** (all 11 layers) -- these are proven effective in the original submission.
2. **`lm_head`** (via tied embeddings) -- adapting the output distribution is high-value.
3. **`c_proj`** (attention output, optional) -- worth experimenting with, adds same parameter count as c_q.
4. **MLP (`fc` or `proj`)** -- not targeted in the original. Could help but significantly increases LoRA parameter count and backward pass cost.

Start with the same targets as the original (c_q, c_v, lm_head). Add c_proj only if you have memory/time budget.

### Step 8: Call TTT Eval After Training

At the end of the SOTA's `main()` function, after all quantization and standard eval:

```python
# LoRA TTT evaluation
torch._dynamo.reset()
torch.cuda.synchronize()
t_ttt = time.perf_counter()
ttt_val_loss, ttt_val_bpb = eval_val_ttt_lora(
    args, base_model, rank, world_size, device,
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
)
torch.cuda.synchronize()
log0(f"final_ttt_lora val_loss:{ttt_val_loss:.4f} val_bpb:{ttt_val_bpb:.4f} "
     f"eval_time:{1000.0 * (time.perf_counter() - t_ttt):.0f}ms")
```

### Memory Concerns

The 11L SOTA model has 27M parameters (vs 17M). During TTT:
- The base model is frozen (no gradients stored for it).
- Only LoRA parameters need gradients. Per document in batch:
  - Q LoRAs: 11 layers x (rank x 512 + 512 x rank) = 11 x 2 x 512 x 8 = 90,112 params
  - V LoRAs: 11 layers x (rank x 512 + 256 x rank) = 11 x (4096 + 2048) = 67,584 params
  - LM head LoRA: (rank x 512 + 1024 x rank) = 12,288 params
  - Total per document: ~170K params
  - Batch of 64: ~10.9M LoRA params (plus Adam states = ~3x = ~33M params total)
- Forward activations for backward pass: with seq_len=1024 and batch=64, this is roughly 64 x 1024 x 512 x 11 layers x ~4 tensors x 2 bytes = ~2.9 GB.

**This should fit on an H100 (80GB)** with plenty of room. The base model in bf16 is only ~54MB. The concern is more about speed than memory.

If memory is tight (e.g., with eval_seq_len=2048), reduce `ttt_batch_size` from 64 to 32.

### Timing Budget Estimate

The LoRA TTT baseline took ~60s on 8x H100 with a 9-layer model and 50K documents.

For the 11L model:
- Forward pass is ~22% slower (11/9 layers).
- Backward pass scales similarly.
- Estimated: ~73s on 8x H100.
- Still well under the 600s budget.

With eval_seq_len=2048 (2x longer contexts):
- Forward/backward roughly 2x slower due to attention being O(n^2) or FlashAttn O(n).
- Estimated: ~120-150s. Still within budget.

---

## Section 5: Optimization Strategies

### Strategy 1: Parallel Eval Across 8 GPUs (Already Built-In)

The existing implementation already splits documents across ranks:
```python
rank_docs = docs[(len(docs) * rank) // world_size : (len(docs) * (rank + 1)) // world_size]
```
With 50K documents and 8 GPUs, each GPU handles ~6,250 documents. This scales linearly.

### Strategy 2: Reduce LoRA Rank

| Rank | Params per doc | Expected quality | Speed |
|------|---------------|-----------------|-------|
| 8 (default) | ~170K | Best (proven) | 1.0x |
| 4 | ~85K | Slight degradation (~0.001 BPB) | ~1.3x faster |
| 2 | ~43K | Noticeable degradation | ~1.5x faster |

Recommendation: Start with rank 8. If you need speed, try rank 4 first -- the quality loss is usually small because most of the adaptation signal is low-rank anyway.

### Strategy 3: Fewer TTT Steps Per Document

Currently: 1 gradient step per chunk (with chunk_size=256, a 1024-token document gets 3 training steps).

Options:
- **Increase chunk_size to 512**: halves the number of steps, but each step uses more context. May be net-positive because longer chunks give better gradient signal.
- **Skip training on short documents**: documents under ~512 tokens get minimal benefit from TTT (only 1-2 adaptation steps).

### Strategy 4: Gradient Checkpointing

If memory becomes an issue with the 11L model:
```python
from torch.utils.checkpoint import checkpoint

# In forward_ttt, wrap each block:
x = checkpoint(self.blocks[i], x, x0, ve, qd, vd, use_reentrant=False)
```
This trades ~2x compute for ~11x memory savings on activations. Only use if needed.

### Strategy 5: torch.compile for the TTT Forward/Backward

The LoRA TTT code currently does NOT use `torch.compile`. Adding it could significantly speed up the forward/backward passes:

```python
# Before the TTT eval loop:
compiled_forward_ttt = torch.compile(base_model.forward_ttt, dynamic=False)
```

Caveats:
- The first call will have compilation overhead (~10-30s).
- Dynamic batch sizes (last batch may be smaller) can trigger recompilation. Handle by padding the last batch.
- FlashAttention 3 may or may not compose well with torch.compile.

### Strategy 6: Use BFloat16 for LoRA Parameters

The LoRA A and B matrices default to float32. Switching to bfloat16:
```python
self.A = nn.Parameter(torch.empty(bsz, rank, in_features, dtype=torch.bfloat16))
self.B = nn.Parameter(torch.zeros(bsz, out_features, rank, dtype=torch.bfloat16))
```
Saves 50% memory and speeds up the matmuls. Quality impact is minimal for rank-8 adapters.

### Strategy 7: Combine TTT with Sliding Window

The SOTA gets a significant boost from sliding-window eval (stride=64). The LoRA TTT code uses its own document-isolated, chunk-strided eval. A hybrid approach is possible:
- Use TTT's document isolation and LoRA adaptation.
- Within each document, use overlapping windows similar to sliding-window eval.
- This is partially what the original TTT already does (chunk_size=256 within eval_seq_len=1024 windows).
- Increasing eval_seq_len to 2048 and reducing chunk_size to 128 would give more sliding-window benefit.

---

## Section 6: Expected Impact

### Arguments for a LARGER Gain on the SOTA Base

1. **Better base = more structure to exploit.** A 1.1233 BPB model has already "understood" more about language. TTT can more effectively fine-tune on document-specific patterns because the base model provides a stronger starting point.

2. **More layers = more adaptation capacity.** With 11 layers of LoRA adapters (vs 9), the model can distribute adaptation across more representational levels.

3. **The SOTA already uses sliding-window eval.** The LoRA TTT's "free" gains from strided eval (-0.034 BPB) overlap with the SOTA's sliding-window eval. However, the *document isolation* benefit (-0.011 BPB) is NOT captured by the SOTA, since the SOTA evaluates a flat token stream without document boundaries.

### Arguments for a SMALLER Gain on the SOTA Base

1. **Diminishing returns.** The lower the base BPB, the harder it is to squeeze out improvements. The "easy" compression gains are already captured.

2. **LoRA TTT's pure adaptation gain was small.** On the baseline, the LoRA adaptation itself (beyond striding) was only -0.003 BPB. On a stronger base, this could shrink to -0.001 or less.

3. **FlashAttention 3 incompatibility with backward pass.** FlashAttention 3 may not support backward-mode efficiently in the same way as standard SDPA. If you have to fall back to standard attention for the TTT pass, you lose the speed advantage.

### Expected BPB Range

| Scenario | Expected BPB | Reasoning |
|----------|-------------|-----------|
| **Best case** | 1.115 - 1.118 | Doc isolation + TTT on SOTA base gives -0.005 to -0.008 |
| **Likely case** | 1.118 - 1.121 | Doc isolation gives -0.002 to -0.004, TTT adds -0.001 |
| **Worst case** | 1.122 - 1.123 | TTT overhead forces compromises, gains are marginal |

The most probable outcome is around **1.119 BPB**, primarily from document-isolated eval on the stronger base.

### Abort Criteria

Stop investing in this approach if:

1. **Document-isolated eval alone (without TTT) does not beat 1.1233 on the SOTA base.** If isolating documents and using strided eval does not help vs the SOTA's sliding-window eval, then the doc-boundary information is already captured, and TTT has little to add.

2. **TTT eval takes more than 300 seconds** on 8x H100. This means you have used half the eval budget, leaving no room for optimization.

3. **The pure LoRA TTT gain (over doc-isolated + strided) is less than 0.001 BPB.** At this point, the complexity is not worth it -- focus on training improvements instead.

4. **Memory OOM on H100 with batch_size=32.** If you cannot fit even 32-document batches, the per-document overhead makes TTT impractically slow.

### Recommended Experiment Sequence

1. **Experiment A (1 hour):** Add `forward_ttt` to the SOTA model. Test with a single document to verify correctness.
2. **Experiment B (2 hours):** Run doc-isolated eval WITHOUT LoRA TTT on the SOTA base. Measure the BPB to establish the doc-isolation baseline.
3. **Experiment C (2 hours):** Run full LoRA TTT eval. Measure BPB and wall-clock time.
4. **Experiment D (if promising):** Sweep `ttt_lora_rank` in {2, 4, 8, 16} and `ttt_chunk_size` in {128, 256, 512}.
5. **Experiment E (if promising):** Try `eval_seq_len=2048` for longer context windows during TTT.

---

## Appendix: File Reference

| File | Description |
|------|-------------|
| `/Users/vaibhavippili/Desktop/Personal/oaipg/parameter-golf/records/track_10min_16mb/2026-03-17_LoRA_TTT/train_gpt.py` | LoRA TTT implementation (source) |
| `/Users/vaibhavippili/Desktop/Personal/oaipg/parameter-golf/records/track_10min_16mb/2026-03-17_LoRA_TTT/README.md` | LoRA TTT method description and ablations |
| `/Users/vaibhavippili/Desktop/Personal/oaipg/parameter-golf/records/track_10min_16mb/2026-03-17_LoRA_TTT/train_v0.txt` | Train log (2x H100), TTT eval = 72.4s |
| `/Users/vaibhavippili/Desktop/Personal/oaipg/parameter-golf/records/track_10min_16mb/2026-03-17_LoRA_TTT/train_v3.txt` | Train log (8x H100), TTT eval = 60.2s |
| `/Users/vaibhavippili/Desktop/Personal/oaipg/parameter-golf/records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py` | SOTA implementation (target) |
| `/Users/vaibhavippili/Desktop/Personal/oaipg/parameter-golf/records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` | SOTA method description |
