---
title: 'pegainfer (6): MoE Expert Parallelism From Scratch'
description: 'Narrowing MoE expert parallelism down to one layer in pure decode, this post walks through dispatch, expert computation, and combine.'
pubDate: 'Jun 06 2026'
---

I have been learning MoE EP recently. Working on EP for Pegainfer has also been giving me a headache. After thinking about it, I might as well write some articles. If I can write it out, it probably means I roughly understand it.

First, narrow the problem down to the MoE part of one layer, and only look at the pure decode scenario, with more focus on latency.

After each attention / DP rank finishes its own attention part, it runs the router for the tokens in this rank's batch, and selects the top-k experts according to router scores.

The real EP starts after this. The whole chain is roughly:

```text
dispatch -> expert compute -> combine
```

`dispatch` sends each token, together with its routing information, to the rank that owns the corresponding expert. The expert owner gathers the tokens and runs the FFN. `combine` then sends the expert outputs back to the original rank, and performs the top-k weighted sum.

After processing the router, when we are ready to enter dispatch, we will get a batch of data like this:

```rust
Vec<(TokenId, HiddenState, Vec<(ExpertId, GateWeight)>)>
```

The length of `Vec<(ExpertId, GateWeight)>` equals the model's MoE top-k, which is `num_experts_per_tok` in Kimi K2.6's configuration.

Using the configuration in `pegainfer/models/Kimi-K2.6/config.json` as an example:

```text
hidden_size = 7168
n_routed_experts = 384
n_shared_experts = 1
num_experts_per_tok = 8
dtype = bfloat16
```

The data corresponding to one token is roughly:

```text
HiddenState: [7168] bf16/fp16
ExpertId:    [8] int32
GateWeight:  [8] fp32
```

That is, the hidden size of a single token is `7168 * 2 = 14336` bytes, about 14 KiB. For example, with dp8 / ep8 and 8 tokens per DP rank, the total hidden size of 64 tokens is about 896 KiB. Gate weight only has MoE top-k scalar values. For Kimi K2.6, this is `8 * 4 = 32` bytes. So the truly large part in communication is hidden. `expert_id` and `gate_weight` are very small routing metadata.

What is `GateWeight`? It is not the expert weight matrix. It is the coefficient assigned by the router to this token's top-k experts. Classic MoE can be simplified first with softmax:

```text
scores = hidden @ router_weight
topk_experts = topk(scores, k)
gate_weights = softmax(scores[topk_experts])
```

Kimi K2.6's actual configuration is not exactly this set. Its `scoring_func` is sigmoid, and it normalizes the gate weights inside top-k. There is also a `routed_scaling_factor = 2.827` afterward. Here, the softmax form is used first to explain the role of gate weight clearly. The concrete model implementation can be replaced with its own gating formula.

After each expert finishes computing its own output, the final token output is a weighted sum:

```text
moe_out[token] =
    gate_weight_0 * expert_0(hidden)
  + gate_weight_1 * expert_1(hidden)
  + ...
  + gate_weight_k * expert_k(hidden)
```

So sending only hidden is not enough. The expert owner rank knows the input vector, but it still does not know two things:

- Which experts this token selected.
- How much gate weight each expert's output should be multiplied by.

## Expert Computation

After the expert owner rank receives a batch of tokens, it first groups them by `expert_id`. For example, if 384 routed experts are split across 8 EP ranks, each rank owns 48 experts. rank0 owns e0..e47, and the received data may contain:

```text
e3:  token_a, token_b, token_c
e7:  token_d
e19: token_e, token_f
```

Each expert is essentially an FFN / MLP. A common SwiGLU structure can be written as:

```text
gate = hidden @ gate_proj
up   = hidden @ up_proj
mid  = silu(gate) * up
out  = mid @ down_proj
```

If GEMM is launched separately for each expert one by one, there will be many small matrix multiplications, and the launch overhead and scheduling overhead are not worthwhile. So in real implementations, multiple experts that need to run on the same rank are usually organized into grouped GEMM:

```text
group 0: tokens_for_e3  @ e3.gate/up/down
group 1: tokens_for_e7  @ e7.gate/up/down
group 2: tokens_for_e19 @ e19.gate/up/down
...
```

After computation, expert output still cannot be thrown into an anonymous buffer directly. It must keep source information:

```text
(original_rank, local_token_id, expert_id, gate_weight)
```

Only then can combine know which original token this expert output belongs to, and which gate weight it should be multiplied by.

## Where Gate Weight Is Multiplied in Combine

The combine stage needs to solve this question: after expert output is computed, should gate weight be multiplied on the expert owner rank, or after returning to the token owner rank?

There are two possible approaches.

The first is that the expert owner multiplies gate weight first. The expert owner receives `(hidden, expert_id, gate_weight)`, and after computing `expert(hidden)`, immediately multiplies it by `gate_weight`, then sends the weighted contribution back to the token owner. The token owner only needs to add the top-k contributions. The current demo uses this semantic:

```text
output[token] += gate_weight * expert(hidden[token])
```

The second is that the token owner multiplies gate weight after the output returns. The expert owner only sends raw `expert(hidden)`. The token owner saves `(token_id, expert_id, gate_weight)`, and after receiving raw outputs, performs weighted summation. DeepEP's normal combine is closer to this semantic: its combine is addition without weights, and returns `combined_x` and `combined_topk_weights` separately.

Because approach B itself needs one additional kernel launch, DeepEP's `low_latency_combine` is closer to approach A: the combine kernel reads `topk_weights` and directly performs weighted accumulation during reduce.

The two approaches only differ in where the multiplication happens. The mathematical result is the same:

```text
moe_out[token] = sum(gate_weight[token, expert] * expert_output[token, expert])
```

## The Current Demo's Teaching Flow

To make the flow clear, the current demo uses the AllGather version, not the efficient version. Suppose there are 8 ranks, and each rank has 8 local tokens:

```text
rank0: hidden[8, hidden_dim], routes[8, num_experts_per_tok]
rank1: hidden[8, hidden_dim], routes[8, num_experts_per_tok]
...
rank7: hidden[8, hidden_dim], routes[8, num_experts_per_tok]
```

Each rank first computes routing only for its own local tokens. Then it performs three AllGathers:

```text
AllGather hidden      -> all ranks get all 8 * 8 token hidden states
AllGather expert_ids  -> all ranks get all tokens' top-k expert ids
AllGather weights     -> all ranks get all tokens' gate weights
```

After getting this information, each EP rank performs local filtering:

```text
if expert_id belongs to the expert range owned by this rank:
    output[token] += gate_weight * expert(hidden[token])
```

Finally, use ReduceScatter to slice the complete token sequence back by original owner rank:

```text
ReduceScatter(sum) -> rank i gets back the MoE output for its own local tokens
```

The benefit of this version is that the flow is very intuitive: all ranks see the complete dispatch table, and then only process the experts that belong to themselves. The downside is also obvious: the data volume is very wasteful, because every rank gets a lot of hidden states and routing metadata that it will not process.

On 8xH200, the current demo's measured data with `./run.sh --gpus 8 --tokens 8` (corresponding to dp8 ep8 above, with bs=8 for each dp rank) is:

```text
dispatch: about 0.922 MB / 0.043 ms / 21.67 GB/s
combine:  about 0.918 MB / 0.259 ms / 3.54 GB/s
```

In later articles, we will optimize this demo little by little and push the bandwidth higher.
