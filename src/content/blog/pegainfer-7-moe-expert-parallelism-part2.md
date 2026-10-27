---
title: 'pegainfer (7): MoE Expert Parallelism From Scratch (Part 2)'
description: 'Continuing MoE EP from scratch, this post walks through dispatch buffers, metadata barriers, expert-major packing, and combine buffers.'
pubDate: 'Jun 06 2026'
series:
  name: pegainfer
  part: 7
---

Continuing from the previous post.

Each rank has `t` local tokens. Each token is selected by the router for `TOP_K = 8` experts, and these 8 experts are scattered across all 8 ranks.

That is, each rank has data like this:

```jsx
Vec<(TokenId, HiddenState, Vec<(ExpertId, GateWeight)>)>
```

The receiver keeps a standing recv buffer `[world_size][max_token_per_rank][hidden]`, covering the occupied worst case. From Rust's perspective, it looks like:

```rust
[Vec<(TokenId, HiddenState, Vec<(ExpertId, GateWeight)>)>; WORLD_SIZE]
```

From the logical abstraction, what each rank needs to do is move `Hidden` from its own `Vec` to the corresponding receiver's buffer according to its own routes. Written as a Rust abstraction, it is:

```rust
type Route = (ExpertId, GateWeight);
type Token = (TokenId, HiddenState, Vec<Route>);

fn dispatch(
    local: Vec<Token>,
    expert_rank: impl Fn(ExpertId) -> usize,   // EP placement: expert -> the rank that owns it
) -> [Vec<Token>; WORLD_SIZE] {
    let mut send: [Vec<Token>; WORLD_SIZE] = std::array::from_fn(|_| Vec::new());

    for (tid, hidden, routes) in local {
        let mut by_rank: [Vec<Route>; WORLD_SIZE] = std::array::from_fn(|_| Vec::new());
        // First aggregate by receiver rank, similar to SQL group by?
        for (eid, w) in routes {
            by_rank[expert_rank(eid)].push((eid, w));
        }
        // Then batch-write to each receiver.
        for (rank, local_routes) in by_rank.into_iter().enumerate() {
            if local_routes.is_empty() { continue; }
            // local_routes is written at once here.
            send[rank].push((tid, hidden.clone(), local_routes)); // this clone is the real copy crossing the wire
        }
    }
    send
}
```

After a rank passes this step, data has been sent. Can it start computing? Not yet, because it still does not know whether all other ranks have completed their sends. We need to notify the receiver.

The sender needs an array `send_expert_count[NUM_EXPERTS]`, used to record that it sent how many tokens to each expert. At the same time, as the receiver, it needs to maintain `recv_expert_count_win[WORLD_SIZE][LOCAL_EXPERT_SIZE]`, a two-dimensional array used to compute information like: rank0 sent 2 tokens to expert0 that I own.

So at this point we need a second wave of I/O, making the receiver's two-dimensional array ready. At the same time, in this wave of I/O we need a barrier. A simple way to understand it is: after sending its own metadata, it starts a busy loop waiting for the flag to jump to `WORLD_SIZE` (actually similar to a CPU barrier, except GPU needs to write it by itself).

After passing the barrier, both data and metadata are complete. The next step is preparing for computation.

According to the `recv_expert_count_win` array, we can know that expert x needs to compute y tokens by simply traversing and summing. Then we start packing them into an expert-major array for grouped GEMM.

Expert-major can be treated as a one-dimensional array. Different experts are mapped into it, and of course another array or cursor is needed to record this mapping. The memory layout can look like this:

```text
memory address 0 -------------------------------------------------------> end of memory
[  Expert 0's Token 0  ]
[  Expert 0's Token 1  ]
...
[  Expert 0's Token 99 ]  <-- suppose expert 0 has 100 tokens, length is 100 * HIDDEN
-------------------------  <-- position of expert_offset[1]
[  Expert 1's Token 0  ]
[  Expert 1's Token 1  ]
...
[  Expert 1's Token 49 ]  <-- suppose expert 1 only has 50 tokens
-------------------------  <-- position of expert_offset[2]
[  Expert 2's Token ... ]
...
```

This method has some memory waste. For example, if `token0 -> expert0,1`, then its hidden is copied twice.

If this feels wasteful, there is another layout, but then an array is needed to dereference indexes, and the kernel will probably become dizzy to write.

Considering WideEP too, a rank's experts are not many to begin with, and the redundancy probability is low. So everyone probably chooses this fully flattened layout without considering redundancy.

There is another tradeoff here: should this one-dimensional array be static or dynamic? We can allocate a static one according to the worst case, or allocate one on-site for every dispatch.

How large is the worst case? It is:

```text
world_size * max_tokens_per_rank * min(num_topk, num_local_experts)
```

For example, Kimi K2.6 has 384 experts. With ep32, each rank has 12 experts, and top-k is 8. Estimating with 32 tokens per rank (actually already quite large), the maximum reserved buffer is:

```text
32 * 32 * 8 * 14KB
```

That is 112MB per card. Actually this is still okay, though the prefill side may be worse.

In short, the most comfortable and most suitable approach for decode seems to be static allocation, ignoring redundancy.

After that comes grouped GEMM computation. This is computation on the operator side.

Skipping this stage, after grouped GEMM finishes, we still get an expert-major array `[TOKENS][HIDDEN]`. But the output dtype may not be the same as the input, depending on the concrete model and the precision tradeoff. For DeepSeek V3, it should be fp8 input and bf16 output. Of course, quantization format support still has many problems, but if we count it as an operator issue, it is not covered here. What I want to express here is that data may become larger after computation.

After that, we still need to send each token's result back. It is like Ekko using his ultimate: roughly follow the previous path in reverse.

Each buffer needs another buffer, call it the combine buffer: `[tokens_per_rank][TOP_K][hidden]`. According to the data exchanged during dispatch, the path each token is sent back on is "static", or in other words, has no conflict. For example, token1's top0 hidden can only be written by one rank, so there is no need to consider others. This buffer is also zero waste and zero redundancy.

After each rank finishes combine send, it enters the barrier again. Once it confirms everything it should receive has arrived, it enters combine reduce. This is the one mentioned in the previous post: compressing the top-k results into one according to some weight ratio, collapsing `[token][TOP_K][hidden]` into `[token][hidden]`.
