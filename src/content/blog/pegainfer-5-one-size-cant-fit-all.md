---
title: "pegainfer (5): One Size Can't Fit All"
description: 'An inference engine should not chase a super abstraction that can swallow every model, but should maintain a set of clearly bounded per-model engines.'
pubDate: 'May 05 2026'
---

> **TL;DR**: An inference engine should not chase a super abstraction that can swallow every model, but should maintain a set of clearly bounded per-model engines. Share the things that are stable, and let go of the things that will truly fork.

## From a Performance Problem to a Maintenance Problem

It has been a long time since the last pegainfer blog update. The previous one was still about integrating CUDA Graph. Now we not only support Qwen3.5 4B, but Qwen3 4B on the 5070ti has also already surpassed vLLM in performance. We also support TP for Qwen3 4B. We even supported full-blooded DeepSeek V3.2, though the code was not merged, partly because V4 was released, and partly because after supporting these models and workloads, I also read a lot of papers and did a lot of experiments in April. I feel that for the next step of Pegainfer, the focus is not supporting more models or more use cases, but how we maintain an inference engine.

The previous few blog posts were more about answering performance questions: can it run from 0, can accuracy align, how should the sampler be done, and how much launch overhead CUDA Graph can really save. This stage was actually quite pleasant, because every problem was very concrete. Profile once, write a kernel, connect graph, run benchmark, watch the numbers get better. The feedback loop was very short.

But as I continued, I increasingly felt that supporting a new model itself is not hard. It is nothing more than going through config, weights, operators, accuracy, and performance round after round. But every time a new model is supported, it breaks a lot of the original code. Multiple models get mixed together, and the complexity is hard to eliminate, because they are essentially different. Qwen3 is dense full attention, Qwen3.5 is 24 layers of linear attention plus 8 layers of full attention and needs recurrent state, and DeepSeek V3.2 is MLA compressed KV plus DSA, MoE routing, FP8 block-scale GEMM, and EP communication. Of course they are all called LLM inference, but when it lands in code, state layout, scheduler, kernel DAG, and communication method are all different. Forcing them under one abstraction is actually quite uncomfortable.

Recently, on the GitHub of vLLM's commercial company, [Inferact](https://github.com/Inferact), the first repo appeared: https://github.com/Inferact/vllm-frontend-rs. Its actual role is also consistent with its name: Rust-ify vLLM's frontend, including all kinds of connection handling, tokenizer and other CPU-intensive workloads moved up to Rust, with Rust and the vLLM engine communicating through ZMQ. For pegainfer, as we said in the first pegainfer blog post, letting Python do these things is really inefficient. Whether Rust, C++, or even Golang, they are all much more efficient. Now that the vLLM commercial company has a team pushing this thing forward, for pegainfer, a "Rust inference project", as its owner, I must answer "What's next". pegainfer has also integrated vllm frontend rs, but the interface has not been carefully checked yet. How much benefit will there be from cutting the backend scheduler and worker into Rust? Honestly, it is very hard to pressure-test out.

The DataFusion ecosystem, the history of OLAP, the papers and experiments of the past month, the maintenance of pegainfer and PegaFlow along the way, and the vibe coding experience kept circling in my mind. I arrived at my thought: one size can't fit all. Every person and every company can use LLMs to customize and optimize their own independent inference engine. This is not rebuilding everything from scratch again, but sharing the things that are truly stable, and then leaving the places that will truly fork to each model.

## What Does an LLM Serving Stack Look Like

What parts does a complete LLM serving stack contain?

After a request comes in, it is first distributed by the router to the corresponding machine. It is worth mentioning that the routers of vLLM and sglang are also both Rust. vLLM initially forked from sglang, and now it looks much cleaner. After reaching a single machine, it may be gRPC or HTTP, though I feel the future may lean toward gRPC. But this is not the focus.

After the request passes through the network, there are all kinds of processing related to request categories: tokenizer, chat template, OpenAI API, stop sequence, streaming, logprobs, usage statistics, and so on. After the frontend is Rust-ified, it seems there is no need to pursue pushing the tokenizer up to the router. Doing it on the local CPU is completely enough. Finally, it is made into a request and sent to the actual scheduler for processing.

Here there is an interface, the bridge for the frontend and scheduler to communicate. In theory, the interface should not perceive scheduler details, such as what model is running under what parallel strategy, whether it is TP or EP, whether it is full attention or linear attention, whether it is P/D disaggregation or single-machine mixed batching. The frontend is only responsible for cleaning up the request, turning it into a request and handing it to the engine behind it, and then translating token events back into OpenAI-compatible responses. The benefit of doing this is that the two sides are independent and modular. The people maintaining the frontend can maintain the frontend with peace of mind, and the scheduler side is the same.

After pegainfer integrated vllm frontend rs, my feeling about this became stronger. We do not need to maintain our own OpenAI server, and we do not need to maintain details like chat template and tokenizer by ourselves. pegainfer only needs to implement a local engine-core bridge, turn requests from the frontend into its own `EngineHandle`, and then send the generated tokens back. Of course, this interface is definitely not perfect yet. Fields like logprobs, usage, and trace headers all need to be supplemented later, but I think the direction is right: frontend is a stable boundary, and it should not be tied to a certain model.

## Is There Only One Scheduler?

Is there only one scheduler? I think as long as there is demand, anyone can customize their own scheduler. The scheduler's responsibilities are forming batches, resource management, and, as the heart of the whole engine, reflecting the health condition of the engine. For example, a full attention scheduler should not perceive the complexity of linear attention or DSA. A dense scheduler also does not want to see MoE operators and communication. A non-multimodal model does not want to perceive multimodal problems. I do not support RL, and schedulers for TP and PP, or even EP, are also written differently. There is no need to force them into one abstraction. The writing method of P/D disaggregation is also different from mixed batching.

This is already not a pure idea in pegainfer. Qwen3's scheduler cares about paged KV, prefill / decode / unified step, batch decode, and TP rank worker. Qwen3.5's scheduler is completely different. Besides KV, it needs to maintain recurrent state, copy the state after prefill into CUDA Graph slots, and do slot compaction after a request ends. Of course they can abstract out some similar words, like request, batch, and decode step, but in real maintenance, the details are completely different.

At first I also thought about whether to make a very general scheduler and stuff all models into it. This idea looks very engineering-like and beautiful, but now I do not really believe in it. Because in the end, every model will put its own special branches into one super scheduler. Full attention has one set of logic, linear attention has one set of logic, MoE has one set of logic, P/D disaggregation has one set of logic, and RL workload has yet another set of logic. In the end it looks like reuse, but actually everyone is maintaining a huge context together, and nobody dares to touch it.

So now I tend more toward this: frontend / protocol can be shared, runtime primitives can be shared, kernels can be shared, data plane can be shared, but the scheduler is allowed to be customized per-model / per-workload.

## Each Model's Own Engine

Further down is the concrete model. After supporting these models, I feel there is one idea: we should maintain code per model. Of course, some components that can be shared can be extracted into reusable crates, such as tensor, CUDA Graph, KV pool, weight loader, kernel wrapper, frontend bridge, and so on. That is, one repo for each model family.

The starting point is also very plain: the current LLM context is large enough and smart enough, writing code is cheap enough, and maintaining a clean context that can be fed in at once can make us iterate development more efficiently. In the past we were very afraid of duplicate code, and felt everything had to be abstracted into one inheritance system or trait. But now what may be more important is letting both LLMs and humans understand at a glance what exactly this model is doing.

For example, Qwen3-4B has now already been split into its own crate. It has its own config, weights, executor, scheduler, kernel plan, and e2e test. The root only talks to it through a generic `EngineHandle`. From the outside, it all looks like "give me prompt tokens, I give you token events", but how Qwen3 internally does prefill, decode, TP, and kernel selection is Qwen3's own business. Qwen3.5 is still in root for now, but from the state, it should also be its own model engine, because hybrid linear attention is not a variant of Qwen3. It is another execution world. DeepSeek V3.2 goes without saying even more.

So what do we need to do to support a new model? The frontend can already be reused, the scheduler can default to TP scheduling, and other concrete scheduling can refer to reference materials from other model repos. The parts that need customization are weight loading and operators. More concretely, a model engine should at least own: how config is interpreted, how weights are loaded, how state is laid out, how KV cache / recurrent state / MLA compressed KV are placed, how prefill / decode / mixed step run, which kernels are selected under what shapes, how accuracy aligns with reference, and how benchmark is run.

This set of things has to be put together for the model's context to be complete. If split too finely, the LLM has to understand across a pile of directories every time. If abstracted too generally, in the end everyone does not know which model a certain branch exists for.

## Kernel Is Not Just a Library

In essence, the inference process of a model is a series of kernels linked together, that is, the operation of mapping from `config.json` to a series of kernels. Kernel is very important. It determines 99% of the factors of final performance and 99% of correctness, so how to maintain and manage a model's kernels is very important. Under different arches, under the shapes in this model's inference process, each kernel's behavior, including GPU behavior, ncu, cupti, and correctness regression, probably all need to use PyTorch as the baseline.

Kernels should be a public repository maintained by everyone, such as `flashinfer`. I feel what is missing is something like LLMArena, letting operator engineers around the world gather together to climb the leaderboard. I do not know whether NVIDIA will do this, or maybe everyone just measures p2p. This also includes Hugging Face pushing their kernels ecosystem, which I think is great.

This also involves a JIT problem. I feel the root of JIT is that the kernel library does not support something like Rust features, causing everything to have to be compiled, with long build time and large artifacts. In fact, one model only references the kernels it needs. There are not many. And in real deployment, it only needs to be compiled once and made into an image. Online there is completely no need to JIT again, right? Because JIT brings service quality jitter.

But having only a kernel library is not enough. The library solves "what kernels exist", but does not solve "which kernel should be used for this model, this shape, and this GPU". The same attention may be a completely different choice under $bs=1, ctx=1024$, $ctx=4096$, and $bs=32$. A kernel being good on 4090 does not mean it is good on 5090; being good for prefill does not mean it is good for decode.

So LLM needs an index to select these kernels and splice together this model's kernel DAG. pegainfer already has a very light kernel plan in Qwen3, telling you what ops are in prefill / decode / unified, where the Rust wrapper is, and whether the backend is CUDA, cuBLAS, or FlashInfer. Later I also want to make a kernel manifest to scan different batch sizes, context lengths, and variants, and generate per-op reports.

Continuing this thing downward, it is actually a kernel ledger. It should record:

- Which dtype, layout, SM, and shape this kernel supports;
- Under a certain GPU / CUDA / commit, roughly what the latency, bandwidth, and tensor core metrics are;
- Who the correctness reference is, how large the error is, and whether it affects greedy token;
- In the model DAG, which phase, which layer, and which op instance it belongs to.

It sounds a bit tedious, but this thing is very suitable for LLMs. Because LLMs are very strong at reading a pile of code, but the premise is that you have to give it an index. Otherwise every time it jumps back and forth between `config.json`, weight name, Rust wrapper, FFI, CUDA kernel, and benchmark JSON, humans get tired, and LLMs also get tired.

Of course, models are getting more and more complex. Some kernels are uncommon and need to be provided by the model vendor itself. If they are not provided, then we can only trust the LLM (laughs).

## Simulator and Tracing

After selecting kernels, theoretically based on the performance of these kernels, we can directly calculate inference performance offline, such as TTFT of 1k, TPOT of bs1, TPOT of bs100, similar to a simulator.

This simulator does not need to be especially accurate at the beginning. I think its most important value is not predicting a beautiful number, but explaining performance. For example, tpot=20ms. 20ms itself has no meaning. What is really meaningful is how the 20ms is composed, which kernels are the main parts, and what they are bound by: memory bandwidth, tensor core, launch overhead, or communication. We can also simulate some curves to find the sweet spot of workloads.

It can also be used to determine online performance regressions. For example, if the estimated 1k pure prefill needs 100ms, but one batch comes back and calculates 500ms, then something must be wrong. At this time tracing needs to appear, dumping the execution trace of this request and handing it to R&D engineers to carefully investigate the problem, similar to an online low-overhead profile.

From router to the various machines in P/D disaggregation, at this stage a request will pass through a lot of I/O. If it is I/O, there is long tail. The long tail itself may not mean much, but the problems it exposes may be more critical. For example, last year I debugged a p999 caused by NUMA, and only then deeply understood the importance of topology. Sometimes a p999 is not a "latency number", but the system structure reminding you where something is wrong.

So I hope pegainfer can later have a closed loop from kernel ledger to simulator and then to request tracing. Offline, I know roughly how fast a model should run. Online, I know what actually happened to a request. If the two do not match, then there is an investigation entry.

## What's Next

So returning to the question at the beginning, what is pegainfer next?

My current thought is that it is not an inference framework that continuously stuffs all models into the same abstraction. It is more like a set of reusable infrastructure, plus some clearly bounded per-model engines.

Stable things are shared: frontend like vllm frontend rs, runtime primitives, CUDA / cuBLAS / NCCL wrappers, tensor types, kernel crates, KV data plane like PegaFlow, benchmark / profiling / tracing tools.

Things that will fork are left to models: config interpretation, weight loading, state layout, scheduler, kernel DAG, TP / EP / PP / P/D disaggregation strategy, accuracy alignment method.

This is not saying that every model should be written completely from 0. Exactly the opposite. I think future LLM code writing will make "maintaining a clean model context" more important than "abstracting a huge general abstraction". Share the things that are stable, and let go of the things that will truly fork. This may be the direction of pegainfer's next stage.
