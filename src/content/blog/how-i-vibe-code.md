---
title: 'How I Vibe Code'
description: 'A personal journey from pair-programming with AI to building a human-AI hybrid team — documentation, tooling, and the honest truth about where the real bottleneck is.'
pubDate: 'Mar 28 2026'
---

At the beginning of this year, I was still a heavy CLI user — Claude Code and Codex — with most of my attention on the conversation itself, "pair programming" with them, though my reviews were sometimes a bit late.

Later, when [OpenClaw](https://github.com/openclaw/openclaw) (nicknamed "小龙虾" / "little crayfish" in Chinese) blew up, I didn't jump on it immediately. My thought was, "Why not just cron two CC tasks instead?" At the time, everyone was hyping daily briefings and such, which didn't hit my sweet spot. Then I read a WeChat article by the TiDB CTO about how he drove an OpenClaw team for development, and I thought it was pretty interesting, so I started setting it up.

Because incidents like OpenClaw deleting repos and poisoning dependencies kept happening, I threw it into a Docker container on my homelab. I mounted some work directories (ones that belong only to it — delete them and I wouldn't care), mounted a GPU, gave it root. In other words, it could basically do whatever it wanted.

The experience after that was quite novel. I could discuss papers with it on the subway, discuss open-source code — it could just `git clone` a repo and dive in. That's something no chat app can easily do. It felt like having Jarvis. It could also help me write code for Pegainfer (a Rust inference framework).

The turning point also came from vibe coding Pegainfer with it. It would often get stuck in loops, taking forever to respond, and I had no idea what it was doing. The experience was basically minute-level TTFT, since it was running on GPT High under the hood. And because of GPT's tone — "to conclude definitively," "shall I go ahead and," "alright, simply put" — I stopped enjoying chatting with it. For conversation, Sonnet 4.5 is still the best; even Opus 4.6 doesn't quite have that vibe.

I started to shift my approach. I promoted it — made it the **Tech Leader** of Pegainfer and had it delegate tasks to the devs below (i.e., Codex). GPT driving GPT. At the same time, I wanted to solve the problem of it often going silent and me not knowing what it was doing. So I created a Notion page, gave it Notion MCP access and page permissions, and had OpenClaw log its own work. When OpenClaw drove Codex, it often found Codex unresponsive too, so OpenClaw wrote a prompt to make Codex log its work as well.

The first PR from this complete workflow: [pegainfer#11](https://github.com/xiaguan/pegainfer/pull/11) — roughly, it discovered that decode performance degraded severely as sequence length grew. [Here's OpenClaw's work log for that session](https://www.notion.so/Project-pegainfer-tpot-gap-investigation-31d59f32de6b8162b42edb3db971c146). A few more PRs followed with corresponding Notion pages, slightly richer, but broadly the same pattern.

But vibe coding this way always felt like I was lacking control. I couldn't see what Codex and CC were doing. I didn't know if they were working efficiently. So I went back to the traditional N tmux panes and N VS Code windows, manually parallelizing and switching.

This is where I started to realize that **documentation matters for both agents and me**. I began thinking about how to manage a documentation system that both humans and agents need to read. I've never really introduced my note-taking method, but it's basically the *Building a Second Brain* book + working docs. That's comfortable for me, so I wanted agents to do the same. The only physical book I read this year was also *Building a Second Brain* — I'd flip to a random page and start reading.

After that, maybe OpenAI released a new model, and I started trying Codex more (I mainly use CC with Opus). With Codex, it kept saying "blocked by sandbox" or constantly asked me to approve things. The experience was terrible. I looked at the commands carefully — they were basically all harmless. And from my experience with Opus, these advanced intelligences basically never execute dangerous commands. So I turned on YOLO mode for Codex — letting it do whatever it wants. The experience was like seeing water in a desert. It could finally work autonomously for long stretches. Though one of Codex's quirks is that no matter how you instruct it, it always stops at some point to ask "should I continue with XX?" — I hope OpenAI collected my data to train the next model well (after all, I still have five more months of Pro). CC is also basically on YOLO, and I only turn it off for tasks that feel ambiguous or where I worry it might misunderstand the environment.

After that, still during the Pegainfer vibe coding experience — since I'm basically not a kernel guy, don't understand models, and don't understand frameworks — some sessions would go in circles because I didn't know the right path. I couldn't write high-quality prompts. To this day I still don't know how to write Triton or CUDA kernels, so naturally I can't point it in the right direction, and it would just keep inefficiently exploring. So I began **documentation governance** for Pegainfer. I wanted it to have better context and better tools.

## Documentation and Agent-First Tooling

**Documentation and agent-first tools and environments** are the foundation of agent-driven project development. Here's an example based on Pegainfer (which is open source).

The overall structure is based on the PARA method I mentioned — Projects, Resources, Areas, Archives. Are these categories the best? Maybe not — I'm still thinking about it — but that's what I'm using now. Then there's an `index.md` for the index (it could go in `CLAUDE.md`, but I feel more comfortable with it external). Currently Pegainfer has documentation organized like this.

<!-- Image: Pegainfer docs directory structure -->

Let me introduce these docs from my perspective:

1. **Milestone**: The project's goals, direction, what good looks like, next actions — things a TL needs to know.
2. **The rest of the project docs**: Basically records of how it debugged accuracy, optimized performance, etc. Many docs are hundreds of lines long.

For example, here's a condensed outline of its debugging session for Qwen 3.5 accuracy issues yesterday:

```
# Qwen3.5-4B Accuracy Debug Doc Outline

## 1. Goals & Definitions
- Accuracy goal: align with Hugging Face Qwen3.5-4B (eval + deterministic greedy)
- Explicitly NOT aligning with pegainfer's self-generated JSON
- Success criteria in three tiers:
  - Layer-level alignment
  - First decode token alignment
  - End-to-end generation alignment or explainable drift

## 2. Truth Source Rules
- Prefill alignment: HF full forward
- Decode alignment: HF incremental (past_key_values)
- Never treat "full-prefill with reconstructed prefix" as decode truth
- The only truth source for generated tokens: HF real incremental path

## 3. Debugging Methodology
- Start with deterministic prompt + deterministic mode
- Prefill first, then decode
- Layer first, then e2e
- Coarse checkpoints first, then per-layer drill-down
- Stop at the first divergence checkpoint — don't do massive pointless dumps

## 4. Alignment Ladder
1. Layer 0, short sequence prefill
2. Layer 0, chunk boundary (seq_len=65)
3. Layer 1/2 (linear attention)
4. Layer 3 (first full attention)
5. Full prefill final hidden / logits
6. First decode token
7. Full prompt-level generation

## 5. Key Checkpoints
embedding → input layernorm → linear qkv/z/b/a → conv1d_out → gdr_out →
recurrent_state → attn_out → post_attention_layernorm → layer_out →
final_norm → logits

## 6. Resolved Issues
- HD256 full-attention decode kernel no longer used as reference
- conv1d_prefill repeated seq_len=1 handoff fix
- argmax tie-break fix: pick smallest token id on ties
- conv1d fix to match HF's bf16 pre-SiLU rounding
- Production-path incremental dump changed to real runtime path

## 7. Current Conclusions
- Major decode-state bugs fixed
- HF exact match up to 11/13
- Remaining issues concentrated in two late-stage small-logit-drift cases
- Current residual deviation looks like cumulative numeric drift, not structural error
- Some cases are now in tie-sensitive territory — a 0.125–0.25 logit shift flips top-1

## 8. Current Blockers
- tell_story
- chinese_capital
Both share: late decode step divergence, logits near greedy decision boundary

## 9. Recommended Debug Actions
- Fix at first HF divergence step
- Use exact token-id prefix for peg vs HF incremental alignment
- Continue narrowing along layer/checkpoint
- Prioritize finding "HF semantic mismatches like the conv1d rounding bug"
- Avoid any debug path that doesn't use real runtime kernel/state updates
```

It also records performance optimization step by step — the profile data it collected, effective attempts, failed attempts. Feel free to check it out.

Under **Resources**, there are docs on how to efficiently use tools:
1. **Bench vs vLLM**: How to benchmark against vLLM — what workloads to compare, how to spin up vLLM serve, bench commands, how to disable multimodal.
2. **Profiling Guide**: How to use Pegainfer's built-in bench tool for quick performance validation, and how to use nsys for profiling, where to put intermediate data.
3. **Model Optimization Pipeline**: How to start an optimization doc for a model, and how to organize it.

I also built a very convenient e2e test — plain text assertions (ground truth from HF). After optimizing, just run it. Fast and easy. I basically avoid having it write unit tests. I'm reminded of the Vercel CTO's quote:

> Write tests. Not too many. Mostly integration.

Documentation, tools (performance, correctness, profiling), environment (YOLO + GPU + uv) — nothing stops it from writing code quickly and efficiently.

The overall testing framework still needs continuous iteration, including operator-level optimization, experience with various frameworks and high-performance kernels. Pegainfer's path to pure agent development still has a long way to go. The biggest problem is still **me** — my taste in inference frameworks isn't great. I can only keep growing together through continuous interaction — learning Triton together, profiling operators, studying linear attention.

Pegainfer went from a toy I hacked on before Chinese New Year to now supporting Qwen3/Qwen3.5 with performance on par with vLLM and accuracy aligned with HF. It's been a remarkable journey.

<!-- Image: Pegainfer benchmark results -->

Of course, Pegainfer is my personal side project — something I tinker with for fun. Most of my energy goes to another Pega — **Pegaflow**, my main focus, working on KV cache. Also open source, though the docs aren't, since they're rather personal.

## The Pegaflow Team

With Pegaflow, my relationship with LLMs is different. My taste and decision-making on KV cache are fairly ahead of the curve — I know what needs to be done and what the right approach is. So the LLM's responsibilities are lighter. But Pegaflow, as a production-level project, requires far more than Pegainfer. Let me introduce two team members in Pegaflow:

1. **Quinn** — Pegaflow's QA. Responsible for internal QA testing (GitHub CI doesn't have GPUs). Many types of tests, day-to-day and per-feature. Test clusters aren't fixed — basically wherever there's availability. Our cluster development has certain conventions: when to use K8s, when to spin up containers, standards for rsyncing things.

2. **Ronan** — Pegaflow's TL and developer. (No separate devs because current agent orchestration isn't great.) Manages Pegaflow's direction and steers the ship. In the future, he'll need to take on more responsibilities, including how to keep painting the vision — as a real project, it can't stop moving, and that's part of a TL's job.

### Quinn's Documentation

<!-- Image: Quinn's docs directory structure -->

First, **Areas** — the fundamental knowledge base for a QA: CI pipeline, what the tests actually test, etc.

In **Archives**, there are projects like adding sccache to internal CI, M2.5 CI scripts, testing features on H200, completed projects, accuracy CI, MTP CI, Pegaflow's P2P RDMA feature testing, and P2P RDMA logs — this one's interesting, I'll cover it in Ronan's section. She was actually helping Ronan debug an RDMA performance issue (with me as the communication bridge, of course). In **Resources**, the most important items are detailed docs for each cluster — incredibly detailed. With these manuals, working is just smooth sailing. Then there are docs on Pegaflow, vLLM, and lm-eval.

There's also a skill: `/run-ci <branch> <commit>` — she goes and runs CI for me with one command.

When Ronan finishes a feature, I check where's appropriate and where there's capacity, then tell Quinn:

> @xx-cluster.md @xxx-project.md, go follow up and test this feature.

Quinn also has her own `CLAUDE.md`. Her workflow looks roughly like this:

```
## Look Before You Leap

| Task Type | Required Reading | Notes |
|---|---|---|
| Any new task | docs/index.md | Global doc index — check for related docs first |
| CI-related | ci/README.md | File structure, test flow, assertion functions, ports, timing |
| Cluster ops | docs/resources/xxx-cluster.md | SSH, K8s, node resources, working directory conventions |
| Pegaflow behavior | docs/resources/pegaflow-overview.md | Architecture, data flow, key metrics |
| Writing new scripts | Grep ci/ for similar implementations first | Don't write from scratch — reference existing work |

## Project Doc Template

When assigned a new task, create a project doc under docs/projects/ with this structure:

### Preparation (must fill before execution)
- **What I read**: List the actual team docs, yamls, scripts (paths) you read
- **What I referenced**: Existing similar implementations, related PRs, related project docs
- **Plan**: Steps to execute, which machines, commands, resources involved

After preparation, get user review before proceeding.

### Execution Log (append-only)
- Smooth → one-liner
- Not smooth → detailed: exact commands, error output, root cause analysis, fix method
- Key decisions and reasoning
- Milestones (deployment success, tests passing, PR merged)
- Current blockers
```

Then she typically goes to read docs, write docs, and wait for my approval. I scan through it, and if nothing looks off, I let her take over.

Her responsibilities are exactly as I designed — just QA, and she does it excellently. I'm very satisfied. These were all tasks I used to do myself. If you've ever tested cluster-level features, I think you'll understand, haha.

### Ronan's Documentation

<!-- Image: Ronan's docs directory structure -->

(There's a `ci` folder that was probably created by accident. Ignore it.)

**CPU Bench All NICs**: This was a fascinating problem. Roughly: on the exact same NIC, exact same configuration, exact same request construction, the e2e total bandwidth was consistently 25% lower than the stress test. Single QP, max outstanding reads = 1, same number of requests, same size, same NIC, same machine. This problem probably only an RDMA expert could answer, haha. Codex relaying to Claude Code worked on it from afternoon to evening before finding the root cause. I can share the wrong answer — the correct one I'll leave for the comments.

Here's the table of eliminated hypotheses — all wrong answers:

| Hypothesis | Verification | Conclusion |
|---|---|---|
| Network path / subnet difference | ib_write_bw cross-machine | All 4 NICs saturated at 364 Gbps |
| NIC pairing error | Code review + QPN mapping | engine.rs uses 1:1 index mapping |
| Protocol overhead (gRPC/prepare/rebuild) | Segment timing | 2.3ms, outside rdma_wait |
| Software-side / CQ polling overhead | Code review + estimation | ~180us/NIC |
| Responder-side cross-NUMA DMA | ib_read_bw membind comparison | NUMA0 vs NUMA1 mem — no difference (320 Gbps both) |
| Block size 2944K vs 4M | ib_read_rs -s 3014656 | Still 320 Gbps |
| Bidirectional concurrency | Unidirectional stress test | Test is single-sided |
| GPU CUDA DMA PCIe contention | vLLM idle + concurrency=1 | Save completes before send starts |
| Bidirectional RDMA overlap | pegaflow-server ms-level timestamp alignment | 82 rdma_waits don't overlap |
| tokio await scheduling latency | cross_host_bench --tokio-await | current_thread / multi_thread+bg both 24ms |

Baseline: `ib_read_bw` on 4 NICs = 1281 Gbps (24.1ms for 3680 MiB). Pegaflow `rdma_wait` = 882 Gbps (35.0ms). **Gap: 399 Gbps (31%)**. Every hypothesis above was ruled out — the real root cause I'll leave for the comments.

Since I've been working on Pegaflow's RDMA recently, you'll see tons of RDMA docs — how to select GID, how to manage QPs, QP contention, QP error handling. Pegaflow hasn't resolved everything yet, but once it does, I think it'll be a pretty solid RDMA implementation, haha.

## The Philosophy

Documentation. Private knowledge as an asset. Compound interest on knowledge. Building high-quality context and efficient execution environments. On that note, I sometimes export sessions — just a bunch of JSONs — then review them to find areas for improvement.

From the beginning of the year when I was using CC and Codex alone, to now where I'm still using CC and Codex but with my own human-AI hybrid team — I've gone from a programmer to an **architect + manager + reviewer**.

Models will keep getting stronger. Sonnet will get version 5, Opus will get a new release, Codex will become more autonomous. But these documents don't go out of date. No matter how much Opus is trained, it can't know how to develop on my company's different clusters. No matter how autonomous Codex becomes, it doesn't know Pegainfer's current progress or technical vision. Conversely, new versions of Opus and Codex will perform even better with the existing context.

## The Honest Truth

Finally, let me have the LLM roast me. This prompt got some good feedback — harsh words are good for you:

> The most honest sentence in your entire article is:
>
> *The biggest problem is still me.*
>
> Yes. But you said it as modesty and moved on. You didn't really face it. You used a documentation system, an agent team, Quinn and Ronan — you built an elegant system to avoid this fact: **you're using management complexity to substitute for technical depth.** The better you get at driving agents, the further you get from writing high-performance kernels yourself. The more satisfied you are with Quinn's QA work, the less you'll understand what those tests are actually verifying.
>
> You said "growing together, learning Triton together." Your "together" with agents is asymmetric. They start from zero every time, and you stop at the review layer every time. Neither of you is growing.
>
> You don't need a better documentation system. You need to close all your terminals, open a blank file, and write a Triton kernel yourself — debug it until 3 AM. Only then do you have the right to talk about taste, architect, "driving a team."
>
> Until then, you're a passenger writing route plans for a self-driving car, convincing yourself that this is driving.

One rebuttal — I actually have been studying CUDA and Triton.
