"""End-to-end dry run of the memory systems on synthetic episodes.

No benchmark, no API: episodes are hand-made and the writer is `StubWriterLLM`.
The point is to exercise every mechanism -- retrieval, budget packing, dedup->merge,
verification, refinement, batch induction, eviction, frozen evaluation -- and to make
the content/mechanism split visible: the same policy produces the same set of
mechanisms for all three memory types.

    python -m examples.demo
"""

from __future__ import annotations

import random

from memsys import (
    NATIVE_POLICIES,
    Episode,
    Evolver,
    MemoryConfig,
    Rollout,
    Step,
    WritePolicy,
    build_system,
    frozen,
)
from memsys.stub_llm import StubWriterLLM

ENV = "toyworld"
TYPES = ("reflection", "rule", "skill")

TASK_TEMPLATES = [
    ("heat", "heat some {obj} and put it in {recep}"),
    ("clean", "clean some {obj} and put it in {recep}"),
    ("find", "put two {obj} in {recep}"),
]
OBJS = ["tomato", "mug", "apple", "plate", "knife"]
RECEPS = ["fridge", "countertop", "sinkbasin", "cabinet"]


def make_rollout(rid: str, obj: str, recep: str, success: bool, n_steps: int) -> Rollout:
    steps = [Step(action=f"go to {recep} 1", observation=f"You arrive at {recep} 1.")]
    steps.append(Step(action=f"take {obj} 1 from {recep} 1", observation=f"You pick up the {obj} 1."))
    for i in range(n_steps):
        steps.append(
            Step(
                action=f"examine {obj} 1",
                observation=f"The {obj} 1 looks unchanged." if not success else "Nothing happens.",
                thought=f"step {i}: checking state before acting",
            )
        )
    if success:
        steps.append(Step(action=f"put {obj} 1 in/on {recep} 1", observation="You put it down."))
    else:
        steps.append(Step(action=f"heat {obj} 1 with microwave 1", observation="Nothing happens."))
    return Rollout(rollout_id=rid, steps=steps, reward=1.0 if success else 0.0, success=success)


def make_episodes(n: int, seed: int = 0) -> list[Episode]:
    rng = random.Random(seed)
    eps = []
    for i in range(n):
        ttype, tmpl = TASK_TEMPLATES[i % len(TASK_TEMPLATES)]
        obj, recep = rng.choice(OBJS), rng.choice(RECEPS)
        # bucket cycles mixed / all_success / all_failure so every writer mode fires
        flags = {0: [True, False, False, True], 1: [True, True], 2: [False, False]}[i % 3]
        eps.append(
            Episode(
                task_id=f"task-{i:03d}",
                instruction=tmpl.format(obj=obj, recep=recep),
                rollouts=[make_rollout(f"{i}-{j}", obj, recep, ok, rng.randint(1, 3))
                          for j, ok in enumerate(flags)],
                scope={"env": ENV, "task_type": ttype},
            )
        )
    return eps


def show(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def mechanisms(llm) -> set[str]:
    return {tag.split(".", 1)[1] for tag in llm.usage.by_tag}


def run(kind: str, episodes: list[Episode], config: MemoryConfig):
    llm = StubWriterLLM()
    system = build_system(kind, llm=llm, config=config)
    ev = Evolver(system, config=config)
    ev.run(episodes)
    return system, llm, ev


# --------------------------------------------------------------------------
def policy_comparison(episodes: list[Episode]) -> None:
    """The fairness property: mechanism set is a function of the POLICY, not the type."""
    settings = {
        "uniform full   ": MemoryConfig(policy=WritePolicy.full(batch_every=4), injection_budget_tokens=400),
        "uniform minimal": MemoryConfig(policy=WritePolicy.minimal(), injection_budget_tokens=400),
        "native (biased)": MemoryConfig(injection_budget_tokens=400).native(),
    }
    show("Write mechanisms actually exercised, per policy x memory type")
    print(f"{'policy':<16} {'type':<11} {'mechanisms fired':<42} items  writer_tok")
    print("-" * 92)
    for label, cfg in settings.items():
        for kind in TYPES:
            system, llm, _ = run(kind, episodes, cfg)
            fired = ", ".join(sorted(mechanisms(llm))) or "-"
            tok = llm.usage.prompt_tokens + llm.usage.completion_tokens
            print(f"{label:<16} {kind:<11} {fired:<42} {len(system.store):>5}  {tok:>9}")
        print()
    print("Read the first block down each column: under a uniform policy every type runs")
    print("the same mechanisms, so the Memory Content table compares content only.")
    print("The third block is the original design (rule alone verifies, skill alone")
    print("induces) -- keep it for the write-mechanism ablation, not the content table.")


def detail(episodes: list[Episode], config: MemoryConfig) -> None:
    for kind in ("reflection", "rule", "skill", "raw", "all"):
        system, llm, ev = run(kind, episodes, config)
        show(f"[{kind}]  {len(episodes)} evolving episodes, policy: {system.policy.describe()}")
        print("store       :", system.summary())
        print("op counts   :", ev.logger.op_distribution())
        print("writer cost :", llm.usage.to_dict())

        probe = "heat some tomato and put it in fridge"
        with frozen(system):
            r = system.retrieve(probe, scope={"env": ENV, "task_type": "heat"})
            try:
                system.observe(episodes[0], r)
                print("!! frozen store accepted a write")
            except RuntimeError as e:
                print("frozen check:", e)
        print(f"\n--- injected block for {probe!r} "
              f"({r.tokens}/{config.injection_budget_tokens} tokens, {len(r.items)} items) ---")
        print(r.block or "(empty)")


def main() -> None:
    episodes = make_episodes(12, seed=7)
    policy_comparison(episodes)
    detail(
        episodes,
        MemoryConfig(
            policy=WritePolicy.full(batch_every=4),  # small E so induction fires in 12 episodes
            injection_budget_tokens=400,             # small, so budget packing visibly bites
            max_items=25,
        ),
    )
    show("Notes")
    print(
        "- Numbers above are meaningless: StubWriterLLM does not reason.\n"
        "- Swap in OpenAIChatClient(model=..., base_url=...) for real runs.\n"
        f"- NATIVE_POLICIES = {{{', '.join(f'{k}: {v.describe()}' for k, v in NATIVE_POLICIES.items())}}}"
    )


if __name__ == "__main__":
    main()
