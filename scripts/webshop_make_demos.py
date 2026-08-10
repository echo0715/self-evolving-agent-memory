#!/usr/bin/env python
"""Record the few-shot demonstrations for the WebShop agent from the live store.

    python scripts/webshop_make_demos.py --server http://localhost:7000 \
        --out memsys/adapters/webshop_examples.json

The *actions and thoughts* below are hand-written; the *observations* are not --
they are replayed out of the real environment and captured verbatim. That is the
point of generating the file instead of writing it: a demonstration whose pages
were written by hand teaches the model a page format the store never emits, and
the mismatch shows up as an agent that clicks buttons which do not exist. Both
trajectories are verified to end at reward 1.0, and the script fails if they
stop doing so -- which is also a cheap end-to-end check that the corpus, the
index and the reward function still agree with each other.

The two goals come from the **train** index range and are excluded from the
evolving manifest, so no demonstration leaks a task the agent is later scored on.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#: (goal index, label, [(thought, action), ...]). Chosen to cover the two shapes
#: an episode takes: a product with no options, and one where the reward depends
#: on selecting them. Both use a query an agent could plausibly write from the
#: instruction alone -- not WebShop's internal `query` field, which the agent
#: never sees.
TRAJECTORIES = [
    (
        3019,
        "buy_without_options",
        [
            (
                "The customer wants a heavy duty computer office desk with a splice board "
                "design, under 140 dollars. I should search with the product words only -- "
                "the price limit is not something the search bar understands.",
                "search[heavy duty splice board computer office desk]",
            ),
            (
                "B09NCB2NC4 is 'A Contrasting Color Splice Board Design Computer Office Desk' "
                "at $100.0, which is under the 140 dollar limit and matches the splice board "
                "design directly. Let me open it to check the details and its options.",
                "click[B09NCB2NC4]",
            ),
            (
                "The page confirms a splice board design office desk at $100.0, under the "
                "limit, and it offers no size or colour options to choose. Nothing else to "
                "select, so I can buy it.",
                "click[Buy Now]",
            ),
        ],
    ),
    (
        3056,
        "buy_with_options",
        [
            (
                "The customer wants a black, medium, long sleeve v neck tunic shirt for women "
                "under 40 dollars. Colour and size are usually options on the product page "
                "rather than search terms, so I will search for the garment itself.",
                "search[women's long sleeve v neck tunic shirt]",
            ),
            (
                "B09BKHRS8G is a 'Long Sleeve Tee Shirts Casual Crewneck Pullover Tops Comfy "
                "Cool Tunic' priced from $9.9, well under 40 dollars, and it is the closest "
                "long sleeve tunic in the results. Let me open it and see which options it "
                "offers.",
                "click[B09BKHRS8G]",
            ),
            (
                "The product offers sizes and colours as buttons. The customer asked for black "
                "in medium, and both are available. The reward is graded on the options I "
                "select, so I must click them before buying. Colour first.",
                "click[black]",
            ),
            (
                "Black is selected. Now the size the customer asked for.",
                "click[medium]",
            ),
            (
                "Black and medium are both selected on a long sleeve tunic under 40 dollars, "
                "so every part of the instruction is satisfied. Time to buy.",
                "click[Buy Now]",
            ),
        ],
    ),
]

MAX_OBS_CHARS = 3000  # must match AgentConfig.max_observation_chars


def post(server: str, route: str, payload: dict) -> dict:
    req = Request(server.rstrip("/") + route, data=json.dumps(payload).encode("utf-8"),
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def render_actions(available: dict) -> str:
    """Must stay byte-identical to WebShopEnvironment.render_actions."""
    clickables = [c for c in available.get("clickables", []) if c != "search"]
    parts = []
    if available.get("has_search_bar"):
        parts.append("search[<keywords>]")
    parts.extend(f"click[{c}]" for c in clickables)
    return "Available actions: " + (", ".join(parts) if parts else "(none)")


def clip(obs: str) -> str:
    obs = obs.strip()
    return obs if len(obs) <= MAX_OBS_CHARS else obs[:MAX_OBS_CHARS] + " ...<truncated>"


def record(server: str, goal_index: int, label: str, steps) -> dict:
    client = f"demo-{label}"
    post(server, "/close", {"client": client})
    r = post(server, "/reset", {"session": goal_index, "client": client})

    messages = [{
        "role": "user",
        "content": f"{clip(r['observation'])}\n{render_actions(r['available_actions'])}",
    }]
    reward = 0.0
    for thought, action in steps:
        messages.append({"role": "assistant", "content": f"Thought: {thought}\nAction: {action}"})
        s = post(server, "/step", {"client": client, "action": action})
        if s["invalid"]:
            raise SystemExit(
                f"[{label}] the store rejected {action!r} -- the recorded page no longer "
                f"offers it. Re-pick the product; do not hand-edit the transcript."
            )
        reward = float(s["reward"])
        if s["done"]:
            break
        messages.append({
            "role": "user",
            "content": f"Observation: {clip(s['observation'])}\n{render_actions(s['available_actions'])}",
        })
    post(server, "/close", {"client": client})

    if reward < 1.0:
        raise SystemExit(
            f"[{label}] goal {goal_index} ended at reward {reward:.3f}, not 1.0. A "
            f"demonstration that does not score full marks teaches the agent to settle."
        )
    return {"task": label, "goal_index": goal_index, "reward": reward, "example": messages}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://localhost:7000")
    ap.add_argument("--out", default="memsys/adapters/webshop_examples.json")
    ap.add_argument("--evolve-manifest", default="manifests/webshop_evolve_train_50_seed42.json",
                    help="checked so a demonstration never leaks an evolving task")
    args = ap.parse_args()

    used = set()
    manifest = Path(args.evolve_manifest)
    if manifest.is_file():
        used = {t["goal_index"] for t in json.loads(manifest.read_text())["tasks"]}

    entries = []
    for goal_index, label, steps in TRAJECTORIES:
        if goal_index in used:
            raise SystemExit(f"goal {goal_index} is in {manifest}; pick another for the demo")
        entry = record(args.server, goal_index, label, steps)
        print(f"[demo] {label}: goal {goal_index}, {len(entry['example'])} messages, "
              f"reward {entry['reward']:.1f}")
        entries.append(entry)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    chars = sum(len(m["content"]) for e in entries for m in e["example"])
    print(f"wrote {len(entries)} demonstrations ({chars} chars) -> {out}")


if __name__ == "__main__":
    main()
