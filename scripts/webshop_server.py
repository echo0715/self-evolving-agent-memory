#!/usr/bin/env python
"""WebShop environment as a small JSON HTTP service.

    $WEBSHOP_ENV/bin/python scripts/webshop_server.py --port 7000 --scale full

Run this with the **WebShop** interpreter (python 3.8, torch 1.11, pyserini),
not the memsys one. That split is the whole point of the file: WebShop is pinned
to a 2022 dependency set (`flask==2.1.2`, `spacy==3.3`, `transformers==4.19`,
`typing_extensions<4.6`) and memsys needs a current `sentence-transformers` and
`openai`, so the two cannot share an interpreter. An HTTP boundary is cheaper
than making them.

It buys two other things that matter more than the isolation:

* **The corpus is loaded once.** `SimServer` holds 1.18M products and the Lucene
  index; building one costs ~2 minutes and ~18 GiB. A sweep runs ten arms, and
  the frozen evaluation phase fans out across processes, so an env-per-worker
  design would pay that cost dozens of times over. Here every agent in every arm
  shares one `SimServer`, and only a `SimBrowser` (a few hundred bytes of session
  state) is per-agent.
* **Determinism across processes.** See `--seed` below -- this is not optional.

Endpoints (all JSON):

    GET  /health                        -> corpus/index identity + goal count
    GET  /goals?start=0&count=100       -> goal metadata, for manifest building
    POST /reset  {session, client}      -> {handle, observation, instruction, ...}
    POST /step   {handle, action}       -> {observation, reward, done, invalid}
    POST /close  {handle}               -> {}
"""

from __future__ import print_function

import argparse
import json
import os
import random
import sys
import threading
import time
import traceback
import uuid
from collections import OrderedDict

try:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
except ImportError:  # pragma: no cover - python 2 / very old 3
    raise SystemExit("this script needs python 3.7+")


# ------------------------------------------------------------------ scales
#: A corpus is only meaningful paired with the index built from it. Upstream
#: derives the index directory from `num_products` alone, so `None` means
#: "`indexes`" whether that directory was built from 1.18M products or 1,000 --
#: and a mismatched pair does not raise, it just returns search hits for products
#: the catalogue does not contain. Naming the pairs here makes the choice one
#: word (`--scale full`) that cannot be half-applied.
SCALES = {
    "full": {
        "file": "items_shuffle.json",
        "attr": "items_ins_v2.json",
        "index": "indexes_full",
    },
    "small": {
        "file": "items_shuffle_1000.json",
        "attr": "items_ins_v2_1000.json",
        "index": "indexes_1k",
    },
}


def ensure_jvm_on_path():
    """Point pyjnius at the JDK that ships inside this conda env.

    Importing WebShop pulls in pyserini, which is a JVM wrapper. pyjnius locates
    the JVM through JAVA_HOME and, failing that, by looking for `javac` on PATH;
    with neither set it dies as `Exception: Unable to find javac` from inside an
    import chain that never mentions Java. The env has openjdk 11, but conda
    installs it under `$PREFIX/lib/jvm` rather than at `$PREFIX`, so deriving
    JAVA_HOME from `sys.prefix` alone also fails.
    """
    jdk = os.path.join(sys.prefix, "lib", "jvm")
    libjvm = os.path.join(jdk, "lib", "server", "libjvm.so")
    if not os.path.isfile(libjvm):
        return  # a system JDK may already be on PATH; let pyjnius decide
    os.environ.setdefault("JAVA_HOME", jdk)
    os.environ.setdefault("JVM_PATH", libjvm)
    os.environ["PATH"] = os.path.join(jdk, "bin") + os.pathsep + os.environ.get("PATH", "")


def build_env_factory(repo, scale, seed, human_goals, observation_mode):
    """Import WebShop with the corpus selected, and return (make_env, server)."""
    ensure_jvm_on_path()
    cfg = SCALES[scale]
    data = os.path.join(repo, "data")
    os.environ["WEBSHOP_FILE_PATH"] = os.path.join(data, cfg["file"])
    os.environ["WEBSHOP_ATTR_PATH"] = os.path.join(data, cfg["attr"])
    os.environ["WEBSHOP_INDEX_DIR"] = cfg["index"]
    sys.path.insert(0, repo)

    # Seed BEFORE the import-and-construct, and do it in every process that ever
    # builds a SimServer. Upstream seeds `random` only just before shuffling the
    # goal list (`random.seed(233)`), which is after two draws that change what
    # the tasks *are*: `generate_product_prices` assigns every product a
    # `random.uniform` price, and `get_human_goals` samples each goal's
    # "price lower than X dollars" clause from that price. Unseeded, two server
    # processes disagree about both the instruction text and the price component
    # of the reward for the same goal index -- so an arm evaluated on server A
    # would not be solving the same tasks as an arm on server B, and memory
    # written during evolving would refer to prices that no longer hold. Nothing
    # in the run would look wrong.
    random.seed(seed)

    from web_agent_site.envs import WebAgentTextEnv  # noqa: E402

    t0 = time.time()
    env = WebAgentTextEnv(
        observation_mode=observation_mode,
        num_products=None,          # the whole catalogue; WEBSHOP_INDEX_DIR picks the index
        human_goals=human_goals,
        session_prefix=None,
    )
    sim_server = env.server
    print(
        "[webshop] loaded scale=%s products=%d goals=%d in %.0fs"
        % (scale, len(sim_server.all_products), len(sim_server.goals), time.time() - t0),
        flush=True,
    )

    def make_env(session_prefix):
        # `server=` is what makes this cheap: the new env reuses the loaded
        # catalogue and Lucene searcher and allocates only a SimBrowser.
        return WebAgentTextEnv(
            observation_mode=observation_mode,
            num_products=None,
            human_goals=human_goals,
            server=sim_server,
            session_prefix=session_prefix,
        )

    return make_env, sim_server


# ------------------------------------------------------------------ sessions
class SessionPool:
    """Live agent sessions, each an env sharing the one loaded SimServer.

    One global lock, deliberately. WebShop's step path runs a Lucene query
    through pyjnius, renders a Jinja template and parses the result with
    BeautifulSoup, over data structures that upstream never intended to be
    touched concurrently -- and `SimServer.receive` mutates `user_sessions`
    in place while it does. The ALFWorld adapter in this repo already paid for
    the lesson that a vendored env's thread-safety cannot be assumed: two
    threads produced `IndexError: pop from empty list` deep inside a parser,
    with nothing in the traceback naming concurrency, and a sequential run never
    reproduced it. A step costs tens of milliseconds against an LLM call of
    seconds, so serialising them is nearly free; if it ever stops being free,
    run several server processes instead of loosening this lock.
    """

    def __init__(self, make_env, max_sessions=4096):
        self._make_env = make_env
        self._lock = threading.Lock()
        self._envs = OrderedDict()
        self._max = max_sessions

    def reset(self, goal_index, client):
        # The handle is the WebShop session id, and it must be unique per live
        # rollout: SimServer keys `user_sessions` by it and *reuses the existing
        # goal* when the id is already present. Using the goal index alone as the
        # id -- the obvious reading of upstream's `reset(session=int)` -- would
        # make two workers evaluating different tasks collide the moment their
        # indices matched. A per-client prefix keeps the id unique while
        # `session_int` still selects the goal.
        prefix = "%s-" % client
        with self._lock:
            env = self._envs.pop(prefix, None)
            if env is None:
                env = self._make_env(prefix)
            self._envs[prefix] = env
            while len(self._envs) > self._max:
                old_prefix, old_env = self._envs.popitem(last=False)
                self._drop(old_prefix, old_env)
            observation, _ = env.reset(session=int(goal_index))
            handle = env.session
            goal = env.server.user_sessions[handle]["goal"]
            return {
                "handle": handle,
                "client": prefix,
                "observation": observation,
                "instruction": env.instruction_text,
                "available_actions": env.get_available_actions(),
                "goal_index": int(goal_index),
                "goal": _goal_summary(goal),
            }

    def step(self, client, action):
        prefix = "%s-" % client
        with self._lock:
            env = self._envs.get(prefix)
            if env is None:
                raise KeyError("unknown client %r; call /reset first" % client)
            before = env.get_available_actions()
            invalid = not _action_is_available(env, action, before)
            observation, reward, done, _ = env.step(action)
            return {
                "observation": observation,
                "reward": float(reward or 0.0),
                "done": bool(done),
                "invalid": invalid,
                "available_actions": env.get_available_actions(),
            }

    def close(self, client):
        prefix = "%s-" % client
        with self._lock:
            env = self._envs.pop(prefix, None)
            if env is not None:
                self._drop(prefix, env)

    def _drop(self, prefix, env):
        # SimServer.user_sessions is never pruned upstream, so a long sweep
        # would accumulate one dict per rollout forever. Small, but this is a
        # process that stays up for a day.
        try:
            for key in [k for k in env.server.user_sessions if k.startswith(prefix)]:
                env.server.user_sessions.pop(key, None)
            env.close()
        except Exception:  # noqa: BLE001 - teardown must not take the server down
            traceback.print_exc()


def _action_is_available(env, action, available):
    """Would WebShop actually act on this, or silently no-op?

    `WebAgentTextEnv.step` falls through to `status = dict(reward=0, done=False)`
    for an unparseable action or an unknown clickable, returning the *unchanged*
    page. To an agent that is indistinguishable from an action that legitimately
    changed nothing, so the distinction is computed here and reported as
    `invalid`, letting the caller say so explicitly.
    """
    from web_agent_site.engine.engine import parse_action

    name, arg = parse_action(action)
    if arg is not None:
        arg = arg.lower()
    if name == "search":
        return bool(available.get("has_search_bar")) and bool(arg)
    if name == "click":
        return arg in [c.lower() for c in available.get("clickables", [])] and arg != "search"
    return False


def _goal_summary(goal):
    """Only the fields a manifest or a scope key needs -- never the answer.

    `goal` also carries `asin`, `attributes` and `goal_options`, which are the
    grading key. Shipping those to the agent's process would put the answer one
    careless log line away from the prompt.
    """
    return {
        "category": goal.get("category", ""),
        "product_category": goal.get("product_category", ""),
        "query": goal.get("query", ""),
        "instruction_text": goal.get("instruction_text", ""),
    }


# --------------------------------------------------------------------- http
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    pool = None
    meta = None

    def log_message(self, fmt, *args):
        pass  # one line per env step would bury the sweep logs

    # -- helpers --
    def _send(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _dispatch(self, route, payload):
        if route == "/health":
            return dict(self.meta, status="ok")
        if route == "/goals":
            start = int(payload.get("start", 0))
            count = int(payload.get("count", 100))
            return {"goals": self.server.goal_slice(start, count)}
        if route == "/reset":
            return self.pool.reset(payload["session"], payload.get("client") or uuid.uuid4().hex)
        if route == "/step":
            return self.pool.step(payload["client"], payload["action"])
        if route == "/close":
            self.pool.close(payload["client"])
            return {}
        raise LookupError(route)

    def do_GET(self):  # noqa: N802
        path, _, query = self.path.partition("?")
        payload = {}
        for part in query.split("&"):
            if "=" in part:
                k, v = part.split("=", 1)
                payload[k] = v
        self._handle(path, payload)

    def do_POST(self):  # noqa: N802
        try:
            payload = self._read_json()
        except Exception as exc:  # noqa: BLE001
            self._send(400, {"error": "bad json: %s" % exc})
            return
        self._handle(self.path, payload)

    def _handle(self, route, payload):
        try:
            self._send(200, self._dispatch(route, payload))
        except LookupError:
            self._send(404, {"error": "no such route: %s" % route})
        except KeyError as exc:
            self._send(400, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            # Return the traceback rather than only logging it: the caller is in
            # a different process and a different interpreter, and a bare 500
            # there is unattributable.
            traceback.print_exc()
            self._send(500, {"error": "%s: %s" % (type(exc).__name__, exc),
                             "traceback": traceback.format_exc()})


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, handler, sim_server):
        ThreadingHTTPServer.__init__(self, addr, handler)
        self._sim = sim_server

    def goal_slice(self, start, count):
        goals = self._sim.goals[start : start + count]
        return [dict(_goal_summary(g), index=start + i) for i, g in enumerate(goals)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=7000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--repo", default=os.environ.get(
        "WEBSHOP_REPO", "/gpfs/radev/scratch/cohan/jw3278/webshop_repo"))
    ap.add_argument("--scale", choices=sorted(SCALES), default="full")
    ap.add_argument("--observation-mode", default="text",
                    choices=("text", "text_rich", "html", "url"))
    ap.add_argument("--human-goals", type=int, default=1,
                    help="1 = the human-written instructions used by the WebShop paper")
    ap.add_argument("--seed", type=int, default=42,
                    help="seeds product prices and goal price bounds; must match across servers")
    args = ap.parse_args()

    make_env, sim_server = build_env_factory(
        args.repo, args.scale, args.seed, bool(args.human_goals), args.observation_mode
    )
    Handler.pool = SessionPool(make_env)
    Handler.meta = {
        "scale": args.scale,
        "index": SCALES[args.scale]["index"],
        "corpus": SCALES[args.scale]["file"],
        "n_products": len(sim_server.all_products),
        "n_goals": len(sim_server.goals),
        "human_goals": bool(args.human_goals),
        "observation_mode": args.observation_mode,
        "seed": args.seed,
    }
    httpd = Server((args.host, args.port), Handler, sim_server)
    print("[webshop] serving on http://%s:%d  %s"
          % (args.host, args.port, json.dumps(Handler.meta)), flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
