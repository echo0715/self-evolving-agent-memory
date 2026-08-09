"""memsys -- memory systems for the self-evolving-agent study.

Implements the three memory systems from memory_strategy_design.md
(Reflection / Rule / Procedural Skill) plus the Raw Trajectory baseline, on a
shared store so that the Memory Content comparison is a controlled one.

    from memsys import build_system, Episode, Rollout, Step, Evolver

    system = build_system("rule", llm=OpenAIChatClient("qwen3.5-9b", base_url=...))
    ev = Evolver(system)
    ev.run(episodes)                       # evolving phase
    with frozen(system):                   # evaluation phase
        block = system.retrieve(task.instruction, scope).block
"""

from .config import NATIVE_POLICIES, MemoryConfig, WritePolicy
from .embedding import HashingEmbedder, SentenceTransformerEmbedder, cosine, default_embedder
from .episode import Episode, Rollout, Step, evidence_supported
from .item import MemoryItem
from .llm import CallableLLM, LLMClient, LLMResponse, OpenAIChatClient, ScriptedLLM, Usage
from .runner import Evolver, RunLogger, frozen
from .schemas import CONTENT_TYPES, ReflectionContent, RuleContent, SkillContent, get_type
from .store import MemoryStore, Scored, render_block
from .systems import (
    CompositeSystem,
    MemorySystem,
    RawTrajectorySystem,
    ReflectionSystem,
    Retrieval,
    RuleSystem,
    SkillSystem,
    WriteReport,
    build_system,
)
from .writers import (
    FOLLOWED_BAD,
    FOLLOWED_OK,
    NOT_APPLICABLE,
    VIOLATED,
    BaseWriter,
    Proposal,
    ReflectionWriter,
    RuleWriter,
    SkillWriter,
    Verdict,
    WriteOp,
    parse_ops,
)

__version__ = "0.1.0"

__all__ = [
    "MemoryConfig",
    "WritePolicy",
    "NATIVE_POLICIES",
    "Verdict",
    "NOT_APPLICABLE",
    "FOLLOWED_OK",
    "FOLLOWED_BAD",
    "VIOLATED",
    "MemoryItem",
    "MemoryStore",
    "Scored",
    "render_block",
    "Episode",
    "Rollout",
    "Step",
    "evidence_supported",
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "default_embedder",
    "cosine",
    "LLMClient",
    "LLMResponse",
    "ScriptedLLM",
    "CallableLLM",
    "OpenAIChatClient",
    "Usage",
    "ReflectionSystem",
    "RuleSystem",
    "SkillSystem",
    "RawTrajectorySystem",
    "CompositeSystem",
    "MemorySystem",
    "Retrieval",
    "WriteReport",
    "build_system",
    "ReflectionWriter",
    "RuleWriter",
    "SkillWriter",
    "BaseWriter",
    "WriteOp",
    "Proposal",
    "parse_ops",
    "CONTENT_TYPES",
    "ReflectionContent",
    "RuleContent",
    "SkillContent",
    "get_type",
    "Evolver",
    "RunLogger",
    "frozen",
]
