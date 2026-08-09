### Main paper
 Analyze How Different memory factors affect the self evolving agent system from several aspect, including what to write(memory content), who write it(memory writing model), base on what to write (Memory Source)

### Main paper
All memory writing can involve append, merge, revise, delete

1. Memory Content：

| Memory Strategy | Representative works | Explanation |
| --- | --- | --- |
| Raw trajectory replay | Synapse, ExpeL, AdaMEM, SkillEvolBench Raw-Trajectory baseline | 保存完整或经过state abstraction/压缩的历史 observation–thought–action trajectory，检索相似任务后作为few-shot exemplar注入 |
| Reflection memory | Reflexion, ExpeL, ReasoningBank, EvoSC, Trajectory-Informed Memory | 从成功、失败或success–failure contrast中总结自然语言lesson、failure cause、rationale和recovery insight |
| Rule memory | AgentEvolver, FORGE, ACE, SkillRevise Principle Memory, TACO | 将经验抽象成带适用条件的局部原则：`trigger/when → action/avoid → exception/constraint` |
| Procedural skill bank | Agent Workflow Memory, Mem(^{p}), ReMe, CODESKILL, SkillEvolBench | 把多条经验整理成可复用的多步workflow/SOP，通常包含trigger、preconditions、ordered steps、verification和fallback |
|  |  |  |
- Reflection：`lesson + rationale + evidence`
- Rule：`trigger + action/avoid + exception`
- Skill：`trigger + preconditions + ordered steps + verification + fallback`

1. Memory source
    
    repetitive evolving:
    
    1. 100 different
    2. 50 different evolving twice
    3. 33 different evolving triple times
2. Task filtering: run gpt on the tasks and check the trajectory length to determine difficulty. 
3.  Memory writing model selection:
4. agent rollout filtering：
    1. all failure
    2. all success
    3. have success half failure.

### Memory Content Study

Qwen3.5-9B ALFWorld: 

| Method | Evolving 50 tasks  | Evolving 100 tasks  | Evolving 150 tasks |
| --- | --- | --- | --- |
| Raw trajectory replay |  |  |  |
| Reflection memory |  |  |  |
| Rule Memory |  |  |  |
| Procedural skill bank |  |  |  |
| With all |  |  |  |

Qwen3.5-9B Webshop

| Method | Evolving 50 tasks  | Evolving 100 tasks  | Evolving 150 tasks |
| --- | --- | --- | --- |
| Raw trajectory replay |  |  |  |
| Reflection memory |  |  |  |
| Rule Memory |  |  |  |
| Procedural skill bank |  |  |  |
| With All  |  |  |  |

Qwen3.5-9B AppWorld

| Method | Evolving 50 tasks  | Evolving 100 tasks  | Evolving 150 tasks |
| --- | --- | --- | --- |
| Raw trajectory replay |  |  |  |
| Reflection memory |  |  |  |
| Rule Memory |  |  |  |
| Procedural skill bank |  |  |  |
| With All |  |  |  |

Maybe Try Gemma With the Same setting. 

|  |  |  |
| --- | --- | --- |
|  |  |  |
|  |  |  |

### Memory Writing Model

ALFWorld:

| Method | Evolving 100 tasks Qwen3.5-9B | Evolving 100 tasks GPT5.5 | Evolving 100 tasks Qwen3.5-27B |
| --- | --- | --- | --- |
| Reflection memory |  |  |  |
| Rule Memory |  |  |  |
| Procedural skill bank |  |  |  |

Webshop

| Method | Evolving 100 tasks Qwen3.5-9B | Evolving 100 tasks GPT5.5 | Evolving 100 tasks Qwen3.5-27B |
| --- | --- | --- | --- |
| Reflection memory |  |  |  |
| Rule Memory |  |  |  |
| Procedural skill bank |  |  |  |

Appworld

| Method | Evolving 100 tasks Qwen3.5-9B | Evolving 100 tasks GPT5.5 | Evolving 100 tasks Qwen3.5-27B |
| --- | --- | --- | --- |
| Reflection memory |  |  |  |
| Rule Memory |  |  |  |
| Procedural skill bank |  |  |  |

### Memory Source

Webshop:

| Method | Evolving 50 tasks 3 times | Evolving 75 tasks twice | Evolving 150 tasks once |
| --- | --- | --- | --- |
| Reflection memory |  |  |  |
| Rule Memory |  |  |  |
| Procedural skill bank |  |  |  |

Webshop:

| Method | Evolving 100 easy tasks | Evolving 100 hard tasks | Evolving 100 tasks  |
| --- | --- | --- | --- |
| Reflection memory |  |  |  |
| Rule Memory |  |  |  |
| Procedural skill bank |  |  |  |

AppWorld:

| Method | Evolving 100 failed tasks | Evolving 100 success tasks | Evolving 100 tasks  |
| --- | --- | --- | --- |
| Reflection memory |  |  |  |
| Rule Memory |  |  |  |
| Procedural skill bank |  |  |  |