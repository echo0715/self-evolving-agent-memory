# Memory System Design: Reflection / Rule / Skill

本文档只定义 plan.md 中三种 memory system 的具体设计（Reflection Memory、Rule Memory、Procedural Skill Bank）。
Raw Trajectory Replay 只作为 baseline，在第 1 节里给出它必须遵守的同一套接口，不单独设计。

设计的第一原则：**三个 system 共用同一个 agent、同一个 retrieval 机制、同一个 injection 预算、同一套写入机制，唯一变化的是 memory item 的内容形态和 writer prompt。**
否则 Memory Content 那张表比的就不是 content，而是实现差异。

这一原则最容易被违反的地方是**写入机制**——见 §1.2，它是本文档第一版存在的一个真实 confound。

---

## 0. 术语与整体循环

一次 self-evolving 实验 = **Evolving 阶段** + **Evaluation 阶段**。

```
Evolving 阶段（构建 memory，重复 N 个 task）:
  for task t in D_evolve:
      M_ret   = Retrieve(M, t)                 # 可选，见 §5.3
      rollout = Agent(t, M_ret)                # K 次 rollout
      cand    = Writer(rollout, M_ret)         # 产出候选 memory item
      M       = Update(M, cand)                # append / merge / revise / delete

Evaluation 阶段（memory 冻结，只读）:
  for task t in D_test:
      M_ret = Retrieve(M, t)
      score = Agent(t, M_ret)
```

- `Agent`：被评测的 policy model（Qwen3.5-9B / Gemma），三个 system 完全相同，prompt 只有 memory block 不同。
- `Writer`：写 memory 的模型（Qwen3.5-9B / GPT5.5 / Qwen3.5-27B），是 Memory Writing Model 那组实验的自变量。
- `M`：memory store，一个 item 列表 + 一个 embedding index。
- Evaluation 阶段严禁写入。所有表格里的数字都是 test set 上的 success rate（AppWorld 另报 TGC/SGC）。

---

## 1. 统一的 Memory Item 接口

三个 system 的 item 共享同一个外壳，只有 `content` 字段的 schema 不同。这样 retrieval、injection、去重、统计代码只写一份。

```python
@dataclass
class MemoryItem:
    id: str                     # uuid
    type: str                   # "reflection" | "rule" | "skill" | "raw"
    content: dict               # type-specific schema，见 §2/§3/§4

    # ---- retrieval ----
    retrieval_key: str          # 用于 embed 的一段文本，由 content 拼出（各 type 定义不同）
    embedding: list[float]      # embed(retrieval_key)
    scope: dict                 # {"env": "alfworld", "task_type": "pick_and_place", ...}

    # ---- provenance（Memory Source 实验必须记）----
    source_task_ids: list[str]
    source_outcome: str         # "success" | "failure" | "mixed"
    writer_model: str
    created_at_step: int        # 第几个 evolving task 时创建
    updated_at_step: int
    version: int                # 每次 revise/merge +1
    superseded_by: str | None   # 被 merge 掉时指向新 id（软删除，便于分析）

    # ---- 使用统计（用于 utility-based 淘汰与分析）----
    n_retrieved: int
    n_retrieved_success: int    # 被检索且该 task 成功的次数
```

**Raw Trajectory baseline** 也用这个壳：`type="raw"`，`content = {"task": ..., "trajectory": [...]}`，`retrieval_key = task instruction`，写策略固定为 append-only、只存成功轨迹（标准做法，如 ExpeL/Synapse）。它不做 merge/revise/delete。

### 1.1 Injection 预算（关键的公平性控制）

三个 system（以及 raw baseline）在 evaluation 时注入 agent context 的 memory block **token 数上限相同**，记为 `B`（建议 ALFWorld/WebShop `B=1500`，AppWorld `B=2500`）。

- 主实验用 **equal token budget**：按 retrieval score 降序填，填不下的丢弃。
- 附录再报一组 **equal item count**（`k=3`）的结果。
- 这一点必须写进论文，否则 raw trajectory 天然吃掉几千 token，比较无效。

### 1.2 写入机制必须与内容形态正交（第二个公平性控制）

本文档的第一版给每种 content 配了不同的写入机制：Rule 有 support/refute 验证回路，Skill 有跨任务批量归纳，Reflection 两样都没有。这样一来三者同时差两件事：

| | 单条写入 | 验证回路 | refinement | 跨任务批量归纳 | utility 淘汰 | grounding 校验 | n_max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Reflection（旧） | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ | 3 |
| Rule（旧） | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | 2 |
| Skill（旧） | ✓ | ✗ | 只有 step 级 | ✓ | ✓ | ✗ | 1 |

那么 Skill 赢了，说明不了是 procedural content 更好，还是"writer 一次看 25 个 episode"更好；Rule 赢了也分不清是规则形态好还是多了一轮验证。**这是 confound，不是设计。**

所以把写入机制从 content type 里抽出来，做成一组对所有类型统一生效的开关 `WritePolicy`：

```
online_write            每个 episode 从当前 rollout 抽取
merge_on_duplicate      重复候选走 LLM merge（否则直接丢弃）
grounding_check         evidence 必须能在轨迹中找到
verify                  episode 结束后判定被注入条目：support / refute
refine                  反例累积后重写该条目
delete_on_low_confidence  confidence 崩掉就删
utility_deletion        检索多但极少伴随成功就删
batch_induction         每 E 个 episode 做一次跨任务归纳
n_max                   每 episode 最多写几条
```

配套的三点改动：

1. **`evidence` 字段加到全部三种 schema**。原来只有 Reflection 有 evidence 和幻觉校验，等于只有它被质量过滤。现在三者都要写 evidence（来自轨迹的原文片段），走同一个 grounding 校验；evidence 不注入 agent prompt，所以不占 injection 预算。
2. **`support / refute / confidence` 从 rule 的 content 里挪到 item 外壳上**。验证是机制不是规则的属性，一旦对所有类型生效，它的记账就该属于所有类型。
3. **三种 writer 看到完全相同的 rollout**。原来 Rule writer 看全部 rollout，Skill writer 只看一条成功的——这是输入信息不对称，和机制不对称一样是 confound。现在统一为：mixed 给 success+failure 各一条，全成功给最短的成功，全失败给最长的失败。

**实验里怎么用：**

- **Memory Content 主表**：policy 固定（推荐 `WritePolicy.full()`，即全开），只换 content type。这才是在比 content。
- **Write Mechanism 消融**（新增的一组，建议进论文）：content type 固定，只开关 policy。这一组能直接回答"验证回路值不值那些额外 token""跨任务归纳的收益来自哪里"。
- **`NATIVE_POLICIES`**：旧的按类型分配机制的设定保留下来，作为"每种 content 配它最自然的机制"这一条对照，但**不能**用来填 Memory Content 表。

剩余的、无法消除的不对称有两处，如实写进论文即可：

- Skill 无法从纯失败的 episode 写出条目（procedure 需要至少一条走通的路径）。这是内容形态本身的性质，而 §6.3 的 rollout 过滤实验正是为暴露它设计的。
- Raw baseline 没有 LLM writer，因而不可能有验证/归纳。它是 baseline，不是机制比较里的一方。

---

## 2. Reflection Memory

> 定位：从单个（或一对成功/失败的）rollout 里抽出 **"这次为什么成 / 为什么败，下次该注意什么"** 的自然语言经验。粒度最细、最贴近具体任务、抽象度最低。

### 2.1 Schema

```yaml
type: reflection
content:
  situation:   str   # 什么任务/什么情境下发生的，1 句
  lesson:      str   # 可迁移的教训，1-2 句，祈使句
  rationale:   str   # 为什么这条成立（因果解释），1-2 句
  evidence:    str   # 来自轨迹的具体锚点：关键 action / observation 片段，≤2 行
  outcome_tag: str   # "from_success" | "from_failure" | "from_contrast"
retrieval_key: situation + " " + lesson
```

约束：整条 ≤120 token。`evidence` 必须是轨迹里真实出现过的字符串片段（写入时做子串校验，校验失败则丢弃该候选——这是抑制 writer 幻觉的主要手段）。

**示例（ALFWorld）**

```yaml
situation: 需要 heat 一个物体但手上已经拿着另一个物体时
lesson: 在 go to microwave 之前先确认手上拿的就是目标物体，否则先 put 掉再 take 目标物体
rationale: 环境只允许单手持物，heat 作用于手上物体，拿错会 heat 失败且不报错
evidence: "> heat tomato 1 with microwave 1  |  Nothing happens."
outcome_tag: from_failure
```

### 2.2 Writer 输入与 prompt 结构

Writer 看到：任务指令 + rollout（做 state abstraction 后的 observation–thought–action 序列）+ 最终 reward + （若可得）同一 task 的一条成功与一条失败轨迹。

三种触发模式，对应 `outcome_tag`：

| 模式 | 输入 | Writer 被要求做的事 |
|---|---|---|
| from_failure | 一条失败轨迹 | 定位**第一个**导致不可恢复的错误 step，写出该错误的可迁移教训 |
| from_success | 一条成功轨迹 | 写出这条轨迹里**非平凡**的决策（"任何 agent 都会做的事"不写） |
| from_contrast | 成功+失败各一条 | 对齐两条轨迹，找到第一个分叉点，解释分叉处的正确选择 |

Prompt 骨架（三个 system 的 writer prompt 都用这个骨架，只换中间的 schema 和 instruction）：

```
[ROLE] 你是一个 memory writer，为 {env} 环境中的 agent 提炼可复用经验。
[INPUT] task / trajectory(ies) / outcome / 已检索到的现有 memory M_ret
[TASK]  产出 0~{n_max} 条 reflection。宁缺毋滥；若本轨迹无可迁移信息，输出空列表。
[SCHEMA] {yaml schema}
[CONSTRAINTS]
  - lesson 必须能迁移到别的 task instance，不能包含具体物体实例名之外的一次性细节
  - evidence 必须逐字来自 trajectory
  - 不得与 M_ret 中已有条目重复；若是对已有条目的修正，改用 REVISE 操作并给出 target_id
[OUTPUT] JSON: {"ops": [{"op": "APPEND"|"REVISE"|"DELETE", "target_id": ..., "content": {...}}]}
```

`n_max = 3`（Reflection）。

### 2.3 Update 策略

机制本身由 §1.2 的 policy 决定，对三种 content 一视同仁；这里只说它在 reflection 上的实例化。

- **APPEND**：新候选与库中同 scope item 的 embedding 最大余弦相似度 `< τ_dup`（建议 0.85）时直接入库。
- **MERGE**：`≥ τ_dup` 时，把新旧两条交给 writer 做一次 merge call，输出一条合并后的 reflection（保留两边 evidence，取更一般的 lesson），旧条目标记 `superseded_by`。
- **REVISE**：writer 显式指定 `target_id` 时，覆盖 content，`version += 1`。用于"上次总结错了"的情况。REVISE 后 `support/refute` 清零——改写过的条目是一条新主张，旧证据不再适用。
- **DELETE**：三个来源——(a) writer 在 contrast 模式下判定某条 lesson 与新证据矛盾；(b) confidence 崩掉（§3.3 的判据，现在对 reflection 同样生效）；(c) §5.4 的 utility-based 淘汰。
- **verify / refine**：§3.3 描述的验证回路对 lesson 同样适用——问题从"这条规则被遵循了吗"变成"这条 lesson 适用吗、跟着它做有没有帮上忙"，judge 的四个选项一字不改。
- **batch induction**：§4.2 的跨任务归纳对 lesson 同样适用——把一批任务里换了说法的同一条教训并成一条更一般的 lesson，并挖出单条轨迹里看不出来的重复失败模式。

Reflection 天然会膨胀，所以它对 §5.4 的容量控制最敏感，这点在论文里值得单独讨论。

### 2.4 Injection 格式

```
## Past experience (lessons)
1. [when 需要 heat 物体但手上已有别的物体] 先确认手持物体是目标物体，否则先 put 再 take。
   why: 环境单手持物，heat 作用于手上物体。
2. ...
```

`evidence` 默认**不注入**（省 token 且它主要用于写入期校验），只在消融时打开。

---

## 3. Rule Memory

> 定位：把经验压成**带适用条件的原子性原则**，`trigger → action/avoid → exception`。比 reflection 更抽象、更短、更可组合，但丢掉了因果叙述。

### 3.1 Schema

```yaml
type: rule
content:
  trigger:    str        # 适用条件，"When ..."，必须可从当前 observation 判断
  directive:  str        # 该做什么 / 该避免什么，祈使句，单一动作
  polarity:   str        # "do" | "avoid"
  exception:  str | null # 不适用的情形
  confidence: float      # 0~1，见 §3.3
  support:    int        # 支持该规则的 episode 数
  refute:     int        # 反例 episode 数
retrieval_key: trigger + " " + directive
```

约束：`trigger + directive + exception` 合计 ≤50 token；**一条规则只能有一个 directive**（writer 若产出复合动作，拆成多条）。

**示例（WebShop）**

```yaml
trigger: When the search results page shows no item matching all attribute constraints
directive: Relax the least-specified attribute and re-search instead of clicking the top result
polarity: do
exception: If fewer than 3 results are returned, click into the top result first to read its options
confidence: 0.78
support: 7
refute: 2
```

### 3.2 Writer 输入与 prompt

输入同 Reflection（rollout + outcome + M_ret），但 instruction 强调：

- trigger 必须是 agent 在决策时**当场可判定**的谓词（可以从 observation / 上一步动作反馈里读出来），禁止使用"当任务很难时"这种不可判定条件。
- directive 必须是环境 action space 内的动作或对动作的选择约束。
- 显式要求 writer 先输出 `candidate_trigger` 的自检："这个 trigger 在多少比例的 step 上会命中？若接近 100%，说明太泛，重写。"
- `n_max = 2`。

### 3.3 Update 策略（统计式 revise：这里定义，但对三种 content 都生效）

Rule 的核心不是"多写"，而是**在后续 episode 里被验证/证伪**。下面这套回路最早是为 rule 设计的，但它不依赖 rule 的任何字段，所以按 §1.2 已经推广到 reflection 和 skill，由 `policy.verify / refine / delete_on_low_confidence` 控制。

1. **APPEND**：新 rule 与已有 rule 的 trigger embedding 相似度 `< τ_dup`（0.88）→ 入库，`support=0, refute=0, confidence=0.5`。注意是 0 不是 1：support 计的是**被验证过的证据**，不是"被写出来"这个动作本身，否则同一条 rule 被反复提出就能白拿 support，且 refinement 永远触发不了。
2. **支持/反驳更新**：每个 evolving episode 结束后，对本次**被检索注入**的每条 rule，让 writer 做一次轻量判定：
   - `followed_success`：agent 遵循了该 rule 且 task 成功 → `support += 1`
   - `followed_failure`：agent 遵循了该 rule 但 task 在该 rule 相关的 step 上失败 → `refute += 1`
   - `not_applicable` / `violated`：不变（前者说明 trigger 没出现，后者说明 agent 压根没照做，两种情况都没有关于这条规则本身的证据）
   `confidence = (support + 1) / (support + refute + 2)`（Laplace 平滑）。
   这四个选项对 lesson 和 procedure 一字不改地适用，这正是它能推广的原因。
3. **REVISE**：当 `refute ≥ 2` 且 `confidence < 0.5` 时，触发一次 refinement call：writer 拿着该 rule + 所有反例摘要，选择 (a) 收窄 trigger、(b) 补一条 exception、(c) 判定该 rule 无效。version += 1，`support/refute` 重置为 refinement 之后的计数。
4. **MERGE**：trigger 相似且 directive 一致 → 合并，support/refute 相加。trigger 相似但 directive 冲突 → **不合并**，交给 writer 判断是否是缺失的 exception，通常应转成一条带 exception 的规则。
5. **DELETE**：refinement 判定无效，或 `confidence < 0.3 且 support+refute ≥ 4`。

注意：这个 support/refute 机制会额外消耗 writer 调用。为公平起见，在 Memory Writing Model 实验中记录并报告每个 system 的 **writer token 成本**，作为附表。policy 统一之后这项成本在三种 content 之间已经基本可比（同一 policy 下差异来自条目长度，不再来自机制多少）。

补充一点 refine/merge 的 grounding：refinement 和 merge 的 prompt 里**没有轨迹**，所以这两处一律沿用原条目的 `evidence`，不接受 writer 新写的 evidence——否则会出现一条 evidence 无从校验的条目。

### 3.4 Injection 格式

按 confidence 降序，紧凑列表：

```
## Rules
- When <trigger>, DO <directive>. (unless <exception>)
- When <trigger>, AVOID <directive>.
```

confidence 不注入（避免 agent 把它当作可以忽略的理由），只用于排序和淘汰。

---

## 4. Procedural Skill Bank

> 定位：把**多条**经验整理成可复用的多步 workflow/SOP。粒度最粗，一条 skill 覆盖一类子任务的完整解法。

### 4.1 Schema

```yaml
type: skill
content:
  name:          str        # 动词短语，如 "heat_object_and_place"
  trigger:       str        # 何时使用这条 skill（任务级条件）
  preconditions: list[str]  # 执行前必须成立的状态
  steps:         list[str]  # 有序步骤，每步是一个动作或一小段动作，含参数占位符 <obj>/<recep>
  verification:  list[str]  # 每个关键步骤后如何确认成功（观察到什么）
  fallback:      list[str]  # verification 失败时的补救动作，与 steps 索引对齐或用 "if ... then ..."
retrieval_key: name + " " + trigger + " " + join(steps[:3])
```

约束：`steps` 长度 3~12；步骤里必须用占位符而不是具体实例名（`take <obj> from <recep>` 而不是 `take tomato 1 from countertop 1`）。整条 ≤400 token。

**示例（AppWorld）**

```yaml
name: resolve_entity_id_before_mutation
trigger: 任务要求对某个命名实体（联系人/歌单/订单）执行修改或删除
preconditions: ["已登录对应 app", "掌握该实体的自然语言名称"]
steps:
  - 调用该 app 的 search/list API，用名称做过滤，拿到候选实体列表
  - 若候选 >1，用任务中的附加约束（时间/所有者/状态）消歧
  - 用消歧后的唯一 id 调用 mutation API
  - 重新读取该实体，确认字段已变更
verification:
  - "search 返回非空且字段与任务描述一致"
  - "mutation API 返回 2xx 且 response 中 id 与步骤 2 一致"
  - "重新读取的字段等于目标值"
fallback:
  - "若 search 为空，改用分页 list 全量拉取后本地匹配"
  - "若候选仍 >1 且无法消歧，向 supervisor 提问而不是随机选一个"
```

### 4.2 Writer 输入（批量归纳：这里定义，但对三种 content 都生效）

Skill 不应该从单条轨迹现编。写入触发方式：

- **成功即抽取（online）**：一条成功轨迹 → 抽出一条 candidate skill（草稿，`version=1`）。
- **批量归纳（每 E 个 evolving task 触发一次，建议 `E=25`）**：把该批次内**同一 task cluster**（用 task instruction embedding 做贪心聚类 / 用 ALFWorld 自带 task type）的所有轨迹（成功 + 失败）连同现有 skill 一起给 writer，要求：
  - 抽公共骨架 → 更新 `steps`
  - 从失败轨迹里抽 `verification` 和 `fallback`（失败轨迹的主要价值在这里）
  - 合并近似 skill

批量归纳最早是为 skill 设计的，但"跨任务看一批轨迹再下结论"和 content 形态无关，所以按 §1.2 它同样对 reflection（把换了说法的同一条教训并成一条更一般的）和 rule（优先保留被多个任务支持的规则、收窄被批次证伪的 trigger）生效，由 `policy.batch_induction / batch_every` 控制。

**这一点是 §1.2 那个 confound 里最重的一项**：writer 一次看 25 个 episode 和只看 1 个，本身就是巨大的信息差。归纳时的 grounding 校验对整批轨迹的并集做，而不是只对第一条。

### 4.3 Update 策略

- **APPEND**：新 skill 与已有 skill 的 `trigger` 相似度 `< τ_dup`（0.90）→ 入库。
- **MERGE**：相似度高时，writer 做 skill merge：steps 取最长公共骨架 + 把差异部分下沉为 fallback / 条件分支；`source_task_ids` 求并集。
- **REVISE**：来自 §3.3 的通用 refinement 回路（反例累积后收窄/加 fallback/判无效），与另外两种 content 相同。
- **DELETE**：`n_retrieved ≥ 5` 且 `n_retrieved_success / n_retrieved < 0.2` → 淘汰（该 skill 起负作用）。这条 utility 判据现在对三种 content 都生效。

**Skill 独有的一项额外机制（默认关闭）**：某条 skill 被检索注入但 task 失败，且失败发生在该 skill 的某一步 → 记录 `failed_step_idx`，累计 2 次同一步失败则触发 step 级 refinement。它对另外两种 content 没有对应物，开了就重新引入不对称，所以默认关；而且它需要 harness 提供"失败发生在第几步"的归因（目前从 `episode.meta["failed_skill_steps"]` 读，没有自己推断）。要用的话在论文里单独说明。

### 4.4 Injection 格式

```
## Applicable procedure: <name>
Use when: <trigger>
Preconditions: ...
Steps:
  1. <step>   [check: <verification>]  [if fails: <fallback>]
  2. ...
```

因为一条 skill 就很长，`B=1500` 下通常只能注入 1~2 条；这本身是一个真实的 trade-off，不要为了塞更多而给 skill 加预算。

---

## 5. 三者共享的机制（必须完全一致）

除本节列的检索/容量机制外，§1.2 的整套写入机制同样属于"必须完全一致"的范畴——那是本文档第一版漏掉的部分。

### 5.1 Embedding 与检索

- Encoder：统一用一个固定的 sentence embedding 模型（如 `gte-modernbert-base` 或 `bge-m3`），全程不变、不 finetune。
- 相似度：cosine over `retrieval_key`。
- Query：test/evolving task 的 instruction（AppWorld 额外拼上 supervisor 的 app 列表）；不使用当前 step 的 observation，保持 **task 级一次性检索**，避免"检索时机"成为混淆变量。
- 打分：`score = cos_sim`（主设置）。附录消融 `score = cos_sim + λ·utility`，`utility = n_retrieved_success / (n_retrieved + 1)`。
- 取 top-`k_pool = 10` 候选，再按 §1.1 的 token 预算填充。

### 5.2 Scope 过滤

检索前按 `scope.env` 硬过滤；ALFWorld 额外可按 task type 过滤（作为消融，主设置不开，避免用上 oracle 信息）。

### 5.3 Evolving 阶段是否检索

**开启**（`M_ret` 注入给 agent，也给 writer）。理由：这才是"self-evolving"——后写的 memory 建立在前面 memory 之上，也才能让 REVISE/DELETE 有触发机会。
需要在论文中明确：这引入了 evolving 顺序依赖，因此**所有实验固定同一个 task 顺序 seed**，并跑 3 个 seed 报均值±std。

### 5.4 容量控制

统一上限：`|M| ≤ 200` items（AppWorld 300）。超出时按下式淘汰最低分者：

```
utility = (n_retrieved_success + 1) / (n_retrieved + 2)
priority = utility  * (1 + log(1 + n_retrieved))      # 用过且有效的优先保留
```

从未被检索过的 item（`n_retrieved = 0`）在超限时优先淘汰**最旧**的。

Rule 额外用 `confidence` 作为一票否决（见 §3.3）。

### 5.5 "With All" 配置

Memory Content 表里的 `With all` 行：三个 store 各自独立维护（writer 每个 episode 依次跑三次写入），检索时**各取各的**，注入预算按 `reflection : rule : skill = 0.3 : 0.2 : 0.5` 切分 `B`。
必须注意：这一行的 writer 调用成本是单一 system 的 3 倍，所以如果它赢了，要在分析里区分"是内容互补带来的收益"还是"更多 writer 计算带来的收益"——建议补一个 **单 system + 3× writer 采样** 的对照。

---

## 6. 与 plan.md 中三组实验的接口

| 实验组 | 自变量 | 在本设计中改什么 | 保持不变 |
|---|---|---|---|
| Memory Content | item 类型 | §2/§3/§4 整块切换 | Agent、Writer 模型、retrieval、`B`、**WritePolicy**、`D_evolve` |
| Memory Writing Model | Writer | `MemoryItem.writer_model` 对应的调用模型 | Agent 恒为 Qwen3.5-9B、schema、WritePolicy |
| Memory Source | `D_evolve` 的构造 | 见下 | 其余全部 |
| **Write Mechanism（新增）** | WritePolicy 开关 | §1.2 的 policy 逐项开关 | content 类型固定、Agent、Writer、retrieval、`B`、`D_evolve` |

Memory Content 那张表必须注明用的是哪个 policy（推荐 `full`，附录再给 `minimal` 的一版）。两个 policy 下 content 的排序若不一致，本身就是论文里值得写的发现：说明"哪种 content 更好"取决于配多少写入机制。

Write Mechanism 这组建议至少跑（三种 content × ALFWorld 或 WebShop 之一，100 tasks）：

| 设置 | 说明 |
| --- | --- |
| minimal | 只有 online write + append/merge |
| + verify | 加验证回路（不加 refine），看单纯的 confidence 排序有没有用 |
| + verify + refine | 加 refinement |
| + batch induction | 在 minimal 基础上只加跨任务归纳，与 verify 分离 |
| full | 全开 |

`+ batch induction` 与 `+ verify` 分开跑很关键：前者给的是**信息**（一次看 25 个 episode），后者给的是**反馈**（后续 episode 的验证），两者带来的收益性质不同，混在一起就说不清楚了。

Memory Source 三个子设置的具体接口：

1. **重复演化**（`50×3 / 75×2 / 150×1`）：同一 task 的第 2、3 次演化时，`M_ret` 中会命中上一次写入的 item → 主要触发 REVISE/MERGE 而非 APPEND。**记录 append/merge/revise/delete 四种操作的次数分布**，这是这组实验最有价值的中间量。
2. **难度过滤**：用 GPT 跑一遍 task，按 trajectory length 分位数切 easy（<P33）/ hard（>P67）。注意 trajectory length 与失败相关，所以 hard 子集里失败率天然更高——报告两个子集的 GPT success rate 以便读者校正。
3. **rollout 结果过滤**：每个 task 跑 `K=4` 次 rollout，按结果分桶为 all-failure / all-success / mixed，各取 100 个 task 构成 `D_evolve`。`mixed` 桶正好是 Reflection 的 `from_contrast` 和批量归纳最能发挥的地方——这是一个可预注册的假设。另外 all-failure 桶下 Skill 写不出任何条目（procedure 需要一条走通的路径，见 §1.2 末尾），这不是 bug 而是这组实验要暴露的东西，报表时把 skill 的 store size 一起给出。

---

## 7. 需要记录的日志（跑实验前就要埋好）

每个 evolving step 落一条 JSONL：

```json
{"step": 17, "task_id": "...", "rollout_rewards": [0,1,0,0],
 "retrieved_ids": [...], "injected_tokens": 1421,
 "writer_ops": [{"op":"APPEND","id":"..."},{"op":"REVISE","id":"...","reason":"..."}],
 "writer_prompt_tokens": 3200, "writer_completion_tokens": 210,
 "store_size": 63}
```

有了它，Memory Source 那组表的分析（操作分布、memory 增长曲线、被检索 item 的 utility 分布）才不用重跑实验。

---

## 8. 实现顺序建议

1. `MemoryItem` + store + embedding index + injection budget（§1、§5.1、§5.4）——三个 system 共用，先做完。
2. Reflection（最简单，append/merge 为主）→ 打通 ALFWorld 全流程，验证 evaluation 不写入、日志完整。
3. Rule（多一个 support/refute 回路）。
4. Skill（多一个批量归纳触发器）。
5. Raw trajectory baseline（最后做，只是一个特例配置）。
6. 扩到 WebShop、AppWorld：只需换 `scope`、action space 描述、以及各 schema 里的示例。
