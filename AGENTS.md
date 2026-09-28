# AGENTS.md — SC2 线性注意力项目协作与实验规范

本文件适用于本仓库根目录及全部子目录。后续 agent、开发者和实验执行者在开始任务前必须先阅读本文件；子目录可以用更具体的 `AGENTS.md` 增补规则，但不得放宽本文件中的可追溯性、安全性和实验完整性要求。

## 1. 项目目标与首要原则

本项目的核心目标是：在公平、可复现的条件下，使线性/混合注意力架构的大模型在 StarCraft II 任务上稳定超越对应的 Qwen 基线，并尽可能证明优势来自模型架构，而不是不对称的外部工具、提示词、执行器或实验预算。

所有工作必须服务于四个首要原则：

1. **可控**：一次只引入有限、明确的变化；代码、配置和运行状态容易理解。
2. **可追溯**：任何结果都能追溯到代码、配置、模型、环境、数据和随机种子。
3. **可回滚**：正式实验只从 Git commit 启动；只有大版本或高风险迭代使用独立 branch/worktree；旧结果不覆盖。
4. **最小实现**：优先实现当前目标所需的最小闭环，不提前建设未经验证的通用框架。

实验可信度优先于短期胜率。不能解释来源的涨分不作为架构证据。

## 2. 每个任务开始前

开始任何修改或服务器操作前，依次执行：

1. 阅读本文件以及目标目录中更具体的 `AGENTS.md`、README 和相关设计文档。
2. 使用 `git rev-parse --show-toplevel` 确认仓库根，不依赖当前 shell 路径猜测。
3. 检查 `git status --short --branch`、当前 branch、HEAD 和已有未提交修改。
4. 保留用户已有修改；不覆盖、不清理、不回退来源不明的工作树变化。
5. 明确本次任务属于代码、文档、实验、诊断还是服务器运维，并只做完成该任务所需的动作。
6. 涉及历史实验时，先定位 experiment record、effective config、deployment 和 run；不得只看仓库当前源码推断历史运行时。
7. 涉及 A100 时，先进行只读健康检查和任务归属检查，再决定是否需要写操作。

如果任务会启动高成本训练、占用多卡、停止进程、修改共享运行时、移动大量资产或产生难以恢复的外部状态，必须先取得用户明确授权。

## 3. 新项目与旧项目的边界

本仓库是新项目的开发入口。旧 SC2 项目是只读证据库和可选择复用的组件来源，不是新项目的可直接修改工作树。

从旧项目复用代码时必须：

1. 记录来源路径、Git SHA 或 deployment 标识、许可证和迁移理由。
2. 一次只迁移一个可独立验证的组件。
3. 为迁移组件补充测试和本仓库文档。
4. 不连同无关历史兼容层、实验脚本和隐式配置整体复制。
5. 通过适配器隔离第三方或旧项目依赖，适配器保持薄且直接。

在 A100 上首次部署新项目时，必须选择与旧项目根目录分离的新路径，并在 `docs/operations/server-layout.md` 记录。未经用户确认，不得自行把新仓库部署进旧项目的 `git`、`deployments`、`experiments` 或 `runs` 目录。

## 4. Coding 规则

### 4.1 最小实现

- 实现当前需求的最小正确闭环，避免为假设中的未来需求增加扩展点。
- 不过度防御性编程。只在真实边界上校验：用户输入、外部服务、文件格式、模型输出和跨进程协议。
- 不为内部不变量层层重复校验，不吞掉异常，不使用无边界的自动重试或静默 fallback。
- 错误应尽早、明确地暴露，并包含足以定位问题的上下文，但不得泄露密钥。
- 没有至少两个明确调用场景时，不提前抽象通用框架。
- 优先使用清晰函数和显式数据流，不滥用继承、注册器、工厂、全局状态、元编程或多层 wrapper。
- 不创建无边界的 `utils.py`、`common.py` 或“万能 Manager”。共享逻辑应按具体职责命名。
- 不以复制代码代替合理复用，但也不为消除少量重复引入复杂抽象。
- 不在同一个变更中顺手重构无关模块。

### 4.2 结构与依赖

仓库结构必须让新成员能快速回答：入口在哪里、配置在哪里、核心协议在哪里、实验记录在哪里、结果在哪里。

优先采用以下职责边界，按实际需要渐进创建，不为空目录搭架子：

```text
src/                    核心代码
configs/                版本化配置
scripts/                薄启动脚本和维护工具
tests/                  单元与小型集成测试
docs/                   分层文档
experiments/records/    实验定义、审计与结论
```

- 依赖方向必须简单、单向、可说明。
- SC2 环境、观测协议、模型策略、动作协议、执行器、奖励和实验编排应保持可辨识边界。
- Planner 是 Core 对比中唯一计划决策来源；Executor 负责执行，不得悄悄注入战略判断。
- 第三方依赖固定到明确版本或 commit，并记录来源。不要无理由跟随 `main`。
- 模型、checkpoint、replay 和大体积运行产物不提交到普通 Git 历史；Git 中保存内容哈希、位置和 provenance。

### 4.3 代码与文档同步

所有非平凡功能必须有配套文档。文档至少说明：

- 功能解决什么问题；
- 公开入口和最小用法；
- 必要配置及默认值；
- 输入、输出和失败行为；
- 与实验结果相关时，哪些指标或协议会受影响。

修改行为、配置、协议或启动方式时，必须在同一变更中更新对应文档。禁止代码已经改变而 README、usage、实验说明长期失真。

### 4.4 测试

- 测试规模与风险匹配，不为了覆盖率数字制造脆弱测试。
- 观测/动作 schema、解析器、奖励、run manifest 和实验状态机必须有单元测试。
- 外部 SC2、模型服务和分布式训练使用小型集成测试或 smoke test，不在普通单元测试中启动重型依赖。
- 修复 bug 时，优先加入能复现该 bug 的最小回归测试。
- 测试结果必须报告实际执行命令和结果；没有运行的测试不得描述为通过。

## 5. 文档层次

### 5.1 README

根 `README.md` 必须简洁，不罗列工程细节。它只承担以下职责：

1. 一段项目目标和当前状态；
2. `Quick Start` / `Usage`：最短可运行路径；
3. `Experiment Guideline`：如何启动一个受控实验；
4. 关键文档索引；
5. 已知限制或当前支持范围。

README 中不得堆放完整服务器清单、模块实现细节、长配置解释、实验流水账或历史结论。细节必须下沉到专门文档，并由 README 链接。

### 5.2 指南、设计与参考

按需使用以下层次：

```text
docs/getting-started/        安装、首次运行、最小实验
docs/guides/                 面向任务的操作指南
docs/architecture/           系统边界、数据流、模型与执行器设计
docs/reference/              配置、schema、接口和指标定义
docs/operations/             A100、部署、运行、恢复和排障
docs/decisions/              重要架构/协议决策记录
experiments/records/<id>/    单次实验定义、审计、结果与结论
```

- 指南回答“怎么做”，架构文档回答“为什么这样设计”，reference 精确定义接口。
- 一个事实应有一个权威位置，其他文档通过链接引用，避免多处复制后漂移。
- 时间敏感状态必须注明核验时间，不把一次服务器检查写成永久事实。
- 重要协议变化应新增 decision record，不静默改写历史理由。
- 文档应随项目推进及时更新；过期文档应修正、标记 deprecated 或删除链接，不能继续误导。

## 6. Git、branch 与回滚

### 6.1 基本规则

- `main` 是默认开发线，始终保持可安装、可测试、可运行。小实验 record、README/文档、配置补充和局部低风险修复直接形成原子 commit 并推送到 `main`。
- branch 是大版本隔离工具，不是每个任务或每次实验的默认容器。不得仅因为“这是一个正式实验”就新建 branch。
- 不在共享服务器的同一个工作树中并行开发和运行正式实验。
- 正式实验必须从 clean working tree 的已提交 SHA 启动；启动器应拒绝 dirty tree，或明确把 dirty diff 保存进 manifest 并将运行标记为非正式。
- commit 保持原子性：一个 commit 只表达一个可解释的变化。
- 不重写共享 branch 历史，不对用户已有提交执行 force push、破坏性 rebase 或 reset。
- `main` 上已完成且验证通过的小任务应作为交付流程的一部分推送到约定的 `origin`；正式实验前必须确保所用 commit 已安全保存在远端。force push、推送到其他远端或发布外部 artifact 仍需明确授权。

### 6.2 只有大版本或高风险变更才新建 branch/worktree

以下任一情况才需要新建 branch；只有确实需要与主线并行开发或冻结运行时，才再创建独立 worktree：

- 大规模迭代、in-place 重写或仓库目录重组；
- 同时修改多个核心子系统；
- 修改观测协议、动作协议、reward、Planner/Executor 职责或训练语义；
- 升级关键模型、训练框架、推理引擎或 SC2 依赖；
- 为正式实验修改共享核心代码，且变更可能破坏当前基线或需要与主线长期并行；
- 预计需要多轮失败尝试或可能影响现有基线。

branch 默认采用 `codex/<type>-<short-slug>`，例如：

```text
codex/feature-action-schema-v1
codex/fix-reward-ledger
codex/refactor-executor-boundary
```

以下工作默认直接在 `main` 完成，不新建 branch/worktree：

- 新增一个只组合现有组件的 experiment record、effective config 或薄 launcher；
- 更新实验状态、结果、结论、README 或其他文档；
- 增加分析脚本、回归测试或局部低风险 bug fix；
- 不改变 Observation/Action/reward/Executor/训练语义的小范围参数实验。

同一方法和协议下的扩量、增加步数、换 seed、重复运行或 smoke → full
晋级，必须复用当前 experiment branch 和原 experiment record，只增加新的
`RUN_ID`、effective config 修订与结果段落；不得仅因数据量或运行预算变化新建
branch 或平行 record。只有方法、共享核心代码、协议或训练语义发生实质变化时，
才升级为新的 `EXPERIMENT_ID` 或 branch。

直接在 `main` 工作仍必须保持原子 commit、测试通过和 clean SHA；实验运行由 commit 与唯一 `RUN_ID` 冻结，而不是靠长期保留 branch 冻结。任务完成后及时删除已经合并的临时 branch 和无用 worktree。原则上除 `main` 外只保留正在开发的大版本分支；历史发布点使用 annotated tag，不用常驻 branch 充当标签。

### 6.3 基线与发布点

- 可复现基线使用 annotated tag 标记，例如 `baseline/qwen-core-v1`。
- 每次协议破坏性变化递增明确版本，不复用旧版本名。
- rollback 通过切换到已知 commit/tag 和对应配置完成，不通过手工“改回大概原样”。

## 7. 实验生命周期与记录

### 7.1 证据对象

新项目沿用并强化以下对象分工：

| 对象 | 作用 | 修改规则 |
|---|---|---|
| Git source | 可开发源码 | 小变更直接在 `main`；大版本在 branch/worktree |
| experiment record | 问题、假设、配置、启动器、审计、结论 | 受控、版本化修改 |
| deployment | 某次实验的冻结运行时 | 创建后只读 |
| run | 日志、checkpoint、replay、metrics、状态 | 唯一目录、只追加、不覆盖 |

判断一次实验实际运行内容的优先级是：

```text
experiment record
  -> effective config / launcher
  -> frozen deployment / source SHA
  -> run manifest / audit / done state
  -> Git provenance
```

仓库当前文件不能代替历史实验 provenance。

### 7.2 启动前必须记录

每个正式实验必须有唯一 `EXPERIMENT_ID` 和 `RUN_ID`。推荐：

```text
EXPERIMENT_ID=YYYYMMDD-<short-purpose>
RUN_ID=<experiment-id>-<model>-s<seed>-r<replicate>
```

启动前的 experiment record 至少记录：

- 状态：`planned`；
- 问题和可证伪假设；
- 与基线相比唯一计划变化；
- 对照组和消融组；
- 主指标、次指标和统计方法；
- 训练/评测数据、地图、难度和种子；
- 资源预算、预计运行时间和停止条件；
- 代码 SHA、branch/tag 和 dirty 状态；
- 模型、tokenizer、checkpoint 的标识与哈希；
- 依赖锁、容器 image digest、SC2 build 和地图哈希；
- 观测、动作、prompt、reward、executor 的版本或哈希；
- 完整 effective config；
- launcher 和唯一 run 输出目录；
- 外部服务依赖，但不包含任何密钥值。

命令行 override 必须合并并保存为 effective config，不能只存在于 shell history。

### 7.3 运行中

- 将状态更新为 `running`，记录开始时间、节点和 GPU 分配。
- 日志、模型输出、环境动作、reward ledger、replay 和 checkpoint 写入该 run 的唯一目录。
- 不边跑边修改 source、deployment、配置或 run 中已有文件。
- 发现代码或配置错误时，停止或让当前运行安全结束，将其标为 `invalid`，记录原因；修复后使用新 commit 和新 `RUN_ID`。
- 训练进程失联、存储故障和代码错误必须区分记录，不把基础设施失败写成模型失败。

### 7.4 结束后

必须显式记录：

- 状态：`completed`、`failed`、`invalid` 或 `aborted`；
- 结束时间、退出码和 `.done`/等价状态文件；
- 原始结果位置和关键 artifact 哈希；
- 预先定义指标及置信区间；
- 与对照组的差异；
- 结论、限制和是否支持原假设；
- 下一步建议。

失败和无提升实验同样保留。不得删除失败记录来美化项目历史。

### 7.5 实验公平性

用于证明架构优势的 Core 实验中，Qwen 与线性/混合注意力模型必须共享：

- 相同观测和动作协议；
- 相同 prompt、约束解码和最大生成预算；
- 相同 Executor、微操和规则系统；
- 相同数据、reward 和环境交互预算；
- 相同评测地图、对手、种子和统计方法；
- 相等的超参数搜索预算。

参数量、训练 FLOPs、推理吞吐、显存和上下文/state 策略必须显式报告。外部记忆、世界模型、verifier、专家提示或额外工具放入独立 System 实验，并同时提供给两个模型。

## 8. A100 连接与只读检查

连接信息核验基准日期为 2026-08-15；节点健康和运行中服务是时间敏感状态，每次使用前必须重新检查。

### 8.1 SSH 入口

- 跳板机：`sc2-a100-jump`，仅用于跳转，不运行训练。
- 共享存储节点：`sc2-a100-42`、`43`、`44`、`46`、`47`、`48`。
- 独立存储节点：`sc2-a100-197`、`198`。
- 所有计算节点都通过配置好的 `ProxyJump sc2-a100-jump` 访问。
- 日常操作只使用 SSH 别名，不在仓库文档或脚本中复制公网地址、私钥路径细节或凭据。
- 不关闭 `StrictHostKeyChecking`；host key 变化时先让管理员确认。

连接后先执行只读检查：

```bash
hostname
date
nvidia-smi
tmux ls 2>/dev/null || true
pgrep -af 'raylet|gcs_server|main_ppo|SC2_x64|vllm|api_server' || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

没有确认任务归属前，不终止进程、tmux、Ray、容器或服务。

### 8.2 两种存储布局

共享节点 42/43/44/46/47/48 使用：

```text
/mnt/data0601/yangjinyue/sc2-verl-agent-experiments
```

访问前使用带超时的检查：

```bash
timeout 5s stat /mnt/data0601/yangjinyue/sc2-verl-agent-experiments
```

若返回超时，不继续运行无超时的递归 `find`、`du` 或 `df`。NFS 不可读不等于 SSH、GPU 或节点故障。

独立节点 197/198 使用：

```text
/data/yangjinyue/sc2-spb2-level7-20260814
```

197/198 不使用 `/mnt/data0601`。两台节点的容器和运行状态不得互相假设；启动前分别检查 Docker、代码、模型、SC2 资产和挂载。

### 8.3 服务器写操作

- 长任务放入 `tmux` 或受控调度器，不依赖前台 SSH 会话。
- 上传先进入明确的 staging 或新项目目录，先核验目标路径，再移动到最终位置。
- 不直接修改冻结 deployment、既有 run、共享环境或来源不明的工作树。
- 不执行归属不明的 `kill`、`pkill`、`ray stop --force`、`docker rm`、递归删除或清理命令。
- 正式运行使用唯一目录和可检查的完成状态；不得复用旧 run 目录。

## 9. 旧项目位置与 provenance

### 9.1 共享目录节点

旧项目统一根：

```text
/mnt/data0601/yangjinyue/sc2-verl-agent-experiments
```

关键目录：

```text
experiments/records/       实验定义、配置、launcher、审计和结论
deployments/               冻结运行时，只读
git/sc2-verl-agent-experiments/  可开发 Git 仓库
git/worktrees/             历史独立 worktree
runs/                      日志、checkpoint、回放和完成状态，只追加
assets/                    模型、checkpoint、SC2 与复现资产
envs/                      Python 环境
staging/                   受控暂存
secrets/                   敏感配置；不得读取、复制、展示或提交
```

旧共享仓库及历史实验可能来自不同 branch、isolated tree 或 deployment。必须从目标实验 record 反向追踪，不能把当前 Git HEAD 当作所有实验的源码权威。

### 9.2 独立存储节点

旧 Level7 项目统一根：

```text
/data/yangjinyue/sc2-spb2-level7-20260814
```

关键位置：

```text
code/              旧可开发仓库；可能存在未提交补丁
model/             本地 SpB2 Planner 模型
SpikingBrain2.0/   模型源码与内核
StarCraftII/       SC2 程序
data/sc2/          RL/SC2 数据
runs/              历史运行结果，只追加
launch_level7.sh   旧顶层启动器
```

使用前必须同时保存并检查：

```bash
git -C /data/yangjinyue/sc2-spb2-level7-20260814/code rev-parse HEAD
git -C /data/yangjinyue/sc2-spb2-level7-20260814/code status --short
git -C /data/yangjinyue/sc2-spb2-level7-20260814/code diff --stat
```

HEAD 不能代表未提交修改，目录名也不能代表运行已经成功。

## 10. 安全与敏感信息

- 不读取、打印、复制、提交或在回复中展示私钥、密码、API Key、W&B Token、服务凭据或 `secrets/` 内容。
- 仅记录环境变量名，例如 `SC2_QWEN_API_KEY`，不记录值。
- 凭据不得出现在命令参数、shell history、进程列表、普通日志、Markdown、Git diff 或 experiment manifest。
- 不把 `.env` 密钥文件、SSH 私钥或服务配置复制到仓库和 run 目录。
- 发现凭据泄露时停止传播，报告泄露位置，并建议管理员轮换；不要在清理过程中重复显示密钥。
- 首次主机指纹、变更后的 host key、陌生密钥或异常权限必须由管理员确认，不猜测处理。

## 11. 完成定义与交接

代码或功能任务只有在以下条件满足后才算完成：

1. 最小需求已经实现，没有夹带无关重构。
2. 相关测试实际运行并记录结果；无法运行的部分明确说明。
3. 配套文档、usage 和配置说明同步更新。
4. `git diff` 已复核，不包含密钥、生成物或用户无关修改。
5. 交接信息说明当前 branch、commit/dirty 状态、修改文件、验证命令、已知限制和下一步。

实验任务只有在 experiment record、effective config、run provenance、结果和结论都已保存后才算完成。仅有日志目录、截图、口头胜率或 W&B 页面不构成完整实验记录。

服务器诊断任务必须区分已验证事实、时间敏感快照和推断；不得把 SSH/NFS/容器/GPU/训练进程故障混为一类。

## 12. 规则维护

- 仓库结构、实验协议、服务器布局或安全流程发生变化时，及时更新本文件和对应专门文档。
- 本文件只保存长期规则和稳定入口；具体实现细节放入 `docs/`，单次实验细节放入 `experiments/records/`。
- 修改本文件属于治理变更，必须说明原因，并检查是否与 README、operations 文档和实验模板一致。
- 本文件依据 2026-08-15 的《A100服务器代码位置索引》和《A100服务器连接指南》编写；其中节点健康、容器和运行服务状态每次操作前重新核验。

## 13. 当前整合边界与晋级

2026-09-06 用户要求将已验证修复纳入主线，默认升级为 Integrated Fixes v1，
具体范围与验证层级以 `docs/integrated-fixes-v1.md` 为准。旧 Canonical r6/r7
由原 commit/实验记录保留，不因主线更新而重标为新结果。
默认路径是 `scripts/train_sc2.sh`，通用入口选择 Qwen3-4B-Instruct-2507；
SpB 入口默认关闭 SWA dropout，两者默认关闭 per-call state shaping。
`configs/system-v1.sha256` 冻结本版已登记源码；变更默认配置必须同步该 manifest、
测试和整合说明。单项通过、整合通过、长训通过必须分别报告，不能长期把已验证
候选误称为已在主线生效，也不能将未验证诊断带入默认训练。
Rush 必须通过 `BO_PROFILE=rush` 显式选择，不默认选择表现最好的 seed。
发布候选的 CPU/dry-run 通过不代表 GPU/SC2 通过，整合验证状态见
`experiments/records/20260906-integrated-fixes-v1/README.md`。
原有 v1524 微操实验保留在历史 records，不启用为默认微操。
