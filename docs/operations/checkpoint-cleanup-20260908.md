# 2026-09-08 退役短测 checkpoint 清理

用户授权清理本项目不再使用的旧权重。本次限定为 `sc2-a100-42` 内存盘中的两份
2026-08-20 SpB 两步短测，不按胜率高低删除实验，也不把成功短测改记成失败。

| RUN_ID | 原状态 | 清理范围 |
|---|---|---|
| `20260820-level5-attention-backend-ab-v1-spb-eager-smoke2-s0-r1` | driver exit 0，step 2 | 八份 model + 八份 optim 分片 |
| `20260820-level5-attention-backend-ab-v1-spb-canonical-eager-level6-smoke2-s0-r4` | driver exit 0，step 2 | 八份 model + 八份 optim 分片 |

运行根为 `/dev/shm/new_scii_level5_attn_ab_assets/runs`。两份均为已结束的旧
attention-backend smoke，不是 r6/r7 长训基线、r4 状态恢复来源或当前 g2 A/B 的
模型输入。当前 A/B 从共享原始 SpB instruct 读取权重，未恢复这些短测 checkpoint。
原始模型、其他 checkpoint、源码、环境、日志、回放、结果、配置、extra state 和
data.pt 均不在清理范围。没有核实到这两份分片的可用备份；清理后不能直接恢复
其训练状态，剩余 checkpoint 目录不能再视作完整 checkpoint。

执行前逐文件记录绝对路径、大小、SHA-256 与 inode/mtime；检查进程参数、cwd、
映射与打开文件是否引用目标 RUN。实际操作仅逐个 unlink 明确枚举的 32 个普通
文件，不递归删除目录，不跟随软链接，不停止任务。审计及删除回执保存在独立
`/root/sc2-checkpoint-cleanup-20260908.UbM4tY`，不改写原结果记录。

清理前 23:14 北京时间快照：42 主机可用内存约 51 GiB，`/dev/shm` 可用约
52 GiB；GPU0–3 为其他任务，4–7 空闲。存在旧的八卡等待队列，本次不终止或
启动它；它还要求整节点空闲且内存盘可用至少 300 GiB，不能仅凭空出四卡启动。
本次不恢复失败 A/B，不改训练配置或 Ray 内存阈值。

状态：**已完成**。2026-09-08 23:25:06–23:25:17 北京时间执行，退出码 0。
32 个 model/optim 分片共 `127107760864` 字节（118.378 GiB），文件分配空间
118.378 GiB，已全部移除。两份各保留八个 extra-state 分片、data.pt、模型配置、
effective config 与日志目录；原始结果、回放及其他目录不在 unlink 清单。

| 即时资源快照 | 清理前 | 清理后 |
|---|---:|---:|
| 主机 MemAvailable | 53,820,092,416 B（约50 GiB） | 180,984,938,496 B（约169 GiB） |
| /dev/shm 可用 | 55,208,730,624 B（约51 GiB） | 182,316,556,288 B（约170 GiB） |
| /dev/shm 使用率 | 90% | 67% |

系统磁盘仍剩约12 GiB：本次释放的是 tmpfs/主机内存，不是系统磁盘。GPU0–3
的其他作业仍在，4–7仍为空；未启动实验，旧等待队列未达到300 GiB门槛。
这只缓解了资源不足，不证明先前A/B启动问题或游戏效果已经通过验证。

逐文件 [manifest](checkpoint-cleanup-20260908/manifest.json) 和
[删除回执](checkpoint-cleanup-20260908/manifest.deleted.jsonl) 已取回。
[一次性脚本](checkpoint-cleanup-20260908/cleanup.py) 固定枚举两个RUN、32个路径；
先以 `audit` 模式核验退出码/无活跃引用并计算SHA-256，再以 `apply` 模式复核
引用及文件元数据、逐文件删除。实际CPU语法检查通过；audit、apply及删除后
不存在性/保留项复查均退出0。不涉及任何模型forward、PPO更新或运行代码修改。
