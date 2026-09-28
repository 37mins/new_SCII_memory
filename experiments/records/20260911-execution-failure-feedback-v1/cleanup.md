# 本次失败产物与临时目录清理

用户授权后于2026-09-11 17:42–17:43北京时间执行。仅处理本record的43短RL
A/B r1和44 supply-order r1；没有启动、重试实验或修改训练机制。

| 节点 | 实际清理 | 释放的已分配空间 |
| --- | --- | --- |
| 43 | A失败step4的4个不完整model分片，无optimizer/extra/data完整集合 | 13.15 GiB本地磁盘 |
| 43 | `sc2_feedback_rl4_A_0911_r1`、`sc2_feedback_rl4_B_0911_r1`临时目录 | 1.83 GiB tmpfs |
| 44 | `sc2_supply_order_0911_r1`临时目录 | 83.17 MiB tmpfs |

43的5个B遗留SC2进程已逐一确认PPID=1、可执行文件和精确tempDir，使用pidfd发送
SIGTERM；随后确认全部退出，未用全局pkill或ray stop。其他历史任务进程不动。
删除前核对`.done`终态、4个分片大小及SHA256与失败记录一致，并检查目标不存在
活跃进程引用。分片永久删除、不可恢复；它们本来就不是可续训的完整checkpoint。

Ray诊断日志、临时SC2数据等已归档并完整读回校验，只有可再生成的编译缓存不备份。
归档和收据在各节点的
`/root/new-scii-execution-feedback-20260911/cleanup-20260911-r1/`。
精确路径、字节数、分片和归档SHA见[43收据](cleanup-node43.json)、[44收据](cleanup-node44.json)。
原始模型、完整checkpoint、源码/deployment、已有日志、回放、结果和失败`.done`
全部保留；原结果内的分片哈希代表删除前状态，不再代表文件仍存在。

清理后43本地盘可用约272GiB，43/44八卡均无显存占用（本次快照）。
临时目录释放不等于已解释或修复957GB主机内存峰值；未调整内存保护、环境或并行预算。
