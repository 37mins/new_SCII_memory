# C1缓存清理结果

2026-09-15 09:33:27–09:33:36 北京时间，sc2-a100-43，执行退出码0。
用户在确认C1归属后批准整理；授权只覆盖C1，不包括其他checkpoint候选或停止进程。

删除31个旧run临时目录内84个HF/Triton/Inductor缓存子目录，共141,605个普通文件。
实际释放5,057,613,824 bytes（4.7103GiB）；`/dev/shm`可用空间由414.7359增至
419.4462GiB，增量与删除分配块一致。这不是系统盘、共享盘或GPU显存释放。
缓存没有逐字备份，不能原样撤销删除，但可由保留的模型源码/环境重新生成。

保留了31个父目录和全部非缓存同级条目；C2–C4、模型、checkpoint、日志、回放、
运行部署未删除或修改。24个旧SC2进程仍引用的三个排除目录完整保留，执行前后引用
PID/路径相同。八个他人GPU任务PID 678703–678710前后均在，停止进程数为0。
本次没有启动或重试实验，不能据此宣称B的主机OOM已解决。

## 执行证据

固定清单为上级 [inventory JSON](../storage-cleanup-candidates-20260915.json) 的
`cache` 字段，SHA-256 `d443210112b3a85de4cb82dfa1b9b624e9995c19ef2b4e71753e646057370740`。
其`retirement_candidates`不在授权范围。原盘点JSON不改写，以保留操作前状态。

[cleanup_c1.py](cleanup_c1.py)只接受该哈希清单。先核对进程cwd/exe、打开文件、映射及
命令行的匹配路径，只保存匹配项、不输出完整参数；对缓存仅查元数据，不读取内容。
拒绝链接、挂载子树、checkpoint或回放等非缓存资产；删除前重查引用及逐项inode/大小/
mtime，使用不跟随软链接的目录删除。审计和执行均84项通过、0项跳过。

服务器独立凭据目录：`/root/sc2-c1-cleanup-20260915.82Zo1S`。

| 文件 | SHA-256 |
| --- | --- |
| `cleanup_c1.py` | `1df96607c7ede11d6892092f2dd9548a165251dc3de867344c44cd6569b63e22` |
| `c1-audit.json`，含清理前逐文件元数据 | `61728264519de8ed049527aa91e1994333920b9dbdc8251246212d3f80301530` |
| [receipt.json](receipt.json)，服务器名`c1-audit.receipt.json` | `20c1ebfb5cf7f5e7d56b24b07fed34fe6dbdff6c8028e43fae85352111bc4490` |

服务器另留`c1-audit.deleted.jsonl`逐目录删除流水。脚本仅为本次操作凭据，不是通用
自动清理工具，不应重复运行。已执行命令为`python3 cleanup_c1.py audit <清单> <审计>`，
人工查看审计后再执行同路径的`apply`，均退出0。

安全回归：`python3 -m unittest discover -s docs/operations/cache-cleanup-20260915 -p 'test_*.py'`，
4项通过：精确allowlist、篡改清单拒绝、缓存变化/软链接识别、模型资产拒绝。
本次仅提交运维脚本与凭据，不变更训练源码。GitHub同步仍待明确权限，未声称已推送。
