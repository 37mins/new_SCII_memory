# SC2 历史窗口核验（2026-09-06）

当前运行代码`1f2d8b5`与KL诊断代码`c49642b`都由启动器固定
`env.history_length=0`、`env.sc2_memory_mode=off`。本次只读核验及本地最小
复现没有修改生产代码、运行中配置或服务器部署；48新增实验尚未启动。

## 历史数字与证据边界

用户提供的`history_window_summary_updated.csv`和`history_window_all_step_metrics.csv`
位于本地`/Users/yangjinyue/Desktop/codex/星际争霸/history_window_outputs`。
使用Spreadsheets的只读流程逐run重算；0132、0133、0134各96条、step1–96无重复，
平均训练胜率分别0.8541666667、0.8684895833、0.87890625，5相对3差2.473958个百分点。
没有修改或导出原表。汇总/逐步CSV的SHA-256分别为
`8500ab9144243d4bde371635d829ca8e0fe64886e1b437a2060ab8410075b226`、
`f7feae472845ac074c90bc196f5852733a9306cb903ad70259dd6e97003cbff5`。

原队列`queue_gpua25_spb2_gigpo_history_window_medium_0132_0137_datafix1_20260624.sh`
明确将`0134 MediumHard 0 5`传给`/data1/yjy/verl-agent/run_spb2_gigpo_history_window_gpua25_20260624.sh`。
队列SHA-256为`402878d1aec11b9aee2be23b18cf30e21710404e5c240c927a30ada248dce351`。
该0134实际运行源码/冻结备份尚未定位；gpua28后续同类启动器不是0134冻结源码。
旧gpu163只读SSH因认证失败，未修改认证配置或尝试其他凭据。

7月20日归档的`02_core_source/agent_system/environments/env_package/sc2_online_restored/manager.py`
SHA-256为`cb6d64dda19a5f23f9b44381d3c4a08a92d9495f6254ba0b3bf090a249fa4e73`。
它的reset/step只传当前text_obs，没有历史读取，归档env_manager.py也路由到此类。
这不能替代6月0134运行证据，不能据此宣判旧实验无效；同样不能仅凭run名称
或history_length配置值断言历史确实进入了模型输入。

## 当前代码已复现的两个边界

`StrategicSC2EnvironmentManagerV2.__init__`在off模式设`self.memory=None`，
`build_text_obs`直接返回当前query，不调用history_length。直接执行该真实方法，
令history_length=5且memory=None，输出仍与当前query逐字相同。

validator的`_bo_candidate_context`及`_strict_legacy_bo_candidate_counts`使用
`text.split("The remaining BO list in this phase:", 1)[1]`提取第一段待办。
用既有测试中的真实validator最小复现：

```text
历史待办：marine x 1；本轮待办：barracks x 1
仅本轮query：- barracks -> True
历史完整query放前面：- barracks -> False
历史完整query放前面：- marine -> True
```

因此不能直接拼五段旧query后保持validator输入不分层。可用的上游GiGPO
`SimpleMemory`在`agent_system/memory/memory.py`，返回最近N条原始观测/动作；
尚未证明0134使用了它。当前`strategic_v2`额外加入规则诊断、动作建议和跨局
策略记录，也不等于纯五轮历史，不应作为静默替代。

当前prompt预算4096且truncation=error。完整五轮历史需先测真实token长度；
不能偷偷截断历史、提高预算或引入总结器后仍称只改了窗口。两次48共享盘/
tokenizer读取超时，尚无可信的五轮实际token计数；没有把预估写成测量结果。

下一步需确认0134实际历史实现，或明确采用现有SimpleMemory作为新窗口实验。
在独立候选中验证历史真正进入token输入、只按本轮待办验动作、reset清空历史、
各环境隔离及prompt预算后，才在48启动新RUN_ID。44两项冻结实验不原位改动。
