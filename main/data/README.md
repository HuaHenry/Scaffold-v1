# 独立数据流程

- `generate.py`：只生成/导入数据；按任务支持CPU进程并行。
- `storage.py`：NPY内存映射读写；03压缩帧流式导入，动作按公共编码转成uint8图。
- `audit.py`：完整冻结文件哈希、shape、任务注册和源码身份校验。

## 入口

在源码目录 `main` 用 `bash generate_data.sh --help` 查看全部参数。从项目根目录可直接：

```bash
python -m main.data.generate --help
python -m main.data.audit --data /路径/冻结数据版本
```

`--episodes`必填，`--tasks all`默认17任务；`--val-episodes`默认32，与当前03真实完整验证集一致；新任务 `--test-scenes`默认128，`--seed`默认20261113，`--workers`默认1。T01测试保留全部512场景，`--test-scenes`不截断它。训练/验证上限读取实际NPZ数组头，不使用旧summary的陈旧shape字段。

数据结果建议写入 `experiments/main/datasets/<版本>/`；整个 `main/` 为源码树，禁止数据或训练结果写入。

## 数据版本结构

```text
<版本>/
  request.json
  generation.lock
  manifest.json
  complete.json
  <任务ID>/
    train/{state,process,partial,shuffle}_{fg,bg,act}.npy
    validation/state_{fg,bg,act}.npy
    train_scenes.json
    validation_scenes.json
    test_scenes.json
    bench.npz
    task_complete.json
```

训练数组为 `[N,64,28,56] uint8`，动作图使用公共127-token编码。phase由条件和片段位置确定，不跨操作重置：state/partial每集56条操作转移，process/shuffle每集31形成＋28操作转移。

bench字段：`ctx_fg/ctx_bg`为 `[M,28,56]`；`act`为 `[M,2,32,28,56]`；`target_fg/target_bg`为 `[M,2,28,56]`；`scene_ids/group`为 `[M]`。统一预测第32步。T01保存原四个group，其余任务group为0。

## T01来源

默认 `experiments/preresearch/03_main/20261111-044057_four_conditions`。首先重算03输入SHA1并匹配原审计，输出不与来源或数据路径重叠。生成请求绑定源audit的SHA256、输入路径和输入哈希，同一路径更换审计或输入版本会拒绝恢复；导入前后再校验源审计。帧直接从NPZ流式取前N集；不是复制旧checkpoint，也不是重新生成不同考题。动作使用相同编码，原有编码限制不被伪称已修复。

## 冻结与中断

只有 `complete.json`存在且验证通过的整体数据版本才可训练。每任务先提交 `task_complete.json`，全部任务成功后提交manifest与整体标记。已提交任务哈希变化即停止。未绑定非空目录拒绝写入；只有匹配 `request.json` 的恢复目录可以保留本生成器的manifest及临时文件。最终提交中断时重新核验全部任务并恢复同一manifest，不重生成任务；只清理根目录已绑定生成器的原子提交临时文件。未提交任务目录仍不会自动删除，需选择新版本，或由用户自行明确处理残留。

数据生成不调用训练和评测模型，不自动安装依赖。完整哈希审计需读取全部NPY文件，TB级数据可能耗时较长。内存映射减少内存峰值，不减少磁盘量；请先dry-run检查规模。
