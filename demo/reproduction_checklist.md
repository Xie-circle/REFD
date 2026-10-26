# REFD 复现准备清单（第 4 周用）

> 整理日期：2026-10-24
> 用途：下下周（约 11-02 起）开始复现时，照着这份清单逐个搞定。

---

## 一、复现完整步骤（来自 README.md）

### RE 阶段（管真假）
```
cd code/ADD2023t3_RE
python preprocess_da_full.py    # ① 离线特征提取（wav2vec2 → .pt）
python main_train.py            # ② 训练 W2VAASIST（学"真实中心"prototype）
python generate_score_da.py     # ③ 生成 RE 分数 → result/result_RE
```

### FD 阶段（管新旧）
```
cd code/ADD2023t3_FD
python main_train.py            # ① 训练多分类模型（mixup 治过自信）
python ood_detector_pre.py      # ② OOD 预处理 → ood_step/traindict.pt + evaldict.pt
python ood_detector.py --ood_detector_name NSD   # ③ 算 NSD 分数 → ood_step/result_oodscore.pt
python generate_f1_e2e.py       # ④ 最终 F1（注意：第 17 行要调 OOD 阈值）
```

---

## 二、复现前必须完成的准备（按顺序）

### ① 数据集（已解压 ✅ 2026-10-26）
- 位置：`D:/REFD/data/ADD2023T3/Track3/`，分卷已合并解压，`test` 目录已重命名为 `eval`
- 结构：`train` 22400 条 / `dev` 8400 条 / `eval` 79490 条，每份含 `wav/` + `label.txt`（标签列是 0~7 数字）
- ⚠️ 关键发现：train/dev 的标签只有 **0~6**（7 类），eval 有 **0~7**（8 类）——**label 7 是训练时没见过的"新型伪造"**，正是 FD 阶段 OOD 要检测的目标

### ② wav2vec2 预训练模型（下载完成 ✅）
- 模型：`facebook/wav2vec2-xls-r-300m`（约 1.2GB）
- 本地：`D:/REFD/models/wav2vec2-xls-r-300m/`（config.json + preprocessor_config.json + pytorch_model.bin 已齐）

### ③ 改路径：服务器路径 → 本地路径（已完成 ✅ 2026-10-26）
实际改动记录：
- 数据集目录 `test` → 重命名为 `eval`（代码里第三段用 `part_='eval'`，避免改代码）
- `raw_dataset.py`（RE 和 FD 各一处）`ADD2023` 类：标签路径从 `label/{part}.txt` 改为直接读 `{part}/label.txt`（适配 zenodo 真实结构）
- `preprocess_da_full.py`：`/data2/Track3` → `D:/REFD/data/ADD2023T3/Track3`、特征路径 → `D:/REFD/data/ADD2023T3/preprocess_xls-r-5-1`、模型 → `D:/REFD/models/wav2vec2-xls-r-300m`、`CUDA_VISIBLE_DEVICES` 3→0
- `main_train.py`（RE）：path 默认值改本地 + **补上缺失的 `--access_type` 参数** + gpu 5→0
- `main_train.py`（FD）：path 默认值改本地 + gpu 1→0
- `generate_score_da.py`：特征路径 + gpu 5→0
- `ood_detector_pre.py`：特征路径 + gpu 1→0 + wav2vec2 从 `facebook/...` 改本地路径
- 遗留：`raw_dataset.py` 里的 `ADD2023_eval*` 类仍含旧路径，但**无任何地方调用**（死代码），不用改

### ④ 已知 bug（复现时需修）
- ✅ 已修：`main_train.py`（RE）缺 `--access_type` 参数（`args.access_type` 会报 AttributeError），已补上
- `generate_score_da.py`：`__main__` 里 `model_path` 和 `args.task` 都未定义（第 2 周发现）——跑到这步再补
- `ood_detector_pre.py`：`args.model_name` 未定义；`txt_file_name='/result/result_FD.txt'` 应改为 `./result/result_FD.txt`
- `main_train.py`（RE/FD）：`SubsetRandomSampler(range(116348))`、`range(8400*5)` 是**硬编码的增强后样本数**（= 原样本 ×5 种增强），和"是否做增强"强相关——决定增强策略后要一起改
- ⚠️ 数据增强依赖外部数据集：`preprocess_da_full.py` 的 `add_rev/add_speech/add_music/add_noise` 需要 RIRS_NOISES + MUSAN（本地没有），见下方"待决策"

### ⑤ 环境：torch 换 GPU 版（⚠️ 版本要选对）
- 现在 d2l-zh 环境是 torch 2.4.1 **CPU 版**，`cuda_available=False`，必须换
- 显卡：RTX 5060 Laptop 8GB，驱动 592.01（CUDA 13.1）
- ⚠️ **RTX 5060 是 Blackwell 架构（sm_120），torch 2.4.1+cu121 不支持**（会报 no kernel image）；需装 **torch 2.7+ / cu128**
- 建议：`pip install torch==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128 --proxy http://127.0.0.1:7892`
- 依赖已装齐：transformers / librosa / soundfile / tqdm（torch 升级后需确认 transformers 兼容）

---

## 三、已确认 + 待决策（2026-10-26 更新）
- ✅ 已确认：标签文件就是每个 part 目录下的 `label.txt`（train/dev/eval 各一份），已解压验证，标签行数与 wav 数量完全一致
- ✅ 已确认：`test` 集就是代码里的 `eval`，已重命名
- ⚠️ **待决策：数据增强怎么办**——`preprocess_da_full.py` 默认跑 5 种处理（original + add_rev + add_speech + add_music + add_noise），后 4 种需要 RIRS_NOISES + MUSAN 数据集（本地没有，需再下载几 GB）。两个选项：
  - **方案 A（推荐先做）：只跑 original**——改循环为 `["original"]`，同时把 `main_train.py` 的 sampler 硬编码数改成实际样本数（train 22400 / dev 8400）。能快速跑通全流程。
  - **方案 B：完整复现**——再下载 RIRS_NOISES + MUSAN 做增强，保持原代码。耗时更长。

---

## 四、涉及的关键文件速查
| 文件 | 作用 |
|------|------|
| `code/ADD2023t3_RE/preprocess_da_full.py` | RE 特征提取 |
| `code/ADD2023t3_RE/main_train.py` | RE 训练 |
| `code/ADD2023t3_RE/generate_score_da.py` | RE 打分 |
| `code/ADD2023t3_FD/main_train.py` | FD 训练 |
| `code/ADD2023t3_FD/ood_detector_pre.py` | FD OOD 预处理 |
| `code/ADD2023t3_FD/ood_detector.py` | FD OOD 打分（NSD） |
| `code/ADD2023t3_FD/generate_f1_e2e.py` | 最终 F1 |
