# 提示词：为数据集编写 profile.yaml

把下面整段提示词连同数据集 CSV 一起发给 AI。AI 只需产出 `profile.yaml`，不要修改 CSV，
不要自己做切分或清洗——那些由 `python -m core.prepare run` 完成。

---

## 任务

你是表格数据的预处理工程师。我会给你一个或多个数据集的 CSV。请为**每一个**数据集写一份
`profile.yaml`，最后把所有文件打包成一个 zip 给我。

profile 是整个生成流程唯一的事实来源：后续代码会按它清洗数据、把日期标准化、切分
train/val/test 并生成元数据，然后用 11 个表格合成模型训练和生成。**profile 写错，产出的
合成数据就是错的**，所以每一列的判断都要有依据，不要凭列名猜。

## 第一步：先统计，再下判断

对每一列必须先算出这些信息，不要跳过（用 pandas 时请 `dtype=str, keep_default_na=False`，
避免 pandas 把 `NA`、`None`、`null` 自动变成缺失）：

1. 不同取值个数、总行数、取值频次前 10；
2. 各种疑似缺失标记的出现次数：空串、`NULL`、`null`、`NA`、`N/A`、`nan`、`NaN`、`None`、`?`、`-`；
3. 去掉缺失标记后，有多少比例能转成数字；能转成数字的，是否全是整数；最小值、最大值；
4. 是否像日期或时间：随机抽 20 个值列出来，判断具体格式；
5. 值里有没有前后空格、有没有换行、有没有逗号或引号。

## 第二步：按规则确定每一列的类型

可选类型只有这些：`continuous`、`integer`、`categorical`、`ordinal`、`boolean`、`id`、
`text`、`date`、`datetime`、`time`。

- **数值**：全部能转成数字。全是整数用 `integer`，否则 `continuous`。
- **类别**：非数值且取值有限用 `categorical`；只有两个取值用 `boolean`；有天然顺序
  （如 low/mid/high）用 `ordinal` 并写出 `order`。
- **id**：近乎唯一的标识串（如主键、编号、文件名）。仍会参与生成，只是模型按类别处理。
- **text**：自由文本，取值很多且没有固定集合（如描述、备注）。
- **日期时间**：`date` 只有日期，`datetime` 有日期和时间，`time` 只有时间。必须写 `format`：
  - 标准 strptime 格式串，如 `"%Y-%m-%d"`、`"%m/%d/%Y"`、`"%m/%d/%Y %I:%M:%S %p"`、`"%Y_%j"`（年_年内第几天）；
  - 或三个特殊值：`iso8601`（形如 `2015-11-16T04:03:21+00:00`，带时区会转成 UTC）、
    `unix_s`（秒级时间戳）、`unix_ms`（毫秒级时间戳）。
  - **数字形式的日期或时间**（如 `80813` 表示 08/08/13，`21` 表示 00:00:21）要加 `pad`，
    即先左补零到指定位数再解析：`{type: date, format: "%d%m%y", pad: 6}`。

**三个最容易出错的地方：**

1. **不要把真实类别值当成缺失。** 比如某列的 `Unknown`、`unknown`、`None`、`NA` 可能是真实
   取值（物种"未识别"、类群"未知"）。只有确定表示"没有数据"的标记才写进 `missing_tokens`。
   拿不准就去看这个值的出现频次和业务含义，并写进 notes。
2. **时间戳列不一定是日期列。** 如果某列在 SQL 里是当数字用的（如 `UnixTimestamp` 会被拿来
   做减法、`dep_time` 是 `1830` 这种 HHMM 数字且可能出现 `2400`），就保持 `integer`，不要
   转成日期，否则会改变语义或解析失败。
3. **一列脏值不该拖垮整个数据集。** 默认 `on_invalid: error`，即出现无法解析的值就报错并
   告诉你例子。如果确认是源数据的个别错误（如时间列里混进一个 `11214`），就对**这一列**
   写 `on_invalid: "null"`，让它变成缺失，并在 notes 里说明。

## 第三步：选目标列

11 个模型里有几个需要一个明确的目标列（X/y 结构），所以**必须且只能指定一列**：

- 分类任务：选 `categorical` / `boolean` / `ordinal` / `integer` 类型、取值数适中（2 到几十个
  最好，超过 256 类很多模型会退化）、缺失很少的列。优先选业务上真正的标签，没有标签就选
  一个有代表性的分组列（如物种、类别、承运人）。
- 回归任务：选 `continuous` 或 `integer` 的关键数值列。
- **不要选**：近乎唯一的 id 列、自由文本列、日期时间列、缺失很多的列、只有一个取值的列。
- 在 notes 里写明为什么选它。

## 第四步：写 profile.yaml

格式如下，`schema_version` 必须原样照抄，`columns` 必须**列出 CSV 里的每一列，不多不少**，
顺序建议与 CSV 一致：

```yaml
schema_version: tqb_dataset_profile_v1
dataset_id: 1117_orcamaster2010_csv        # 用我在下面指定的 id
description: 虎鲸目击记录
source: 数据来源说明
data_file: 1117_orcamaster2010_csv.csv     # CSV 文件名，与 profile.yaml 同目录
format: {delimiter: ",", encoding: utf-8}  # 可省略，这是默认值
missing_tokens: ["", "NULL"]               # 精确匹配（去空格后），区分大小写
on_invalid: error                          # 数据集级默认；可在列上覆盖
target: {column: Pod, task_type: classification}
split: {train: 0.8, val: 0.1, test: 0.1, seed: 42, stratify: true}
generation: {max_train_rows: null, num_rows: train}
columns:
  - {name: SightDate, type: date, format: "%m/%d/%Y"}
  - {name: Time2, type: time, format: "%H%M", pad: 4, on_invalid: "null", description: HHMM 数字形式}
  - {name: Pod, type: categorical}
  - {name: Lat, type: continuous}
  - {name: Quadrant, type: integer}
  - {name: Notes, type: text}
```

其他说明：

- `split`：分类任务用 `stratify: true`（按目标列分层），回归任务用 `false`。比例和 seed 保持默认。
- `generation.max_train_rows`：行数在 20 万以内写 `null`；超过 20 万写 `50000`，并在 notes
  里提醒大表某些模型很慢。
- `drop`：默认**不要删列**，因为下游要用完整表结构跑 SQL 查询。只有当某列确实无法使用时才
  写 `{name: X, type: text, drop: true}`，并说明原因。
- 列名里有点号、空格、中文都没关系，保持和 CSV 表头完全一致即可，后续流程会自动处理。

## 第五步：自检（必须做）

写完后，用代码逐列验证，**覆盖全部行，不要只看前几行**：

1. 每个非缺失值都能按声明的类型解析：数值列能转成数字、`integer` 列没有小数、日期时间列
   能按 `format`（含 `pad`）解析成功；
2. 统计每列的缺失数量，确认和你判断的缺失标记一致；
3. 目标列：确认存在、缺失极少、取值数合理；
4. `columns` 的列名集合与 CSV 表头完全一致。

把每列的解析成功率、缺失率、不同取值数整理成一张表放进 notes。有任何不确定的判断（尤其是
某个值到底算不算缺失、某列到底是类别还是 id），都要写出来，不要静默决定。

## 第六步：打包

zip 里按数据集分目录，只放 profile 和说明，**不要放 CSV**：

```
profiles.zip
  <dataset_id>/profile.yaml
  <dataset_id>/notes.md        # 每列的统计结果、判断依据、存疑项
  ...
```

文件用 UTF-8 编码、不要 BOM。

## 本次的数据集 id 和文件名

| dataset_id | CSV 文件名 |
|---|---|
| `1010_birds_csv` | 1010_birds_csv.csv |
| `790_table_extr_accessory_v_csv` | 790_table_extr_accessory_v_csv.csv |
| `1117_orcamaster2010_csv` | 1117_orcamaster2010_csv.csv |
| `1267_h2_w_2_csv` | 1267_h2_w_2_csv.csv |
| `1059_sds_view` | 1059_sds_view.csv |
| `1059_stats_view` | 1059_stats_view.csv |
| `372_flights09_part` | 372_flights09_part.csv |

---

## 拿到 zip 之后

解压到 `Extra-Experiment/WorkloadTest/datasets/`，每个 `<dataset_id>/` 下和 CSV 放在一起，然后：

```bash
cd Synthesizing/code && export PYTHONPATH=src
python -m core.prepare check ../../Extra-Experiment/WorkloadTest/datasets   # 只校验格式
python -m core.prepare run   ../../Extra-Experiment/WorkloadTest/datasets   # 清洗 + 切分 + 元数据
```

`check` 会报 profile 本身的结构错误，`run` 会报数据层面的错误（类型对不上、日期解析失败等），
并对每个数据集打印警告（全空列、常数列、高基数列、大表）。两步都通过后就可以跑生成。
