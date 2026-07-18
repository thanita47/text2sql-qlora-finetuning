# Text2SQL-Thai-Instruct

Fine-tuning `Qwen2.5-Coder-1.5B-Instruct` with QLoRA (via [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)) to translate natural language questions into SQL queries.

This started as a fairly standard fine-tuning exercise, but the more interesting part ended up being a debugging process: the first version looked fine on similarity metrics but was actually failing badly on exact match. Digging into why led to a fix that raised accuracy quite a bit. Details below.

## Setup

| | |
|---|---|
| Base model | [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) (Apache 2.0) |
| Method | QLoRA (4-bit quantization + LoRA, rank 16 / alpha 32) |
| Framework | [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) (Apache 2.0) |
| Hardware | Google Colab, single T4 GPU |

## Dataset

**Round 1** used [Spider](https://huggingface.co/datasets/xlangai/spider) (Yu et al., 2018, CC BY-SA 4.0), the standard cross-domain text-to-SQL benchmark — 7,000 training examples across 140 databases. Quick EDA before training:

- 39.8% of queries use JOIN
- 25.4% use GROUP BY, 23.3% use ORDER BY
- 6.5% are nested queries
- Average query length: 110 characters

**Round 2** switched to [b-mc2/sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context) (b-mc2, 2023, CC BY-4.0) (5,000 train / 500 val), which pairs each question with its schema directly — the dataset itself was built from Spider and WikiSQL specifically to reduce this kind of schema hallucination, which matches what I ran into below. Reason for the switch is explained below.

## What went wrong in Round 1 (and why)

Training on Spider gave a ROUGE-L of 64.93%, which looked reasonable. But exact-match accuracy was only **11.70% (121/1034)** — a big gap that was worth investigating rather than ignoring.

Looking at the actual failures made the problem obvious:

```
Question: How many singers do we have?
Database: concert_singer

Predicted: SELECT DISTINCT T1.country FROM people AS T1 
           JOIN band AS T2 ON T1.people_id = T2.people_id 
           WHERE T1.age > 20
Correct:   SELECT DISTINCT country FROM singer WHERE age > 20
```

The model is joining `people` and `band` tables that don't even exist in this database. It's making up a schema.

**The reason:** the instruction I used only told the model the database *name* (`'concert_singer'`), never the actual schema — no table names, no columns. So the model had to guess table/column structure from the question alone, and it guessed wrong most of the time whenever a JOIN was needed. ROUGE/BLEU stayed high because those metrics reward word overlap, not structural correctness, so they didn't catch this.

**The fix:** switch to a dataset that includes the schema in every example (`sql-create-context`), and put the schema directly in the instruction:

```
Given the database schema: CREATE TABLE singer (singer_id, name, age, country, ...)
Translate this question into a SQL query: How many singers do we have?
```

## Results

| Metric | v1 (no schema) | v2 (with schema) |
|---|---|---|
| Exact-match accuracy | 11.70% (121/1034) | **72.80% (364/500)** |
| ROUGE-L | 64.93% | 97.13% |
| BLEU-4 | 62.27% | 95.54% |
| Eval loss | 0.8838 | 0.0621 |

Also worth noting: in Round 1, eval_loss (0.88) was much higher than train_loss (0.14) — a pretty clear generalization gap. In Round 2 eval_loss actually came in slightly *below* train_loss, which suggests the model wasn't just memorizing but genuinely learned to use the schema.

**One honest caveat:** this isn't a clean ablation. v1 and v2 differ in more than just "schema present or not" — they're different datasets with different query difficulty (sql-create-context is mostly single-table, Spider has a lot of cross-table joins) and different validation sets. So the jump from 11.7% to 72.8% is best read as *evidence supporting* the schema-hallucination explanation, not a fully isolated measurement of that one variable. A tighter version of this experiment would add schema strings directly into the Spider instructions and re-evaluate on the same Spider validation set — noted below as future work.

## Error analysis (v2)

Of the ~27% that still failed exact-match in v2, most of what I checked were harmless formatting differences — SQL keyword casing, or WHERE conditions written in a different but equivalent order — rather than actual logical errors. This is a known weakness of exact-match as a metric. Execution accuracy (actually running the query and comparing results) would be a better measure and is listed below as something to add.

## Limitations / what I'd do next

- Haven't measured execution accuracy yet, only exact string match — some of the "failures" are probably correct queries written differently
- Only tested single-GPU training; understand the ideas behind distributed/multi-GPU training (data parallel, DeepSpeed) but haven't had a chance to run it
- Only tried one LoRA rank (16) — didn't get to compare against rank 8 or 32
- The schema-fix comparison above isn't a controlled ablation (see caveat) — doing it properly on the same dataset/val-set is the next thing I'd want to try
- sql-create-context is mostly single-table; haven't tested how well this holds up on more complex multi-table schemas

## Running it

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .

llamafactory-cli train configs/qwen_lora_sft_v2.yaml       # train
llamafactory-cli train configs/predict_config_v2.yaml      # evaluate
```

Configs and data prep scripts are in `configs/` and `data/`.

## Model weights

LoRA adapter (v2): [link here]

## License

The code in this repository is released under the [MIT License](LICENSE) — see the LICENSE file for details.

This does **not** cover the base model or the datasets used, which each have their own license (see Credits below). Anyone reusing this repo should check those separately before using the fine-tuned weights or the training data.

## Credits

**Framework:** [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) (Apache 2.0) — used for the full training and evaluation pipeline.

**Base model:** [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) by Alibaba Cloud/Qwen Team (Apache 2.0).

**Datasets:**
- [Spider](https://huggingface.co/datasets/xlangai/spider) — Yu et al., 2018, Yale University (CC BY-SA 4.0)
- [b-mc2/sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context) — b-mc2, 2023 (CC BY-4.0), built from Spider and WikiSQL (Zhong et al., 2017)

```bibtex
@article{yu2018spider,
  title={Spider: A large-scale human-labeled dataset for complex and cross-domain semantic parsing and text-to-sql task},
  author={Yu, Tao and Zhang, Rui and Yang, Kai and Yasunaga, Michihiro and Wang, Dongxu and Li, Zifan and Ma, James and Li, Irene and Yao, Qingning and Roman, Shanelle and others},
  journal={arXiv preprint arXiv:1809.08887},
  year={2018}
}

@misc{b-mc2_2023_sql-create-context,
  title = {sql-create-context Dataset},
  author = {b-mc2},
  year = {2023},
  url = {https://huggingface.co/datasets/b-mc2/sql-create-context},
  note = {Built from Spider (Yu et al., 2018) and WikiSQL (Zhong et al., 2017)}
}

@article{hui2024qwen2,
  title={Qwen2.5-Coder Technical Report},
  author={Hui, Binyuan and Yang, Jian and Cui, Zeyu and Yang, Jiaxi and Liu, Dayiheng and Zhang, Lei and Liu, Tianyu and Zhang, Jiajun and Yu, Bowen and Dang, Kai and others},
  journal={arXiv preprint arXiv:2409.12186},
  year={2024}
}
```

---
Thanita — [github.com/thanita47](https://github.com/thanita47)
