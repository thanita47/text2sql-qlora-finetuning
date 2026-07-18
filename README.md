# Text2SQL-Thai-Instruct

Fine-tuning `Qwen2.5-Coder-1.5B-Instruct` with QLoRA (via [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)) to translate natural language questions into SQL queries.

The first version I trained scored well on similarity metrics but poorly on exact match. This README covers what I found when I looked into why, and what I changed as a result.

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

**Round 2** switched to [b-mc2/sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context) (b-mc2, 2023, CC BY-4.0) — 5,000 train / 500 val. Each example here comes with its schema attached. The reason for switching is explained below.

## What went wrong in Round 1

Training on Spider gave a ROUGE-L of 64.93%. But exact-match accuracy was only **11.70% (121/1034)**. That gap seemed worth checking.

Looking at a few failed examples:

```
Question: How many singers do we have?
Database: concert_singer

Predicted: SELECT DISTINCT T1.country FROM people AS T1 
           JOIN band AS T2 ON T1.people_id = T2.people_id 
           WHERE T1.age > 20
Correct:   SELECT DISTINCT country FROM singer WHERE age > 20
```

The model is joining `people` and `band` tables that don't exist in this database. It's guessing at a schema it never saw.

**Reason:** the instruction only gave the database *name* (`'concert_singer'`), not the schema — no table or column names. So the model guessed the table structure from the question alone, and got it wrong whenever a JOIN was needed. ROUGE/BLEU stayed high because those metrics score word overlap, not whether the SQL is structurally correct, so this didn't show up there.

**Fix:** use a dataset where the schema is included with every example (`sql-create-context`), and put the schema in the instruction directly:

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

A side note on loss: in Round 1, eval_loss (0.88) was much higher than train_loss (0.14) — a sign the model wasn't generalizing well. In Round 2, eval_loss (0.062) was actually a bit lower than train_loss (0.072), which is a better sign.

**Caveat:** this isn't a clean controlled comparison. v1 and v2 use different datasets, not just "schema vs. no schema" — sql-create-context is mostly single-table and generally easier than Spider, and the two use different validation sets. So the 11.7% → 72.8% jump supports the schema explanation, but it isn't an isolated measurement of that one variable. A more controlled version of this would add schema text into the original Spider instructions and re-evaluate on the same Spider validation set — noted below as something to try next.

## Error analysis (v2)

Of the ~27% that still failed exact-match, most of what I checked were formatting differences rather than actual mistakes — different SQL keyword casing, or WHERE conditions in a different but equivalent order. This is a known limitation of exact-match as a metric. Execution accuracy (running the query and comparing output) would give a more accurate picture — listed below as something to add.

## Limitations / what I'd do next

- Only measured exact string match, not execution accuracy — some "failures" are probably correct queries written differently
- Only trained on a single GPU; know the ideas behind distributed/multi-GPU training (data parallel, DeepSpeed) but haven't tried it
- Only tried LoRA rank 16, didn't compare against other ranks (8, 32)
- The schema-fix comparison isn't a controlled ablation (see caveat above) — doing that properly is the next thing worth trying
- sql-create-context is mostly single-table; haven't checked how this holds up on more complex, multi-table schemas

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
