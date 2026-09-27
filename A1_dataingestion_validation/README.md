# ML Production Track - Module A1

## Real data ingestion and validation

Every line explained. No `sklearn` toy datasets.

### What you actually do in a job

Nobody hands you a clean CSV. You get access to a database or a data lake, a vague business problem, and the freedom to figure out the rest. The real work looks like this in order of time spent: you spend the most time understanding and validating data - checking for leakage, missing values, unexpected categories, and drift between training and production data. You build pipelines, not notebooks: code that runs unattended on a schedule, handles bad input gracefully, and logs what happened. You collaborate constantly with data engineers on where the data comes from, with product on what "good enough" means, and with backend engineers on how the model gets called. You maintain what you ship - a model released six months ago can start drifting, and finding out why its predictions got worse becomes part of the job. You are measured on business impact, not model accuracy alone: "we reduced churn by 8%" matters more than "F1 improved by 0.02" on its own.

### What changes when you run your own thing

The technical skills are the same, but you own every layer. There is no infrastructure team, so you are the infrastructure team. Cost becomes personal: every cloud job, storage request, and API call comes out of a budget you control. You ship faster and rougher because an early product does not need five-nines reliability; it needs to prove that people want it. Every technical decision is also a business decision. "Should I fine-tune or use a better prompt?" is really "Do I have three days or three hours?"

Both paths need the same foundation. This module starts with the part that makes everything else possible: getting trustworthy data into the system.

---

## What I am building in this module

I am building a small production-style data pipeline around the **Telco Customer Churn** dataset. It is a public business dataset with customer information, service choices, account details, and a churn label.

The goal is not to train the best churn model yet. The goal is to prove that I can:

- obtain data from a real external source;
- inspect it before making assumptions;
- identify missing values that are disguised as normal text;
- validate column names, types, ranges, and allowed categories;
- separate useful data checks from model-specific decisions;
- reject a bad input instead of silently passing it downstream; and
- save a clean, documented output that the next module can trust.

That distinction matters. A model cannot rescue a pipeline that quietly ingests the wrong columns or turns missing values into fake customer behavior.

## Part 1 - Why this is the actual starting point

Nobody in industry starts with:

```python
df = pd.read_csv("clean_data.csv")
```

That line may eventually exist, but it is not the interesting part. Before I can trust the dataframe, I need to know where it came from, when it was downloaded, what each column means, which values are valid, and what should happen when the source changes.

The first production ML component is often not a model. It is a controlled path from an unreliable source to a validated dataset. If that path is weak, every metric after it gives me false confidence.

## Part 2 - The dataset and its problems

The Telco Customer Churn dataset looks simple at first because it arrives as a table. Real tables are rarely simple.

I should expect to find:

- numeric values stored as strings;
- blank strings that represent missing values;
- categorical columns with a fixed set of expected values;
- identifier columns that should not be treated as predictive features;
- a target column whose meaning must be defined clearly; and
- possible changes to the source file that could break the pipeline later.

The important habit is to inspect first and clean second. I do not replace every suspicious value immediately. I first record what I found and decide whether the value is missing, invalid, meaningful, or evidence that the source contract has changed.

## Part 3 - The workflow I am following

### 1. Acquire the raw data

I download the source data into a raw-data location and keep the original file unchanged. The raw copy gives me something to audit and reproduce later. I also record the source URL, download time, file name, and a checksum when the pipeline is run outside a learning environment.

### 2. Inspect before transforming

I look at the shape, column names, data types, sample rows, unique values, and missingness. I am trying to answer basic questions before writing validation rules:

- What is one row supposed to represent?
- Which column is the target?
- Which columns identify a customer but do not explain churn?
- Which values are unexpectedly blank or inconsistent?
- Are the types I received the types I actually need?

### 3. Validate the data contract

I turn those observations into explicit checks. The pipeline should fail with a useful message when a required column disappears, a numeric field contains impossible values, or a categorical field contains a new unexpected category.

Validation is not the same as cleaning. Cleaning changes data. Validation decides whether the data is trustworthy enough to continue. Sometimes the correct response to a validation failure is to stop and investigate instead of automatically fixing it.

### 4. Apply careful transformations

Only after the checks pass do I normalize types and represent missing values consistently. I keep transformations predictable and documented. I do not use information from the future or from the target to make the input look better than it really is.

### 5. Write a validated output

The output of this module is a dataset and a validation result that a later training step can consume. Someone else should be able to tell what passed, what failed, how many rows were processed, and where the output came from without opening the code and guessing.

## Part 4 - What I should be able to explain afterwards

By the end of this module, I should be able to explain:

1. Why raw data and processed data are stored separately.
2. Why missingness is a data-quality question before it is a modeling question.
3. Why schema, type, range, and category checks catch different failures.
4. Why an identifier can be useful for tracing a row but harmful as a model feature.
5. Why validation rules should be visible, testable, and version-controlled.
6. What the pipeline does when a check fails.
7. Which decisions are general data engineering decisions and which belong in the modeling module.

## Definition of done

This module is complete when I have a repeatable ingestion and validation flow that:

- starts from the public source rather than a hand-cleaned copy;
- preserves the raw input;
- produces a clear schema and data-quality report;
- handles disguised missing values explicitly;
- checks required columns, types, ranges, and categories;
- fails loudly on invalid input;
- writes a validated dataset for the next module; and
- explains its decisions well enough that another engineer can review them.

The model comes later. First, I need to know that the data entering the model is real, understood, and safe to use.

## Module structure

The implementation is organized by responsibility so that each part can be tested, reused, and explained independently:

```text
churn_pipeline/
├── config.py              # Settings dataclass from Part 2
├── logging_setup.py       # setup_logging() from Part 3
├── data/
│   ├── __init__.py
│   ├── loader.py          # DataLoader from Part 4
│   ├── validator.py       # DataValidator and ValidationIssue from Part 5
│   └── profiler.py        # DataProfiler from Part 6
├── splitting.py           # SplitStrategy from Part 7
└── pipeline_ingest.py     # run_ingestion_pipeline(), the Part 8 orchestrator
```
