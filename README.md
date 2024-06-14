## Prerequisites

Before starting, ensure that you have the following installed on your system:
- Python (version 3.6 or later)
- Poetry (version 1.0.0 or later)

## Step 1: Install Poetry

Poetry can be installed using the following command:

```sh
curl -sSL https://install.python-poetry.org | python3 -
```

## Step 2: Install Dependencies

```
poetry install
poetry shell
```

## Step 3: Edif config
Edit the `drug_ner/config.json` file and change the `dbname`, `user`, `password`, `host`, `port` and `open_ai_api_key` variables to match your local environment.

## Step 4: Crawl summaries from www.clinicaltrials.gov
Collect the brief summaries of clinical trials corresponding to the nct_ids found in `data/input_nct_ids.txt`. Run the following: 

```
poetry install
poetry shell

```
cd drug_ner/
poetry run python ctgov_crawler.py
```
