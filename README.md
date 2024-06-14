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
cd drug_ner/
poetry run python ctgov_crawler.py
```

## Step 5: Run the GPT-based drug NER

```
cd drug_ner/
poetry run python openai_ner.py
```

## Step 6: Run the T5 drug NER

First download the trained model from here: https://drive.google.com/file/d/10o8zzj8QxL5hogCDqE_nAiEwayyKdlmq/view

Extract the file and edit the `drug_ner/config.json` file by changing the `t5_check_point` variable to point to the location of the `t5-drug-ner-checkpoint` subfolder which is inside the file you have previously extracted. 

```
"t5_check_point": "results_t5small/t5-drug-ner-checkpoint"
```


Finally run: 
 
```
cd drug_ner/
poetry run python t5_inference.py
```
