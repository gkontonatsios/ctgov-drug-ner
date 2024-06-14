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
Change config file to your local env variables. For this edit the `drug_ner/config.json` file and change the `dbname`, `user`, `password`, `host`, `port` and `open_ai_api_key` variables to match your local environment.
