import requests
import pandas as pd
import numpy as np
from ctgov_crawler import craw_ctgov
import os
from file_utils import get_gold_standard_file
def fetch_nct_ids():
    base_url = "https://clinicaltrials.gov/api/query/study_fields"
    query_params = {
        "expr": "*",  # This fetches all studies
        "fields": "NCTId",
        "min_rnk": 1,
        "max_rnk": 1000,
        "fmt": "json"
    }

    all_nct_ids = []
    current_rank = 1
    total_studies = None

    while True:
        query_params["min_rnk"] = current_rank
        query_params["max_rnk"] = current_rank+200
        # Make the API request
        response = requests.get(base_url, params=query_params)

        # Check for request errors
        if response.status_code != 200:
            print(f"Error: Received response code {response.status_code}")
            print(response.text)
            break

        data = response.json()

        # Extract total number of studies if not already done
        if total_studies is None:
            total_studies = data.get("StudyFieldsResponse", {}).get("NStudiesFound", 0)
            print(f"Total studies found: {total_studies}")

        study_fields = data.get("StudyFieldsResponse", {}).get("StudyFields", [])
        if not study_fields:
            print("No more study fields found.")
            break

        # Extract NCT IDs
        nct_ids = [field["NCTId"][0] for field in study_fields]
        all_nct_ids.extend(nct_ids)

        current_rank += len(study_fields)
        if len(all_nct_ids) > 3000:
            break

    return all_nct_ids

def main():

    df_gold = pd.read_csv(get_gold_standard_file()).replace(np.nan, '')
    eval_nct_ids = df_gold['nct_id'].tolist()
    # Fetch all NCT IDs
    nct_ids = fetch_nct_ids()
    nct_ids = list(set(nct_ids))

    # remove eval nct_ids
    nct_ids = [token for token in nct_ids if token not in eval_nct_ids]

    # Display the NCT IDs using pandas
    df = pd.DataFrame(nct_ids, columns=["NCT ID"])
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(project_root, 'data')
    craw_ctgov(list_nct_id=nct_ids, output_file=os.path.join(data_dir, 'brief_summaries.training.csv'))
if __name__ == '__main__':
    main()

