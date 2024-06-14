from typing import List

import pandas as pd
import requests
from loguru import logger
import os
import json
from tqdm import tqdm


def get_brief_summary_by_nct_id(nct_id: str) -> str:
    """
       Fetches the brief summary of a clinical trial from ClinicalTrials.gov using the provided NCT ID.

       Args:
           nct_id (str): The NCT ID of the clinical trial to retrieve.

       Returns:
           str: The brief summary of the clinical trial if the request is successful.
                An error message is returned if the request fails.

       Example:
           summary = get_brief_summary_by_nct_id("NCT00109707")
           print(summary)
       """

    # logger.info(f'Getting brief summary for: {nct_id}')
    url = f"https://clinicaltrials.gov/api/v2/studies?query.term=AREA[NCTId]{nct_id}"

    # Send a GET request to the API
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        return data['studies'][0]['protocolSection']['descriptionModule']['briefSummary']
    else:
        logger.warning(f"No brief summary found for: {nct_id}")


def craw_ctgov(list_nct_id: List[str], output_file: str):
    brief_summaries = [get_brief_summary_by_nct_id(nct_id) for nct_id in tqdm(list_nct_id, total=len(list_nct_id))]
    df = pd.DataFrame()
    df['nct_id'] = list_nct_id
    df['brief_summary'] = brief_summaries

    df.to_csv(output_file, index=False)


def main():
    # Load database configuration from JSON file
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Get the path to the data directory
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(project_root, 'data')
    NCT_IDS_FILE_NAME = os.path.join(data_dir, config['input_nctid_list_file'])

    # get input list of nct_ids
    # Read the file and create a list of NCT IDs using list comprehensions
    with open(NCT_IDS_FILE_NAME, 'r') as file:
        nct_ids = [line.strip() for line in file if line.strip()]

    craw_ctgov(list_nct_id=nct_ids, output_file=os.path.join(data_dir, 'brief_summaries.csv'))


if __name__ == '__main__':
    main()
