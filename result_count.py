import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import requests


URL = "http://127.0.0.1:8000/similar-styles-recommendations"


df = pd.read_csv('ASHI_FINAL_DATA.csv')


def get_result(item_id):
    url = f"{URL}/{item_id}"
    r = requests.get(url)
    if r.status_code == 200:
        print('Waiting for result-----------------------')
        data = r.json()
        result_array = data.get('array', [])
        if len(result_array) > 0:
            return {"ITEM_ID": item_id, "result": result_array}
        else:
             return {"ITEM_ID": item_id, "result": []}
    else:
        print(f"Error fetching {url}: {r.status_code}")
        return {"ITEM_ID": item_id, "result": []}


import json
with ThreadPoolExecutor(max_workers=100) as executor:
        results = list(tqdm(executor.map(get_result, [row['ITEM_ID'] for _, row in df.iterrows()]), total=len(df)))

json_object = json.dumps(results, indent=4)


# Writing to sample.json
with open("sample.json", "w") as outfile:
    outfile.write(json_object)


print('Done-----------------------------------')

# success_count = results.count("success")
# failure_count = results.count("failure")

# print(f"Successes: {success_count}, Failures: {failure_count}")