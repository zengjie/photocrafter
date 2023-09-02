import os
import requests
import json
import time


from dotenv import load_dotenv

load_dotenv()

RUNPOD_API_TOKEN = os.getenv("RUNPOD_API_TOKEN")
CONTENT_TYPE = "application/json"
AUTHORIZATION = f"Bearer {RUNPOD_API_TOKEN}"
PROMPT = "realistic professional headshot photo of an asian sks business man"
DATASET_URL = os.getenv("DATASET_URL")

API_ENDPOINT = "https://api.runpod.ai/v2/dream-booth-v1"


def prepare_headers():
    headers = {"Content-Type": CONTENT_TYPE, "Authorization": AUTHORIZATION}
    return headers


def prepare_payload():
    payload = {
        "input": {
            "train": {
                "data_url": DATASET_URL,
            },
            "inference": [{"prompt": PROMPT}] * 10,
        }
    }
    return payload


def send_post_request(headers, payload):
    response = requests.post(
        f"{API_ENDPOINT}/run", headers=headers, data=json.dumps(payload)
    )
    return response


def handle_response(response):
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(f"Message: {response.text}")
        return
    response_data = response.json()
    print(response_data)
    return response_data["id"]


def check_job_status(job_id, headers):
    delay = 1  # initial delay is 1 second
    while True:
        status_response = requests.get(
            f"{API_ENDPOINT}/status/{job_id}", headers=headers
        )
        if status_response.status_code != 200:
            print(f"Error: {status_response.status_code}")
            print(f"Message: {status_response.text}")
            break
        status_data = status_response.json()
        print(f"Status: {status_data['status']}")
        if status_data["status"] == "COMPLETED":
            print("Job ID: ", status_data['id'])
            print("Status: ", status_data['status'])
            print("Delay Time: ", status_data['delayTime'])
            print("Execution Time: ", status_data['executionTime'])

            for inference in status_data['output']['inference']:
                for image in inference['images']:
                    print("Image URL: ", image)

            print(json.dumps(status_data, indent=4))            
            break
        elif status_data["status"] == "FAILED":
            print("Job failed with ID: ", status_data["id"])
            print("Delay Time: ", status_data["delayTime"])
            print("Execution Time: ", status_data["executionTime"])
            print("Error: ", status_data["error"])
            break
        else:
            time.sleep(delay)
            delay = min(10, delay * 2)  # increase delay, but cap at 10 seconds


def main():
    try:
        job_id = JOB_ID
    except NameError:
        job_id = None
    headers = prepare_headers()
    payload = prepare_payload()
    if job_id is None:
        response = send_post_request(headers, payload)
        job_id = handle_response(response)
    if job_id:
        check_job_status(job_id, headers)


if __name__ == "__main__":
    main()
