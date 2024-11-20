#!/usr/bin/env python

import requests
import json


def interact_with_openwebui():
    # OpenWebUI API endpoint
    url = "https://your-openwebui-instance.com/api/endpoint"

    # Your request payload
    payload = {"prompt": "Your prompt or command here", "other_parameters": "as needed"}

    # Headers, including authentication if required
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer YOUR_API_KEY",
    }

    try:
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        response.raise_for_status()  # Raises an HTTPError for bad responses

        # Process the response
        result = response.json()
        print("Interaction successful:", result)

        # Perform any additional actions with the result

    except requests.exceptions.RequestException as e:
        print("An error occurred:", e)


if __name__ == "__main__":
    interact_with_openwebui()
