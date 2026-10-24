import requests
import os 

bimo_token = os.getenv("BIMO_TOKEN")

if not bimo_token:
    raise RuntimeError("BIMO_TOKEN is not set!")

headers = {
    "x-bimo-token": bimo_token
}

prompt = input("You: ")

try:
    # send this prompt to my server and give me its response
    response = requests.post(
        "http://localhost:8000/ask",
        json={"prompt": prompt},
        headers=headers,
        timeout=30
    )

    response.raise_for_status()
    data = response.json()
    print("BIMO:", data["answer"])

except requests.exceptions.HTTPError as e:
    if response.status_code == 401:
        print("Error: Invalid or missing BIMO token.")
    else:
        print(f"Server returned an error: {e}")

except requests.exceptions.RequestException as e:
    print(f"Could not connect to BIMO's server: {e}")

