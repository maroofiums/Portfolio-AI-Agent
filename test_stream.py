import requests

url = "https://portfolio-ai-agent-dd0ba526.fastapicloud.dev/chat"
payload = {
    "messages": [
        {"role": "user", "content": "Reply with Hello"}
    ]
}

print("Sending request...")
try:
    response = requests.post(
        url,
        json=payload,
        headers={"Accept": "text/event-stream"},
        stream=True,
        timeout=(15, 90),
    )

    print("HTTP:", response.status_code)
    print("Content-Type:", response.headers.get("content-type"))
    print("Waiting for first event...")

    for line in response.iter_lines(decode_unicode=True):
        if line:
            print("EVENT:", line)
            break
    else:
        print("Stream ended without an event.")

except Exception as error:
    print(type(error).__name__ + ":", error)
