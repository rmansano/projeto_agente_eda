import requests
from config import DEEPSEEK_API_KEY

headers = {
    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    "Accept": "application/json",
}

resp = requests.get(
    "https://api.deepseek.com/user/balance",
    headers=headers,
    timeout=20,
)

print(resp.status_code)
print(resp.json())