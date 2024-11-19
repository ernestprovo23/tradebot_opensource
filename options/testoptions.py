import requests

url = "https://paper-api.alpaca.markets/v2/options/contracts"

headers = {
    "accept": "application/json",
    "APCA-API-KEY-ID": "PK3V2G6KZNNTDDBZKH5T",
    "APCA-API-SECRET-KEY": "erOfDZMg7iSkCrhoabx57ZUdXvdi9ZsbHYXiJ6gc"
}

response = requests.get(url, headers=headers)

print(response.text)