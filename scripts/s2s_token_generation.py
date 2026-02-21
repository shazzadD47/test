import httpx

url = "https://api-dev.delineate.pro/v1/s2s/oauth/validate"

headers = {
    "Content-Type": "application/json",
}

data = {
    "clientId": "16e203deb836f246a133cee649236d7c",
    "clientSecret": "ea7af5512d821a61ecc468f4483a58bbc526ba47585f363fd9a8a6247461ea5a",
}

response = httpx.post(url, headers=headers, json=data)

token = response.json()["token"]

print(token)
