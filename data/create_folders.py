import os

for l in ["bg", "cz", "de", "en", "es", "fr", "nl", "pl", "sl"]:
    try:
        os.mkdir(l)
        os.mkdir(l + "/data")
    except:
        print(f"{l} already exists")