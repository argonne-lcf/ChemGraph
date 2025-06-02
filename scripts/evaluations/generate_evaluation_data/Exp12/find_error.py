import json
import sys

with open(sys.argv[1], "r") as rf:
    data = json.load(rf)

for item in data:
    if "ERROR" in data[item]["manual_workflow"]["result"]['value']:
        print(item)
