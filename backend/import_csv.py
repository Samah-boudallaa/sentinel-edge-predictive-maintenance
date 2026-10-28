import sys
import requests

API = "http://localhost:8000/measures/batch"
path = sys.argv[1]

rows = []
for line in open(path, encoding="utf-8"):
    p = line.strip().split(",")
    if len(p) == 7 and p[0].isdigit():
        rows.append({"ts_ms": int(p[0]), "mode": p[1], "std_g": float(p[2]),
                     "p2p_g": float(p[3]), "ratio": float(p[4]),
                     "freq_hz": float(p[5]), "level": p[6]})

print(requests.post(API, json=rows).json())