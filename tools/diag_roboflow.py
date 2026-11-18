import os, sys, json, traceback
sys.path.insert(0, "src")
import common  # carrega .env
import requests
key = os.environ["ROBOFLOW_API_KEY"]
for ws, proj in [("pothole-skc","pothole-kvso8"),("sidewalk-detection","sidewalk-walkability"),("blindspot-fmcu0","sidewalk-obstacles")]:
    print("=====", ws, proj)
    r = requests.get(f"https://api.roboflow.com/{ws}/{proj}", params={"api_key": key}, timeout=60)
    print("status", r.status_code)
    try:
        d = r.json()
        p = d.get("project", {})
        print("project:", {k: p.get(k) for k in ("name","type","images","classes","versions","license")})
        for v in d.get("versions", [])[:10]:
            print("  version:", v.get("id"), "imagens:", v.get("images"), "export:", list((v.get("exports") or [])))
        if "error" in d: print("error:", d["error"])
    except Exception:
        print(r.text[:500])
