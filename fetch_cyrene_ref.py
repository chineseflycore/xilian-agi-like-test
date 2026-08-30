# -*- coding: utf-8 -*-
"""抓取 cyrene.skill 参考数据（昔涟人设 skill）到 data_cache/reference_cyrene/"""
import os
import base64
import time
import requests

H = {"User-Agent": "Mozilla/5.0"}
REPO = "HeartEase1/cyrene.skill"
# 输出目录基于脚本所在目录（勿硬编码绝对路径，勿提交到仓库）
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache", "reference_cyrene")
os.makedirs(OUT, exist_ok=True)

FILES = ["SKILL.md", "personality.md", "profile.md", "background_story.md",
         "interaction.md", "relations.md", "memory.md", "conflicts.md"]

for f in FILES:
    url = f"https://api.github.com/repos/{REPO}/contents/{f}"
    ok = False
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=20, headers=H)
            if r.status_code == 200:
                content = base64.b64decode(r.json()["content"]).decode("utf-8", "replace")
                with open(os.path.join(OUT, f), "w", encoding="utf-8") as fh:
                    fh.write(content)
                print(f"OK {f}: {len(content)} 字符")
                ok = True
                break
            print(f"{f}: HTTP {r.status_code}")
        except Exception as e:
            print(f"{f}: attempt{attempt} FAIL {str(e)[:60]}")
            time.sleep(2)
    if not ok:
        print(f"{f}: 抓取失败")
