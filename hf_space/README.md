---
title: P09 LLM Eval Framework
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: false
---

# P09 · LLM Eval Framework

Evaluate LLM responses using ROUGE + LLM-as-judge + custom SRE rubric.
Scores stored in SQLite for regression tracking. CI gate blocks deploys on score drop >5%.

Part of the [Staff SRE · AI Engineer Portfolio](https://github.com/amarshiv86).

> Runs locally inside this Space — no external API calls.
