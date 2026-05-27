# What is Normal? Measuring Status Quo Bias in LLM Embedding Geometry

**Fairness and Collective Decision-Making in AI — Research Project**
University of Konstanz, SS 2026

Jorge · Emre

---

## Research question

Do large language models encode a structural bias toward the social status quo — treating concepts associated with social change as *deviant* or *disruptive*, while concepts associated with existing social order appear *normal* and *stable*? And does this vary across models trained on different cultural corpora?

This project is distinct from prior work measuring left/right political orientation in LLMs. We measure a different axis: whether the *geometry* of model representations places change-adjacent concepts (redistribution, protest, solidarity) closer to semantic fields of deviance and disruption, independently of explicit political valence. We interpret findings through Althusser's Ideological State Apparatuses and Marcuse's one-dimensional thought.

---

## Method

- **WEAT (Word Embedding Association Test)** — measures cosine distance asymmetries between target and attribute word sets in embedding space
- **Masked token completion** — probes what generative models predict in politically framed sentence templates
- **Cross-model comparison** — same word sets applied across models representing distinct training corpora and geographic origins

---

## Models (planned)

| Model | Origin | Access |
|-------|--------|--------|
| Llama 3 | US — Meta | Ollama |
| Gemma 4 | US — Google | Ollama |
| Claude (API) | US — Anthropic | API |
| LatamGPT | Chile/LatAm — CENIA | HuggingFace / GPU cluster |
| Sea-Lion | Singapore — AISG | HuggingFace / GPU cluster |
| Qwen | China — Alibaba | Ollama |

> Infrastructure note: LatamGPT and Sea-Lion require GPU access. University cluster access pending.

---

## Repository structure

```
/
├── notebooks/
│   └── 01_weat_political_values.ipynb   # Main analysis notebook
├── data/
│   └── word_sets.json                   # Target and attribute word sets
├── results/                             # Output files (ignored by git except .gitkeep)
├── requirements.txt
└── README.md
```

---

## Status

| Task | Status |
|------|--------|
| WEAT notebook (BERT baseline) | ✅ Done — effect size d = −1.12 |
| Word set validation | 🔄 In progress |
| Masked token analysis | 🔄 In progress |
| Ollama model integration | ⬜ Pending |
| GPU cluster access | ⬜ Pending application |
| LatamGPT / Sea-Lion runs | ⬜ Pending infrastructure |

---

## Key references

- Caliskan et al. (2017). Semantics derived automatically from language corpora contain human-like biases. *Science.*
- Tang et al. (2023). What Do Llamas Really Think? *ArXiv.*
- Bai et al. (2025). Explicitly unbiased LLMs still form biased associations. *PNAS.*
- Kronlund-Drouault (2024). Propaganda is all you need. *ArXiv.*
- Althusser (1970). Ideology and Ideological State Apparatuses.
- Marcuse (1964). One-Dimensional Man.
