# data/

This directory holds the sample corpus (public arXiv papers + text summaries) and the auto-generated demo SQLite database.

| File | Description |
|---|---|
| `rag_lewis_2020.pdf` | Lewis et al. (2020) — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (arXiv:2005.11401) |
| `react_yao_2022.pdf` | Yao et al. (2022) — *ReAct: Synergizing Reasoning and Acting in Language Models* (arXiv:2210.03629) |
| `climate_overview.txt` | Short summary of climate change science |
| `renewable_energy.txt` | Overview of solar, wind, and storage technologies |
| `sample.db` | Auto-generated SQLite demo database (created on first run by `SQLSpecialistAgent`) |

You can index `data/` instead of `docs/sample_documents` by setting `KB_DOC_DIR=data`, or upload individual `.txt`, `.pdf`, or `.docx` files via `POST /upload`.
