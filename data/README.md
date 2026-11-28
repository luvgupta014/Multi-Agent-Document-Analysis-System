# data/

This directory holds the sample corpus used for indexing and retrieval.

| File | Description |
|------|-------------|
| `climate_overview.txt` | Short summary of climate change science |
| `renewable_energy.txt` | Overview of solar, wind and storage technologies |
| `sample.db` | Auto-generated SQLite demo database (created on first run) |

You can add your own `.txt`, `.pdf`, or `.docx` files here and re-ingest them
with the `/upload` endpoint or by calling `DocumentLoader.load_directory("data/")`.
