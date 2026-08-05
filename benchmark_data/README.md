# Benchmark data

This directory separates reproducibility metadata from source datasets.

Included:

- `manifest.json`: benchmark schemas, field types, deterministic sampling
  settings, and expected converted filenames
- `convert_to_single_table.py`: conversion logic for supported cross-source
  datasets
- `fake_1000.csv` and `sim_er_data.csv`: synthetic smoke datasets

Not included:

- NC voter snapshots
- BPID and ICE-ID records
- DeepMatcher, Magellan, or FAMER benchmark records
- Converted copies of those source datasets

Those datasets have distinct access, privacy, and redistribution terms. Obtain
them from their original publishers, retain their expected source directory
names, and run:

```bash
uv run python benchmark_data/convert_to_single_table.py
```

Converted files are written under `benchmark_data/converted/`. The exact
filenames expected by the evaluation scripts are recorded in `manifest.json`.
The manifest is also the authoritative record of the field types and
deterministic entity sampling used for the paper.

Absence of a converted file is intentional and produces a local file-not-found
error rather than silently substituting a different dataset.
