# Third-party notices

IdentityPFN includes or adapts the following third-party work.

## nanoTabPFN

Parts of the model, tokenizer, and training implementation began as an
adaptation of nanoTabPFN.

Copyright 2025 Prior Labs GmbH, AutoML Freiburg, and Alexander Pfefferle.

nanoTabPFN is licensed under the Apache License 2.0. IdentityPFN substantially
modifies that implementation for synthetic entity-resolution training and
pairwise adjacency prediction. The Apache 2.0 license text is included in
`LICENSE`.

## PyTorch

`model.py` contains a modified form of PyTorch's
`torch.nn.TransformerEncoderLayer` implementation, adapted to alternate
attention across table columns and records. PyTorch is distributed under a
BSD-style license. The applicable copyright and license text is included in
`licenses/PYTORCH.txt`.

## GeoNames postal-code data

`dgp_data/geonames_us_zip_data/US.txt` is derived from the GeoNames postal-code
dataset and is distributed under the Creative Commons Attribution 4.0
license. Source and attribution details are recorded in the adjacent
`readme.txt`.

## Carlton Northern nickname data

`dgp_data/corruption/nicknames/names.csv` comes from the `nicknames` dataset by
Carlton Northern and is distributed under the Apache License 2.0. Its license
and upstream documentation are included beside the data file.

## North American area-code data

`dgp_data/area_codes/area_codes_bennetyeedogorg_ucsd_pages_area_html.csv`
contains factual area-code mappings derived from the source documented in the
adjacent `README.md`. IdentityPFN uses the mapping only to preserve plausible
state/phone dependencies in synthetic records.

## Installed dependencies

Other libraries listed in `pyproject.toml` are installed as dependencies and
are not vendored into this repository. Their respective upstream licenses
apply.
