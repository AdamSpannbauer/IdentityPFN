"""Create non-destructive single-table benchmark CSVs from raw benchmark data.

The current diagnostics expect one table with an entity ID column. This script
keeps the raw linkage files untouched and writes derived CSVs under
benchmark_data/converted/.

Assumptions made here:
- For two-table linkage data, rows connected by any match edge are one entity.
  This uses connected components rather than assuming one-to-one matches.
- Unmatched source rows are retained as singleton entities.
- Source tables are assumed deduplicated enough that unmatched rows from the
  same source table should not be assigned shared entity IDs.
- Dataset-specific column mappings are intentionally conservative and exclude
  obvious label/helper columns from the model-facing fields.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "converted"


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


@dataclass(frozen=True)
class LinkageConversion:
    name: str
    directory: str
    left_id_column: str
    right_id_column: str
    match_left_column: str
    match_right_column: str
    build_records: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype=str, encoding="latin1")


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.replace({"": pd.NA, "nan": pd.NA})


def combine_text(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    available = [column for column in columns if column in frame]
    if not available:
        return pd.Series(pd.NA, index=frame.index, dtype="object")
    values = frame[available].fillna("").astype(str)
    combined = values.apply(
        lambda row: " ".join(value.strip() for value in row if value.strip()),
        axis=1,
    )
    return combined.replace("", pd.NA)


def clean_numeric_text(values: pd.Series) -> pd.Series:
    numeric_text = values.astype("string").str.extract(r"([-+]?\d+(?:\.\d+)?)")[0]
    return pd.to_numeric(numeric_text, errors="coerce")


def add_metadata(
    frame: pd.DataFrame, source_table: str, source_id_column: str
) -> pd.DataFrame:
    frame = frame.copy()
    frame.insert(0, "source_id", frame[source_id_column].astype(str))
    frame.insert(0, "source_table", source_table)
    return frame


def build_amazon_google(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    left_records = add_metadata(
        pd.DataFrame(
            {
                "id": left["id"],
                "title": left["title"],
                "description": left["description"],
                "manufacturer": left["manufacturer"],
                "price": left["price"],
            }
        ),
        "A",
        "id",
    )
    right_records = add_metadata(
        pd.DataFrame(
            {
                "id": right["id"],
                "title": right["name"],
                "description": right["description"],
                "manufacturer": right["manufacturer"],
                "price": right["price"],
            }
        ),
        "B",
        "id",
    )
    return pd.concat([left_records, right_records], ignore_index=True)


def make_numeric_price_variant(records: pd.DataFrame) -> pd.DataFrame:
    variant = records.copy()
    variant["price"] = clean_numeric_text(variant["price"])
    return variant


def make_both_price_variant(records: pd.DataFrame) -> pd.DataFrame:
    variant = records.rename(columns={"price": "price_text"}).copy()
    variant["price_numeric"] = clean_numeric_text(variant["price_text"])
    return variant


def build_bibliographic(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    columns = ["id", "title", "authors", "venue", "year"]
    left_records = add_metadata(left[columns], "A", "id")
    right_records = add_metadata(right[columns], "B", "id")
    return pd.concat([left_records, right_records], ignore_index=True)


def build_fodors_zagats(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    # The raw "class" columns are helper/label artifacts, so they are excluded.
    columns = ["id", "name", "addr", "city", "phone", "type"]
    left_records = add_metadata(left[columns], "A", "id")
    right_records = add_metadata(right[columns], "B", "id")
    return pd.concat([left_records, right_records], ignore_index=True)


def build_walmart_amazon(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    left_records = pd.DataFrame(
        {
            "id": left["custom_id"],
            "title": left["title"],
            "brand": left["brand"],
            "modelno": left["modelno"],
            "category": left["groupname"],
            "price": left["price"],
            "description": combine_text(
                left, ["shelfdescr", "shortdescr", "longdescr"]
            ),
        }
    )
    right_records = pd.DataFrame(
        {
            "id": right["custom_id"],
            "title": right["title"],
            "brand": right["brand"],
            "modelno": right["modelno"],
            "category": right["category1"],
            "price": right["price"],
            "description": combine_text(
                right,
                ["prodfeatures", "techdetails", "proddescrshort", "proddescrlong"],
            ),
        }
    )
    return pd.concat(
        [
            add_metadata(left_records, "A", "id"),
            add_metadata(right_records, "B", "id"),
        ],
        ignore_index=True,
    )


CONVERSIONS = (
    LinkageConversion(
        name="amazon_google",
        directory="deepmatcher_provided/Amazon-Google",
        left_id_column="id",
        right_id_column="id",
        match_left_column="idAmazon",
        match_right_column="idGoogleBase",
        build_records=build_amazon_google,
    ),
    LinkageConversion(
        name="dblp_acm",
        directory="deepmatcher_provided/DBLP-ACM",
        left_id_column="id",
        right_id_column="id",
        match_left_column="idDBLP",
        match_right_column="idACM",
        build_records=build_bibliographic,
    ),
    LinkageConversion(
        name="dblp_google_scholar",
        directory="deepmatcher_provided/DBLP-GoogleScholar",
        left_id_column="id",
        right_id_column="id",
        match_left_column="idDBLP",
        match_right_column="idScholar",
        build_records=build_bibliographic,
    ),
    LinkageConversion(
        name="fodors_zagats",
        directory="deepmatcher_provided/Fodors-Zagats",
        left_id_column="id",
        right_id_column="id",
        match_left_column="fodors_id",
        match_right_column="zagats_id",
        build_records=build_fodors_zagats,
    ),
    LinkageConversion(
        name="walmart_amazon",
        directory="deepmatcher_provided/Walmart-Amazon",
        left_id_column="custom_id",
        right_id_column="custom_id",
        match_left_column="id1",
        match_right_column="id2",
        build_records=build_walmart_amazon,
    ),
)


NC_VOTER_FILES = (
    "nc_voters/ncvr_numrec_1000000_modrec_2_ocp_20_myp_0_nump_5.csv",
    "nc_voters/ncvr_numrec_1000000_modrec_2_ocp_20_myp_1_nump_5.csv",
    "nc_voters/ncvr_numrec_1000000_modrec_2_ocp_20_myp_2_nump_5.csv",
    "nc_voters/ncvr_numrec_1000000_modrec_2_ocp_20_myp_3_nump_5.csv",
    "nc_voters/ncvr_numrec_1000000_modrec_2_ocp_20_myp_4_nump_5.csv",
)


BPID_MATCHING_PATH = ROOT / "bpid" / "matching_dataset.jsonl"
BPID_MATCHING_BALANCED_TRUE_PAIRS = 500
BPID_MATCHING_BALANCED_FALSE_PAIRS = 500
ICE_ID_PEOPLE_PATH = ROOT / "ice_id" / "raw_data" / "people.csv"


def source_key(source_table: str, source_id) -> str:
    return f"{source_table}:{source_id}"


def convert_linkage_dataset(config: LinkageConversion) -> pd.DataFrame:
    directory = ROOT / config.directory
    left = clean_frame(read_csv(directory / "tableA.csv"))
    right = clean_frame(read_csv(directory / "tableB.csv"))
    matches = clean_frame(read_csv(directory / "matches.csv"))

    records = config.build_records(left, right)
    records["source_key"] = [
        source_key(source_table, source_id)
        for source_table, source_id in zip(records["source_table"], records["source_id"])
    ]

    union_find = UnionFind()
    for key in records["source_key"]:
        union_find.find(key)
    for match in matches.itertuples(index=False):
        left_id = getattr(match, config.match_left_column)
        right_id = getattr(match, config.match_right_column)
        union_find.union(source_key("A", left_id), source_key("B", right_id))

    roots = records["source_key"].map(union_find.find)
    root_ids = {root: f"{config.name}:{index}" for index, root in enumerate(sorted(set(roots)))}
    records.insert(0, "entity_id", roots.map(root_ids))
    records = records.drop(columns=["source_key"])
    return records


def convert_nc_voters() -> Path:
    output_path = OUTPUT_DIR / "nc_voters.csv"
    fieldnames = [
        "entity_id",
        "source_table",
        "source_id",
        "givenname",
        "surname",
        "suburb",
        "postcode",
    ]
    rows_written = 0
    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for index, relative_path in enumerate(NC_VOTER_FILES):
            source_table = f"myp_{index}"
            with (ROOT / relative_path).open(newline="") as input_file:
                reader = csv.DictReader(input_file)
                for row in reader:
                    writer.writerow(
                        {
                            "entity_id": row["recid"],
                            "source_table": source_table,
                            "source_id": row["recid"],
                            "givenname": row["givenname"],
                            "surname": row["surname"],
                            "suburb": row["suburb"],
                            "postcode": row["postcode"],
                        }
                    )
                    rows_written += 1
    print(f"nc_voters: wrote {rows_written:,} records to {output_path}")
    return output_path


def join_bpid_values(values: list[str]) -> str | pd.NA:
    joined = " | ".join(value for value in values if value)
    return joined if joined else pd.NA


def bpid_profile_row(pair_index: int, side: str, profile: dict, match: bool) -> dict:
    entity_id = (
        f"bpid_matching:{pair_index}"
        if match
        else f"bpid_matching:{pair_index}:{side}"
    )
    return {
        "entity_id": entity_id,
        "pair_id": pair_index,
        "source_table": side,
        "source_id": f"{pair_index}:{side}",
        "match_label": str(match),
        "fullname": profile.get("fullname") or pd.NA,
        "email": join_bpid_values(profile.get("email", [])),
        "phone": join_bpid_values(profile.get("phone", [])),
        "addr": join_bpid_values(profile.get("addr", [])),
        "dob": profile.get("dob") or pd.NA,
    }


def convert_bpid_matching_balanced() -> Path:
    output_path = OUTPUT_DIR / "bpid_matching_balanced.csv"
    rows = []
    true_pairs = 0
    false_pairs = 0
    with BPID_MATCHING_PATH.open() as input_file:
        for pair_index, line in enumerate(input_file):
            record = json.loads(line)
            match = record["match"] == "True"
            if match:
                if true_pairs >= BPID_MATCHING_BALANCED_TRUE_PAIRS:
                    continue
                true_pairs += 1
            else:
                if false_pairs >= BPID_MATCHING_BALANCED_FALSE_PAIRS:
                    continue
                false_pairs += 1

            rows.append(bpid_profile_row(pair_index, "profile1", record["profile1"], match))
            rows.append(bpid_profile_row(pair_index, "profile2", record["profile2"], match))
            if (
                true_pairs >= BPID_MATCHING_BALANCED_TRUE_PAIRS
                and false_pairs >= BPID_MATCHING_BALANCED_FALSE_PAIRS
            ):
                break

    records = pd.DataFrame(rows)
    records.to_csv(output_path, index=False)
    print(
        f"bpid_matching_balanced: wrote {len(records):,} records, "
        f"{records['entity_id'].nunique():,} entities, "
        f"{true_pairs:,} true pairs and {false_pairs:,} false pairs to {output_path}"
    )
    return output_path


def convert_ice_id_people() -> Path:
    output_path = OUTPUT_DIR / "ice_id_people.csv"
    columns = [
        "id",
        "heimild",
        "nafn_norm",
        "first_name",
        "middle_name",
        "patronym",
        "surname",
        "birthyear",
        "sex",
        "status",
        "marriagestatus",
        "person",
        "partner",
        "father",
        "mother",
        "farm",
        "county",
        "parish",
        "district",
    ]
    records = pd.read_csv(ICE_ID_PEOPLE_PATH, usecols=columns, dtype=str)
    records = records[records["person"].notna()].copy()
    records.insert(0, "source_table", "people")
    records = records.rename(
        columns={
            "id": "source_id",
            "person": "entity_id",
            "heimild": "census_year",
            "nafn_norm": "full_name",
        }
    )
    ordered_columns = [
        "entity_id",
        "source_table",
        "source_id",
        "census_year",
        "full_name",
        "first_name",
        "middle_name",
        "patronym",
        "surname",
        "birthyear",
        "sex",
        "status",
        "marriagestatus",
        "partner",
        "father",
        "mother",
        "farm",
        "county",
        "parish",
        "district",
    ]
    records = records[ordered_columns]
    records.to_csv(output_path, index=False)
    print(
        f"ice_id_people: wrote {len(records):,} records, "
        f"{records['entity_id'].nunique():,} entities to {output_path}"
    )
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    for config in CONVERSIONS:
        records = convert_linkage_dataset(config)
        output_path = OUTPUT_DIR / f"{config.name}.csv"
        records.to_csv(output_path, index=False)
        if config.name == "amazon_google":
            numeric_path = OUTPUT_DIR / "amazon_google_price_numeric.csv"
            both_path = OUTPUT_DIR / "amazon_google_price_both.csv"
            make_numeric_price_variant(records).to_csv(numeric_path, index=False)
            make_both_price_variant(records).to_csv(both_path, index=False)
            print(f"amazon_google_price_numeric: wrote {len(records):,} records to {numeric_path}")
            print(f"amazon_google_price_both: wrote {len(records):,} records to {both_path}")

        source_counts = (
            records.groupby(["entity_id", "source_table"]).size().unstack(fill_value=0)
        )
        print(
            f"{config.name}: wrote {len(records):,} records, "
            f"{records['entity_id'].nunique():,} entities to {output_path}"
        )
        for source_table in sorted(records["source_table"].unique()):
            max_per_entity = int(source_counts[source_table].max())
            entities_over_one = int((source_counts[source_table] > 1).sum())
            print(
                f"  source {source_table}: max records/entity={max_per_entity}; "
                f"entities >1={entities_over_one:,}"
        )
    convert_nc_voters()
    convert_bpid_matching_balanced()
    convert_ice_id_people()


if __name__ == "__main__":
    main()
