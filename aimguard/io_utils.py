"""Consistent ID handling: IDs are ALWAYS strings, so '007' in a CSV matches '007' in JSON evidence."""
import pandas as pd


def norm_id(x):
    return str(x).strip()


def read_engagement_csv(path):
    df = pd.read_csv(path, dtype={"player_id": str, "match_id": str}, skipinitialspace=True)
    for c in ("player_id", "match_id"):
        if c in df.columns:
            df[c] = df[c].map(norm_id)
    return df


def normalize_reports(reports: dict) -> dict:
    return {norm_id(k): v for k, v in reports.items()}


def read_id_list(path):
    with open(path) as f:
        return {norm_id(l.split("#")[0]) for l in f if l.split("#")[0].strip()}
