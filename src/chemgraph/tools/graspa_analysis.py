"""Record normalization and deterministic gRASPA analysis shared by clients and MCP."""

from collections import defaultdict
import csv
import json
import math
from pathlib import Path, PureWindowsPath
import uuid

from chemgraph.schemas.graspa_analysis import GraspaAnalysis


def write_json(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_records(path: str) -> list[dict]:
    from chemgraph.tools.ase_core import _resolve_existing_path

    source = Path(_resolve_existing_path(path))
    with source.open(encoding="utf-8") as stream:
        if source.suffix.lower() == ".csv":
            return [normalize_record(row) for row in csv.DictReader(stream)]
        return [normalize_record(json.loads(line)) for line in stream if line.strip()]


def _number(value):
    if isinstance(value, bool):
        raise ValueError("Boolean values are not measurements")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Measurements must be finite")
    return result


def normalize_record(record: dict) -> dict:
    if not isinstance(record, dict):
        raise ValueError("Simulation records must be objects")
    result = dict(record)
    source = result.get("input_structure_file") or result.get("cif_path")
    if not source and result.get("cif_base_path") and result.get("cif_filename"):
        base = result["cif_base_path"]
        source = (str(PureWindowsPath(base) / result["cif_filename"])
                  if PureWindowsPath(base).is_absolute()
                  else str(Path(base) / result["cif_filename"]))
    if not isinstance(source, str) or not source:
        raise ValueError("Missing original structure identity; a CIF basename is insufficient")
    result["input_structure_file"] = source
    for canonical, old in (("temperature_in_K", "temperature"), ("pressure_in_Pa", "pressure")):
        value = result.get(canonical)
        if value in (None, ""):
            value = result.get(old)
        try:
            result[canonical] = _number(value)
        except (TypeError, ValueError):
            result[canonical] = None
    status = result.get("status") or "success"  # Historical CSVs omit status.
    result["status"] = status
    result["is_mock"] = result.get("is_mock") in (True, "True", "true", "1")
    try:
        value = _number(result.get("uptake_in_mol_kg"))
        if status != "success" or value < 0 or result["is_mock"]:
            raise ValueError("Failed, negative, or mock uptake")
        if result["temperature_in_K"] is None or result["temperature_in_K"] <= 0:
            raise ValueError("Missing or invalid temperature")
        if result["pressure_in_Pa"] is None or result["pressure_in_Pa"] < 0:
            raise ValueError("Missing or invalid pressure")
        result["uptake_in_mol_kg"] = value
    except (TypeError, ValueError) as exc:
        result.update(status="failure", uptake_in_mol_kg=None)
        result["message"] = result.get("message") or str(exc)
    return result


def write_records(path: Path, records: list[dict]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in records))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


RECORD_COLUMNS = [
    "input_structure_file", "job_id", "status", "temperature_in_K", "pressure_in_Pa",
    "uptake_in_mol_kg", "is_mock", "run_dir", "stdout_path", "stderr_path",
    "results_path", "cif_path", "error_type", "message",
]


def ranking_columns(analysis: GraspaAnalysis) -> list[str]:
    """Use the same ranking fields in JSON rows and both CSV interfaces."""
    if analysis.desorption is not None:
        return ["input_structure_file", "uptake_ads", "uptake_des", "working_capacity"]
    return ["input_structure_file", "absolute_uptake"]


def rank_records(records: list[dict], analysis: GraspaAnalysis) -> tuple[list[dict], list[dict]]:
    """Rank exact conditions; every requested repeat at those conditions must succeed."""
    grouped = defaultdict(lambda: defaultdict(list))
    for raw in records:
        row = normalize_record(raw)
        grouped[row["input_structure_file"]][
            (row["temperature_in_K"], row["pressure_in_Pa"])
        ].append(row)
    ads = (analysis.adsorption.temperature, analysis.adsorption.pressure)
    des = ((analysis.desorption.temperature, analysis.desorption.pressure)
           if analysis.desorption else None)
    ranked, excluded = [], []
    for source, groups in grouped.items():
        if any(t is None or p is None for t, p in groups):
            # An unidentified legacy failure could be a repeat at either point.
            excluded.append({"input_structure_file": source, "reason": "Record has missing conditions"})
            continue
        ads_rows, des_rows = groups.get(ads, []), groups.get(des, []) if des else []
        needed = [ads_rows, des_rows] if des else [ads_rows]
        if any(not rows or any(r["status"] != "success" for r in rows) for rows in needed):
            excluded.append({"input_structure_file": source, "reason": "Missing or failed requested condition/repeat"})
            continue
        mean_ads = _mean_uptake(ads_rows)
        row = {"input_structure_file": source}
        if des:
            mean_des = _mean_uptake(des_rows)
            row.update(uptake_ads=mean_ads, uptake_des=mean_des, working_capacity=mean_ads - mean_des)
        else:
            row["absolute_uptake"] = mean_ads
        ranked.append(row)
    metric = "working_capacity" if des else "absolute_uptake"
    ranked.sort(key=lambda row: (-row[metric], row["input_structure_file"]))
    return ranked, excluded


def _mean_uptake(rows):
    # Scale finite nonnegative values so their sum cannot overflow.
    scale = max(row["uptake_in_mol_kg"] for row in rows)
    return scale * (math.fsum(row["uptake_in_mol_kg"] / scale for row in rows) / len(rows)) if scale else 0.0


def analyze_records(records: list[dict], root: Path, analysis: GraspaAnalysis | None) -> dict:
    """Write complete artifacts and return only a bounded, model-facing summary."""
    rows = [normalize_record(row) for row in records]
    write_records(root / "results.jsonl", rows)
    write_csv(root / "results.csv", rows, RECORD_COLUMNS)
    failed = sum(row["status"] != "success" for row in rows)
    summary = {"total_records": len(rows), "failed_records": failed, "units": "mol/kg",
               "results_jsonl": str(root / "results.jsonl"), "results_csv": str(root / "results.csv")}
    if analysis is not None:
        ranked, excluded = rank_records(rows, analysis)
        selected = ranked[:math.ceil(analysis.top_fraction * len(ranked))]
        columns = ranking_columns(analysis)
        write_csv(root / "rankings.csv", ranked, columns)
        write_csv(root / "top_candidates.csv", selected, columns)
        write_json(root / "excluded.json", excluded)
        summary.update(conditions=analysis.model_dump(mode="json"),
                       valid_structures=len(ranked), excluded_structures=len(excluded),
                       selected_structures=len(selected), top_fraction=analysis.top_fraction,
                       rankings_csv=str(root / "rankings.csv"),
                       top_candidates_csv=str(root / "top_candidates.csv"),
                       excluded_json=str(root / "excluded.json"), preview=selected[:5])
    else:
        summary["preview"] = [{key: row.get(key) for key in RECORD_COLUMNS[:6]} for row in rows[:5]]
    summary["status"] = ("failed" if not rows or failed == len(rows)
                         else "partial" if failed or summary.get("excluded_structures")
                         else "completed")
    write_json(root / "analysis.json", summary)
    return summary
