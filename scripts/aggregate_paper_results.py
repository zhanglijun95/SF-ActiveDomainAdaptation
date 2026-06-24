#!/usr/bin/env python3
"""Aggregate paper-facing DAOD results by dataset pair and detector.

The wave summaries are operational logs. This script builds the paper table:
source/oracle anchors, pure SFDA, +random anchors, and realistic plugin
families grouped by dataset pair and detector.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


OUT_DIR = Path("runs/negative_results_summary")
BASELINES = ("ddt_daod", "lpld_daod", "lpu_daod", "pets_daod")
SEEDS = (42, 43, 44)

METHOD_ORDER = {
    "source_only": 0,
    "oracle": 1,
    "pure_sfod": 10,
    "random": 20,
    "selection/threshold_calibration": 30,
    "selection/threshold_mapping": 31,
    "selection/pseudo_score_reweight": 32,
    "completion/query_recovery_scorer": 40,
    "completion/query_recovery_multiview": 41,
    "completion/query_revival_scorer": 42,
    "completion/query_revival_multiview": 43,
    "control/pcgrad_pseudo_only": 50,
    "control/sparse_loss_balance": 51,
    "control/label_guided_aema": 52,
}


@dataclass(frozen=True)
class RunRecord:
    dataset: str
    detector: str
    sfod_method: str
    family: str
    method: str
    label_ratio: float | None
    seed: int | None
    exp_name: str
    run_dir: str
    metric_source: str
    ap50: float
    ap: float
    ap75: float | None


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _bbox(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    bbox = payload.get("bbox")
    return bbox if isinstance(bbox, dict) else {}


def _float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _seed(exp_name: str) -> int | None:
    match = re.search(r"(?:^|_)seed(\d+)(?:_|$)", exp_name)
    return int(match.group(1)) if match else None


def _budget(exp_name: str) -> float | None:
    match = re.search(r"(?:^|_)budget(\d+)(?:_|$)", exp_name)
    if not match:
        return None
    raw = match.group(1)
    # Historical config names use budget001/005/010 for 1%/5%/10%.
    return int(raw) / 100.0


def _classify(exp_name: str) -> tuple[str, str, float | None]:
    lower = exp_name.lower()
    budget = _budget(lower)
    # BDD/Foggy correction tags live between ``random`` and the actual plugin
    # name. Strip them for classification while keeping the original exp_name
    # for priority and traceability.
    normalized = lower
    normalized = re.sub(
        r"_random_(?:dynfix|bddfix|static_anchor|officialdt)"
        r"(?:_pst\d+(?:p\d+)?_lscl\d+(?:p\d+)?)?_",
        "_random_",
        normalized,
    )
    if "_pilot_" in lower:
        return "unclassified", "unclassified", budget
    if "_random_selection_threshold_calibration_" in normalized:
        return "selection", "selection/threshold_calibration", budget
    if "_random_selection_threshold_mapping_" in normalized:
        return "selection", "selection/threshold_mapping", budget
    if "_random_selection_pseudo_score_reweight_" in normalized:
        return "selection", "selection/pseudo_score_reweight", budget
    if "_random_query_recovery_scorer_" in normalized:
        return "completion", "completion/query_recovery_scorer", budget
    if "_random_query_recovery_multiview_" in normalized:
        return "completion", "completion/query_recovery_multiview", budget
    if "_random_query_revival_scorer_" in normalized:
        return "completion", "completion/query_revival_scorer", budget
    if "_random_query_revival_multiview_" in normalized:
        return "completion", "completion/query_revival_multiview", budget
    if "_random_control_target_anchored_pcgrad_pseudo_only_" in normalized:
        return "optimization_control", "control/pcgrad_pseudo_only", budget
    if "_random_control_sparse_loss_balance_" in normalized:
        return "optimization_control", "control/sparse_loss_balance", budget
    if "_random_control_label_guided_aema_" in normalized:
        return "optimization_control", "control/label_guided_aema", budget
    if re.search(
        r"_random(?:_(?:dynfix|bddfix|static_anchor|officialdt))"
        r"*(?:_pst\d+(?:p\d+)?_lscl\d+(?:p\d+)?)?_budget\d+_seed\d+$",
        lower,
    ):
        return "random_supervised_anchor", "random", budget
    if re.search(
        r"_(?:dino|deta)(?:_(?:dynfix|bddfix|static_anchor|officialdt))"
        r"*(?:_pst\d+(?:p\d+)?_lscl\d+(?:p\d+)?)?_seed\d+$",
        lower,
    ):
        return "pure_sfda_anchor", "pure_sfod", 0.0
    if "_oracle_" in lower:
        return "oracle_diagnostic", "oracle_diagnostic", budget
    return "unclassified", "unclassified", budget


def _record_priority(record: RunRecord) -> int:
    """Prefer corrected full BDD reruns over the earlier broken BDD configs."""
    lower = record.exp_name.lower()
    metric_priority = {
        "best_intermediate": 1,
        "summary": 2,
        "student_target_val": 3,
        "target_val": 4,
    }.get(record.metric_source, 0)
    exp_priority = 1
    if "_dynfix_" in lower or "_bddfix_" in lower or "_static_anchor_" in lower or "_officialdt_" in lower:
        exp_priority = 2
    return exp_priority * 10 + metric_priority


def _metric_source(path: Path) -> str:
    if path.name == "best_student_target_val_metrics.json":
        return "best_intermediate"
    if path.name == "student_target_val_metrics.json":
        return "student_target_val"
    if path.name == "summary.json":
        return "summary"
    return "target_val"


def _baseline_bbox(path: Path) -> dict[str, Any]:
    if path.name == "summary.json":
        payload = _load_json(path)
        for key in ("final_target_val_metrics", "teacher_target_val_metrics", "student_target_val_metrics"):
            metrics = payload.get(key)
            if isinstance(metrics, dict) and isinstance(metrics.get("bbox"), dict):
                return metrics["bbox"]
        return {}
    return _bbox(path)


def _record_from_baseline_metrics(path: Path) -> RunRecord | None:
    parts = list(path.parts)
    if "baselines" not in parts:
        return None
    idx = parts.index("baselines")
    if idx + 5 >= len(parts):
        return None
    sfod_method = parts[idx + 1]
    exp_name = parts[idx + 2]
    dataset = parts[idx + 3]
    detector = parts[idx + 4]
    if sfod_method not in BASELINES:
        return None
    family, method, label_ratio = _classify(exp_name)
    if method in {"unclassified", "oracle_diagnostic"}:
        return None
    metrics = _baseline_bbox(path)
    ap50 = _float(metrics.get("AP50"))
    ap = _float(metrics.get("AP"))
    if ap50 is None or ap is None:
        return None
    return RunRecord(
        dataset=dataset,
        detector=detector,
        sfod_method=sfod_method,
        family=family,
        method=method,
        label_ratio=label_ratio,
        seed=_seed(exp_name),
        exp_name=exp_name,
        run_dir=str(path.parent),
        metric_source=_metric_source(path),
        ap50=ap50,
        ap=ap,
        ap75=_float(metrics.get("AP75")),
    )


def _anchor_records(root: Path, family: str, method: str) -> list[RunRecord]:
    records: list[RunRecord] = []
    for path in sorted(root.glob("*/*/target_val_metrics.json")):
        dataset = path.parent.parent.name
        detector = path.parent.name
        metrics = _bbox(path)
        ap50 = _float(metrics.get("AP50"))
        ap = _float(metrics.get("AP"))
        if ap50 is None or ap is None:
            continue
        records.append(
            RunRecord(
                dataset=dataset,
                detector=detector,
                sfod_method="all",
                family=family,
                method=method,
                label_ratio=None,
                seed=None,
                exp_name=method,
                run_dir=str(path.parent),
                metric_source="target_val",
                ap50=ap50,
                ap=ap,
                ap75=_float(metrics.get("AP75")),
            )
        )
    return records


def _scan_records() -> list[RunRecord]:
    records: list[RunRecord] = []
    baseline_run_dirs_with_final_metrics: set[Path] = set()
    for path in sorted(Path("runs/baselines").glob("*/*/*/*/target_val_metrics.json")):
        record = _record_from_baseline_metrics(path)
        if record is not None:
            records.append(record)
            baseline_run_dirs_with_final_metrics.add(path.parent)
    for path in sorted(Path("runs/baselines").glob("*/*/*/*/student_target_val_metrics.json")):
        if path.parent in baseline_run_dirs_with_final_metrics:
            continue
        record = _record_from_baseline_metrics(path)
        if record is not None:
            records.append(record)
            baseline_run_dirs_with_final_metrics.add(path.parent)
    for path in sorted(Path("runs/baselines").glob("*/*/*/*/summary.json")):
        if path.parent in baseline_run_dirs_with_final_metrics:
            continue
        record = _record_from_baseline_metrics(path)
        if record is not None:
            records.append(record)
            baseline_run_dirs_with_final_metrics.add(path.parent)
    for path in sorted(Path("runs/baselines").glob("*/*/*/*/best_student_target_val_metrics.json")):
        if path.parent in baseline_run_dirs_with_final_metrics:
            continue
        record = _record_from_baseline_metrics(path)
        if record is not None:
            records.append(record)
    records.extend(_anchor_records(Path("runs/daod_source"), "detector_anchor", "source_only"))
    records.extend(_anchor_records(Path("runs/daod_oracle"), "detector_anchor", "oracle"))
    return records


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def _label(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}"


def _seed_values(records: list[RunRecord], attr: str) -> str:
    pairs = []
    for record in sorted(records, key=lambda row: (-1 if row.seed is None else row.seed)):
        value = getattr(record, attr)
        seed = "NA" if record.seed is None else str(record.seed)
        pairs.append(f"s{seed}={value:.3f}")
    return ", ".join(pairs)


def _seed_sources(records: list[RunRecord]) -> str:
    pairs = []
    for record in sorted(records, key=lambda row: (-1 if row.seed is None else row.seed)):
        seed = "NA" if record.seed is None else str(record.seed)
        pairs.append(f"s{seed}={record.metric_source}")
    return ", ".join(pairs)


def _source_counts(records: list[RunRecord]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[record.metric_source] += 1
    return ", ".join(f"{name}={counts[name]}" for name in sorted(counts))


def _aggregate(records: list[RunRecord]) -> list[dict[str, Any]]:
    deduped: dict[tuple[Any, ...], RunRecord] = {}
    for record in records:
        dedupe_key = (
            record.dataset,
            record.detector,
            record.sfod_method,
            record.family,
            record.method,
            record.label_ratio,
            record.seed,
        )
        previous = deduped.get(dedupe_key)
        if previous is None or _record_priority(record) >= _record_priority(previous):
            deduped[dedupe_key] = record
    records = list(deduped.values())

    grouped: dict[tuple[Any, ...], list[RunRecord]] = defaultdict(list)
    for record in records:
        key = (
            record.dataset,
            record.detector,
            record.sfod_method,
            record.family,
            record.method,
            record.label_ratio,
        )
        grouped[key].append(record)

    rows: list[dict[str, Any]] = []
    for (dataset, detector, sfod_method, family, method, label_ratio), group in grouped.items():
        ap50s = [record.ap50 for record in group]
        aps = [record.ap for record in group]
        ap75s = [record.ap75 for record in group if record.ap75 is not None]
        seeds = sorted(record.seed for record in group if record.seed is not None)
        missing = [seed for seed in SEEDS if seed not in seeds] if sfod_method != "all" else []
        rows.append(
            {
                "dataset": dataset,
                "detector": detector,
                "sfod_method": sfod_method,
                "family": family,
                "method": method,
                "label_ratio": label_ratio,
                "n": len(group),
                "missing_seeds": ",".join(str(seed) for seed in missing),
                "AP50_mean": _mean(ap50s),
                "AP50_std": _std(ap50s),
                "AP_mean": _mean(aps),
                "AP_std": _std(aps),
                "AP75_mean": _mean(ap75s),
                "AP75_std": _std(ap75s),
                "seed_AP50": _seed_values(group, "ap50"),
                "seed_AP": _seed_values(group, "ap"),
                "metric_sources": _source_counts(group),
                "seed_metric_sources": _seed_sources(group),
                "run_dirs": " | ".join(sorted(record.run_dir for record in group)),
            }
        )

    source = {
        (row["dataset"], row["detector"]): row["AP50_mean"]
        for row in rows
        if row["method"] == "source_only"
    }
    oracle = {
        (row["dataset"], row["detector"]): row["AP50_mean"]
        for row in rows
        if row["method"] == "oracle"
    }
    pure = {
        (row["dataset"], row["detector"], row["sfod_method"]): row["AP50_mean"]
        for row in rows
        if row["method"] == "pure_sfod"
    }
    random_anchor = {
        (row["dataset"], row["detector"], row["sfod_method"], row["label_ratio"]): row["AP50_mean"]
        for row in rows
        if row["method"] == "random"
    }

    for row in rows:
        bench = (row["dataset"], row["detector"])
        pure_key = (row["dataset"], row["detector"], row["sfod_method"])
        random_key = (
            row["dataset"],
            row["detector"],
            row["sfod_method"],
            row["label_ratio"],
        )
        ap50 = row["AP50_mean"]
        row["delta_vs_source_AP50"] = (
            None if ap50 is None or source.get(bench) is None else ap50 - source[bench]
        )
        row["gap_to_oracle_AP50"] = (
            None if ap50 is None or oracle.get(bench) is None else oracle[bench] - ap50
        )
        if ap50 is None or source.get(bench) is None or oracle.get(bench) is None:
            row["oracle_gap_closed_pct"] = None
        else:
            denom = oracle[bench] - source[bench]
            row["oracle_gap_closed_pct"] = None if denom == 0 else 100.0 * (ap50 - source[bench]) / denom
        row["delta_vs_pure_AP50"] = (
            None if ap50 is None or pure.get(pure_key) is None else ap50 - pure[pure_key]
        )
        row["delta_vs_random_AP50"] = (
            None if ap50 is None or random_anchor.get(random_key) is None else ap50 - random_anchor[random_key]
        )

    rows.sort(
        key=lambda row: (
            str(row["dataset"]),
            str(row["detector"]),
            BASELINES.index(row["sfod_method"]) if row["sfod_method"] in BASELINES else -1,
            METHOD_ORDER.get(row["method"], 999),
            -1.0 if row["label_ratio"] is None else float(row["label_ratio"]),
        )
    )
    return rows


CSV_COLUMNS = [
    "dataset",
    "detector",
    "sfod_method",
    "family",
    "method",
    "label_ratio",
    "n",
    "missing_seeds",
    "AP50_mean",
    "AP50_std",
    "AP_mean",
    "AP_std",
    "AP75_mean",
    "AP75_std",
    "delta_vs_source_AP50",
    "delta_vs_pure_AP50",
    "delta_vs_random_AP50",
    "gap_to_oracle_AP50",
    "oracle_gap_closed_pct",
    "seed_AP50",
    "seed_AP",
    "metric_sources",
    "seed_metric_sources",
    "run_dirs",
]


DISPLAY_COLUMNS = [
    "sfod_method",
    "family",
    "method",
    "label_ratio",
    "n",
    "metric_sources",
    "AP50_mean",
    "AP50_std",
    "AP_mean",
    "AP_std",
    "delta_vs_source_AP50",
    "delta_vs_pure_AP50",
    "delta_vs_random_AP50",
    "gap_to_oracle_AP50",
    "oracle_gap_closed_pct",
    "seed_AP50",
]


BENCHMARK_OVERVIEW_COLUMNS = [
    "benchmark",
    "sfod_method",
    "source_AP50",
    "oracle_AP50",
    "pure_AP50",
    "pure_std",
    "random_AP50",
    "random_std",
    "random_minus_pure",
    "random_gap_to_oracle",
    "random_oracle_gap_closed_pct",
]


BEST_PLUGIN_COLUMNS = [
    "benchmark",
    "sfod_method",
    "pure_AP50",
    "random_AP50",
    "best_family",
    "best_plugin",
    "best_AP50",
    "best_std",
    "best_delta_vs_random",
    "best_delta_vs_pure",
    "best_gap_to_oracle",
    "worst_plugin",
    "worst_AP50",
    "worst_delta_vs_random",
    "positive_plugins",
    "total_plugins",
    "mean_plugin_delta",
    "median_plugin_delta",
]


FAMILY_SUMMARY_COLUMNS = [
    "benchmark",
    "family",
    "settings",
    "seed_runs",
    "mean_delta_vs_random",
    "median_delta_vs_random",
    "positive_settings",
    "positive_rate",
    "best_plugin",
    "best_sfod_method",
    "best_delta_vs_random",
    "best_AP50",
    "worst_plugin",
    "worst_sfod_method",
    "worst_delta_vs_random",
    "worst_AP50",
]


VARIANT_SUMMARY_COLUMNS = [
    "benchmark",
    "family",
    "plugin",
    "settings",
    "seed_runs",
    "mean_delta_vs_random",
    "median_delta_vs_random",
    "positive_settings",
    "positive_rate",
    "best_sfod_method",
    "best_delta_vs_random",
    "best_AP50",
    "worst_sfod_method",
    "worst_delta_vs_random",
    "worst_AP50",
]


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def _md_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, float):
                values.append(_fmt(value))
            elif value is None:
                values.append("")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_md(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    by_sheet = _benchmark_sheets(rows)
    lines = [f"# {title}", ""]
    for sheet, sheet_rows in by_sheet.items():
        lines.extend([f"## {sheet}", "", _md_table(sheet_rows, DISPLAY_COLUMNS), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _cell_ref(row_idx: int, col_idx: int) -> str:
    letters = ""
    col = col_idx
    while col:
        col, rem = divmod(col - 1, 26)
        letters = chr(65 + rem) + letters
    return f"{letters}{row_idx}"


def _sheet_xml(table: list[list[Any]]) -> str:
    rows_xml = []
    for row_idx, row in enumerate(table, start=1):
        cells = []
        for col_idx, value in enumerate(row, start=1):
            ref = _cell_ref(row_idx, col_idx)
            if value is None:
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                cells.append(f'<c r="{ref}"><v>{value}</v></c>')
            else:
                text = escape(str(value))
                cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>')
        rows_xml.append(f'<row r="{row_idx}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetData>'
        + "".join(rows_xml)
        + "</sheetData></worksheet>"
    )


def _safe_sheet_name(name: str, used: set[str]) -> str:
    cleaned = re.sub(r"[\[\]:*?/\\\\]", "_", name)[:31] or "Sheet"
    candidate = cleaned
    idx = 2
    while candidate in used:
        suffix = f"_{idx}"
        candidate = cleaned[: 31 - len(suffix)] + suffix
        idx += 1
    used.add(candidate)
    return candidate


def _write_xlsx(path: Path, sheets: dict[str, list[list[Any]]]) -> None:
    used: set[str] = set()
    named_sheets = [(_safe_sheet_name(name, used), table) for name, table in sheets.items()]
    workbook_sheets = []
    workbook_rels = []
    content_overrides = []
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
            + "".join(
                f'<Override PartName="/xl/worksheets/sheet{i}.xml" '
                'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                for i in range(1, len(named_sheets) + 1)
            )
            + "</Types>",
        )
        zf.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>",
        )
        zf.writestr(
            "xl/styles.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
            '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
            '<borders count="1"><border/></borders>'
            '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
            '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
            "</styleSheet>",
        )
        for idx, (name, table) in enumerate(named_sheets, start=1):
            workbook_sheets.append(f'<sheet name="{escape(name)}" sheetId="{idx}" r:id="rId{idx}"/>')
            workbook_rels.append(
                f'<Relationship Id="rId{idx}" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
                f'Target="worksheets/sheet{idx}.xml"/>'
            )
            content_overrides.append(idx)
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", _sheet_xml(table))
        zf.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets>'
            + "".join(workbook_sheets)
            + "</sheets></workbook>",
        )
        zf.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(workbook_rels)
            + "</Relationships>",
        )


def _benchmark_name(dataset: str, detector: str) -> str:
    if dataset == "cityscapes__to__foggy_cityscapes" and detector == "dino_r50_4scale_12ep":
        return "Foggy_DINO"
    if dataset == "cityscapes__to__foggy_cityscapes" and detector == "deta_r50_5scale_12ep_bs8":
        return "Foggy_DETA"
    if dataset == "cityscapes__to__bdd100k" and detector == "dino_r50_4scale_12ep":
        return "BDD100K_DINO"
    return f"{dataset}__{detector}"


def _benchmark_sheets(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    sheets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = _benchmark_name(str(row["dataset"]), str(row["detector"]))
        sheets.setdefault(key, []).append(row)
    return sheets


def _rows_to_table(rows: list[dict[str, Any]], columns: list[str]) -> list[list[Any]]:
    table = [columns]
    for row in rows:
        table.append([row.get(column) for column in columns])
    return table


def _plugin_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["family"] in {"selection", "completion", "optimization_control"}
        and row.get("delta_vs_random_AP50") is not None
    ]


def _benchmark_overview_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    overview: list[dict[str, Any]] = []
    benchmark_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        benchmark_groups[(str(row["dataset"]), str(row["detector"]))].append(row)

    for (dataset, detector), group in sorted(benchmark_groups.items()):
        source = next((row for row in group if row["method"] == "source_only"), None)
        oracle = next((row for row in group if row["method"] == "oracle"), None)
        for sfod_method in BASELINES:
            pure = next(
                (row for row in group if row["sfod_method"] == sfod_method and row["method"] == "pure_sfod"),
                None,
            )
            random = next(
                (row for row in group if row["sfod_method"] == sfod_method and row["method"] == "random"),
                None,
            )
            if pure is None and random is None:
                continue
            overview.append(
                {
                    "benchmark": _benchmark_name(dataset, detector),
                    "dataset": dataset,
                    "detector": detector,
                    "sfod_method": sfod_method,
                    "source_AP50": None if source is None else source.get("AP50_mean"),
                    "oracle_AP50": None if oracle is None else oracle.get("AP50_mean"),
                    "pure_AP50": None if pure is None else pure.get("AP50_mean"),
                    "pure_std": None if pure is None else pure.get("AP50_std"),
                    "random_AP50": None if random is None else random.get("AP50_mean"),
                    "random_std": None if random is None else random.get("AP50_std"),
                    "random_minus_pure": None
                    if pure is None
                    or random is None
                    or pure.get("AP50_mean") is None
                    or random.get("AP50_mean") is None
                    else random["AP50_mean"] - pure["AP50_mean"],
                    "random_gap_to_oracle": None if random is None else random.get("gap_to_oracle_AP50"),
                    "random_oracle_gap_closed_pct": None
                    if random is None
                    else random.get("oracle_gap_closed_pct"),
                }
            )
    return overview


def _method_best_plugin_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    plugins = _plugin_rows(rows)
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    anchors = {
        (row["dataset"], row["detector"], row["sfod_method"], row["method"]): row
        for row in rows
        if row["method"] in {"pure_sfod", "random"}
    }
    for row in plugins:
        groups[(str(row["dataset"]), str(row["detector"]), str(row["sfod_method"]))].append(row)

    summary: list[dict[str, Any]] = []
    for (dataset, detector, sfod_method), group in sorted(groups.items()):
        best = max(group, key=lambda row: float(row["AP50_mean"]))
        worst = min(group, key=lambda row: float(row["AP50_mean"]))
        deltas = [float(row["delta_vs_random_AP50"]) for row in group]
        pure = anchors.get((dataset, detector, sfod_method, "pure_sfod"))
        random = anchors.get((dataset, detector, sfod_method, "random"))
        summary.append(
            {
                "benchmark": _benchmark_name(dataset, detector),
                "dataset": dataset,
                "detector": detector,
                "sfod_method": sfod_method,
                "pure_AP50": None if pure is None else pure.get("AP50_mean"),
                "random_AP50": None if random is None else random.get("AP50_mean"),
                "best_family": best["family"],
                "best_plugin": best["method"],
                "best_AP50": best["AP50_mean"],
                "best_std": best["AP50_std"],
                "best_delta_vs_random": best["delta_vs_random_AP50"],
                "best_delta_vs_pure": best["delta_vs_pure_AP50"],
                "best_gap_to_oracle": best["gap_to_oracle_AP50"],
                "worst_plugin": worst["method"],
                "worst_AP50": worst["AP50_mean"],
                "worst_delta_vs_random": worst["delta_vs_random_AP50"],
                "positive_plugins": sum(delta > 0 for delta in deltas),
                "total_plugins": len(deltas),
                "mean_plugin_delta": _mean(deltas),
                "median_plugin_delta": _median(deltas),
            }
        )
    return summary


def _family_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    plugins = _plugin_rows(rows)
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in plugins:
        groups[(str(row["dataset"]), str(row["detector"]), str(row["family"]))].append(row)

    summary: list[dict[str, Any]] = []
    for (dataset, detector, family), group in sorted(groups.items()):
        deltas = [float(row["delta_vs_random_AP50"]) for row in group]
        best = max(group, key=lambda row: float(row["delta_vs_random_AP50"]))
        worst = min(group, key=lambda row: float(row["delta_vs_random_AP50"]))
        summary.append(
            {
                "benchmark": _benchmark_name(dataset, detector),
                "dataset": dataset,
                "detector": detector,
                "family": family,
                "settings": len(group),
                "seed_runs": sum(int(row["n"]) for row in group),
                "mean_delta_vs_random": _mean(deltas),
                "median_delta_vs_random": _median(deltas),
                "positive_settings": sum(delta > 0 for delta in deltas),
                "positive_rate": sum(delta > 0 for delta in deltas) / max(len(deltas), 1),
                "best_plugin": best["method"],
                "best_sfod_method": best["sfod_method"],
                "best_delta_vs_random": best["delta_vs_random_AP50"],
                "best_AP50": best["AP50_mean"],
                "worst_plugin": worst["method"],
                "worst_sfod_method": worst["sfod_method"],
                "worst_delta_vs_random": worst["delta_vs_random_AP50"],
                "worst_AP50": worst["AP50_mean"],
            }
        )
    return summary


def _variant_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    plugins = _plugin_rows(rows)
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in plugins:
        groups[(str(row["dataset"]), str(row["detector"]), str(row["method"]))].append(row)

    summary: list[dict[str, Any]] = []
    for (dataset, detector, method), group in sorted(groups.items()):
        deltas = [float(row["delta_vs_random_AP50"]) for row in group]
        best = max(group, key=lambda row: float(row["delta_vs_random_AP50"]))
        worst = min(group, key=lambda row: float(row["delta_vs_random_AP50"]))
        summary.append(
            {
                "benchmark": _benchmark_name(dataset, detector),
                "dataset": dataset,
                "detector": detector,
                "family": group[0]["family"],
                "plugin": method,
                "settings": len(group),
                "seed_runs": sum(int(row["n"]) for row in group),
                "mean_delta_vs_random": _mean(deltas),
                "median_delta_vs_random": _median(deltas),
                "positive_settings": sum(delta > 0 for delta in deltas),
                "positive_rate": sum(delta > 0 for delta in deltas) / max(len(deltas), 1),
                "best_sfod_method": best["sfod_method"],
                "best_delta_vs_random": best["delta_vs_random_AP50"],
                "best_AP50": best["AP50_mean"],
                "worst_sfod_method": worst["sfod_method"],
                "worst_delta_vs_random": worst["delta_vs_random_AP50"],
                "worst_AP50": worst["AP50_mean"],
            }
        )
    return summary


def _wave8_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["dataset"] == "cityscapes__to__foggy_cityscapes"
        and row["detector"] == "deta_r50_5scale_12ep_bs8"
        and row["method"] in {"source_only", "oracle", "pure_sfod", "random"}
    ]


def _wave9_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["dataset"] == "cityscapes__to__foggy_cityscapes"
        and row["detector"] == "deta_r50_5scale_12ep_bs8"
        and (
            row["method"] in {"source_only", "oracle", "pure_sfod", "random"}
            or row["family"] in {"selection", "completion", "optimization_control"}
        )
    ]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = _scan_records()
    rows = _aggregate(records)

    final_csv = OUT_DIR / "paper_results_by_dataset_detector.csv"
    final_md = OUT_DIR / "paper_results_by_dataset_detector.md"
    final_xlsx = OUT_DIR / "paper_results_by_dataset_detector.xlsx"
    wave8_csv = OUT_DIR / "wave8_deta_random005_summary.csv"
    wave8_md = OUT_DIR / "wave8_deta_random005_summary.md"
    wave9_csv = OUT_DIR / "wave9_deta_plugins_summary.csv"
    wave9_md = OUT_DIR / "wave9_deta_plugins_summary.md"

    _write_csv(final_csv, rows, CSV_COLUMNS)
    _write_md(final_md, rows, "Paper Results by Dataset and Detector")

    sheets: dict[str, list[list[Any]]] = {
        "all_results": _rows_to_table(rows, CSV_COLUMNS),
        "benchmark_overview": _rows_to_table(
            _benchmark_overview_rows(rows),
            BENCHMARK_OVERVIEW_COLUMNS,
        ),
        "best_plugin_by_method": _rows_to_table(
            _method_best_plugin_rows(rows),
            BEST_PLUGIN_COLUMNS,
        ),
        "family_summary": _rows_to_table(
            _family_summary_rows(rows),
            FAMILY_SUMMARY_COLUMNS,
        ),
        "variant_summary": _rows_to_table(
            _variant_summary_rows(rows),
            VARIANT_SUMMARY_COLUMNS,
        ),
    }
    for sheet, sheet_rows in _benchmark_sheets(rows).items():
        sheets[sheet] = _rows_to_table(sheet_rows, DISPLAY_COLUMNS)
    sheets["wave8_deta_random005"] = _rows_to_table(_wave8_rows(rows), DISPLAY_COLUMNS)
    sheets["wave9_deta_plugins"] = _rows_to_table(_wave9_rows(rows), DISPLAY_COLUMNS)
    _write_xlsx(final_xlsx, sheets)

    wave8 = _wave8_rows(rows)
    _write_csv(wave8_csv, wave8, CSV_COLUMNS)
    _write_md(wave8_md, wave8, "Wave 8 DETA +5% Random Summary")
    wave9 = _wave9_rows(rows)
    _write_csv(wave9_csv, wave9, CSV_COLUMNS)
    _write_md(wave9_md, wave9, "Wave 9 DETA Plugin Summary")

    print(f"records={len(records)} aggregated_rows={len(rows)}")
    print(f"wrote {final_csv}")
    print(f"wrote {final_md}")
    print(f"wrote {final_xlsx}")
    print(f"wrote {wave8_csv}")
    print(f"wrote {wave8_md}")
    print(f"wrote {wave9_csv}")
    print(f"wrote {wave9_md}")


if __name__ == "__main__":
    main()
