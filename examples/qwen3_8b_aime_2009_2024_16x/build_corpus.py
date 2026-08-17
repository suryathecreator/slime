#!/usr/bin/env python3
"""Build and verify the frozen 2009--2024, 30-problem/year AIME corpus."""

from __future__ import annotations

import argparse
import csv
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
INPUT_MANIFEST = HERE / "source_inputs.json"
DEFAULT_CORPUS = HERE / "aime_2009_2024.jsonl"
DEFAULT_MANIFEST = HERE / "corpus_manifest.json"
YEARS = tuple(range(2024, 2008, -1))
PARTS = ("I", "II")
EXPECTED_TOTAL = 480
EXPECTED_PRIMARY = 455
EXPECTED_SUPPLEMENT = 25
SPECIAL_ACCEPTED = {"2022-II-8": [80, 81]}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    text = "".join(canonical_json(row) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


class NextDataParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_next_data = False
        self.fragments: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "script" and dict(attrs).get("id") == "__NEXT_DATA__":
            self.in_next_data = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self.in_next_data:
            self.in_next_data = False

    def handle_data(self, data: str) -> None:
        if self.in_next_data:
            self.fragments.append(data)


def expected_ids() -> list[str]:
    return [
        f"{year}-{part}-{number}"
        for year in YEARS
        for part in PARTS
        for number in range(1, 16)
    ]


def parse_primary(path: Path, spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    actual_hash = sha256_file(path)
    if actual_hash != spec["sha256"]:
        raise RuntimeError(f"primary CSV hash mismatch: {actual_hash}")
    selected: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            year = int(row["Year"])
            if year not in YEARS:
                continue
            problem_id = row["ID"]
            if problem_id in selected:
                raise RuntimeError(f"duplicate primary ID: {problem_id}")
            selected[problem_id] = row
    return selected


def parse_live_pages(
    source_dir: Path, specs: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for year_part, spec in sorted(specs.items()):
        path = source_dir / spec["file"]
        actual_hash = sha256_file(path)
        if actual_hash != spec["sha256"]:
            raise RuntimeError(f"LIVE page hash mismatch for {year_part}: {actual_hash}")
        parser = NextDataParser()
        parser.feed(path.read_text(encoding="utf-8"))
        payload = json.loads("".join(parser.fragments))
        page = payload["props"]["pageProps"]
        if page["year"] != year_part or page["contest"] != "aime":
            raise RuntimeError(f"unexpected LIVE page identity for {year_part}")
        questions = page["baseQuestions"]
        if len(questions) != 15:
            raise RuntimeError(f"LIVE page {year_part} has {len(questions)} problems")
        year = int(year_part[:4])
        part = year_part[4:]
        # The provider's aimeProblemNumber metadata is not reliable on older
        # pages. The published page order is the contest problem order.
        for number, question in enumerate(questions, start=1):
            problem_id = f"{year}-{part}-{number}"
            rows[problem_id] = {
                "answer": str(question["answer"]),
                "problem": str(question["question"]),
                "source_page_sha256": actual_hash,
                "source_url": spec["url"],
            }
    return rows


def accepted_answers(problem_id: str, raw_answer: str) -> list[int]:
    if problem_id in SPECIAL_ACCEPTED:
        return SPECIAL_ACCEPTED[problem_id]
    try:
        answer = int(raw_answer)
    except ValueError as error:
        raise RuntimeError(f"non-integer AIME gold for {problem_id}: {raw_answer!r}") from error
    if not 0 <= answer <= 999:
        raise RuntimeError(f"out-of-range AIME gold for {problem_id}: {answer}")
    return [answer]


def source_record_hash(problem: str, raw_answer: str) -> str:
    return sha256_bytes(
        canonical_json({"answer": raw_answer, "problem": problem}).encode("utf-8")
    )


def build(primary_csv: Path, live_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inputs = json.loads(INPUT_MANIFEST.read_text(encoding="utf-8"))
    primary = parse_primary(primary_csv, inputs["primary"])
    live = parse_live_pages(live_dir, inputs["supplement"]["pages"])
    requested = expected_ids()
    missing = [problem_id for problem_id in requested if problem_id not in primary]
    if len(missing) != EXPECTED_SUPPLEMENT:
        raise RuntimeError(f"expected 25 primary gaps, found {len(missing)}: {missing}")
    unavailable = [problem_id for problem_id in missing if problem_id not in live]
    if unavailable:
        raise RuntimeError(f"LIVE supplement is missing: {unavailable}")

    # Answer agreement on all overlapping rows guards page-order alignment.
    overlap_checks = 0
    for problem_id in sorted(set(primary).intersection(live)):
        raw = primary[problem_id]["Answer"]
        try:
            primary_answer = int(raw)
        except ValueError:
            continue
        overlap_checks += 1
        if primary_answer != int(live[problem_id]["answer"]):
            raise RuntimeError(f"source answer disagreement for {problem_id}")
    if overlap_checks < 90:
        raise RuntimeError(f"insufficient cross-source alignment checks: {overlap_checks}")

    rows: list[dict[str, Any]] = []
    for eval_index, problem_id in enumerate(requested):
        year_text, part, number_text = problem_id.split("-")
        if problem_id in primary:
            item = primary[problem_id]
            problem = item["Question"]
            raw_answer = item["Answer"]
            source = {
                "kind": "primary_huggingface",
                "dataset": inputs["primary"]["dataset"],
                "revision": inputs["primary"]["revision"],
                "url": inputs["primary"]["url"],
                "source_file_sha256": inputs["primary"]["sha256"],
            }
        else:
            item = live[problem_id]
            problem = item["problem"]
            raw_answer = item["answer"]
            source = {
                "kind": "supplement_live_poshenloh",
                "dataset": "LIVE by Po-Shen Loh AIME past contests",
                "revision": item["source_page_sha256"],
                "url": item["source_url"],
                "source_file_sha256": item["source_page_sha256"],
            }
        accepted = accepted_answers(problem_id, raw_answer)
        rows.append(
            {
                "accepted_answer_integers": accepted,
                "answer_canonical": "/".join(f"{value:03d}" for value in accepted),
                "answer_raw": raw_answer,
                "contest": f"AIME {part}",
                "eval_index": eval_index,
                "has_asy": "[asy]" in problem,
                "part": part,
                "problem": problem,
                "problem_id": problem_id,
                "problem_number": int(number_text),
                "source": source,
                "source_record_sha256": source_record_hash(problem, raw_answer),
                "year": int(year_text),
            }
        )
    verify_rows(rows)
    counts = {
        kind: sum(row["source"]["kind"] == kind for row in rows)
        for kind in ("primary_huggingface", "supplement_live_poshenloh")
    }
    manifest = {
        "corpus_schema_version": "qwen3_aime_2009_2024_corpus_v1",
        "ordering": "year_descending_then_part_I_II_then_problem_ascending",
        "problem_count": len(rows),
        "source_counts": counts,
        "source_inputs": inputs,
        "source_overlap_answer_checks": overlap_checks,
        "supplemented_problem_ids": missing,
        "year_count": len(YEARS),
    }
    return rows, manifest


def verify_rows(rows: list[dict[str, Any]]) -> None:
    if len(rows) != EXPECTED_TOTAL:
        raise RuntimeError(f"expected {EXPECTED_TOTAL} rows, found {len(rows)}")
    identifiers = [row["problem_id"] for row in rows]
    if identifiers != expected_ids():
        raise RuntimeError("corpus IDs or order differ from the contract")
    for index, row in enumerate(rows):
        if row["eval_index"] != index:
            raise RuntimeError(f"non-canonical eval index at row {index}")
        if not row["problem"].strip():
            raise RuntimeError(f"empty problem at row {index}")
        accepted = row["accepted_answer_integers"]
        if not accepted or any(not 0 <= int(value) <= 999 for value in accepted):
            raise RuntimeError(f"bad accepted answer list at row {index}")
    for year in YEARS:
        year_rows = [row for row in rows if row["year"] == year]
        if len(year_rows) != 30:
            raise RuntimeError(f"year {year} has {len(year_rows)} rows")
        for part in PARTS:
            part_rows = [row for row in year_rows if row["part"] == part]
            if [row["problem_number"] for row in part_rows] != list(range(1, 16)):
                raise RuntimeError(f"year {year} part {part} is incomplete")
    source_counts = {
        kind: sum(row["source"]["kind"] == kind for row in rows)
        for kind in ("primary_huggingface", "supplement_live_poshenloh")
    }
    if source_counts != {
        "primary_huggingface": EXPECTED_PRIMARY,
        "supplement_live_poshenloh": EXPECTED_SUPPLEMENT,
    }:
        raise RuntimeError(f"unexpected source counts: {source_counts}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--primary-csv", type=Path, required=True)
    build_parser.add_argument("--live-dir", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, default=DEFAULT_CORPUS)
    build_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    args = parser.parse_args()

    if args.command == "build":
        rows, manifest = build(args.primary_csv, args.live_dir)
        write_jsonl(args.output, rows)
        manifest["corpus_sha256"] = sha256_file(args.output)
        args.manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            f"AIME_CORPUS_BUILT rows={len(rows)} sha256={manifest['corpus_sha256']}"
        )
    else:
        rows = read_jsonl(args.corpus)
        verify_rows(rows)
        print(f"AIME_CORPUS_VALID rows={len(rows)} sha256={sha256_file(args.corpus)}")


if __name__ == "__main__":
    main()
