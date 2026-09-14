"""Import the audited corpus and reuse existing extractors without running ASR."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import yaml

from sara_audio.config import PipelineSettings
from sara_audio.domain import ExtractionResult, FeatureValue, Transcript
from sara_audio.linguistic import LinguisticFeatureExtractor
from sara_audio.output import ArtifactWriter
from sara_audio.pipeline import FeaturePipeline, PipelineRun
from sara_audio.registry import LinguisticFeatureRegistry

CONFIG_PATH = Path("config/external_delivery.yaml")
TURN_PATTERN = re.compile(r"^\[(\d+):(\d{2})\]\s+([^:]+):\s?(.*)$")
TIME_CODES = (
    "TIME_first_pause_s",
    "TIME_mean_internal_pause_s",
    "TIME_replica_duration_s",
    "TIME_audio_duration_s",
    "TIME_active_speech_duration_s",
)
RATE_CODES = (
    "RATE_words_per_second_active",
    "RATE_chars_per_second_active",
    "RATE_syllables_per_second_active",
    "RATE_phonemes_per_second_active",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_child(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes source root: {relative}")
    return candidate


@dataclass(frozen=True)
class ParsedTurn:
    line: int
    start_s: int
    role: str
    text: str
    analysis_text: str
    selected: bool
    artifact_replacement: bool


class ImportedTranscriptReader:
    """Select a role and remove only silence markers documented by the sender."""

    def __init__(self, source_root: Path, role: str) -> None:
        self._root = source_root
        self._role = role
        journal = source_root / "Обработанные_записи (1)/_Проверка/Замены_тишины.tsv"
        with journal.open(encoding="utf-8-sig", newline="") as stream:
            self._edits = {
                (row["Исходный путь"], int(row["Строка"])): row
                for row in csv.DictReader(stream, delimiter="\t")
            }

    def read(self, entry: dict) -> tuple[Transcript, list[ParsedTurn]]:
        relative = entry["processed"][0]["path"]
        path = safe_child(self._root, relative)
        journal_key = Path(relative).relative_to("Обработанные_записи (1)").as_posix()
        turns = self.parse(path.read_text(encoding="utf-8-sig"), journal_key)
        selected = [
            turn.analysis_text
            for turn in turns
            if turn.selected and turn.analysis_text.strip()
        ]
        return Transcript(
            text="\n".join(selected),
            source="external-whisper+documented-role-cleanup:operator",
            is_verified=False,
        ), turns

    def parse(self, text: str, journal_key: str) -> list[ParsedTurn]:
        turns = []
        for number, line in enumerate(text.splitlines(), 1):
            match = TURN_PATTERN.fullmatch(line)
            if not match:
                raise ValueError(
                    f"Unrecognized transcript line: {journal_key}:{number}"
                )
            minute, second, role, spoken = match.groups()
            if role not in {"Оператор", "Клиент"} or int(second) >= 60:
                raise ValueError(f"Invalid role or timestamp: {journal_key}:{number}")
            cleaned = spoken
            edit = self._edits.get((journal_key, number))
            if edit:
                if spoken != edit["Итоговый текст"]:
                    raise ValueError(
                        f"Replacement journal mismatch: {journal_key}:{number}"
                    )
                cleaned = re.sub(r"\bТишина\b", "", spoken)
                if not re.search(r"\w", cleaned):
                    cleaned = ""
            turns.append(
                ParsedTurn(
                    number,
                    int(minute) * 60 + int(second),
                    role,
                    spoken,
                    cleaned,
                    role == self._role,
                    edit is not None,
                )
            )
        return turns


class AuditedCorpus:
    """Validate source fingerprints and resolve on-disk or ZIP-backed WAVs."""

    def __init__(self, root: Path, inventory: dict) -> None:
        self.root = root.resolve()
        self.inventory = inventory
        self.records = inventory["records"]
        if len(self.records) != 390:
            raise ValueError(
                "This delivery profile expects the audited 390-record corpus"
            )

    def validate(self) -> None:
        needed = set()
        for record_id, entry in self.records.items():
            if not re.fullmatch(r"\d{14}_Agent_\d{2}", record_id):
                raise ValueError(f"Invalid record ID: {record_id}")
            for variant in ("processed", "before_roles"):
                if len(entry.get(variant, [])) != 1:
                    raise ValueError(
                        f"Expected exactly one {variant} transcript: {record_id}"
                    )
                needed.add(entry[variant][0]["path"])
            if entry.get("audio_disk"):
                needed.add(entry["audio_disk"][0]["path"])
            elif entry.get("audio_zip"):
                needed.add(entry["audio_zip"][0]["archive"])
        needed.update(
            item["path"]
            for item in self.inventory["files"]
            if item.get("kind") != "macos_metadata"
            and Path(item["path"]).suffix in {".csv", ".tsv", ".xlsx", ".md"}
        )
        indexed = {item["path"]: item for item in self.inventory["files"]}
        for relative in sorted(needed):
            path = safe_child(self.root, relative)
            if file_hash(path) != indexed[relative]["sha256"]:
                raise ValueError(f"Input changed since audit: {relative}")

    def audio(self, entry: dict, cache: Path) -> tuple[Path | None, str | None]:
        if entry.get("audio_disk"):
            relative = entry["audio_disk"][0]["path"]
            return safe_child(self.root, relative), relative
        if not entry.get("audio_zip"):
            return None, None
        item = entry["audio_zip"][0]
        archive_path = safe_child(self.root, item["archive"])
        cache.mkdir(parents=True, exist_ok=True)
        destination = safe_child(cache, item["path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            member = archive.getinfo(item["path"])
            if (
                f"{member.CRC:08x}" != item["crc32"]
                or member.file_size != item["bytes"]
            ):
                raise ValueError(f"Archive member changed: {item['path']}")
            with archive.open(member) as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target)
        return destination, f"{item['archive']}::{item['path']}"


class ExistingFeatureDelivery:
    """Process one record at a time and publish only complete record folders."""

    def __init__(self, corpus: AuditedCorpus, config: dict, destination: Path) -> None:
        self.corpus = corpus
        self.config = config
        self.destination = destination
        self.work = Path("artifacts/external_delivery_cache") / destination.name
        self.registry = LinguisticFeatureRegistry.load_default()
        self.extractor = LinguisticFeatureExtractor(self.registry)
        self.reader = ImportedTranscriptReader(corpus.root, config["analysis_role"])
        self.writer = ArtifactWriter()
        self.settings = PipelineSettings.from_mapping(config)

    def build(self) -> dict:
        if self.destination.exists() or self.destination.with_suffix(".zip").exists():
            raise FileExistsError(f"Delivery already exists: {self.destination}")
        self.corpus.validate()
        self.destination.mkdir(parents=True)
        (self.destination / "records").mkdir()
        self.work.mkdir(parents=True, exist_ok=False)
        rows, errors = [], []
        started = time.monotonic()
        workers = self.config.get("workers", 1)
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError("workers must be a positive integer")
        for number, (record_id, row, error) in enumerate(self.iter_results(workers), 1):
            if error is None:
                rows.append(row)
            else:
                errors.append({"record_id": record_id, "error": error})
            summary = {
                "record_count": len(rows),
                "expected_records": len(self.corpus.records),
                "records": sorted(rows, key=lambda row: row["record_id"]),
                "errors": errors,
                "complete": False,
            }
            write_json(self.destination / "run_summary.json", summary)
            if number == 1 or number % 10 == 0 or errors:
                print(
                    f"{number}/{len(self.corpus.records)} records; errors={len(errors)}; elapsed={time.monotonic() - started:.0f}s",
                    flush=True,
                )
        summary["complete"] = not errors and len(rows) == len(self.corpus.records)
        summary["elapsed_seconds"] = round(time.monotonic() - started, 2)
        write_json(self.destination / "run_summary.json", summary)
        if errors:
            raise RuntimeError(
                f"{len(errors)} records failed; see run_summary.json; no archive published"
            )
        return summary

    def retry_failed(self) -> dict:
        """Recover a finished failed batch without rerunning successful recordings."""
        summary = read_json(self.destination / "run_summary.json")
        if summary["complete"] or not summary["errors"]:
            raise ValueError("Expected a finished batch with failed records")
        rows = summary["records"]
        failed_ids = [error["record_id"] for error in summary["errors"]]
        all_ids = [row["record_id"] for row in rows] + failed_ids
        if len(all_ids) != len(self.corpus.records) or set(all_ids) != set(
            self.corpus.records
        ):
            raise ValueError("Batch is still running, incomplete, or has duplicate IDs")
        if self.destination.with_suffix(".zip").exists():
            raise FileExistsError("Cannot change an archived delivery")
        self.corpus.validate()
        self.work = self.work / f"retry-{time.time_ns()}"
        self.work.mkdir(parents=True)
        recovery = {"previous_errors": summary["errors"], "retried_records": failed_ids}
        errors = []
        for record_id in failed_ids:
            try:
                if (self.destination / "records" / record_id).exists():
                    raise FileExistsError(
                        "A partial published record exists; inspect it before recovery"
                    )
                rows.append(
                    self.process_record(record_id, self.corpus.records[record_id])
                )
            except Exception as error:
                errors.append(
                    {
                        "record_id": record_id,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
        summary.update(
            records=sorted(rows, key=lambda row: row["record_id"]),
            record_count=len(rows),
            errors=errors,
            complete=not errors,
        )
        write_json(self.destination / "run_summary.json", summary)
        write_json(self.destination / "recovery.json", recovery)
        if errors:
            raise RuntimeError("Recovery failed; see run_summary.json")
        return summary

    def iter_results(self, workers: int):
        """Keep extraction failures isolated; only the parent publishes the summary."""
        if workers == 1:
            for record_id, entry in sorted(self.corpus.records.items()):
                try:
                    row = self.process_record(record_id, entry)
                except Exception as error:
                    yield record_id, None, f"{type(error).__name__}: {error}"
                else:
                    yield record_id, row, None
            return
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_worker,
            initargs=(
                self.corpus.root,
                self.corpus.inventory,
                self.config,
                self.destination,
            ),
        ) as executor:
            pending = {
                executor.submit(_process_in_worker, record_id, entry): record_id
                for record_id, entry in sorted(self.corpus.records.items())
            }
            for future in as_completed(pending):
                record_id = pending[future]
                try:
                    row = future.result()
                except Exception as error:
                    yield record_id, None, f"{type(error).__name__}: {error}"
                else:
                    yield record_id, row, None

    def process_record(self, record_id: str, entry: dict) -> dict:
        folder = self.work / record_id
        folder.mkdir()
        transcript, turns = self.reader.read(entry)
        has_operator_text = bool(transcript.text.strip())
        audio_path, audio_source = self.corpus.audio(entry, folder / "audio_cache")
        analysis = None
        frames = None
        coverage = None
        diagnostics = [
            "Text scope: operator turns from the externally role-labelled transcript; not manually verified.",
            "No ASR was executed. External Whisper model/version is unknown.",
            "Speech rate not computed: full-dialogue timestamps and supplied WAV do not share a verified timeline.",
            "All 92 registered linguistic codes use the portable heuristic backend; confidence is method-specific.",
        ]
        if not has_operator_text:
            diagnostics.append(
                "No operator text in the supplied role-labelled transcript; all linguistic values are unavailable, not zero."
            )
        if audio_path:
            pipeline = FeaturePipeline(
                folder / "audio_cache", self.settings, self.registry
            )
            acoustic = pipeline.extract(audio_path, transcript=None)
            analysis, frames, coverage = (
                acoustic.result.pause_analysis,
                acoustic.frames,
                acoustic.opensmile_coverage,
            )
            timing = [
                replace(
                    feature,
                    evidence={
                        **feature.evidence,
                        "time_axis": "supplied_wav",
                        "original_call_timing_available": False,
                    },
                )
                for feature in acoustic.result.features
            ]
            if frames is not None:
                frames = frames.rename(
                    index=lambda value: Path(str(value)).name, level="file"
                )
            diagnostics.append(
                "Timing and acoustics describe the supplied WAV, not the original full call or verified operator-only intervals."
            )
            if analysis.audio_duration_s < 5:
                diagnostics.append(
                    "Very short WAV (<5 s); inspect descriptor validity before interpretation."
                )
        else:
            timing = [
                self.unavailable(code, "s", "Audio is absent from the supplied corpus.")
                for code in TIME_CODES
            ]
            diagnostics.append(
                "Text-only record: source audio not present; no acoustic or timing measurements."
            )
        rates = [
            self.unavailable(
                code,
                "phonemes/s" if "phonemes" in code else "items/s",
                "G2P backend is not configured."
                if "phonemes" in code
                else "Audio/text timeline and speech-content correspondence are unverified; rate would be misleading.",
            )
            for code in RATE_CODES
        ]
        linguistic = tuple(
            replace(
                feature,
                evidence={
                    **feature.evidence,
                    "analysis_scope": "operator",
                    "analysis_text_sha256": hashlib.sha256(
                        transcript.text.encode()
                    ).hexdigest(),
                },
            )
            for feature in (
                self.extractor.extract(transcript)
                if has_operator_text
                else tuple(
                    self.unavailable(
                        definition.code,
                        "count",
                        "No operator text in the supplied role-labelled transcript.",
                    )
                    for definition in self.registry.definitions
                )
            )
        )
        logical_input = Path(audio_source or entry["processed"][0]["path"])
        result = ExtractionResult(
            logical_input,
            tuple(timing) + tuple(rates) + linguistic,
            analysis,
            transcript,
            tuple(diagnostics),
            record_id,
        )
        run = PipelineRun(
            result,
            frames,
            coverage,
            acoustic.normalized_audio_path if audio_path else logical_input,
            acoustic.analysis_audio_path if audio_path else logical_input,
        )
        for extension in ("csv", "json"):
            self.writer.write_feature_rows(folder / f"features.{extension}", [run])
        if frames is not None and self.config["include_frames"]:
            self.writer.write_frames(folder / "frames.csv", [run])
        self.writer.write_transcript(
            folder / "transcript.txt", folder / "transcript.json", transcript
        )
        write_json(
            folder / "transcript_turns.json",
            {
                "time_axis": "external_full_dialogue",
                "timestamp_precision_s": 1,
                "end_timestamps_available": False,
                "turns": [asdict(turn) for turn in turns],
            },
        )
        write_json(
            folder / "segments.json",
            {
                "file_id": record_id,
                "time_axis": "supplied_wav",
                "audio_available": bool(audio_path),
                "speech_segments": [asdict(s) for s in analysis.speech_segments]
                if analysis
                else [],
                "pauses": [asdict(s) for s in analysis.pauses] if analysis else [],
                "original_full_call_intervals_available": False,
            },
        )
        for variant, name in (
            ("processed", "dialogue_processed.txt"),
            ("before_roles", "dialogue_before_roles.txt"),
        ):
            shutil.copy2(
                safe_child(self.corpus.root, entry[variant][0]["path"]), folder / name
            )
        calculated = sum(feature.value is not None for feature in linguistic)
        if (
            calculated != (92 if has_operator_text else 0)
            or len({f.code for f in linguistic}) != 92
        ):
            raise ValueError(f"Unexpected linguistic coverage: {record_id}")
        manifest = {
            "record_id": record_id,
            "split": entry.get("split", "text_only"),
            "audio_source": audio_source,
            "audio_sha256": file_hash(audio_path) if audio_path else None,
            "audio_status": "available" if audio_path else "missing",
            "analysis_scope": "operator",
            "analysis_text_status": "available"
            if has_operator_text
            else "no_operator_text",
            "audio_speaker_scope": "unverified",
            "transcript_sources": {
                name: entry[name][0] for name in ("processed", "before_roles")
            },
            "role_review": entry.get("review", []),
            "expert_sheet_row": entry.get("expert_sheet_row"),
            "linguistic_registered": 92,
            "linguistic_computed": calculated,
            "linguistic_not_computed": 92 - calculated,
            "speech_rate_status": "not-computed",
            "opensmile_semantic_coverage": coverage,
            "diagnostics": diagnostics,
        }
        write_json(folder / "manifest.json", manifest)
        write_json(
            folder / "report.json",
            {
                **manifest,
                "temporal_features": [f.as_dict() for f in timing],
                "speech_rate": [f.as_dict() for f in rates],
                "linguistic_features": [
                    {
                        **f.as_dict(),
                        "description": self.registry.get(f.code).description,
                    }
                    for f in linguistic
                ],
                "transcript": {
                    "available": has_operator_text,
                    "is_verified": False,
                    "source": transcript.source,
                    "artifacts": [
                        "transcript.txt",
                        "transcript.json",
                        "transcript_turns.json",
                    ],
                },
                "confidence_note": "Heuristic confidence values are not calibrated accuracy estimates.",
            },
        )
        # Keep temporary WAVs out of the published record.
        destination = self.destination / "records" / record_id
        destination.mkdir()
        for path in folder.iterdir():
            if path.is_file():
                path.rename(destination / path.name)
        return {
            "record_id": record_id,
            "directory": f"records/{record_id}",
            "split": manifest["split"],
            "audio_available": bool(audio_path),
            "audio_duration_s": analysis.audio_duration_s if analysis else None,
            "linguistic_computed": calculated,
            "linguistic_not_computed": 92 - calculated,
            "speech_rate_status": "not-computed",
            "analysis_scope": "operator",
            "text_characters": len(transcript.text),
            "source_turns": len(turns),
            "selected_turns": sum(
                t.selected and bool(t.analysis_text.strip()) for t in turns
            ),
        }

    @staticmethod
    def unavailable(code: str, unit: str, reason: str) -> FeatureValue:
        return FeatureValue(
            code,
            None,
            unit,
            "not-computed",
            None,
            "external-corpus-import-v1",
            {"reason": reason},
        )


_worker_delivery: ExistingFeatureDelivery | None = None


def _initialize_worker(
    root: Path, inventory: dict, config: dict, destination: Path
) -> None:
    global _worker_delivery
    _worker_delivery = ExistingFeatureDelivery(
        AuditedCorpus(root, inventory), config, destination
    )


def _process_in_worker(record_id: str, entry: dict) -> dict:
    if _worker_delivery is None:
        raise RuntimeError("Delivery worker was not initialized")
    return _worker_delivery.process_record(record_id, entry)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the external-corpus delivery using config/external_delivery.yaml"
    )
    parser.add_argument("input", type=Path, help="Audited corpus directory")
    parser.add_argument("output", nargs="?", type=Path, help="New delivery directory")
    args = parser.parse_args(argv)
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["analysis_role"] != "Оператор":
        raise ValueError("This fixed delivery profile uses operator turns")
    destination = args.output or Path(config["output_root"]) / "real-data-existing19"
    inventory = read_json(Path(config["inventory"]))
    delivery = ExistingFeatureDelivery(
        AuditedCorpus(args.input, inventory), config, destination
    )
    delivery.build()
    # Imported here so importing the parser does not initialize archive-building code.
    from sara_audio.external_package import ExternalPackageWriter

    ExternalPackageWriter(
        destination, CONFIG_PATH, Path(config["inventory"]), args.input
    ).build()
    print(f"Delivery complete: {destination.with_suffix('.zip')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
