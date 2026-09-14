"""Package a completed server extraction without rerunning audio processing."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from sara_audio.feature_filter import filter_wide_table, load_excluded_feature_codes
from sara_audio.registry import LinguisticFeatureRegistry

IDENTITY_COLUMNS = frozenset({"file_id", "input_path", "analysis_audio_path"})


class ServerDeliveryBuilder:
    """Create a compact, auditable hand-off from a completed server result."""

    def __init__(self, source: Path, destination: Path) -> None:
        self.source = source.resolve()
        self.destination = destination.resolve()
        self.registry = LinguisticFeatureRegistry.load_default()

    def build(self) -> Path:
        if self.destination.exists() or self.destination.with_suffix(".zip").exists():
            raise FileExistsError(f"Delivery output already exists: {self.destination}")
        summary_path = self.source / "run_summary.json"
        wide_path = self.source / "features_wide.csv"
        if not summary_path.is_file() or not wide_path.is_file():
            raise FileNotFoundError(
                "Source must be a completed server result with run_summary.json and features_wide.csv"
            )

        summary = self._read_json(summary_path)
        records = summary.get("records", [])
        wide_rows, columns = self._read_wide(wide_path)
        excluded_feature_codes = load_excluded_feature_codes()
        wide_rows, columns = filter_wide_table(
            wide_rows, columns, excluded_feature_codes
        )
        if len(records) != len(wide_rows):
            raise ValueError("Record count does not match features_wide.csv row count")
        if not wide_rows:
            raise ValueError("features_wide.csv has no rows")

        self.destination.mkdir(parents=True)
        tables = self.destination / "tables"
        provenance = self.destination / "provenance"
        tables.mkdir()
        provenance.mkdir()
        shutil.copy2(summary_path, provenance / "run_summary.json")
        self._copy_provenance(provenance)
        index_rows, transcript_rows, diagnostic_rows = self._record_tables(records)
        dictionary_rows = self._dictionary_rows(columns)
        self._write_csv(tables / "features_wide.csv", wide_rows, columns)
        self._write_csv(
            tables / "transcripts.csv", transcript_rows, ["record_id", "transcript"]
        )
        self._write_csv(
            tables / "records_index.csv",
            index_rows,
            ["record_id", "input_path", "operator_speech_path", "transcript_available"],
        )
        self._write_csv(
            tables / "record_diagnostics.csv",
            diagnostic_rows,
            ["record_id", "diagnostics", "opensmile_semantic_coverage"],
        )
        self._write_csv(
            tables / "feature_dictionary.csv",
            dictionary_rows,
            ["feature_code", "group", "description"],
        )
        prediction_tables: list[tuple[str, list[str], list[dict[str, str]]]] = []
        for filename, sheet_name in (
            ("predictions.csv", "Predictions"),
            ("interpretation_input.csv", "Interpretation"),
            ("filter_predictions.csv", "Filtering"),
        ):
            path = self.source / filename
            if path.is_file():
                rows, fields = self._read_table(path)
                self._write_csv(tables / filename, rows, fields)
                prediction_tables.append((sheet_name, fields, rows))
        self._write_workbook(
            tables / "delivery.xlsx",
            [
                ("Features", columns, wide_rows),
                ("Transcripts", ["record_id", "transcript"], transcript_rows),
                (
                    "Records",
                    [
                        "record_id",
                        "input_path",
                        "operator_speech_path",
                        "transcript_available",
                    ],
                    index_rows,
                ),
                (
                    "Diagnostics",
                    ["record_id", "diagnostics", "opensmile_semantic_coverage"],
                    diagnostic_rows,
                ),
                (
                    "Dictionary",
                    ["feature_code", "group", "description"],
                    dictionary_rows,
                ),
                *prediction_tables,
            ],
        )
        self._write_readme(
            len(records),
            len(columns) - len(IDENTITY_COLUMNS),
            excluded_feature_codes,
        )
        self._write_manifest(len(records), columns, excluded_feature_codes)
        archive = shutil.make_archive(
            str(self.destination),
            "zip",
            root_dir=self.destination.parent,
            base_dir=self.destination.name,
        )
        digest = self._sha256(Path(archive))
        Path(f"{archive}.sha256").write_text(
            f"{digest}  {Path(archive).name}\n", encoding="ascii"
        )
        return Path(archive)

    def _read_wide(self, path: Path) -> tuple[list[dict[str, str]], list[str]]:
        with path.open(encoding="utf-8", newline="") as stream:
            header = stream.readline()
            stream.seek(0)
            reader = csv.DictReader(stream, delimiter=";" if ";" in header else ",")
            rows = list(reader)
            columns = reader.fieldnames or []
        if not {"file_id", "input_path"} <= set(columns):
            raise ValueError("features_wide.csv is missing identity columns")
        if len({row["file_id"] for row in rows}) != len(rows):
            raise ValueError("features_wide.csv has duplicate file_id values")
        return rows, columns

    def _copy_provenance(self, destination: Path) -> None:
        input_manifest = self.source / "input_manifest.json"
        if input_manifest.is_file():
            shutil.copy2(input_manifest, destination / input_manifest.name)
        batches = self.source / "batches"
        if batches.is_dir():
            shutil.copytree(batches, destination / "batches")

    @staticmethod
    def _read_table(path: Path) -> tuple[list[dict[str, str]], list[str]]:
        with path.open(encoding="utf-8", newline="") as stream:
            header = stream.readline()
            stream.seek(0)
            reader = csv.DictReader(stream, delimiter=";" if ";" in header else ",")
            return list(reader), reader.fieldnames or []

    def _record_tables(
        self, records: list[dict[str, object]]
    ) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
        index_rows: list[dict[str, str]] = []
        transcript_rows: list[dict[str, str]] = []
        diagnostic_rows: list[dict[str, str]] = []
        for record in records:
            directory = Path(str(record["directory"]))
            record_id = directory.name
            index_rows.append(
                {
                    "record_id": record_id,
                    "input_path": str(record["input_path"]),
                    "operator_speech_path": str(record.get("operator_speech") or ""),
                    "transcript_available": str(
                        bool(record.get("transcripts"))
                    ).lower(),
                }
            )
            transcript_path = directory / "transcript.txt"
            if transcript_path.is_file():
                transcript_rows.append(
                    {
                        "record_id": record_id,
                        "transcript": transcript_path.read_text(
                            encoding="utf-8"
                        ).strip(),
                    }
                )
            report_path = directory / "report.json"
            if report_path.is_file():
                report = self._read_json(report_path)
                diagnostic_rows.append(
                    {
                        "record_id": record_id,
                        "diagnostics": json.dumps(
                            report.get("diagnostics", []), ensure_ascii=False
                        ),
                        "opensmile_semantic_coverage": json.dumps(
                            report.get("acoustic_features", {}).get(
                                "semantic_coverage"
                            ),
                            ensure_ascii=False,
                        ),
                    }
                )
        return index_rows, transcript_rows, diagnostic_rows

    def _dictionary_rows(self, columns: list[str]) -> list[dict[str, str]]:
        definitions = {item.code: item for item in self.registry.definitions}
        rows = []
        for code in columns:
            if code in IDENTITY_COLUMNS:
                continue
            definition = definitions.get(code)
            if definition is not None:
                group, description = definition.group_title, definition.description
            elif code.startswith("ACOUSTIC_"):
                group, description = (
                    "Acoustic",
                    "Aggregated openSMILE ComParE_2016 feature.",
                )
            elif code.startswith("ASR_"):
                group, description = (
                    "ASR",
                    "Token-level metric from faster-whisper output.",
                )
            elif code.startswith("TIME_"):
                group, description = (
                    "Temporal",
                    "VAD-based temporal metric for the selected operator speech.",
                )
            elif code.startswith("RATE_"):
                group, description = (
                    "Speech rate",
                    "Transcript rate normalized by VAD active speech duration.",
                )
            else:
                group, description = "Other", "Pipeline scalar feature."
            rows.append(
                {"feature_code": code, "group": group, "description": description}
            )
        return rows

    def _write_readme(
        self,
        record_count: int,
        feature_count: int,
        excluded_feature_codes: frozenset[str],
    ) -> None:
        excluded_note = (
            f"- Excluded feature columns: {len(excluded_feature_codes)} "
            "from `exclude_features.txt`.\n"
            if excluded_feature_codes
            else ""
        )
        (self.destination / "README.md").write_text(
            "# SARA server delivery\n\n"
            f"- Records: {record_count}\n"
            f"- Scalar feature columns: {feature_count}\n"
            f"{excluded_note}"
            "- `tables/features_wide.csv`: model-ready matrix, semicolon-delimited.\n"
            "- `tables/transcripts.csv`: ASR transcript per recording, semicolon-delimited.\n"
            "- `tables/records_index.csv`: input and generated-audio index, semicolon-delimited.\n"
            "- `tables/record_diagnostics.csv`: processing diagnostics, semicolon-delimited.\n"
            "- `tables/feature_dictionary.csv`: feature-column definitions, semicolon-delimited.\n"
            "- `tables/predictions.csv`: compact MIC predictions, when prediction is enabled.\n"
            "- `tables/interpretation_input.csv`: full MIC output for interpretation, when available.\n"
            "- `tables/filter_predictions.csv`: filter outcome for every input recording.\n"
            "- `tables/delivery.xlsx`: Excel workbook containing the same five tables.\n"
            "  - `Features`: one row per recording, ready for modelling.\n"
            "  - `Transcripts`: ASR transcript per recording.\n"
            "  - `Records`: input and generated-audio index.\n"
            "  - `Diagnostics`: processing diagnostics and openSMILE coverage.\n"
            "  - `Dictionary`: feature-column definitions.\n"
            "- All CSV tables use `;`, so commas in transcript text cannot shift fields.\n"
            "- `provenance/`: input and batch manifests.\n"
            "- `delivery_manifest.json`: SHA-256 checksums for all delivery files.\n",
            encoding="utf-8",
        )

    def _write_manifest(
        self,
        record_count: int,
        columns: list[str],
        excluded_feature_codes: frozenset[str],
    ) -> None:
        files = []
        for path in sorted(
            item for item in self.destination.rglob("*") if item.is_file()
        ):
            if path.name == "delivery_manifest.json":
                continue
            files.append(
                {
                    "path": str(path.relative_to(self.destination)).replace("\\", "/"),
                    "bytes": path.stat().st_size,
                    "sha256": self._sha256(path),
                }
            )
        payload = {
            "source_result": str(self.source),
            "record_count": record_count,
            "feature_column_count": len(columns) - len(IDENTITY_COLUMNS),
            "excluded_features": sorted(excluded_feature_codes),
            "files": files,
        }
        (self.destination / "delivery_manifest.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    @staticmethod
    def _read_json(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, delimiter=";")
            writer.writeheader()
            writer.writerows(rows)

    @classmethod
    def _write_workbook(
        cls, path: Path, sheets: list[tuple[str, list[str], list[dict[str, str]]]]
    ) -> None:
        """Write delivery data as a dependency-free, standards-compliant XLSX workbook."""
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as workbook:
            workbook.writestr("[Content_Types].xml", cls._content_types(len(sheets)))
            workbook.writestr("_rels/.rels", cls._root_relationships())
            workbook.writestr("xl/workbook.xml", cls._workbook_xml(sheets))
            workbook.writestr(
                "xl/_rels/workbook.xml.rels", cls._workbook_relationships(len(sheets))
            )
            workbook.writestr("xl/styles.xml", cls._styles_xml())
            for index, (_, fields, rows) in enumerate(sheets, start=1):
                workbook.writestr(
                    f"xl/worksheets/sheet{index}.xml", cls._worksheet_xml(fields, rows)
                )

    @staticmethod
    def _content_types(sheet_count: int) -> str:
        overrides = "".join(
            f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            for index in range(1, sheet_count + 1)
        )
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/styles.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
            f"{overrides}</Types>"
        )

    @staticmethod
    def _root_relationships() -> str:
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="xl/workbook.xml"/>'
            "</Relationships>"
        )

    @staticmethod
    def _workbook_xml(sheets: list[tuple[str, list[str], list[dict[str, str]]]]) -> str:
        sheet_nodes = "".join(
            f'<sheet name="{escape(name)}" sheetId="{index}" r:id="rId{index}"/>'
            for index, (name, _, _) in enumerate(sheets, start=1)
        )
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f"<sheets>{sheet_nodes}</sheets></workbook>"
        )

    @staticmethod
    def _workbook_relationships(sheet_count: int) -> str:
        nodes = "".join(
            f'<Relationship Id="rId{index}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{index}.xml"/>'
            for index in range(1, sheet_count + 1)
        )
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f'{nodes}<Relationship Id="rId{sheet_count + 1}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
            'Target="styles.xml"/></Relationships>'
        )

    @staticmethod
    def _styles_xml() -> str:
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
            '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
            '<borders count="1"><border/></borders>'
            '<cellStyleXfs count="1"><xf/></cellStyleXfs><cellXfs count="1"><xf xfId="0"/>'
            "</cellXfs></styleSheet>"
        )

    @classmethod
    def _worksheet_xml(cls, fields: list[str], rows: list[dict[str, str]]) -> str:
        all_rows = [{field: field for field in fields}, *rows]
        row_nodes = "".join(
            f'<row r="{row_number}">'
            + "".join(
                cls._cell_xml(column_number, row_number, row.get(field, ""))
                for column_number, field in enumerate(fields, start=1)
            )
            + "</row>"
            for row_number, row in enumerate(all_rows, start=1)
        )
        last_cell = f"{cls._excel_column(len(fields))}{len(all_rows)}"
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f'<dimension ref="A1:{last_cell}"/><sheetViews><sheetView workbookViewId="0">'
            '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
            "</sheetView></sheetViews><sheetData>"
            f'{row_nodes}</sheetData><autoFilter ref="A1:{last_cell}"/></worksheet>'
        )

    @classmethod
    def _cell_xml(cls, column_number: int, row_number: int, value: object) -> str:
        text = cls._xml_safe_text(str(value))
        reference = f"{cls._excel_column(column_number)}{row_number}"
        return f'<c r="{reference}" t="inlineStr"><is><t xml:space="preserve">{escape(text)}</t></is></c>'

    @staticmethod
    def _xml_safe_text(value: str) -> str:
        if len(value) > 32767:
            raise ValueError("Excel cells cannot contain more than 32767 characters")
        return "".join(
            character
            for character in value
            if ord(character) >= 32 or character in "\t\n\r"
        )

    @staticmethod
    def _excel_column(number: int) -> str:
        label = ""
        while number:
            number, remainder = divmod(number - 1, 26)
            label = chr(65 + remainder) + label
        return label

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="sara-package-server-run")
    parser.add_argument(
        "source", type=Path, help="Completed results/server-* directory."
    )
    parser.add_argument(
        "destination", nargs="?", type=Path, help="New delivery directory."
    )
    args = parser.parse_args(argv)
    destination = args.destination or args.source.with_name(
        f"{args.source.name}-delivery"
    )
    archive = ServerDeliveryBuilder(args.source, destination).build()
    print(f"Delivery complete: {archive}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
