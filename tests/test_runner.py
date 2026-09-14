from __future__ import annotations

import json
import struct
import tempfile
import unittest
import wave
from pathlib import Path

from sara_audio.config import ApplicationSettings
from sara_audio.predictions import EXPERT_MODEL_NAMES
from sara_audio.runner import BatchProcessor


class BatchProcessorTests(unittest.TestCase):
    def test_directory_creates_a_clear_output_directory_per_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            self._write_wav(source / "first.wav")
            self._write_wav(source / "second.wav")
            settings = ApplicationSettings.from_mapping(
                {
                    "opensmile": {"enabled": False},
                    "execution": {
                        "output_root": str(root / "results"),
                        "feature_formats": ["csv", "json"],
                        "frame_formats": ["csv"],
                        "write_frames": False,
                        "write_segments": True,
                    },
                }
            )

            result = BatchProcessor(settings).process(source)

            self.assertEqual(result.directory, root / "results" / "source")
            self.assertEqual(len(result.records), 2)
            self.assertTrue(result.summary_path.is_file())
            self.assertTrue(result.input_manifest_path and result.input_manifest_path.is_file())
            self.assertEqual(len(result.batch_manifest_paths), 1)
            self.assertTrue(result.wide_features_path and result.wide_features_path.is_file())
            wide_header = result.wide_features_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("TIME_active_speech_duration_s", wide_header)
            self.assertNotIn("TIME_audio_duration_s", wide_header)
            for record in result.records:
                self.assertTrue(record.features_path.is_file())
                self.assertEqual(
                    {path.suffix for path in record.feature_paths}, {".csv", ".json"}
                )
                self.assertTrue(record.segments_path and record.segments_path.is_file())
                self.assertEqual(record.transcript_paths, ())
                self.assertTrue(record.manifest_path.is_file())
                self.assertTrue(record.report_path.is_file())
                self.assertTrue(record.operator_speech_path and record.operator_speech_path.is_file())
                self.assertIsNone(record.frames_path)

    def test_manual_transcript_is_loaded_from_configured_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "speech.wav"
            self._write_wav(audio)
            transcripts = root / "transcripts"
            transcripts.mkdir()
            (transcripts / "speech.txt").write_text(
                "Я очень рад, потому что это прекрасно!",
                encoding="utf-8",
            )
            settings = ApplicationSettings.from_mapping(
                {
                    "opensmile": {"enabled": False},
                    "execution": {
                        "output_root": str(root / "results"),
                        "output_format": "jsonl",
                        "write_frames": False,
                        "transcript_mode": "manual_directory",
                        "transcript_directory": str(transcripts),
                    },
                }
            )

            result = BatchProcessor(settings).process(audio)
            rows = [
                json.loads(line)
                for line in result.records[0].features_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            report = json.loads(
                result.records[0].report_path.read_text(encoding="utf-8")
            )
            transcript_text = result.records[0].transcript_paths[0].read_text(
                encoding="utf-8"
            ).strip()
            transcript_metadata = json.loads(
                result.records[0].transcript_paths[1].read_text(encoding="utf-8")
            )
            wide_header = result.wide_features_path.read_text(encoding="utf-8").splitlines()[0]

        linguistic = [
            row
            for row in rows
            if row["feature_code"].startswith(
                ("STR_", "LEX_", "MOR_", "ORT_", "PUN_", "SYN_", "EMO_")
            )
        ]
        self.assertEqual(len(linguistic), 92)
        self.assertIn("RATE_words_per_second_active", {row["feature_code"] for row in rows})
        self.assertTrue(report["transcript"]["available"])
        self.assertEqual(
            [path.name for path in result.records[0].transcript_paths],
            ["transcript.txt", "transcript.json"],
        )
        self.assertEqual(
            transcript_text,
            "Я очень рад, потому что это прекрасно!",
        )
        self.assertEqual(transcript_metadata["source"], "manual-text-file")
        self.assertEqual(len(report["linguistic_features"]), 92)
        self.assertIn("LEX_EmotLex", wide_header)
        self.assertIn("word_count", wide_header)

    def test_directory_ignores_appledouble_audio_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            self._write_wav(source / "record.wav")
            (source / "._record.wav").write_bytes(b"AppleDouble metadata")
            settings = ApplicationSettings.from_mapping({"opensmile": {"enabled": False}})

            files = BatchProcessor(settings)._find_audio_files(source)

        self.assertEqual(files, [source / "record.wav"])

    def test_prediction_outputs_include_compact_and_interpretation_tables(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            self._write_wav(source / "record.wav")
            model_path = root / "filter.json"
            model_path.write_text(
                json.dumps({"Intercept": 0.6}), encoding="utf-8"
            )
            settings = ApplicationSettings.from_mapping(
                {
                    "opensmile": {"enabled": False},
                    "execution": {
                        "output_root": str(root / "results"),
                        "write_frames": False,
                        "write_segments": False,
                    },
                    "prediction": {
                        "enabled": True,
                        "filter_model_path": str(model_path),
                        "formats": ["csv", "json"],
                    },
                }
            )

            result = BatchProcessor(settings).process(source)

            self.assertEqual(
                {path.name for path in result.prediction_paths},
                {
                    "predictions.csv",
                    "predictions.json",
                    "interpretation_input.csv",
                    "interpretation_input.json",
                },
            )
            compact = (result.directory / "predictions.json").read_text(
                encoding="utf-8"
            )
            full = (result.directory / "interpretation_input.json").read_text(
                encoding="utf-8"
            )
            self.assertEqual(json.loads(compact)[0]["filter_remove"], 1)
            self.assertIn("filter_missing_features", full)
            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(len(summary["predictions"]), 4)

    def test_methodology_outputs_complete_prediction_cascade(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "20241118162438_Agent_15.wav"
            self._write_wav(source)
            models = root / "models"
            models.mkdir()
            model_payloads = {
                "filter": {"Intercept": 0.0},
                "agent_sex": {"Intercept": 1.0},
                "burnout": {"Intercept": 0.0, "m": 1.0},
                **{
                    name: {"Intercept": 0.0}
                    for name in EXPERT_MODEL_NAMES
                },
            }
            for name, payload in model_payloads.items():
                (models / f"{name}.json").write_text(
                    json.dumps(payload),
                    encoding="utf-8",
                )
            settings = ApplicationSettings.from_mapping(
                {
                    "opensmile": {"enabled": False},
                    "execution": {
                        "output_root": str(root / "results"),
                        "write_frames": False,
                        "write_segments": False,
                    },
                    "prediction": {
                        "enabled": True,
                        "filter_model_path": str(models / "filter.json"),
                        "methodology_model_directory": str(models),
                        "formats": ["csv", "json"],
                    },
                }
            )

            result = BatchProcessor(settings).process(source)

            self.assertEqual(
                {path.name for path in result.prediction_paths},
                {
                    "predictions.csv",
                    "predictions.json",
                    "interpretation_input.csv",
                    "interpretation_input.json",
                    "filter_predictions.csv",
                    "filter_predictions.json",
                },
            )
            final = json.loads(
                (result.directory / "predictions.json").read_text(encoding="utf-8")
            )
            self.assertEqual(final[0]["agent_year_id"], "15_2024")
            self.assertEqual(final[0]["agent_sex"], 1)
            self.assertEqual(final[0]["burnout_final"], 1)

    @staticmethod
    def _write_wav(path: Path) -> None:
        sample_rate = 16000
        samples = [10000] * sample_rate
        with wave.open(str(path), "wb") as destination:
            destination.setnchannels(1)
            destination.setsampwidth(2)
            destination.setframerate(sample_rate)
            destination.writeframes(struct.pack(f"<{len(samples)}h", *samples))
