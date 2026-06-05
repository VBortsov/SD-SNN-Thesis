from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.services.paths import REAL_DATA_DIR


@dataclass(frozen=True)
class EdfMetadata:
    """Basic metadata read from an EDF file."""
    path: Path
    channel_names: list[str]
    sample_rates: list[float]
    samples_per_record: list[int]
    total_samples: list[int]
    physical_mins: list[float]
    physical_maxs: list[float]
    digital_mins: list[float]
    digital_maxs: list[float]
    record_duration: float
    num_records: int
    header_bytes: int
    record_bytes: int

    @property
    def duration_seconds(self) -> float:
        if not self.total_samples or not self.sample_rates or self.sample_rates[0] <= 0:
            return 0.0
        return float(self.total_samples[0] / self.sample_rates[0])


@dataclass(frozen=True)
class RealSignalSegment:
    """Windowed real signal segment plus timing metadata."""
    source: str
    channel_name: str
    time_axis: np.ndarray
    raw_signal: np.ndarray
    sampling_rate: float
    total_duration_seconds: float
    start_seconds: float


def list_real_data_files() -> list[Path]:
    """List real data files."""
    if not REAL_DATA_DIR.exists():
        return []
    return sorted(REAL_DATA_DIR.glob("*.edf"))


def _parse_int_field(raw: bytes, default: int = 0) -> int:
    text = raw.decode("latin-1", errors="ignore").strip()
    if not text:
        return default
    try:
        return int(text)
    except ValueError:
        return default


def _parse_float_field(raw: bytes, default: float = 0.0) -> float:
    text = raw.decode("latin-1", errors="ignore").strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _read_fixed_width_fields(handle, count: int, width: int) -> list[str]:
    raw = handle.read(count * width)
    return [raw[idx * width : (idx + 1) * width].decode("latin-1", errors="ignore").strip() for idx in range(count)]


@lru_cache(maxsize=32)
def read_edf_metadata(path_str: str) -> EdfMetadata:
    """Read edf metadata.
    
    Args:
        path_str: Path supplied by the user.
    """
    path = Path(path_str)
    with path.open("rb") as handle:
        fixed = handle.read(256)
        header_bytes = _parse_int_field(fixed[184:192], default=256)
        num_records = _parse_int_field(fixed[236:244], default=-1)
        record_duration = _parse_float_field(fixed[244:252], default=1.0)
        signal_count = _parse_int_field(fixed[252:256], default=0)

        channel_names = _read_fixed_width_fields(handle, signal_count, 16)
        _read_fixed_width_fields(handle, signal_count, 80)  # transducer
        _read_fixed_width_fields(handle, signal_count, 8)  # physical dimension
        physical_mins = [_parse_float_field(item.encode("latin-1")) for item in _read_fixed_width_fields(handle, signal_count, 8)]
        physical_maxs = [_parse_float_field(item.encode("latin-1")) for item in _read_fixed_width_fields(handle, signal_count, 8)]
        digital_mins = [_parse_float_field(item.encode("latin-1")) for item in _read_fixed_width_fields(handle, signal_count, 8)]
        digital_maxs = [_parse_float_field(item.encode("latin-1")) for item in _read_fixed_width_fields(handle, signal_count, 8)]
        _read_fixed_width_fields(handle, signal_count, 80)  # prefiltering
        samples_per_record = [_parse_int_field(item.encode("latin-1")) for item in _read_fixed_width_fields(handle, signal_count, 8)]
        _read_fixed_width_fields(handle, signal_count, 32)  # reserved

    record_bytes = int(sum(samples_per_record) * 2)
    if num_records < 0:
        data_bytes = max(0, path.stat().st_size - header_bytes)
        num_records = int(data_bytes // record_bytes) if record_bytes else 0
    sample_rates = [samples / record_duration if record_duration > 0 else 0.0 for samples in samples_per_record]
    total_samples = [samples * num_records for samples in samples_per_record]
    return EdfMetadata(
        path=path,
        channel_names=channel_names,
        sample_rates=sample_rates,
        samples_per_record=samples_per_record,
        total_samples=total_samples,
        physical_mins=physical_mins,
        physical_maxs=physical_maxs,
        digital_mins=digital_mins,
        digital_maxs=digital_maxs,
        record_duration=record_duration,
        num_records=num_records,
        header_bytes=header_bytes,
        record_bytes=record_bytes,
    )


def _signal_scale(metadata: EdfMetadata, channel_index: int) -> tuple[float, float]:
    digital_min = metadata.digital_mins[channel_index]
    digital_max = metadata.digital_maxs[channel_index]
    physical_min = metadata.physical_mins[channel_index]
    physical_max = metadata.physical_maxs[channel_index]
    denominator = digital_max - digital_min
    if denominator == 0:
        return 1.0, 0.0
    scale = (physical_max - physical_min) / denominator
    offset = physical_min - (digital_min * scale)
    return scale, offset


def load_edf_segment(path: str | Path, channel_name: str, start_seconds: float, duration_seconds: float) -> RealSignalSegment:
    """Load edf segment.
    
    Args:
        path: File or directory path.
    """
    metadata = read_edf_metadata(str(path))
    if channel_name not in metadata.channel_names:
        raise ValueError(f"Channel not found in EDF: {channel_name}")

    channel_index = metadata.channel_names.index(channel_name)
    sampling_rate = metadata.sample_rates[channel_index]
    if sampling_rate <= 0:
        raise ValueError(f"Invalid sampling rate for channel {channel_name}.")

    total_samples = metadata.total_samples[channel_index]
    if total_samples <= 0:
        raise ValueError(f"No samples available for channel {channel_name}.")

    start_sample = int(max(0, round(start_seconds * sampling_rate)))
    if start_sample >= total_samples:
        raise ValueError("Start position is beyond the end of the file.")

    requested_samples = int(max(1, round(duration_seconds * sampling_rate)))
    stop_sample = min(total_samples, start_sample + requested_samples)
    record_samples = metadata.samples_per_record[channel_index]
    first_record = start_sample // record_samples
    last_record = (stop_sample - 1) // record_samples
    record_offset_samples = int(sum(metadata.samples_per_record[:channel_index]))
    channel_bytes = record_samples * 2

    chunks: list[np.ndarray] = []
    with metadata.path.open("rb") as handle:
        for record_index in range(first_record, last_record + 1):
            byte_offset = metadata.header_bytes + (record_index * metadata.record_bytes) + (record_offset_samples * 2)
            handle.seek(byte_offset)
            raw = handle.read(channel_bytes)
            if len(raw) != channel_bytes:
                raise ValueError("Unexpected end of EDF data while reading channel samples.")
            chunks.append(np.frombuffer(raw, dtype="<i2").astype(np.float32))

    joined = np.concatenate(chunks, axis=0)
    local_start = start_sample - (first_record * record_samples)
    local_stop = local_start + (stop_sample - start_sample)
    digital_signal = joined[local_start:local_stop]
    scale, offset = _signal_scale(metadata, channel_index)
    physical_signal = (digital_signal * scale) + offset
    time_axis = np.arange(physical_signal.size, dtype=float) / sampling_rate
    return RealSignalSegment(
        source=metadata.path.name,
        channel_name=channel_name,
        time_axis=time_axis,
        raw_signal=physical_signal.astype(np.float32),
        sampling_rate=float(sampling_rate),
        total_duration_seconds=metadata.duration_seconds,
        start_seconds=float(start_sample / sampling_rate),
    )


@lru_cache(maxsize=4)
def parse_summary_annotations(summary_path: str | None = None) -> dict[str, dict]:
    """Parse summary annotations.
    
    Args:
        summary_path: File-system location.
    """
    path = Path(summary_path) if summary_path else REAL_DATA_DIR / "chb01-summary.txt"
    if not path.exists():
        return {}

    annotations: dict[str, dict] = {}
    current_file: str | None = None
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("File Name:"):
            current_file = line.split(":", 1)[1].strip()
            annotations.setdefault(current_file, {"seizure_count": 0, "seizures": []})
        elif current_file and line.startswith("Number of Seizures in File:"):
            annotations[current_file]["seizure_count"] = _parse_int_field(line.split(":", 1)[1].encode("latin-1"), default=0)
        elif current_file and line.startswith("Seizure Start Time:"):
            start_sec = _parse_float_field(line.split(":", 1)[1].replace("seconds", "").encode("latin-1"), default=0.0)
            annotations[current_file]["seizures"].append({"start_sec": start_sec, "end_sec": start_sec})
        elif current_file and line.startswith("Seizure End Time:"):
            end_sec = _parse_float_field(line.split(":", 1)[1].replace("seconds", "").encode("latin-1"), default=0.0)
            seizures = annotations[current_file]["seizures"]
            if seizures:
                seizures[-1]["end_sec"] = end_sec
    return annotations
