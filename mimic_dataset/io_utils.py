from __future__ import annotations

import io
import os
import zipfile
from dataclasses import dataclass
from typing import BinaryIO, Dict, Iterable, Iterator, List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class FileRef:
    """Reference to a table-like file, possibly inside a ZIP archive."""

    source_path: str
    member_path: Optional[str] = None  # for zip members

    def is_zip_member(self) -> bool:
        return self.member_path is not None

    def display_name(self) -> str:
        if self.member_path:
            return f"{self.source_path}::{self.member_path}"
        return self.source_path


def iter_files_recursive(root_dir: str) -> Iterator[str]:
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for fn in filenames:
            yield os.path.join(dirpath, fn)


def iter_table_file_refs(root_dir: str) -> Iterator[FileRef]:
    """Yield FileRef for any CSV-ish resources (csv, csv.gz, zip containing csv)."""
    for path in iter_files_recursive(root_dir):
        lower = path.lower()
        if lower.endswith(".csv") or lower.endswith(".csv.gz"):
            yield FileRef(source_path=os.path.abspath(path))
        elif lower.endswith(".zip"):
            # include members that look like CSVs
            try:
                with zipfile.ZipFile(path) as zf:
                    for name in zf.namelist():
                        nlow = name.lower()
                        if nlow.endswith(".csv") or nlow.endswith(".csv.gz"):
                            yield FileRef(source_path=os.path.abspath(path), member_path=name)
            except zipfile.BadZipFile:
                continue


def open_fileref_binary(fr: FileRef) -> BinaryIO:
    """Open a FileRef for binary reading (caller must close)."""
    if not fr.is_zip_member():
        return open(fr.source_path, "rb")
    zf = zipfile.ZipFile(fr.source_path)
    # ZipExtFile will be closed by caller; zipfile must also remain open.
    # We wrap both into a single handle-like object to ensure closure.
    member_f = zf.open(fr.member_path)  # type: ignore[arg-type]

    class _ZipHandle(io.BufferedReader):
        def close(self) -> None:  # noqa: D401
            try:
                member_f.close()
            finally:
                zf.close()

    return _ZipHandle(member_f)  # type: ignore[arg-type]


def read_csv_schema_sample(
    fr: FileRef,
    nrows: int = 200,
    *,
    encoding_errors: str = "replace",
) -> Tuple[List[str], Dict[str, str]]:
    """Read a tiny sample to infer column names and rough dtypes."""
    # pandas can read from file-like object
    handle = open_fileref_binary(fr)
    try:
        df = pd.read_csv(
            handle,
            nrows=nrows,
            low_memory=True,
            encoding_errors=encoding_errors,
        )
    finally:
        handle.close()

    cols = [str(c) for c in df.columns.tolist()]
    dtypes = {str(k): str(v) for k, v in df.dtypes.to_dict().items()}
    return cols, dtypes


def iter_csv_chunks(
    fr: FileRef,
    *,
    usecols: Optional[List[str]] = None,
    chunksize: int = 200_000,
    encoding_errors: str = "replace",
) -> Iterable[pd.DataFrame]:
    """Stream CSV rows as pandas DataFrames."""
    handle = open_fileref_binary(fr)
    # Note: handle must remain open for the generator lifetime.
    reader = pd.read_csv(
        handle,
        usecols=usecols,
        chunksize=chunksize,
        low_memory=True,
        encoding_errors=encoding_errors,
    )
    try:
        for chunk in reader:
            yield chunk
    finally:
        handle.close()


