"""Data loader for CIC-IDS-2018 dataset."""

import os
import logging
import threading
import time
import json
import shutil
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd

from src.phase1_preprocessing.io_utils import (
    validate_and_resolve_path,
    validate_filename,
    ensure_within_dir,
    downcast_df,
)

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and combine CIC-IDS-2018 CSV files."""
    
    # Allowed file extensions for security
    ALLOWED_EXTENSIONS = {'.csv', '.CSV'}
    MAX_FILENAME_LENGTH = 255
    
    def __init__(
        self,
        data_dir: str,
        io_retries: int = 1,
        io_retry_delay_seconds: int = 1,
        read_timeout_seconds: int = 0,
        quarantine_dir: Optional[str] = None,
        skip_on_error: bool = False,
    ):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Directory containing CIC-IDS-2018 CSV files
            
        Raises:
            FileNotFoundError: If data directory does not exist
            ValueError: If data_dir contains path traversal sequences
        """
        self.data_dir = validate_and_resolve_path(data_dir)
        self.io_retries = max(1, io_retries)
        self.io_retry_delay_seconds = max(0, io_retry_delay_seconds)
        self.read_timeout_seconds = max(0, read_timeout_seconds)
        self.skip_on_error = skip_on_error
        self.quarantine_dir = Path(quarantine_dir) if quarantine_dir else None
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        if not self.data_dir.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {data_dir}")

    def _quarantine(self, csv_file: Path) -> None:
        if not self.quarantine_dir:
            return
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        dest = self.quarantine_dir / csv_file.name
        try:
            shutil.copy2(csv_file, dest)
            logger.warning(f"Quarantined file to {dest}")
        except Exception as exc:
            logger.error(f"Failed to quarantine {csv_file}: {exc}")

    def _with_retries(self, desc: str, func):
        last_exc = None
        for attempt in range(1, self.io_retries + 1):
            try:
                return func()
            except Exception as exc:
                last_exc = exc
                if attempt < self.io_retries:
                    logger.warning(f"{desc} failed (attempt {attempt}/{self.io_retries}), retrying in {self.io_retry_delay_seconds}s: {exc}")
                    time.sleep(self.io_retry_delay_seconds)
                else:
                    logger.error(f"{desc} failed after {self.io_retries} attempts: {exc}")
                    raise last_exc

    def _read_csv_with_timeout(self, csv_file: Path, **kwargs) -> pd.DataFrame:
        def reader():
            # POSIX-only timeout via signal; ignored on non-POSIX or non-main threads
            use_alarm = (
                self.read_timeout_seconds > 0
                and hasattr(signal, "SIGALRM")
                and threading.current_thread() is threading.main_thread()
            )
            if use_alarm:
                def _timeout_handler(signum, frame):
                    raise TimeoutError(f"read_csv timed out after {self.read_timeout_seconds}s for {csv_file}")
                previous = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(self.read_timeout_seconds)
                try:
                    return pd.read_csv(csv_file, **kwargs)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, previous)
            else:
                return pd.read_csv(csv_file, **kwargs)

        return self._with_retries(f"read_csv:{csv_file.name}", reader)
    

    def smart_load(
        self,
        pattern: str = "*.csv",
        memory_threshold_mb: int = 2000,
        chunksize: int = 200_000,
        use_chunking: bool = True,
        io_workers: int = 1
    ) -> pd.DataFrame:
        """
        Smart data loading with automatic chunking for large files.
        
        Prevents memory exhaustion by using chunked loading for files exceeding
        memory threshold. This is now the recommended default loading method.
        
        Args:
            pattern: Glob pattern for CSV files
            memory_threshold_mb: Auto-chunk if total file size exceeds this (MB)
            chunksize: Rows per chunk for large files
            use_chunking: Force chunking regardless of file size
            io_workers: Parallel I/O workers (files level). Use 1 to disable.
            
        Returns:
            Combined DataFrame with memory-efficient loading
        """
        if ".." in pattern or "/" in pattern or "\\" in pattern:
            raise ValueError("Pattern cannot contain path traversal sequences or path separators")
        
        csv_files = list(self.data_dir.glob(pattern))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")
        
        # Calculate total file size
        total_size_mb = sum(f.stat().st_size for f in csv_files) / (1024 * 1024)
        logger.info(json.dumps({
            "event": "metric",
            "metric": "data.files.total_size_mb",
            "value": round(total_size_mb, 3),
            "files": len(csv_files)
        }))
        
        # Decide loading strategy
        if use_chunking or total_size_mb > memory_threshold_mb:
            logger.info(
                f"Using chunked loading (size {total_size_mb:.1f} MB > "
                f"threshold {memory_threshold_mb} MB)"
            )
            return self._load_chunked_memory_efficient(
                csv_files, 
                chunksize=chunksize,
                io_workers=io_workers
            )
        else:
            logger.info(f"Using direct loading (size {total_size_mb:.1f} MB)")
            return self.load_csv_files_direct(csv_files, io_workers=io_workers)
    
    def _load_chunked_memory_efficient(
        self,
        csv_files: List[Path],
        chunksize: int = 200_000,
        io_workers: int = 1
    ) -> pd.DataFrame:
        """
        Memory-efficient chunked loading with streaming processing.
        
        Processes data in chunks to minimize peak memory usage.
        
        Args:
            csv_files: List of CSV file paths
            chunksize: Rows per chunk
            
        Returns:
            Combined DataFrame
        """
        all_chunks: List[pd.DataFrame] = []
        total_rows = 0
        lock = threading.Lock()

        def process_file(csv_file: Path):
            try:
                ensure_within_dir(csv_file, self.data_dir)
            except ValueError:
                logger.warning(f"Skipping file outside data directory: {csv_file}")
                return [], 0, str(csv_file), 0

            logger.info(f"Streaming {csv_file.name} in chunks of {chunksize:,} rows...")

            def stream_chunks():
                local_chunks: List[pd.DataFrame] = []
                local_rows = 0
                chunk_count = 0
                for chunk in pd.read_csv(
                    csv_file,
                    encoding='utf-8',
                    low_memory=False,
                    chunksize=chunksize
                ):
                    chunk = downcast_df(chunk)
                    local_chunks.append(chunk)
                    local_rows += len(chunk)
                    chunk_count += 1
                return local_chunks, local_rows, chunk_count

            try:
                local_chunks, local_rows, chunk_count = self._with_retries(
                    f"chunked_stream:{csv_file.name}", stream_chunks
                )
            except Exception as exc:
                self._quarantine(csv_file)
                logger.error(f"Failed to stream {csv_file}: {exc}")
                if self.skip_on_error:
                    logger.warning(f"Skipping {csv_file.name} after failures")
                    return [], 0, str(csv_file), 0
                raise

            return local_chunks, local_rows, csv_file.name, chunk_count

        if io_workers <= 1:
            for csv_file in csv_files:
                chunks, rows, name, chunk_count = process_file(csv_file)
                all_chunks.extend(chunks)
                total_rows += rows
                logger.info(f"  Loaded {chunk_count} chunks, {rows:,} rows from {name}")
        else:
            logger.info(f"Parallel chunked loading with {io_workers} workers")
            with ThreadPoolExecutor(max_workers=io_workers) as executor:
                futures = {executor.submit(process_file, f): f for f in csv_files}
                for future in as_completed(futures):
                    chunks, rows, name, chunk_count = future.result()
                    with lock:
                        all_chunks.extend(chunks)
                        total_rows += rows
                    logger.info(f"  Loaded {chunk_count} chunks, {rows:,} rows from {name}")

        logger.info(f"Concatenating {len(all_chunks)} chunks...")
        t0 = time.perf_counter()
        combined_df = pd.concat(all_chunks, ignore_index=True)
        concat_time = time.perf_counter() - t0
        combined_df = downcast_df(combined_df)
        mem_mb = combined_df.memory_usage(deep=True).sum() / 1_000_000
        logger.info(json.dumps({
            "event": "metric",
            "metric": "data.load.chunked",
            "rows": len(combined_df),
            "cols": len(combined_df.columns),
            "memory_mb": round(mem_mb, 3),
            "concat_seconds": round(concat_time, 3)
        }))
        
        return combined_df
    
    def load_csv_files_direct(self, csv_files: List[Path], io_workers: int = 1) -> pd.DataFrame:
        """
        Direct loading for small datasets (legacy method).
        
        Use smart_load() instead for automatic memory management.
        
        Args:
            csv_files: List of CSV file paths
            
        Returns:
            Combined DataFrame
        """
        def load_file(csv_file: Path) -> Tuple[pd.DataFrame, str]:
            try:
                ensure_within_dir(csv_file, self.data_dir)
            except ValueError:
                logger.warning(f"Skipping file outside data directory: {csv_file}")
                return pd.DataFrame(), str(csv_file)

            logger.info(f"Loading {csv_file.name}...")
            try:
                df = self._read_csv_with_timeout(csv_file, encoding='utf-8', low_memory=False)
                return df, csv_file.name
            except pd.errors.ParserError as e:
                self._quarantine(csv_file)
                logger.error(f"CSV parsing error in {csv_file}: {e}")
                if self.skip_on_error:
                    logger.warning(f"Skipping {csv_file.name} due to parse error")
                    return pd.DataFrame(), csv_file.name
                raise ValueError(f"Invalid CSV file format: {csv_file.name}") from e
            except Exception as e:
                self._quarantine(csv_file)
                logger.error(f"Error loading {csv_file}: {e}")
                if self.skip_on_error:
                    logger.warning(f"Skipping {csv_file.name} after failures")
                    return pd.DataFrame(), csv_file.name
                raise

        dfs: List[pd.DataFrame] = []
        if io_workers <= 1:
            for csv_file in csv_files:
                df, name = load_file(csv_file)
                if not df.empty:
                    dfs.append(df)
                    logger.info(f"  Loaded {len(df):,} rows from {name}")
        else:
            logger.info(f"Parallel direct loading with {io_workers} workers")
            with ThreadPoolExecutor(max_workers=io_workers) as executor:
                futures = {executor.submit(load_file, f): f for f in csv_files}
                for future in as_completed(futures):
                    df, name = future.result()
                    if not df.empty:
                        dfs.append(df)
                        logger.info(f"  Loaded {len(df):,} rows from {name}")
        
        combined_df = pd.concat(dfs, ignore_index=True)
        mem_mb = combined_df.memory_usage(deep=True).sum() / 1_000_000
        logger.info(json.dumps({
            "event": "metric",
            "metric": "data.load.direct",
            "rows": len(combined_df),
            "cols": len(combined_df.columns),
            "memory_mb": round(mem_mb, 3)
        }))
        
        return combined_df
    
    def load_csv_files(self, pattern: str = "*.csv", use_smart_loading: bool = True, io_workers: int = 1) -> pd.DataFrame:
        """
        Load CSV files with optional smart memory management.
        
        Args:
            pattern: Glob pattern for CSV files
            use_smart_loading: Use automatic chunking for large files (recommended)
            
        Returns:
            Combined DataFrame
        """
        if use_smart_loading:
            return self.smart_load(pattern=pattern, io_workers=io_workers)
        else:
            # Legacy direct loading
            if ".." in pattern or "/" in pattern or "\\" in pattern:
                raise ValueError("Pattern cannot contain path traversal sequences or path separators")
            csv_files = list(self.data_dir.glob(pattern))
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in {self.data_dir}")
            return self.load_csv_files_direct(csv_files, io_workers=io_workers)

    def load_csv_files_chunked(
        self,
        pattern: str = "*.csv",
        chunksize: int = 200_000,
        usecols: Optional[List[str]] = None,
        categorical_threshold: int = 50,
        write_intermediate: bool = True,
        intermediate_format: str = "feather",
        return_dataframe: bool = True
    ) -> Tuple[Optional[pd.DataFrame], Dict[str, int]]:
        """
        Load CSV files in chunks with projection and downcasting.

        Args:
            pattern: Glob pattern for CSV files.
            chunksize: Rows per chunk for streamed reading.
            usecols: Optional list of columns to project.
            categorical_threshold: Max unique values to convert object→category.
            write_intermediate: If True, write each processed chunk to disk.
            intermediate_format: 'feather' or 'parquet'.
            return_dataframe: If True, concatenate and return a DataFrame; else returns None.

        Returns:
            (DataFrame or None, stats dict with rows_written and chunks)
        """
        if ".." in pattern or "/" in pattern or "\\" in pattern:
            raise ValueError("Pattern cannot contain path traversal sequences or path separators")
        csv_files = list(self.data_dir.glob(pattern))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")

        logger.info(f"Found {len(csv_files)} CSV files (chunked mode)")
        intermediate_dir = self.data_dir / "_intermediate"
        if write_intermediate:
            intermediate_dir.mkdir(parents=True, exist_ok=True)

        chunk_paths = []
        dfs = []
        total_rows = 0
        chunks = 0

        for csv_file in csv_files:
            try:
                ensure_within_dir(csv_file, self.data_dir)
            except ValueError:
                logger.warning(f"Skipping file outside data directory: {csv_file}")
                continue

            logger.info(f"Streaming {csv_file.name} with chunksize={chunksize}...")

            def stream_and_process():
                local_chunk_paths = []
                local_dfs = []
                local_total_rows = 0
                local_chunks = 0

                def _iterate_chunks():
                    if self.read_timeout_seconds > 0 and hasattr(signal, "SIGALRM"):
                        def _timeout_handler(signum, frame):
                            raise TimeoutError(f"chunked read timed out after {self.read_timeout_seconds}s for {csv_file}")
                        previous = signal.signal(signal.SIGALRM, _timeout_handler)
                        signal.alarm(self.read_timeout_seconds)
                        try:
                            for chunk in pd.read_csv(
                                csv_file,
                                encoding='utf-8',
                                low_memory=False,
                                usecols=usecols,
                                chunksize=chunksize
                            ):
                                yield chunk
                        finally:
                            signal.alarm(0)
                            signal.signal(signal.SIGALRM, previous)
                    else:
                        for chunk in pd.read_csv(
                            csv_file,
                            encoding='utf-8',
                            low_memory=False,
                            usecols=usecols,
                            chunksize=chunksize
                        ):
                            yield chunk

                for chunk in _iterate_chunks():
                    chunk = downcast_df(chunk, categorical_threshold=categorical_threshold)
                    local_chunks += 1
                    local_total_rows += len(chunk)

                    if write_intermediate:
                        out_path = intermediate_dir / f"{csv_file.stem}_chunk{local_chunks}.{ 'feather' if intermediate_format=='feather' else 'parquet'}"
                        if intermediate_format == "feather":
                            chunk.reset_index(drop=True).to_feather(out_path)
                        else:
                            chunk.reset_index(drop=True).to_parquet(out_path, index=False)
                        local_chunk_paths.append(out_path)
                    if return_dataframe:
                        local_dfs.append(chunk)

                return local_total_rows, local_chunks, local_chunk_paths, local_dfs

            try:
                local_total_rows, local_chunks, local_chunk_paths, local_dfs = self._with_retries(
                    f"chunked_projection:{csv_file.name}", stream_and_process
                )
            except Exception as exc:
                self._quarantine(csv_file)
                logger.error(f"Failed during chunked stream {csv_file}: {exc}")
                if self.skip_on_error:
                    logger.warning(f"Skipping {csv_file.name} after failures")
                    continue
                raise

            total_rows += local_total_rows
            chunks += local_chunks
            chunk_paths.extend(local_chunk_paths)
            dfs.extend(local_dfs)

        combined_df = pd.concat(dfs, ignore_index=True) if return_dataframe else None
        logger.info(f"Chunked load complete: {total_rows} rows across {chunks} chunks")
        if combined_df is not None:
            logger.info(f"Combined in-memory frame: {combined_df.shape}")
        return combined_df, {"rows_written": total_rows, "chunks": chunks, "intermediate_paths": [str(p) for p in chunk_paths]}
    
    def load_single_file(self, filename: str) -> pd.DataFrame:
        """
        Load a single CSV file with path validation.
        
        Args:
            filename: Name of CSV file (without directory path)
            
        Returns:
            DataFrame
            
        Raises:
            ValueError: If filename is invalid or contains traversal sequences
            FileNotFoundError: If file does not exist
        """
        # Validate filename
        validated_filename = validate_filename(filename)
        
        filepath = self.data_dir / validated_filename
        
        # Verify resolved path is within data_dir
        try:
            filepath_resolved = ensure_within_dir(filepath, self.data_dir)
        except ValueError:
            raise ValueError(f"Path traversal detected: {filename}")

        if not filepath_resolved.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        if not filepath_resolved.is_file():
            raise ValueError(f"Path is not a file: {filename}")
        
        logger.info(f"Loading {filename}...")
        try:
            df = pd.read_csv(filepath, encoding='utf-8', low_memory=False)
            logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            return df
        except pd.errors.ParserError as e:
            logger.error(f"CSV parsing error: {e}")
            raise ValueError(f"Invalid CSV file format: {filename}") from e
    
    def get_label_distribution(self, df: pd.DataFrame, label_col: str = 'Label') -> pd.Series:
        """
        Get distribution of labels.
        
        Args:
            df: Input DataFrame
            label_col: Name of label column
            
        Returns:
            Series with label counts
        """
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found")
        
        distribution = df[label_col].value_counts()
        logger.info(f"\nLabel distribution:\n{distribution}")
        
        return distribution
