# Module 0 — Performance Optimizations (Historical Notes)

Rationale for the non-obvious shapes in `module0_analysis/`. Each
optimization is a deliberate "before → after" with a measurable reason.
Listed here instead of in code so the modules read clean. To inspect the
pre-optimization shape of any item, see `git log -S <symbol>`.

## Opt-1 — `descriptive_stats`: per-group bulk aggregation
[analyzer.py::StatisticsAnalyzer.descriptive_stats](analyzer.py)

**Before:** One Python loop over F numeric columns, each iteration making
2-5 separate pandas reductions (`.mean()`, `.median()`, `.std()`, `.min()`,
`.max()`) → 2-5×F pandas reduction calls, each paying C → Python boundary
overhead.

**After:** Partition columns once into `net_cols` / `bio_cols`, then one
`DataFrame.agg([...])` call per group. Two C-level passes total (one per
group) regardless of F.

## Opt-2 — Avoid post-loop second pass over biometric counts
[analyzer.py::StatisticsAnalyzer.descriptive_stats](analyzer.py)

**Before:** After the main loop, `sum(1 for c in stats if c in BIOMETRIC_COLUMNS)`
walked the result dict a second time to log the biometric channel count.

**After:** `len(bio_cols)` is already known from the partition built for
Opt-1 — no second pass.

## Opt-3 — `missing_values`: single vectorised null count
[analyzer.py::StatisticsAnalyzer.missing_values](analyzer.py)

**Before:** Per-column `df[col].isna().sum()` inside a Python loop. C →
Python transition C times.

**After:** `df.isna().sum()` returns the full per-column series in one
C-level pass; iterate only the non-zero entries.

## Opt-4 — `high_correlation_pairs`: numpy triu + boolean mask
[analyzer.py::CorrelationAnalyzer.high_correlation_pairs](analyzer.py)

**Before:** O(F²) Python double-loop with per-pair NaN check and threshold
compare.

**After:** `np.triu_indices(F, k=1)` plus vectorised `~np.isnan(r) & |r|>τ`
mask. Zero Python-level per-pair work — filtering executes in C via numpy.

## Opt-5 — `outlier_report`: one quantile() call for the whole frame
[analyzer.py::OutlierAnalyzer.outlier_report](analyzer.py)

**Before:** Per-column `dropna()` + two separate `.quantile()` calls →
2F sort operations.

**After:** Single `numeric_df.quantile([0.25, 0.75])` returns both bounds
for all columns at once. One C-level pass. Inline counter `n_with`
replaces a post-loop `sum(1 for r in report if ...)` second pass.

## Opt-6 — Module-level `_CORRELATION_INTERPRETATIONS` constant
[quality_report.py](quality_report.py)

**Before:** `_correlation_interpretation()` built a 7-entry dict with 7
`frozenset()` constructions on every call. With dozens of pairs to
interpret in a single report render, that's 7×N allocations.

**After:** Constant built once at module import; lookup is a single
`dict.get(frozenset({a, b}))`.

## Opt-7 — `_section_outliers`: single-pass counter
[quality_report.py::_section_outliers](quality_report.py)

**Before:** Built a `features_with_outliers` list inside the table-emit
loop, then `len()`'d it separately.

**After:** Inline `n_with` counter incremented during construction. No
throwaway list.

## Opt-8 — `log_phase0_event`: set-intersection redaction
[security.py::log_phase0_event](security.py)

**Before:** Iterated `list(payload.keys())` and checked each key against
`BIOMETRIC_COLUMNS`. O(|payload|) name comparisons + a list allocation.

**After:** `BIOMETRIC_COLUMNS & payload.keys()` gives only the keys that
need redacting in O(min(|payload|, |BIOMETRIC_COLUMNS|)). No allocation.

## Opt-9 — Phase 0 public key lru_cache
[security.py::_load_phase0_public_key](security.py)

**Before:** `_read_metadata_verified()` called
`serialization.load_pem_public_key(public_path.read_bytes())` on every
invocation — one PEM disk read + parse per dataset load.

**After:** `@functools.lru_cache(maxsize=1)` on a wrapper function; PEM
is read and parsed at most once per process.

## Opt-10 — Reproducibility report: hyphen-normalised package lookup
[reproducibility_report.py::_section_environment](reproducibility_report.py)

**Before:** Two sequential `dict.get(pkg) or dict.get(pkg.replace("-", "_"))`
calls per key package, with a string allocation in between.

**After:** Build a normalised lookup dict once
(`{k.replace("-", "_"): v for k, v in packages.items()}`), then one
`dict.get(pkg)` per package. Also `_KEY_PACKAGES` is a module-level
constant, not a per-call local list.

## Opt-11 — Output directory created at ReportExporter init
[exporter.py::ReportExporter.__init__](exporter.py)

**Before:** Every `export_*` method called
`path.parent.mkdir(parents=True, exist_ok=True)` → redundant syscall on
every export.

**After:** `config.output_dir.mkdir(...)` once in `__init__`. Individual
exporters still mkdir their parent for standalone use, but inside the
orchestrator path the syscall is a no-op cache hit.
