"""Query helpers for `.code_index/callgraph.db`.

Usage:
  python3 query_call_graph.py callers <name>     # who calls <name>?
  python3 query_call_graph.py callees <qname>    # what does <qname> call?
  python3 query_call_graph.py hot   [N]          # most-called names (top N, default 30)
  python3 query_call_graph.py orphan             # functions with 0 internal callers
  python3 query_call_graph.py file <path>        # outgoing calls from a file
  python3 query_call_graph.py refactor-check     # impact for the duplicate-cluster targets
"""

from __future__ import annotations

import sqlite3
import sys

DB = ".code_index/callgraph.db"


def conn() -> sqlite3.Connection:
    return sqlite3.connect(DB)


def cmd_callers(name: str) -> None:
    c = conn()
    rows = c.execute(
        """
        SELECT caller_file, caller_qualified, caller_line, call_line, callee_full_name
        FROM calls
        WHERE callee_name = ? OR callee_full_name = ? OR callee_full_name LIKE ?
        ORDER BY caller_file, call_line
        """,
        (name, name, f"%.{name}"),
    ).fetchall()
    if not rows:
        print(f"No callers for '{name}'")
        return
    print(f"=== {len(rows)} call sites referring to '{name}' ===")
    for cf, cq, _cl, call_line, full in rows:
        print(f"  {cf}:{call_line}  {cq}   ->  {full}")


def cmd_callees(qname: str) -> None:
    c = conn()
    rows = c.execute(
        """
        SELECT call_line, callee_full_name, callee_resolved, call_type
        FROM calls
        WHERE caller_qualified = ? OR caller_qualified LIKE ?
        ORDER BY caller_qualified, call_line
        """,
        (qname, f"%.{qname}"),
    ).fetchall()
    if not rows:
        print(f"No outgoing calls from '{qname}'")
        return
    print(f"=== {qname} outgoing calls ===")
    for line, full, resolved, ctype in rows:
        tail = f"   [{resolved}]" if resolved else ""
        print(f"  L{line:>5}  {ctype:<6}  {full}{tail}")


def cmd_hot(top: int = 30) -> None:
    c = conn()
    rows = c.execute(
        """
        SELECT callee_name, COUNT(*) AS n,
               COUNT(DISTINCT caller_file) AS files
        FROM calls
        WHERE callee_name NOT IN ('print','len','range','isinstance','str','int','float',
                                  'list','dict','set','tuple','bool','enumerate','zip',
                                  'open','sorted','min','max','sum','any','all','map','filter',
                                  'getattr','setattr','hasattr','type')
        GROUP BY callee_name
        ORDER BY n DESC
        LIMIT ?
        """,
        (top,),
    ).fetchall()
    print(f"=== Hottest callee names (top {top}, builtins filtered) ===")
    for name, n, files in rows:
        print(f"  {n:5d}  ({files:3d} files)  {name}")


def cmd_orphan() -> None:
    """Functions whose name does not appear as any callee_name anywhere."""
    c = conn()
    fdb = sqlite3.connect(".code_index/functions.db")
    rows = fdb.execute(
        """
        SELECT name, qualified_name, file, line_start, num_lines
        FROM functions
        WHERE name NOT LIKE '\\_\\_%' ESCAPE '\\'
          AND name <> 'main'
          AND name NOT LIKE 'test\\_%' ESCAPE '\\'
          AND file NOT LIKE 'tests/%'
          AND file NOT LIKE 'experiments/%'
        ORDER BY num_lines DESC
        """
    ).fetchall()
    fdb.close()
    print("=== Functions with no caller anywhere in the call graph ===")
    print("    (excludes tests/, experiments/, dunders, main, test_*)\n")
    flagged = 0
    for name, qname, f, l, n in rows:
        cnt = c.execute(
            "SELECT COUNT(*) FROM calls WHERE callee_name = ?", (name,)
        ).fetchone()[0]
        if cnt == 0:
            print(f"  {n:4d} lines  {f}:{l}  {qname}")
            flagged += 1
            if flagged >= 40:
                print("  ... (truncated, 40 shown)")
                break


def cmd_file(path: str) -> None:
    c = conn()
    rows = c.execute(
        """
        SELECT caller_qualified, call_line, callee_full_name, callee_resolved
        FROM calls
        WHERE caller_file = ?
        ORDER BY call_line
        """,
        (path,),
    ).fetchall()
    print(f"=== Outgoing calls from {path} ({len(rows)} edges) ===")
    for cq, line, full, resolved in rows:
        tail = f"   [{resolved}]" if resolved else ""
        print(f"  L{line:>5}  {cq}  ->  {full}{tail}")


def cmd_refactor_check() -> None:
    """Spotlight refactor safety for the Tier-1 duplicate clusters."""
    c = conn()

    targets = [
        # (label, callee_name, optional class filter on callee_full_name)
        ("Detector.fit",                "fit",            ("Detector",)),
        ("Detector.evaluate",           "evaluate",       ("Detector",)),
        ("Detector.predict",            "predict",        ("Detector",)),
        ("Detector.predict_proba",      "predict_proba",  ("Detector",)),
        ("Detector._build_pipeline",    "_build_pipeline", ()),
        ("Detector._find_optimal_threshold", "_find_optimal_threshold", ()),
        ("phase1.get_report",           "get_report",     ()),
        ("export_response_policy",      "export_response_policy", ()),
        ("export_feature_concepts",     "export_feature_concepts", ()),
        ("export_nlg_templates",        "export_nlg_templates", ()),
        ("_drop_non_feature_cols",      "_drop_non_feature_cols", ()),
    ]

    for label, name, hints in targets:
        print(f"\n=== {label}  (callee_name='{name}') ===")
        rows = c.execute(
            """
            SELECT caller_file, caller_qualified, call_line, callee_full_name
            FROM calls
            WHERE callee_name = ?
            ORDER BY caller_file, call_line
            """,
            (name,),
        ).fetchall()
        # Filter for hints (e.g. "Detector" in callee_full_name)
        if hints:
            rows = [r for r in rows if any(h in (r[3] or "") for h in hints)]
        if not rows:
            print("  (no internal callers — safe to refactor freely)")
            continue
        files_seen = {}
        for cf, cq, line, full in rows:
            files_seen.setdefault(cf, []).append((cq, line, full))
        print(f"  {sum(len(v) for v in files_seen.values())} call sites in {len(files_seen)} files:")
        for cf in sorted(files_seen):
            print(f"   {cf}:")
            for cq, line, full in files_seen[cf]:
                print(f"     L{line:<5}  {cq}  ->  {full}")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 1
    cmd = argv[1]
    if cmd == "callers" and len(argv) >= 3:
        cmd_callers(argv[2])
    elif cmd == "callees" and len(argv) >= 3:
        cmd_callees(argv[2])
    elif cmd == "hot":
        cmd_hot(int(argv[2]) if len(argv) >= 3 else 30)
    elif cmd == "orphan":
        cmd_orphan()
    elif cmd == "file" and len(argv) >= 3:
        cmd_file(argv[2])
    elif cmd == "refactor-check":
        cmd_refactor_check()
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
