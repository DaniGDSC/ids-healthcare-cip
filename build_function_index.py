"""Index toàn bộ function vào SQLite để query nhanh."""

import ast
import sqlite3
import hashlib
from pathlib import Path

def normalize_ast(node):
    """Chuẩn hóa AST: xóa tên biến để detect duplicate logic."""
    if isinstance(node, ast.Name):
        return "VAR"
    if isinstance(node, ast.arg):
        return "ARG"
    if isinstance(node, ast.Constant):
        return f"CONST_{type(node.value).__name__}"

    if isinstance(node, ast.AST):
        result = type(node).__name__
        children = [normalize_ast(c) for c in ast.iter_child_nodes(node)]
        if children:
            result += "(" + ",".join(children) + ")"
        return result
    return str(node)

def get_function_hash(node):
    """Hash của structure function (không tính tên biến)."""
    normalized = normalize_ast(node)
    return hashlib.md5(normalized.encode()).hexdigest()

def init_db(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        DROP TABLE IF EXISTS functions;
        CREATE TABLE functions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            qualified_name TEXT NOT NULL,
            file TEXT NOT NULL,
            line_start INTEGER,
            line_end INTEGER,
            signature TEXT,
            docstring TEXT,
            source TEXT,
            structure_hash TEXT,
            is_async INTEGER DEFAULT 0,
            class_name TEXT,
            num_lines INTEGER,
            num_args INTEGER,
            has_return_type INTEGER DEFAULT 0
        );
        CREATE INDEX idx_name ON functions(name);
        CREATE INDEX idx_hash ON functions(structure_hash);
        CREATE INDEX idx_file ON functions(file);
        CREATE INDEX idx_qualified ON functions(qualified_name);
    """)
    return conn

def index_file(filepath: Path, root: Path, conn):
    try:
        source = filepath.read_text(encoding='utf-8')
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"  SKIP {filepath}: {e}")
        return 0

    rel_path = str(filepath.relative_to(root))
    count = 0

    def process_function(node, class_name=None):
        nonlocal count
        is_async = isinstance(node, ast.AsyncFunctionDef)

        args = [arg.arg for arg in node.args.args]
        sig = f"{'async ' if is_async else ''}def {node.name}({', '.join(args)})"
        if node.returns:
            sig += f" -> {ast.unparse(node.returns)}"

        qualified = f"{class_name}.{node.name}" if class_name else node.name
        source_code = ast.get_source_segment(source, node) or ""

        conn.execute("""
            INSERT INTO functions
            (name, qualified_name, file, line_start, line_end, signature,
             docstring, source, structure_hash, is_async, class_name,
             num_lines, num_args, has_return_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            node.name,
            qualified,
            rel_path,
            node.lineno,
            node.end_lineno or node.lineno,
            sig,
            ast.get_docstring(node) or "",
            source_code,
            get_function_hash(node),
            int(is_async),
            class_name,
            (node.end_lineno or node.lineno) - node.lineno + 1,
            len(node.args.args),
            int(node.returns is not None)
        ))
        count += 1

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            process_function(node)
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    process_function(item, class_name=node.name)

    return count

def main(root_path: str = '.'):
    root = Path(root_path).resolve()
    db_path = root / '.code_index' / 'functions.db'
    db_path.parent.mkdir(exist_ok=True)

    conn = init_db(str(db_path))

    exclude = {'.git', '__pycache__', 'venv', '.venv', 'node_modules',
               'build', 'dist', '.tox', '.pytest_cache', 'migrations'}

    py_files = [
        f for f in root.rglob('*.py')
        if not any(part in exclude for part in f.parts)
    ]

    print(f"Indexing {len(py_files)} files...")
    total = 0
    for i, filepath in enumerate(py_files, 1):
        count = index_file(filepath, root, conn)
        total += count
        if i % 50 == 0:
            print(f"  {i}/{len(py_files)} files, {total} functions")

    conn.commit()

    cursor = conn.execute("SELECT COUNT(*) FROM functions")
    print(f"\n[OK] Indexed {cursor.fetchone()[0]} functions")

    cursor = conn.execute("""
        SELECT structure_hash, COUNT(*) as cnt
        FROM functions
        GROUP BY structure_hash
        HAVING cnt > 1
    """)
    duplicates = cursor.fetchall()
    print(f"[WARN] Found {len(duplicates)} groups of duplicate function structures")

    conn.close()
    print(f"\nDB: {db_path}")

if __name__ == '__main__':
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else '.')
