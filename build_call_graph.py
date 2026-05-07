"""Build call graph cho Python codebase."""

import ast
import sqlite3
import json
from pathlib import Path
from collections import defaultdict

class CallExtractor(ast.NodeVisitor):
    """Extract function calls trong scope của 1 function."""

    def __init__(self):
        self.calls = []

    def visit_Call(self, node):
        # foo() -> "foo"
        if isinstance(node.func, ast.Name):
            self.calls.append({
                'name': node.func.id,
                'type': 'direct',
                'line': node.lineno
            })
        # obj.method() -> "method" hoac "Class.method"
        elif isinstance(node.func, ast.Attribute):
            attr_chain = []
            current = node.func
            while isinstance(current, ast.Attribute):
                attr_chain.append(current.attr)
                current = current.value

            if isinstance(current, ast.Name):
                attr_chain.append(current.id)

            attr_chain.reverse()
            self.calls.append({
                'name': attr_chain[-1],
                'full_name': '.'.join(attr_chain),
                'type': 'method',
                'line': node.lineno
            })

        self.generic_visit(node)

class FunctionVisitor(ast.NodeVisitor):
    """Visit tat ca function definition va extract calls."""

    def __init__(self, filepath: str, source: str):
        self.filepath = filepath
        self.source = source
        self.functions = []
        self.class_stack = []
        self.imports = {}  # alias -> module.name

    def visit_Import(self, node):
        for alias in node.names:
            name = alias.asname or alias.name
            self.imports[name] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        module = node.module or ''
        for alias in node.names:
            name = alias.asname or alias.name
            self.imports[name] = f"{module}.{alias.name}"
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node):
        self._process_function(node)

    def visit_AsyncFunctionDef(self, node):
        self._process_function(node)

    def _process_function(self, node):
        extractor = CallExtractor()
        for stmt in node.body:
            extractor.visit(stmt)

        class_name = '.'.join(self.class_stack) if self.class_stack else None
        qualified = f"{class_name}.{node.name}" if class_name else node.name

        resolved_calls = []
        for call in extractor.calls:
            resolved = dict(call)
            base = call.get('full_name', call['name']).split('.')[0]
            if base in self.imports:
                resolved['resolved_module'] = self.imports[base]
            resolved_calls.append(resolved)

        self.functions.append({
            'file': self.filepath,
            'qualified_name': qualified,
            'name': node.name,
            'class_name': class_name,
            'line_start': node.lineno,
            'line_end': node.end_lineno or node.lineno,
            'calls': resolved_calls
        })

        for stmt in node.body:
            self.visit(stmt)


def build_call_graph(root: Path, db_path: str):
    """Build call graph cho toan repo."""

    conn = sqlite3.connect(db_path)
    conn.executescript("""
        DROP TABLE IF EXISTS calls;
        DROP TABLE IF EXISTS callers;

        CREATE TABLE calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            caller_file TEXT,
            caller_qualified TEXT,
            caller_line INTEGER,
            callee_name TEXT,
            callee_full_name TEXT,
            callee_resolved TEXT,
            call_type TEXT,
            call_line INTEGER
        );

        CREATE INDEX idx_caller ON calls(caller_qualified);
        CREATE INDEX idx_callee ON calls(callee_name);
        CREATE INDEX idx_callee_full ON calls(callee_full_name);
        CREATE INDEX idx_resolved ON calls(callee_resolved);
    """)

    exclude = {'.git', '__pycache__', 'venv', '.venv', 'node_modules',
               'build', 'dist', '.tox', '.pytest_cache', 'migrations'}

    py_files = [
        f for f in root.rglob('*.py')
        if not any(part in exclude for part in f.parts)
    ]

    print(f"Building call graph for {len(py_files)} files...")

    total_functions = 0
    total_calls = 0

    for i, filepath in enumerate(py_files, 1):
        try:
            source = filepath.read_text(encoding='utf-8')
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            continue

        rel_path = str(filepath.relative_to(root))
        visitor = FunctionVisitor(rel_path, source)
        visitor.visit(tree)

        for func in visitor.functions:
            total_functions += 1
            for call in func['calls']:
                conn.execute("""
                    INSERT INTO calls
                    (caller_file, caller_qualified, caller_line,
                     callee_name, callee_full_name, callee_resolved,
                     call_type, call_line)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    func['file'],
                    func['qualified_name'],
                    func['line_start'],
                    call['name'],
                    call.get('full_name', call['name']),
                    call.get('resolved_module'),
                    call['type'],
                    call['line']
                ))
                total_calls += 1

        if i % 50 == 0:
            print(f"  {i}/{len(py_files)} files, {total_calls} calls")

    conn.commit()

    print(f"\n[OK] Indexed {total_functions} functions")
    print(f"[OK] Extracted {total_calls} call edges")
    print(f"[OK] DB: {db_path}")

    conn.close()


if __name__ == '__main__':
    import sys
    root = Path(sys.argv[1] if len(sys.argv) > 1 else '.').resolve()
    db_path = root / '.code_index' / 'callgraph.db'
    db_path.parent.mkdir(exist_ok=True)
    build_call_graph(root, str(db_path))
