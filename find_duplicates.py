"""Tìm các function có cấu trúc giống nhau."""

import sqlite3


def find_duplicates(db_path: str, min_lines: int = 5):
    conn = sqlite3.connect(db_path)

    cursor = conn.execute("""
        SELECT structure_hash, COUNT(*) as cnt
        FROM functions
        WHERE num_lines >= ?
        GROUP BY structure_hash
        HAVING cnt > 1
        ORDER BY cnt DESC
    """, (min_lines,))

    duplicate_groups = cursor.fetchall()

    print(f"Tìm thấy {len(duplicate_groups)} nhóm function trùng cấu trúc\n")

    for hash_val, count in duplicate_groups[:20]:
        cursor = conn.execute("""
            SELECT qualified_name, file, line_start, num_lines
            FROM functions
            WHERE structure_hash = ?
            ORDER BY file
        """, (hash_val,))

        funcs = cursor.fetchall()
        print(f"[*] {count} functions giong nhau ({funcs[0][3]} dong):")
        for name, file, line, _ in funcs:
            print(f"   {file}:{line}  ->  {name}")
        print()

    conn.close()


def find_similar_names(db_path: str):
    """Tìm function có tên tương tự (có thể merge được)."""
    conn = sqlite3.connect(db_path)

    cursor = conn.execute("""
        SELECT name, COUNT(DISTINCT file) as files, COUNT(*) as total
        FROM functions
        GROUP BY name
        HAVING files > 1
        ORDER BY total DESC
        LIMIT 20
    """)

    print("\n[?] Function trung ten o nhieu file:")
    for name, files, total in cursor.fetchall():
        print(f"   {name}: {total} lan o {files} files")

    conn.close()


if __name__ == '__main__':
    db = '.code_index/functions.db'
    find_duplicates(db)
    find_similar_names(db)
