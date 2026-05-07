# AGENTS.md

> **Hướng dẫn bắt buộc cho mọi AI coding agent** (Claude Code, Cursor, Aider, Cline, Continue, Windsurf, GitHub Copilot Workspace, etc.) làm việc trên codebase này.
>
> File này là **single source of truth**. Khi có xung đột giữa instruction của user và quy tắc trong file này, **agent PHẢI báo cáo xung đột** và đợi user xác nhận trước khi vi phạm quy tắc.

---

## 📌 Project Context

**Codebase**: Python ~50k LOC
**Python version**: 3.11+
**Type checker**: Pyright (basic mode, target: strict)
**Linter/Formatter**: Ruff
**Test framework**: pytest
**Package manager**: pip (hoặc poetry/uv nếu có `pyproject.toml`)

### Scope rules (project-specific)

The pipeline rules in `research_spec.yaml` define what this prototype is.
A few do-not-build categories listed there have project-specific carve-outs
for the evaluation track:

- **UI / frontend.** Building a deployable UI product is out of scope, but
  UX work in `module6_evaluation/module6_app.py` is **in scope** when it
  directly supports (a) the Phase 2 user study, (b) the M1–M8 evaluation
  views, or (c) the thesis-defense demo. This includes role selectors,
  page navigation, mode/status indicators, and demo-curation toggles —
  i.e. anything that lets a researcher or examiner exercise the pipeline.
  It does **not** extend to general-purpose UI features (multi-tenant
  workspaces, theming systems, plugin frameworks, etc.).

**Architecture** _(TODO: agent đọc cấu trúc thực tế từ root)_:
```
src/           # Main source code
tests/         # Test files (mirror src/ structure)
scripts/       # CLI utilities, one-off scripts
docs/          # Documentation
.code_index/   # Generated: function index, call graph, embeddings (gitignored)
```

---

## 🚨 Nguyên Tắc Vàng (KHÔNG được vi phạm)

### Rule 1: SEARCH FIRST, CODE LATER
**Trước khi viết bất kỳ function nào**, agent BẮT BUỘC search xem đã có function tương tự chưa. Code duplicate là **lỗi nghiêm trọng nhất**, đắt hơn cả bug.

### Rule 2: IMPACT ANALYSIS BEFORE EDIT
**Trước khi sửa hàm đã tồn tại**, agent BẮT BUỘC query call graph để biết ai gọi nó. Sửa code mà không biết blast radius là **gambling**, không phải engineering.

### Rule 3: VERIFY BEFORE CLAIM DONE
**Trước khi báo "đã xong"**, agent BẮT BUỘC chạy type check + test. "I think it works" không phải là done. Pyright + pytest pass mới là done.

### Rule 4: NEVER SKIP RULES, EVEN IF USER ASKS
Nếu user nói "nhanh thôi, skip test đi" → agent **giải thích rủi ro** và **đề xuất minimum viable verification** thay vì bỏ qua hoàn toàn. Senior không cắt góc, chỉ chọn shortcut an toàn.

### Rule 5: REPORT, DON'T HIDE
Nếu agent gặp lỗi/uncertainty → **báo rõ ràng** với user. KHÔNG được:
- Im lặng skip phần khó
- Generate code "trông giống đúng" mà không verify
- Claim đã làm thứ chưa làm

### Rule 6: ARCHITECTURE.md LÀ SOURCE OF TRUTH

**`ARCHITECTURE.md` ở project root là canonical design spec**. Code phải bám sát doc này — module boundaries, data flow, design invariants, step → code map, naming conventions, configuration files đều phải khớp.

**Trước khi viết hoặc sửa code**, agent BẮT BUỘC:

1. Đọc section liên quan trong `ARCHITECTURE.md` (ví dụ: sửa Module 3 → đọc Step [9] Composite Risk Scoring + invariant tương ứng)
2. Implementation phải khớp với spec — module boundaries, file paths, function names, invariants, config file locations
3. Nếu phát hiện code đã drift khỏi doc → **báo rõ với user**, propose 1 trong 3:
   - **A**: Update doc để khớp với code (nếu code phản ánh quyết định mới)
   - **B**: Update code để khớp với doc (nếu doc là target spec)
   - **C**: Hybrid — doc-update ngắn hạn, code-update khi an toàn

**KHÔNG được tự ý đi lệch**. Ví dụ: doc nói "DAE on raw 25 features only (no cascade)" nhưng code vẫn cascade → đây là drift cần báo, không phải bug để giấu hoặc tự "fix" code.

**Ngoại lệ**: nếu user explicitly chấp nhận drift (ví dụ Phase B post-defense), document drift đó vào `docs/` thay vì xóa khỏi `ARCHITECTURE.md`.

**Why**: Doc và code lệch nhau là silent technical debt. Người đọc doc tin một thứ, người đọc code thấy thứ khác — cả paper, defense slides, và onboarding đều dựa trên doc. Bám sát doc là cách rẻ nhất giữ cho codebase reasonable.

**How to apply**:

- Mọi PR sửa Module N phải có note: "Affected ARCHITECTURE.md sections: [...]" hoặc "No doc impact"
- Mọi review module phải bắt đầu từ doc, không từ code
- Khi spec mơ hồ → ưu tiên hỏi user, không suy diễn

---

## 🛠️ Available Tools (Agent BẮT BUỘC dùng)

Agent có sẵn các tool sau trong codebase này. KHÔNG được giả vờ không biết.

### Search Tools

```bash
# 1. Hybrid search - lexical + symbol + semantic
python code_search.py "<natural language query>"

# 2. Lexical search nhanh
rg "<pattern>" --type py
rg "def <function_name>" --type py

# 3. Symbol jump (ctags)
# Trong Vim: Ctrl-]
# CLI: vim -t <symbol_name>
```

### Call Graph Tools

```bash
# Ai gọi function này?
python query_callgraph.py callers <function_name>

# Function này gọi gì?
python query_callgraph.py callees <function_name>

# Blast radius khi sửa
python query_callgraph.py impact <function_name>

# Tìm dead code
python query_callgraph.py dead

# Trace data flow
python query_callgraph.py trace <from_func> <to_func>
```

### Quality Tools

```bash
# Type check (oracle)
pyright <file_or_directory>

# Lint + format
ruff check --fix <file>
ruff format <file>

# Test
pytest <test_file> -v
pytest --cov=<module> --cov-report=term-missing

# Duplicate detection
python find_duplicates.py
```

### Index Maintenance

```bash
# Rebuild sau khi codebase thay đổi nhiều
python build_function_index.py .
python build_call_graph.py .
python build_embeddings.py
```

---

## 📋 Mandatory Workflows

### 🟢 Workflow ADD: Thêm function/feature mới

Agent BẮT BUỘC follow đúng thứ tự, KHÔNG skip step:

#### Step 1: SEARCH (báo cáo cho user)

```
ACTION:
- Run: python code_search.py "<feature description>"
- Run: rg "def.*<keyword>" --type py
- Đọc top 5 function tương tự nhất

REPORT TO USER:
"Đã search codebase, tìm thấy N function tương tự:
1. <name1> ở <file>:<line> - <signature>
2. <name2> ở <file>:<line> - <signature>
...

Đề xuất: [reuse X / extend Y / viết mới với reasoning]"
```

#### Step 2: PLAN (đợi user confirm)

```
REPORT TO USER:
"Plan implementation:
- File: <path>
- Function name: <name>
- Signature: def foo(x: int, y: str) -> Result
- Behavior: <description>
- Dependencies: <list>
- Test plan: <test cases>

Confirm để tôi bắt đầu code?"
```

⚠️ **KHÔNG code trước khi user confirm**, trừ khi task quá đơn giản (typo, format, comment).

#### Step 3: IMPLEMENT

Quy tắc viết code:

```python
# ✅ ĐÚNG: Type hint đầy đủ + docstring
def calculate_tax(amount: Decimal, region: str) -> Decimal:
    """Calculate tax for a transaction.

    Args:
        amount: Pre-tax amount
        region: ISO region code (e.g., 'VN', 'US-CA')

    Returns:
        Tax amount

    Raises:
        UnknownRegionError: If region not in DB
    """
    ...

# ❌ SAI: Thiếu type hint, không docstring
def calculate_tax(amount, region=None):
    ...
```

**Quy tắc bắt buộc:**
- [ ] Type hint cho mọi argument và return value
- [ ] Docstring cho public function (Google/NumPy style)
- [ ] Single Responsibility - 1 function = 1 việc
- [ ] Function > 30 dòng → suy nghĩ lại, có nên tách không
- [ ] Reuse utility có sẵn, KHÔNG copy-paste logic
- [ ] Error handling rõ ràng (raise specific exception, không bare `except`)

#### Step 4: TEST (viết song song, không sau cùng)

```python
# tests/<module>/test_<file>.py
def test_calculate_tax_valid_region():
    assert calculate_tax(Decimal("100"), "VN") == Decimal("10")

def test_calculate_tax_unknown_region_raises():
    with pytest.raises(UnknownRegionError):
        calculate_tax(Decimal("100"), "XX")

def test_calculate_tax_zero_amount():
    assert calculate_tax(Decimal("0"), "VN") == Decimal("0")
```

**Coverage tối thiểu:**
- Happy path: ≥ 1 test
- Edge cases: ≥ 2 test (zero, negative, max value)
- Error cases: ≥ 1 test cho mỗi exception
- **Total coverage cho function mới: ≥ 80%**

#### Step 5: VERIFY (chạy theo thứ tự, dừng nếu fail)

```bash
# 5.1. Format
ruff format <file>

# 5.2. Lint
ruff check --fix <file>

# 5.3. Type check
pyright <file>

# 5.4. Test
pytest <test_file> -v

# 5.5. Coverage
pytest --cov=<module> --cov-report=term-missing

# 5.6. Duplicate check
python build_function_index.py .
python find_duplicates.py | grep <new_function_name>
# Phải KHÔNG có match
```

#### Step 6: REPORT

```
DEFINITION OF DONE:
✓ Pyright: 0 error
✓ Ruff: 0 warning
✓ Tests: <N> passed, 0 failed
✓ Coverage: <X>%
✓ No duplicate detected
✓ Docstring + type hints complete

FILES CHANGED:
- src/<module>/<file>.py (+<N> lines)
- tests/<module>/test_<file>.py (+<M> lines)

READY TO COMMIT.
```

---

### 🟡 Workflow EDIT: Sửa function đã tồn tại

#### Step 1: IMPACT ANALYSIS (BẮT BUỘC)

```bash
# Run TẤT CẢ các lệnh sau:
python query_callgraph.py callers <function_name>
python query_callgraph.py impact <function_name>
python query_callgraph.py callees <function_name>
rg -l "<function_name>" --type py | grep -i test
```

```
REPORT TO USER:
"Impact analysis cho '<function_name>':
- Direct callers: N nơi (list)
- Blast radius (3 levels): M functions
- Test files liên quan: K files (list)
- Risk level: [LOW/MEDIUM/HIGH/CRITICAL]

Loại thay đổi: [Internal refactor / Signature change / Behavior change]
"
```

**Phân loại risk:**

| Callers | Level | Approach |
|---|---|---|
| 0 | 🟢 LOW | Sửa thoải mái, có thể là dead code |
| 1-3 | 🟡 MEDIUM | Sửa + update caller + test |
| 4-10 | 🟠 HIGH | Comprehensive test, cân nhắc deprecation |
| 10+ | 🔴 CRITICAL | RFC, deprecation plan, migration guide |

#### Step 2: PHÂN LOẠI THAY ĐỔI

**Type A: Internal Refactor (an toàn nhất)**
- Đổi tên biến local, tách helper, optimize
- KHÔNG đổi signature, KHÔNG đổi behavior
- Test cũ phải pass nguyên si

**Type B: Signature Change (BREAKING)**
- Đổi tên hàm, thêm/bớt arg, đổi return type
- BẮT BUỘC update tất cả caller trong cùng PR
- Cân nhắc deprecation thay vì breaking thẳng:

```python
def calculate_tax_v2(amount: Decimal, region: str, currency: str = "VND") -> Decimal:
    """New version with multi-currency support."""
    ...

def calculate_tax(amount: Decimal, region: str) -> Decimal:
    """DEPRECATED: Use calculate_tax_v2.

    .. deprecated:: 2.5.0
        Will be removed in 3.0.0
    """
    import warnings
    warnings.warn("Use calculate_tax_v2", DeprecationWarning, stacklevel=2)
    return calculate_tax_v2(amount, region)
```

**Type C: Behavior Change (NGUY HIỂM NHẤT)**
- Cùng signature, logic khác
- Caller không biết → silent breakage
- BẮT BUỘC: changelog rõ ràng + migration test + announce

#### Step 3: PLAN (đợi user confirm với HIGH/CRITICAL risk)

```
REPORT TO USER:
"Plan edit cho '<function_name>':
- Change type: [A/B/C]
- Files cần sửa: <list>
- Callers cần update: <list>
- Tests cần update/thêm: <list>
- Migration strategy (nếu B/C): <plan>

Confirm để tiến hành?"
```

#### Step 4: IMPLEMENT

**Quy tắc sửa:**
- Atomic commits: 1 commit = 1 logical change
- KHÔNG mix refactor với feature change
- Update caller song song với function chính
- Backward compat khi có thể (optional param > breaking)

#### Step 5: VERIFY

```bash
# 5.1. Baseline test trước khi sửa
pytest > /tmp/before.txt

# 5.2. Sau khi sửa
pytest > /tmp/after.txt
diff /tmp/before.txt /tmp/after.txt

# 5.3. Type check toàn module
pyright src/<module>/

# 5.4. Integration test
pytest tests/integration/ -v

# 5.5. Coverage không giảm
pytest --cov=<module> --cov-fail-under=<previous>
```

#### Step 6: REPORT

```
DEFINITION OF DONE:
✓ All callers updated (verified via callgraph)
✓ Old tests pass + new tests for new behavior
✓ Pyright pass
✓ Coverage: <before>% → <after>% (≥ before)
✓ Docstring updated
✓ Migration path clear (if B/C)

CALLERS UPDATED: <N> files
- file1.py: <description>
- file2.py: <description>

READY TO COMMIT.
```

---

### 🔴 Workflow DELETE: Xóa function/feature

> **Triết lý**: Xóa code là **tài sản**, không phải mất mát. Nhưng phải xóa đúng cách, không thì là **sabotage**.

#### Step 1: VERIFY THỰC SỰ KHÔNG DÙNG

Agent BẮT BUỘC chạy 5 check sau, **TẤT CẢ phải clean**:

```bash
# 1.1. Static call graph
python query_callgraph.py callers <function_name>

# 1.2. Lexical search trong code
rg "<function_name>" --type py

# 1.3. Search trong config files
rg "<function_name>" --type yaml --type json --type toml --type ini

# 1.4. Dynamic reference (getattr, string lookup)
rg "getattr.*<function_name>" --type py
rg "['\"]<function_name>['\"]" --type py

# 1.5. Entry points & build configs
grep -r "<function_name>" pyproject.toml setup.py setup.cfg \
  Makefile .github/ scripts/ 2>/dev/null
```

#### Step 2: CHECK FALSE POSITIVES

⚠️ **KHÔNG xóa nếu function match một trong các pattern sau** (dù callgraph báo 0 callers):

| Pattern | Lý do |
|---|---|
| `@app.route(...)`, `@router.get(...)` | Flask/FastAPI route handler |
| `class Meta:` trong Django model | Django ORM dùng |
| `def __init_subclass__`, `def __init__` | Python magic methods |
| `def setUp`, `def tearDown`, `def test_*` | Test framework gọi |
| Có trong `entry_points` của setup.py | CLI command |
| Có trong YAML/JSON config | Plugin/handler system |
| Inherit từ ABC/Protocol | Subclass dùng implicit |
| Có decorator `@register`, `@command`, etc. | Registry pattern |
| Function được pass as callback | Higher-order usage |

```
REPORT TO USER:
"Verify dead code cho '<function_name>':
✓ Callgraph: 0 callers
✓ Lexical: 0 references in .py
✓ Config: 0 references
✓ Dynamic: 0 getattr/string matches
✓ Entry points: clean

False positive checks:
- Decorator: <pass/fail>
- Magic method: <pass/fail>
- Registry: <pass/fail>

Risk level: [SAFE / NEEDS DEPRECATION / DO NOT DELETE]
Recommendation: <action>"
```

#### Step 3: DEPRECATION (nếu là public API hoặc > 5 callers)

```python
# Release N: Mark deprecated
def old_function(x: int) -> int:
    """DEPRECATED: Use new_function instead.

    .. deprecated:: 2.5.0
        Will be removed in 3.0.0
    """
    warnings.warn(
        "old_function is deprecated, use new_function",
        DeprecationWarning,
        stacklevel=2
    )
    return new_function(x)

# Release N+1: Bump warning level
# Release N+2: Xóa thật
```

KHÔNG xóa thẳng nếu:
- Là public API (export trong `__init__.py`)
- Có > 5 caller
- Module được external package import
- Có tài liệu reference

#### Step 4: SAFE DELETE

```bash
# 4.1. Tạo branch riêng (KHÔNG xóa trên main)
git checkout -b chore/remove-<function_name>

# 4.2. Xóa function + test + import liên quan

# 4.3. Verify không vỡ gì
pyright src/
pytest
ruff check

# 4.4. Rebuild index
python build_function_index.py .
python build_call_graph.py .

# 4.5. Confirm thật sự dead
python query_callgraph.py callers <function_name>
# → "0 callers" or "not found"

# 4.6. Final search
rg "<function_name>" --type py
# → empty
```

#### Step 5: COMMIT & DOCUMENT

```bash
git commit -m "refactor: remove unused function <function_name>

- Confirmed 0 callers via static analysis
- No dynamic references (getattr/string lookups)
- No config references
- Deprecated since v<X.Y.Z> (if applicable)
- Removed associated tests and imports

BREAKING CHANGE: <function_name> was deprecated since <version>
Migration: use <new_function> instead"
```

Update `CHANGELOG.md`:
```markdown
## [Unreleased]

### Removed
- `module.old_function` (deprecated since 2.5.0). Use `new_function` instead.
```

---

## 🎯 Code Quality Standards

### Type Hints (BẮT BUỘC)

```python
# ✅ ĐÚNG
def process(data: dict[str, Any], config: Config | None = None) -> Result:
    ...

def fetch_users(ids: list[int]) -> list[User]:
    ...

async def get_user(user_id: int) -> User | None:
    ...

# ❌ SAI - không có type hint
def process(data, config=None):
    ...

# ❌ SAI - dùng Any vô tội vạ
def process(data: Any) -> Any:
    ...
```

**Quy tắc:**
- Mọi public function: type hint đầy đủ
- Internal helper: ít nhất return type
- Tránh `Any` - dùng `object` hoặc generic nếu cần
- Dùng `| None` thay vì `Optional[X]` (Python 3.10+)
- Dùng `list[X]` thay vì `List[X]` (Python 3.9+)

### Docstring Style

```python
def calculate_discount(
    price: Decimal,
    user_tier: str,
    promo_code: str | None = None,
) -> Decimal:
    """Calculate final price after applying discounts.

    Discounts are stacked in order: tier discount first, then promo code.
    Maximum total discount is capped at 50%.

    Args:
        price: Original price (must be positive)
        user_tier: User membership tier ('free', 'pro', 'enterprise')
        promo_code: Optional promo code from active campaign

    Returns:
        Final price after all discounts, never negative

    Raises:
        InvalidTierError: If user_tier not in allowed values
        InvalidPromoCodeError: If promo_code expired or invalid

    Example:
        >>> calculate_discount(Decimal("100"), "pro", "SUMMER20")
        Decimal('72.00')
    """
```

### Error Handling

```python
# ✅ ĐÚNG: Specific exception
def get_user(user_id: int) -> User:
    user = db.query(User).get(user_id)
    if user is None:
        raise UserNotFoundError(f"User {user_id} not found")
    return user

# ❌ SAI: Bare except
try:
    do_something()
except:  # KHÔNG bao giờ bare except
    pass

# ❌ SAI: Catch quá rộng
try:
    do_something()
except Exception:  # Quá generic
    pass

# ✅ ĐÚNG: Catch specific, re-raise nếu cần
try:
    do_something()
except (ValueError, KeyError) as e:
    logger.error(f"Failed: {e}")
    raise ProcessingError("Could not process") from e
```

### Naming Conventions

```python
# Functions: snake_case, verb-based
def calculate_tax(): ...
def get_user_by_id(): ...
def is_valid_email(): ...  # Bool returns: is_/has_/can_

# Classes: PascalCase, noun-based
class UserRepository: ...
class TaxCalculator: ...

# Constants: SCREAMING_SNAKE_CASE
MAX_RETRY_COUNT = 3
DEFAULT_TIMEOUT_SECONDS = 30

# Private: leading underscore
def _internal_helper(): ...
class _PrivateClass: ...

# Avoid:
def data(): ...           # Quá generic
def do_stuff(): ...       # Vô nghĩa
def func1(): ...          # Không descriptive
```

---

## 🚫 Anti-Patterns (Agent KHÔNG được làm)

### ❌ Anti-pattern 1: Code mà không search

```
USER: "Thêm function validate email"

❌ AI XẤU:
def validate_email(email: str) -> bool:
    # Viết luôn không search
    ...

✅ AI TỐT:
"Để tôi check codebase trước..."
[runs: python code_search.py "validate email"]
"Tôi tìm thấy 3 function tương tự:
1. is_valid_email() trong utils/validation.py
2. check_email_format() trong auth/helpers.py
3. EmailValidator class trong validators/

Bạn muốn:
A. Reuse is_valid_email() (recommended)
B. Generalize EmailValidator
C. Viết mới với reason cụ thể?"
```

### ❌ Anti-pattern 2: Sửa mà không impact analysis

```
USER: "Đổi calculate_tax để support multi-currency"

❌ AI XẤU:
[Sửa luôn signature]

✅ AI TỐT:
"Trước khi sửa, để tôi check impact..."
[runs: python query_callgraph.py callers calculate_tax]
"Function này được gọi ở 23 nơi trong 8 files.
Đề xuất 2 approaches:
A. Backward compat: thêm currency param với default='VND'
B. Breaking change: tạo calculate_tax_v2, deprecate v1

A an toàn hơn, B sạch hơn. Chọn approach nào?"
```

### ❌ Anti-pattern 3: Xóa "dead code" mà không verify

```
USER: "Xóa function unused này"

❌ AI XẤU:
[Xóa luôn]

✅ AI TỐT:
"Check trước khi xóa..."
[runs 5 verification commands]
"Function này có decorator @app.route('/api/legacy/...').
Mặc dù callgraph báo 0 callers, nó vẫn được Flask gọi qua HTTP route.
NOT SAFE TO DELETE. Bạn có chắc đã deprecate route này không?"
```

### ❌ Anti-pattern 4: Skip verification

```
❌ AI XẤU:
"Đã xong! Code chạy ổn rồi."
[không chạy pytest, không chạy pyright]

✅ AI TỐT:
"Đã implement xong. Verifying...
- Pyright: 0 errors ✓
- Pytest: 12 passed, 0 failed ✓
- Coverage: 87% ✓
- No duplicates detected ✓

DONE. Ready to commit."
```

### ❌ Anti-pattern 5: Generate test sau cùng

```
❌ AI XẤU:
[Viết function 100 dòng]
[Sau đó mới viết test cho function đó - test mù theo code]

✅ AI TỐT:
[Viết test case từ spec trước]
[Implement function]
[Verify test pass]
```

### ❌ Anti-pattern 6: Mix concerns trong 1 commit

```
❌ AI XẤU:
git commit -m "Add tax calculation, fix login bug, refactor utils, update deps"

✅ AI TỐT:
git commit -m "feat(tax): add multi-currency tax calculation"
git commit -m "fix(auth): handle expired token correctly"
git commit -m "refactor(utils): extract date helpers to dedicated module"
git commit -m "chore(deps): bump pyright to 1.1.350"
```

---

## 📦 Commit & PR Standards

### Conventional Commits (BẮT BUỘC)

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: Tính năng mới
- `fix`: Bug fix
- `refactor`: Refactor không đổi behavior
- `perf`: Optimize performance
- `test`: Thêm/sửa test
- `docs`: Tài liệu
- `chore`: Maintenance (deps, config)
- `style`: Format, không ảnh hưởng code
- `revert`: Revert commit trước

**Examples:**
```
feat(tax): add multi-currency tax calculation

Adds support for USD, EUR, JPY in addition to VND.
Tax rates are fetched from external API with 1h cache.

Closes #123
```

```
refactor(auth): extract token validation to separate module

No behavior change. Improves testability and prepares for
multi-provider OAuth integration.
```

```
fix(payment): handle decimal rounding edge case

Previously: amounts like 99.999 would round to 100.00 then to 100
Now: properly uses ROUND_HALF_EVEN for financial accuracy

BREAKING CHANGE: Tax calculations now return Decimal with
exactly 2 decimal places. Callers passing Float will see
TypeError instead of silent conversion.

Fixes #456
```

### PR Description Template

```markdown
## What
<Mô tả ngắn gọn thay đổi>

## Why
<Lý do/context. Link issue nếu có>

## How
<Approach kỹ thuật. Trade-offs nếu có>

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing scenarios: <list>

## Impact Analysis
- Callers affected: <N>
- Breaking changes: <Y/N>
- Migration required: <Y/N>

## Checklist
- [ ] Pyright: 0 errors
- [ ] Ruff: 0 warnings
- [ ] Coverage: ≥ previous
- [ ] No new duplicates
- [ ] Docs updated
- [ ] CHANGELOG.md updated (if user-facing)
```

---

## 🤝 Communication Protocol

### Khi nào agent PHẢI hỏi user

1. **Ambiguous requirement**: Spec không rõ, có > 1 cách interpret
2. **HIGH/CRITICAL risk change**: > 4 callers bị ảnh hưởng
3. **Breaking change**: Đổi public API
4. **Trade-off decisions**: Performance vs readability, etc.
5. **Discovery của duplicate**: Tìm thấy code tương tự, hỏi reuse hay refactor
6. **Scope creep**: Task ban đầu nhỏ, agent thấy cần làm thêm thứ khác

### Format câu hỏi

```
❌ MƠ HỒ:
"Bạn muốn làm gì?"

✅ RÕ RÀNG:
"Tôi tìm thấy 3 cách approach cho task này:

A. <option A>
   - Pros: <list>
   - Cons: <list>
   - Effort: 30 min

B. <option B>
   - Pros: <list>
   - Cons: <list>
   - Effort: 2 hours

C. <option C>
   - Pros: <list>
   - Cons: <list>
   - Effort: 1 day

Recommendation: A nếu cần ship nhanh, B nếu codebase này còn phát triển dài.

Bạn chọn approach nào?"
```

### Khi nào agent CHỦ ĐỘNG báo cáo

- Sau mỗi major step (search done, plan ready, implementation done, tests pass)
- Khi gặp blocker không tự giải quyết được
- Khi phát hiện vấn đề ngoài scope (vd: tìm thấy security bug khi đang refactor)
- Khi quyết định khác plan ban đầu

---

## 🔧 Maintenance Tasks

Agent có thể chủ động đề xuất các task này khi thấy phù hợp:

### Weekly
- [ ] Run `python query_callgraph.py dead` → propose cleanup
- [ ] Run `python find_duplicates.py` → propose dedup
- [ ] Check `pyright` errors trend

### Monthly
- [ ] Rebuild full index
- [ ] Review god functions: `python query_callgraph.py god`
- [ ] Check circular deps: `python module_deps.py`
- [ ] Update embeddings nếu codebase thay đổi nhiều

### Per-PR
- [ ] Update index sau merge
- [ ] Verify coverage không giảm
- [ ] Check no new duplicates introduced

---

## 🆘 Escalation: Khi nào STOP và hỏi user

Agent BẮT BUỘC dừng và hỏi user trong các trường hợp sau:

1. **Verification failed** sau 3 lần thử fix
2. **Breaking change** ngoài scope ban đầu
3. **Security concern** (auth, crypto, PII handling)
4. **Performance regression** > 20% trong benchmark
5. **Database schema change** (migration cần plan)
6. **External API change** (cần coordinate với team khác)
7. **License/legal issue** (dependency mới, code copy từ đâu đó)

Format escalation:

```
⚠️ ESCALATION REQUIRED

Issue: <mô tả>
Severity: <LOW/MEDIUM/HIGH/CRITICAL>
Why I'm stopping: <lý do không tự quyết được>

Options:
A. <option>
B. <option>
C. <option>

Recommendation: <if any>

Cần guidance từ bạn để tiếp tục.
```

---

## 📚 References & Project-Specific Notes

_(TODO: User điền các thông tin specific của project)_

### Domain knowledge
- <Link đến domain docs, business rules>

### External dependencies
- <Database schema docs>
- <API contracts>
- <Third-party services>

### Convention exceptions
- <Nếu có chỗ codebase không follow rule chuẩn, document ở đây>

### Known tech debt
- <List các phần code cần cải thiện nhưng chưa có thời gian>

---

## 🎓 Triết Lý Cuối

> **Senior không phải là người viết code nhanh nhất.**
> **Senior là người ngần ngại nhất trước mỗi thay đổi**, vì họ hiểu blast radius.

3 câu thần chú khi muốn skip quy tắc:

1. *"Code này sẽ tồn tại lâu hơn tôi nghĩ."*
2. *"Người sửa nó sau này có thể là tôi của 6 tháng sau, đã quên hết context."*
3. *"Test không có không có nghĩa là không cần test."*

**Nguyên tắc cuối:**

> Khi không chắc chắn → **hỏi user**, đừng đoán.
> Khi đoán → **báo rõ là đang đoán**, đừng giả vờ chắc chắn.
> Khi sai → **thừa nhận và sửa**, đừng cover up.

---

## ✅ Pre-Action Checklist (Agent print ra trước mỗi action)

```
[ ] Đã hiểu requirement chưa?
[ ] Đã search codebase chưa?
[ ] Đã impact analysis chưa? (nếu sửa/xóa)
[ ] Đã có plan rõ ràng chưa?
[ ] User đã confirm chưa? (nếu HIGH risk)
[ ] Có available tool nào support task này?
[ ] Verification strategy là gì?
```

Nếu BẤT KỲ ô nào chưa check → **DỪNG** và xử lý ô đó trước.

---

**END OF AGENTS.md**

> File này được maintain bởi tech lead. Đề xuất thay đổi qua PR với label `agents-md`.
> Last updated: 2026-05-07
> Version: 1.0