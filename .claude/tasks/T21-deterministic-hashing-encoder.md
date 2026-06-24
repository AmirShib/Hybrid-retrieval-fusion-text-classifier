# T21 — Deterministic test double: replace `hash()` in HashingEncoder with `hashlib`

status: todo
tier: 2
depends_on: T01

## Goal
Replace the builtin `hash()` call in `tests/_doubles.py::HashingEncoder` with
`hashlib.sha256` so that the test double produces **identical embeddings across
Python processes, versions, and platforms**, regardless of `PYTHONHASHSEED`.

## Why
The current implementation pins `PYTHONHASHSEED=0` in CI to get deterministic
embeddings.  This is fragile: `PYTHONHASHSEED` only applies to the current
process, it silently stops working if a subprocess or parallel worker is spawned,
and it is Python-version-specific.  A `hashlib`-based encoder is portable and
removes the env-var dependency entirely.

## Background
`hash(tok)` with `PYTHONHASHSEED=0` gives a deterministic integer for a given
Python version, but that integer differs between Python 3.10 and 3.11 for the
same token.  `hashlib.sha256(tok.encode()).digest()` is stable across all
environments.

## Files to change
- `tests/_doubles.py` — update `HashingEncoder.encode`
- `.github/workflows/ci.yml` — remove `PYTHONHASHSEED: "0"` from the env block
  (or keep it as an explicit no-op comment explaining it's no longer needed)
- Any test that asserts exact embedding values derived from the old encoder must
  be updated to match the new values (run the suite; update golden values).

## Implementation

Replace:
```python
h = hash(tok)
out[i, h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0
```
With:
```python
import hashlib
digest = hashlib.sha256(tok.encode()).digest()
h = int.from_bytes(digest[:4], "little")
sign_bit = digest[4] & 1
out[i, h % self.dim] += 1.0 if sign_bit else -1.0
```
(Use the first 4 bytes as the bucket index and byte 4 bit 0 as the sign, giving
a stable, uniform distribution.)

## Tests to add/update (`tests/unit/test_doubles.py` or similar)

- [ ] **Cross-process stability:** spawn a subprocess that imports `HashingEncoder`
      without setting `PYTHONHASHSEED` and asserts it produces the same embedding
      as the parent process for a known token.
- [ ] **Known-value regression:** encode `["hello world"]` and assert the resulting
      vector matches a hardcoded expected array (pin the golden value after the
      implementation is written).
- [ ] **L2-normalization preserved:** output rows have unit norm (existing invariant,
      re-assert explicitly).
- [ ] **Smoke:** existing unit + integration suite passes without `PYTHONHASHSEED=0`.

## Acceptance criteria
- [ ] `HashingEncoder` produces identical results in two separate Python invocations
      without any env-var pinning.
- [ ] CI workflow no longer sets `PYTHONHASHSEED`.
- [ ] Golden-value regression test present and passing.
- [ ] Full suite green after the change (including updated exact-value tests in T03).

## Out of scope
Changing the token-splitting strategy (still whitespace `.split()`).
Making `HashingEncoder` faster (T30 territory if ever needed).
