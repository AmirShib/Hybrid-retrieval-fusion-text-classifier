# T50 — Pin dependencies for air-gapped reproducibility

status: todo
tier: 5
depends_on: —

## Goal
A reproducible, hash-verified dependency set so an install on the air-gapped host
yields byte-identical wheels to the one that was tested.

## Why
The package targets air-gapped deployment, where "it worked on my machine" is not
acceptable — the exact transitive dependency tree (including torch, which
sentence-transformers pulls in) must be reproducible offline. `pyproject.toml`
currently declares only conservative *lower bounds* (done in T60); that is enough
to install but not to reproduce.

## Scope
- Produce a fully pinned, hashed lockfile (e.g. `pip-compile`/`uv` →
  `requirements.lock` with `--generate-hashes`) for a reference platform/Python.
- Document the offline install flow: build/download a wheelhouse, then
  `pip install --no-index --find-links wheelhouse -r requirements.lock`.
- Decide and document the policy for the heavy ML wheels (torch/xgboost): which
  versions are validated, and how to refresh the lock.
- Optionally add a CI job that installs from the lockfile to prove it resolves.

## Acceptance criteria
- [ ] A hashed lockfile exists and installs offline from a wheelhouse.
- [ ] The README documents the air-gapped install procedure.
- [ ] CI verifies the lockfile resolves on the reference platform.

## Notes
Version floors in `pyproject.toml` were added in T60; this ticket is the
remaining hash-pinned lockfile work.
