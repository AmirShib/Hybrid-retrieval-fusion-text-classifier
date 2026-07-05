# Releasing

The version is single-sourced in `text_classifier/_version.py`
(`__version__ = "..."`). `pyproject.toml` declares `version` as `dynamic` and
reads that literal via `[tool.setuptools.dynamic]` — nothing else needs to
change, and the two cannot drift out of sync.

## Cutting a release

1. Update `CHANGELOG.md`: move the `[Unreleased]` entries under a new
   `## [X.Y.Z] - YYYY-MM-DD` heading; leave `[Unreleased]` empty above it.
2. Bump `__version__` in `text_classifier/_version.py` to `X.Y.Z`.
3. Commit: `Release X.Y.Z`.
4. Tag: `git tag -a vX.Y.Z -m "Release X.Y.Z"` and `git push origin vX.Y.Z`.
5. Build the distribution:
   ```bash
   python -m pip install --upgrade build
   python -m build            # writes dist/*.whl and dist/*.tar.gz
   ```
6. Attach `dist/*.whl` and `dist/*.tar.gz` to the GitHub release for the tag
   (the `release.yml` workflow does this automatically on tag push; see
   below). Publishing to a package index (e.g. PyPI) is a separate decision —
   not done by default, since the project targets air-gapped installs from a
   wheelhouse (see the README's "Air-gapped / reproducible install" section)
   at least as often as a public index.

## Automated build on tag push

`.github/workflows/release.yml` builds the wheel + sdist and attaches them to
a GitHub release whenever a `v*` tag is pushed. It does not publish to any
package index — that step is intentionally left to a human decision per
release.

## Versioning policy

[Semantic Versioning](https://semver.org/): breaking changes to the public
API (`text_classifier`'s top-level exports, `PipelineConfig`'s schema, the
CLI flags, or the on-disk model-directory format) bump the major version;
new backwards-compatible capability bumps the minor version; fixes bump the
patch version.
