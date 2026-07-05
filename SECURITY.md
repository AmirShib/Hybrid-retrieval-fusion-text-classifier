# Security Policy

## Supported versions

This project is pre-1.0; only the latest released version on the default
branch receives security fixes. There is no long-term-support branch.

## Reporting a vulnerability

Please **do not** open a public GitHub issue for a suspected vulnerability.
Instead, use GitHub's private reporting:

1. Go to the repository's **Security** tab.
2. Click **Report a vulnerability** to open a private advisory.

Include, if known: the affected version/commit, a minimal reproduction, and
the impact you'd expect (e.g. what an attacker could do with it).

We'll acknowledge new reports as soon as we can and follow up once a fix or
mitigation is available. There is no guaranteed SLA on response time — this
is a small open-source project rather than a commercially supported product.

## Scope notes specific to this package

- A trained model directory is loaded with `ArtifactRepository.load()`. Model
  directories are currently persisted with stdlib `pickle` for some
  components — **only load model directories you trust**, the same caution
  that applies to any pickle-based ML artifact format. Loading an untrusted
  model directory can execute arbitrary code.
- The CLIs read CSV/JSON files supplied by the user running them; they are
  not designed to process untrusted input from an unauthenticated remote
  party without additional sandboxing.
