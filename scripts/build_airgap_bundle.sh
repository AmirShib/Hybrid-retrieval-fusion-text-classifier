#!/usr/bin/env bash
# Build a single downloadable archive for air-gapped installs.
#
# Run on a connected machine matching the reference platform (Linux x86_64,
# CPython 3.11 — see requirements.lock). Produces one file,
# text-classifier-airgap-bundle.tar.gz, containing every wheel plus a
# one-command installer. Move that one file to the air-gapped host and run:
#
#   tar xzf text-classifier-airgap-bundle.tar.gz
#   ./wheelhouse/install.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

rm -rf wheelhouse
mkdir -p wheelhouse

echo "Downloading pinned dependencies (hash-verified)..."
pip download --require-hashes -r requirements.lock -d wheelhouse/

echo "Building the text-classifier wheel..."
pip wheel . --no-deps -w wheelhouse/

cp requirements.lock wheelhouse/

cat > wheelhouse/install.sh <<'EOF'
#!/usr/bin/env bash
# Offline install — run from inside the extracted wheelhouse/ directory.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

pip install --no-index --find-links . --require-hashes -r requirements.lock
pip install --no-index --find-links . --no-deps text-classifier
EOF
chmod +x wheelhouse/install.sh

tar czf text-classifier-airgap-bundle.tar.gz wheelhouse/
rm -rf wheelhouse

echo "Built text-classifier-airgap-bundle.tar.gz"
echo "Move it to the air-gapped host, then run:"
echo "  tar xzf text-classifier-airgap-bundle.tar.gz && ./wheelhouse/install.sh"
