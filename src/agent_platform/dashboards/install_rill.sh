#!/bin/sh
# Install the Rill binary for `make dashboard`, into this repo (never system-wide).
#
#   src/agent_platform/dashboards/install_rill.sh <install dir> [version]      FORCE=1 reinstalls
#
# Downloads the release zip for this platform from Rill's CDN (~120 MB, with a
# progress bar), verifies its SHA-256 against the release checksums, and puts
# the binary at <install dir>/rill by rename: a console or `make dashboard`
# running meanwhile sees the old complete binary or the new one, never half of
# one. Skips everything when that version is already installed. No shell
# profile is edited and nothing is installed outside <install dir>.
set -eu

DEST="${1:?usage: install_rill.sh <install dir> [version]}"
VERSION="${2:-latest}"
FORCE="${FORCE:-}"
CDN="https://cdn.rilldata.com/rill"

case "$(uname -s)/$(uname -m)" in
  Darwin/arm64) PLATFORM=darwin_arm64 ;;
  Darwin/x86_64) PLATFORM=darwin_amd64 ;;
  Linux/x86_64) PLATFORM=linux_amd64 ;;
  Linux/arm64|Linux/aarch64) PLATFORM=linux_arm64 ;;
  *) echo "Rill has no release for $(uname -s)/$(uname -m)" >&2; exit 1 ;;
esac

if [ "$VERSION" = "latest" ]; then
  echo "Resolving the latest Rill version…"
  VERSION="$(curl -fsS "$CDN/latest.txt")"
fi

if [ -z "$FORCE" ] && [ -x "$DEST/rill" ] && "$DEST/rill" version 2>/dev/null | grep -q " $VERSION "; then
  echo "Rill $VERSION is already installed at $DEST/rill (FORCE=1 to reinstall)."
  exit 0
fi

mkdir -p "$DEST"
WORK="$(mktemp -d "$DEST/.install.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

echo "Downloading Rill $VERSION for $PLATFORM: $CDN/$VERSION/rill_$PLATFORM.zip"
curl -fL --progress-bar "$CDN/$VERSION/rill_$PLATFORM.zip" -o "$WORK/rill_$PLATFORM.zip"
curl -fsS "$CDN/$VERSION/checksums.txt" -o "$WORK/checksums.txt"
echo "Verifying the SHA-256 checksum…"
(cd "$WORK" && shasum -a 256 --ignore-missing --check checksums.txt)
echo "Unpacking…"
unzip -oq "$WORK/rill_$PLATFORM.zip" -d "$WORK/unpacked"
chmod +x "$WORK/unpacked/rill"
mv -f "$WORK/unpacked/rill" "$DEST/rill"
echo "Rill $VERSION installed at $DEST/rill"
