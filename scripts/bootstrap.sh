#!/usr/bin/env bash
#
# Mazemaker — bootstrap installer
#
# This file is the source of truth for what https://api.mazemaker.dev/install.sh
# serves. Publish this exact file at that path; do not hand-edit the hosted copy.
#
#   curl -fsSL https://api.mazemaker.dev/install.sh | bash
#   curl -fsSL https://api.mazemaker.dev/install.sh | bash -s -- --hash-backend
#
# What it does:
#   1. Preflight  — OS, shell, python3, git/curl, disk, root check
#   2. Fetch      — clone (or update) the repo into ~/.mazemaker/src
#   3. Deps       — pip install -r requirements.txt
#   4. Hand off   — bash install.sh install [passthrough options]
#
# The whole body lives in functions and main() is only called on the very last
# line, so a truncated download can never execute a half-read script.
#
set -euo pipefail

BOOTSTRAP_VERSION="1.0.0"

REPO_DEFAULT="https://github.com/itsXactlY/mazemaker"
REF_DEFAULT="master"
DIR_DEFAULT="$HOME/.mazemaker/src"

# Free space we want before pulling FastEmbed's ONNX model (~500MB) plus wheels.
MIN_FREE_MB=2048
# Below this, install.sh auto-selects the hash backend anyway — say so up front.
LOW_RAM_MB=3072

# -------------------------------------------------------------------
# Output
# -------------------------------------------------------------------
setup_colors() {
    if [ -n "${NO_COLOR:-}" ] || [ "${MAZEMAKER_NO_COLOR:-0}" = "1" ] || [ ! -t 1 ]; then
        GREEN=''; YELLOW=''; RED=''; CYAN=''; BOLD=''; NC=''
    else
        GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'
        CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
    fi
}

log()  { printf '%b→%b %s\n' "$CYAN" "$NC" "$*"; }
ok()   { printf '%b✓%b %s\n' "$GREEN" "$NC" "$*"; }
warn() { printf '%b⚠%b %s\n' "$YELLOW" "$NC" "$*" >&2; }
die()  { printf '%b✗%b %s\n' "$RED" "$NC" "$*" >&2; exit 1; }

banner() {
    printf '%b' "$BOLD"
    echo "╔══════════════════════════════════════════════════╗"
    echo "║   Mazemaker — bootstrap installer                ║"
    echo "║   Community Edition · free forever               ║"
    echo "╚══════════════════════════════════════════════════╝"
    printf '%b\n' "$NC"
}

usage() {
    cat <<'EOF'
Mazemaker bootstrap installer

Usage:
  curl -fsSL https://api.mazemaker.dev/install.sh | bash
  curl -fsSL https://api.mazemaker.dev/install.sh | bash -s -- [OPTIONS]
  bash scripts/bootstrap.sh [OPTIONS]

Options:
  --dir PATH            Checkout location            (default: ~/.mazemaker/src)
  --ref REF             Branch, tag or commit        (default: master)
  --repo URL            Source repository            (default: github.com/itsXactlY/mazemaker)
  --hermes-agent PATH   Explicit hermes-agent path (else auto-detected)
  --hash-backend        Hash embeddings — instant, no model download, low RAM
  --with-mssql          Also set up the MSSQL cold store
  --skip-deps           Do not pip install requirements.txt
  --fetch-only          Fetch the source, skip install.sh
  --force               Overwrite a dirty/foreign checkout at --dir
  --no-color            Plain output
  --help                This message
  --version             Print bootstrap version

Environment (options win over env):
  MAZEMAKER_DIR, MAZEMAKER_REF, MAZEMAKER_REPO, MAZEMAKER_SKIP_DEPS=1,
  MAZEMAKER_ALLOW_ROOT=1, PYTHON, NO_COLOR

Uninstall:
  bash ~/.mazemaker/src/install.sh uninstall
EOF
}

# -------------------------------------------------------------------
# Preflight
# -------------------------------------------------------------------
have() { command -v "$1" >/dev/null 2>&1; }

check_platform() {
    local os
    os="$(uname -s 2>/dev/null || echo unknown)"
    case "$os" in
        Linux)  ok "Platform: Linux ($(uname -m))" ;;
        Darwin) ok "Platform: macOS ($(uname -m))" ;;
        *)      die "Unsupported platform: $os (Linux and macOS only — on Windows use WSL2)" ;;
    esac
}

check_not_root() {
    if [ "$(id -u)" -eq 0 ] && [ "${MAZEMAKER_ALLOW_ROOT:-0}" != "1" ]; then
        die "Do not run this as root. It writes to \$HOME and uses your venv.
  Re-run as your normal user, or set MAZEMAKER_ALLOW_ROOT=1 if you really mean it."
    fi
}

# Sets PYTHON_BIN. Requires 3.9+; 3.10+ is what CI covers.
check_python() {
    local candidate
    for candidate in "${PYTHON:-}" python3 python; do
        [ -n "$candidate" ] || continue
        have "$candidate" || continue
        if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
            PYTHON_BIN="$candidate"
            break
        fi
    done

    [ -n "${PYTHON_BIN:-}" ] || die "Python 3.9+ not found. Install python3 (and python3-venv on Debian/Ubuntu), then re-run."

    local ver
    ver="$("$PYTHON_BIN" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')"
    ok "Python: $ver ($PYTHON_BIN)"

    if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
        warn "Python $ver is below the 3.10 baseline CI tests against — expect rough edges."
    fi
}

check_fetcher() {
    if have git; then
        ok "Fetcher: git"
    elif have curl && have tar; then
        warn "git not found — falling back to a tarball download (no in-place updates later)."
    else
        die "Need either git, or curl + tar, to fetch the source."
    fi
}

check_disk() {
    local target="$1" probe free_mb
    probe="$target"
    while [ ! -d "$probe" ] && [ "$probe" != "/" ] && [ -n "$probe" ]; do
        probe="$(dirname "$probe")"
    done

    free_mb="$(df -Pk "$probe" 2>/dev/null | awk 'NR==2 {printf "%d", $4/1024}')"
    [ -n "$free_mb" ] || return 0

    if [ "$free_mb" -lt "$MIN_FREE_MB" ]; then
        warn "Only ${free_mb}MB free at $probe — the embedding model needs ~${MIN_FREE_MB}MB. Use --hash-backend to skip the download."
    else
        ok "Disk: ${free_mb}MB free at $probe"
    fi
}

# Purely informational: install.sh makes the final backend call itself.
check_ram() {
    local ram_mb=""
    if [ -r /proc/meminfo ]; then
        ram_mb="$(awk '/MemTotal/ {printf "%d", $2/1024}' /proc/meminfo)"
    elif have sysctl; then
        ram_mb="$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%d", $1/1048576}')"
    fi

    [ -n "$ram_mb" ] && [ "$ram_mb" -gt 0 ] || return 0
    ok "RAM: $((ram_mb / 1024))GB"

    if [ "$ram_mb" -lt "$LOW_RAM_MB" ] && [ "$HASH_BACKEND" != "true" ]; then
        warn "Under $((LOW_RAM_MB / 1024))GB RAM — the installer will select the hash embedding backend."
    fi
}

# -------------------------------------------------------------------
# Fetch
# -------------------------------------------------------------------
# A directory is ours if it is a checkout of $REPO, or an unpacked tarball
# carrying the files install.sh needs.
looks_like_mazemaker() {
    local dir="$1"
    [ -f "$dir/install.sh" ] && [ -d "$dir/python" ]
}

# Move an existing checkout onto $REF — shallow when the host allows it.
sync_to_ref() {
    local dir="$1"

    if git -C "$dir" fetch --quiet --depth 1 origin "$REF" 2>/dev/null; then
        # Shallow histories cannot always fast-forward, and the tree is
        # verified clean (or --force), so reset is the honest equivalent.
        # Untracked files are left alone.
        git -C "$dir" reset --quiet --hard FETCH_HEAD
        return 0
    fi

    # Hosts that refuse "SHA in want" need the fuller history.
    git -C "$dir" fetch --quiet --unshallow origin 2>/dev/null \
        || git -C "$dir" fetch --quiet origin 2>/dev/null \
        || die "git fetch failed from $REPO — check network access."

    git -C "$dir" reset --quiet --hard "$REF" 2>/dev/null \
        || git -C "$dir" reset --quiet --hard "origin/$REF" 2>/dev/null \
        || die "Ref '$REF' not found in $REPO."
}

fetch_git() {
    local dir="$1"

    if [ -d "$dir/.git" ] && git -C "$dir" rev-parse --verify --quiet HEAD >/dev/null 2>&1; then
        log "Updating existing checkout: $dir"

        if [ "$FORCE" != "true" ] && ! git -C "$dir" diff --quiet HEAD; then
            die "Local changes in $dir. Commit/stash them, pass --force to discard, or pick another --dir."
        fi
    else
        log "Cloning $REPO ($REF) → $dir"
        if [ -e "$dir" ]; then rm -rf "${dir:?}"; fi
        mkdir -p "$(dirname "$dir")"

        if git clone --quiet --depth 1 --branch "$REF" "$REPO" "$dir" 2>/dev/null; then
            ok "Source at $dir ($(git -C "$dir" rev-parse --short HEAD))"
            return 0
        fi

        # --branch only accepts branches and tags — clone, then land on the ref.
        rm -rf "${dir:?}"
        git clone --quiet --depth 1 "$REPO" "$dir" \
            || die "git clone failed. Check network access to $REPO."
    fi

    sync_to_ref "$dir"
    ok "Source at $dir ($(git -C "$dir" rev-parse --short HEAD))"
}

fetch_tarball() {
    local dir="$1"
    local url="${REPO%.git}/archive/${REF}.tar.gz"
    local tmp

    tmp="$(mktemp -d "${TMPDIR:-/tmp}/mazemaker.XXXXXX")"
    # shellcheck disable=SC2064  # expand $tmp now, not at trap time
    trap "rm -rf '$tmp'" EXIT

    log "Downloading $url"
    curl -fsSL --retry 3 --retry-delay 2 "$url" -o "$tmp/src.tar.gz" \
        || die "Download failed: $url"

    mkdir -p "$dir"
    tar -xzf "$tmp/src.tar.gz" --strip-components=1 -C "$dir" \
        || die "Extraction failed — the archive may be truncated."

    rm -rf "$tmp"
    trap - EXIT
    ok "Source at $dir (tarball $REF)"
}

fetch_source() {
    local dir="$1"

    if [ -e "$dir" ] && [ ! -d "$dir" ]; then
        die "$dir exists and is not a directory."
    fi

    if [ -d "$dir" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ] \
       && [ ! -d "$dir/.git" ] && ! looks_like_mazemaker "$dir"; then
        if [ "$FORCE" != "true" ]; then
            die "$dir is not empty and does not look like a Mazemaker checkout.
  Pass --dir PATH to install elsewhere, or --force to overwrite it."
        fi
        warn "--force: clearing $dir"
        rm -rf "${dir:?}"
    fi

    if have git; then
        fetch_git "$dir"
    else
        fetch_tarball "$dir"
    fi

    looks_like_mazemaker "$dir" || die "Fetched tree at $dir is missing install.sh or python/ — aborting."
}

# -------------------------------------------------------------------
# Dependencies
# -------------------------------------------------------------------
# install.sh installs the heavy optional stack (numpy, fastembed, hnswlib …)
# into whichever venv it finds. Here we only cover requirements.txt: the two
# hard runtime deps the engine will not start without.
install_requirements() {
    local dir="$1"
    local req="$dir/requirements.txt"

    [ -f "$req" ] || { warn "No requirements.txt in $dir — skipping."; return 0; }

    local py="$PYTHON_BIN"
    local pip_args=""

    if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python3" ]; then
        py="$VIRTUAL_ENV/bin/python3"
        log "Installing runtime deps into the active venv: $VIRTUAL_ENV"
    else
        pip_args="--user"
        log "Installing runtime deps (--user; no venv active)"
    fi

    if ! "$py" -m pip --version >/dev/null 2>&1; then
        warn "pip is unavailable for $py — skipping requirements.txt.
  Install them yourself: $py -m pip install -r $req"
        return 0
    fi

    # shellcheck disable=SC2086  # pip_args is a single optional flag
    if "$py" -m pip install --quiet $pip_args -r "$req"; then
        ok "Runtime dependencies installed"
    else
        warn "pip install failed (PEP 668 / externally-managed environment?).
  Create a venv and re-run, or install manually:
      python3 -m venv ~/.mazemaker/venv
      ~/.mazemaker/venv/bin/pip install -r $req"
    fi
}

# -------------------------------------------------------------------
# Hand off to the repo installer
# -------------------------------------------------------------------
run_installer() {
    local dir="$1"
    shift

    log "Running install.sh from $dir"
    echo ""
    # install.sh resolves its own paths from its location; run it in place.
    bash "$dir/install.sh" install ${1+"$@"}
}

next_steps() {
    local dir="$1"
    echo ""
    printf '%b═══════════════════════════════════════════════%b\n' "$BOLD" "$NC"
    printf '%b  Mazemaker is installed.%b\n' "$GREEN" "$NC"
    printf '%b═══════════════════════════════════════════════%b\n' "$BOLD" "$NC"
    echo ""
    echo "  Source:    $dir"
    echo "  Update:    bash $dir/install.sh update"
    echo "  Verify:    bash $dir/install.sh verify"
    echo "  Uninstall: bash $dir/install.sh uninstall"
    echo ""
    echo "  Docs:      https://github.com/itsXactlY/mazemaker#documentation"
    echo "  Console:   https://mazemaker.dev"
    echo ""
}

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
main() {
    setup_colors

    REPO="${MAZEMAKER_REPO:-$REPO_DEFAULT}"
    REF="${MAZEMAKER_REF:-$REF_DEFAULT}"
    DIR="${MAZEMAKER_DIR:-$DIR_DEFAULT}"
    SKIP_DEPS="${MAZEMAKER_SKIP_DEPS:-0}"
    FETCH_ONLY="false"
    FORCE="false"
    HASH_BACKEND="false"

    # Options forwarded verbatim to install.sh.
    INSTALL_ARGS=()

    while [ $# -gt 0 ]; do
        case "$1" in
            --dir)           [ $# -ge 2 ] || die "--dir needs a path"; DIR="$2"; shift 2 ;;
            --dir=*)         DIR="${1#*=}"; shift ;;
            --ref)           [ $# -ge 2 ] || die "--ref needs a value"; REF="$2"; shift 2 ;;
            --ref=*)         REF="${1#*=}"; shift ;;
            --repo)          [ $# -ge 2 ] || die "--repo needs a URL"; REPO="$2"; shift 2 ;;
            --repo=*)        REPO="${1#*=}"; shift ;;
            --hermes-agent)  [ $# -ge 2 ] || die "--hermes-agent needs a path"; INSTALL_ARGS[${#INSTALL_ARGS[@]}]="$2"; shift 2 ;;
            --hermes-agent=*) INSTALL_ARGS[${#INSTALL_ARGS[@]}]="${1#*=}"; shift ;;
            --hash-backend)  HASH_BACKEND="true"; INSTALL_ARGS[${#INSTALL_ARGS[@]}]="--hash-backend"; shift ;;
            --with-mssql)    INSTALL_ARGS[${#INSTALL_ARGS[@]}]="--with-mssql"; shift ;;
            --skip-deps)     SKIP_DEPS=1; shift ;;
            --fetch-only)    FETCH_ONLY="true"; shift ;;
            --force)         FORCE="true"; shift ;;
            --no-color)      MAZEMAKER_NO_COLOR=1; setup_colors; shift ;;
            --help|-h)       usage; return 0 ;;
            --version)       echo "mazemaker-bootstrap $BOOTSTRAP_VERSION"; return 0 ;;
            *)               die "Unknown option: $1 (try --help)" ;;
        esac
    done

    # ~ in an env var or --dir value is literal until we expand it.
    case "$DIR" in
        "~") DIR="$HOME" ;;
        "~/"*) DIR="$HOME/${DIR#\~/}" ;;
    esac
    DIR="${DIR%/}"

    # --force deletes whatever sits at $DIR; never let that be a home or root.
    case "$DIR" in
        ""|"/"|"$HOME"|"$HOME/.mazemaker")
            die "--dir must be a dedicated directory (got '$DIR'), not a home or system root." ;;
    esac

    banner

    check_platform
    check_not_root
    check_python
    check_fetcher
    check_disk "$DIR"
    check_ram

    echo ""
    fetch_source "$DIR"

    if [ "$FETCH_ONLY" = "true" ]; then
        echo ""
        ok "--fetch-only: source ready at $DIR"
        echo "  Next: bash $DIR/install.sh install"
        return 0
    fi

    echo ""
    if [ "$SKIP_DEPS" = "1" ]; then
        log "--skip-deps: not touching requirements.txt"
    else
        install_requirements "$DIR"
    fi

    run_installer "$DIR" ${INSTALL_ARGS[@]+"${INSTALL_ARGS[@]}"}

    next_steps "$DIR"
}

main "$@"
