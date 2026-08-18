#!/usr/bin/env bash
#
# Tests for scripts/bootstrap.sh — the script published at
# https://api.mazemaker.dev/install.sh
#
# Everything runs against a local fixture repo in a temp dir: no network, no
# writes outside $TMPDIR, and the real install.sh is never executed (a stub
# stands in so we can assert on the arguments it receives).
#
#   bash tests/test_bootstrap.sh
#
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BOOTSTRAP="$REPO_ROOT/scripts/bootstrap.sh"

# The suite must work in root containers and CI images alike; the root guard
# itself is exercised explicitly in test_root_guard.
export MAZEMAKER_ALLOW_ROOT=1
export NO_COLOR=1

WORK="$(mktemp -d "${TMPDIR:-/tmp}/mazemaker-bootstrap-test.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

PASS=0
FAIL=0

pass() { PASS=$((PASS + 1)); echo "  ok   — $1"; }
fail() { FAIL=$((FAIL + 1)); echo "  FAIL — $1"; [ $# -gt 1 ] && echo "         $2"; }

assert_contains() {
    local haystack="$1" needle="$2" label="$3"
    case "$haystack" in
        *"$needle"*) pass "$label" ;;
        *) fail "$label" "expected to find: $needle" ;;
    esac
}

assert_eq() {
    local actual="$1" expected="$2" label="$3"
    if [ "$actual" = "$expected" ]; then
        pass "$label"
    else
        fail "$label" "expected [$expected], got [$actual]"
    fi
}

# -------------------------------------------------------------------
# Fixture: a minimal repo shaped like mazemaker, with a stub install.sh
# that echoes the arguments it was handed.
# -------------------------------------------------------------------
make_fixture() {
    local repo="$WORK/fixture-repo"
    mkdir -p "$repo/python"
    cat > "$repo/install.sh" <<'STUB'
#!/bin/bash
echo "install.sh argc=$#"
for a in "$@"; do echo "install.sh arg=[$a]"; done
STUB
    echo "# fixture" > "$repo/python/memory_client.py"
    echo "PyJWT>=2.8.0" > "$repo/requirements.txt"

    git -C "$repo" init -q -b master
    git -C "$repo" add -A
    git -C "$repo" -c user.email=test@example.com -c user.name=test commit -qm "fixture"
    echo "$repo"
}

FIXTURE="$(make_fixture)"

run_bootstrap() {
    bash "$BOOTSTRAP" --repo "file://$FIXTURE" "$@" 2>&1
}

# -------------------------------------------------------------------
echo "syntax"
if bash -n "$BOOTSTRAP"; then pass "bootstrap.sh parses"; else fail "bootstrap.sh parses"; fi

echo "usage"
out="$(bash "$BOOTSTRAP" --help 2>&1)"
assert_eq "$?" "0" "--help exits 0"
assert_contains "$out" "api.mazemaker.dev/install.sh" "--help documents the published URL"
assert_contains "$out" "--hash-backend" "--help lists passthrough options"

out="$(bash "$BOOTSTRAP" --version 2>&1)"
assert_contains "$out" "mazemaker-bootstrap" "--version prints a version"

out="$(bash "$BOOTSTRAP" --not-a-flag 2>&1)"; rc=$?
assert_eq "$rc" "1" "unknown option exits 1"
assert_contains "$out" "Unknown option" "unknown option explains itself"

out="$(bash "$BOOTSTRAP" --dir 2>&1)"; rc=$?
assert_eq "$rc" "1" "--dir without a value exits 1"

echo "root guard"
out="$(MAZEMAKER_ALLOW_ROOT=0 bash "$BOOTSTRAP" --fetch-only --dir "$WORK/root-guard" 2>&1)"; rc=$?
if [ "$(id -u)" -eq 0 ]; then
    assert_eq "$rc" "1" "refuses to run as root"
    assert_contains "$out" "root" "root refusal names the reason"
else
    pass "root guard skipped (not root)"
    pass "root guard skipped (not root)"
fi

echo "fetch"
dest="$WORK/dest"
out="$(run_bootstrap --ref master --dir "$dest" --fetch-only)"; rc=$?
assert_eq "$rc" "0" "--fetch-only clones the source"
if [ -f "$dest/install.sh" ] && [ -d "$dest/python" ]; then
    pass "checkout has install.sh and python/"
else
    fail "checkout has install.sh and python/"
fi
assert_contains "$out" "source ready" "--fetch-only stops before installing"

out="$(run_bootstrap --ref master --dir "$dest" --fetch-only)"; rc=$?
assert_eq "$rc" "0" "re-running updates the existing checkout"
assert_contains "$out" "Updating existing checkout" "update path is taken on re-run"

echo "guards"
echo "local edit" >> "$dest/install.sh"
out="$(run_bootstrap --ref master --dir "$dest" --fetch-only)"; rc=$?
assert_eq "$rc" "1" "dirty checkout is refused"
assert_contains "$out" "Local changes" "dirty checkout names the reason"

out="$(run_bootstrap --ref master --dir "$dest" --fetch-only --force)"; rc=$?
assert_eq "$rc" "0" "--force overwrites local changes"

foreign="$WORK/foreign"
mkdir -p "$foreign"
echo "someone else's data" > "$foreign/important.txt"
out="$(run_bootstrap --dir "$foreign" --fetch-only)"; rc=$?
assert_eq "$rc" "1" "non-mazemaker directory is refused"
if [ -f "$foreign/important.txt" ]; then
    pass "refused directory is left untouched"
else
    fail "refused directory is left untouched"
fi

echo "handoff"
dest2="$WORK/dest2"
out="$(run_bootstrap --dir "$dest2" --skip-deps --hash-backend --with-mssql --hermes-agent "/opt/my hermes")"; rc=$?
assert_eq "$rc" "0" "full run completes"
assert_contains "$out" "install.sh argc=4" "install.sh receives exactly four arguments"
assert_contains "$out" "install.sh arg=[install]" "install.sh gets the install command"
assert_contains "$out" "install.sh arg=[--hash-backend]" "--hash-backend is forwarded"
assert_contains "$out" "install.sh arg=[--with-mssql]" "--with-mssql is forwarded"
assert_contains "$out" "install.sh arg=[/opt/my hermes]" "a path with a space survives as one argument"
assert_contains "$out" "not touching requirements.txt" "--skip-deps skips pip"
assert_contains "$out" "Mazemaker is installed" "success summary is printed"

dest3="$WORK/dest3"
out="$(run_bootstrap --dir "$dest3" --skip-deps)"
assert_contains "$out" "install.sh argc=1" "a bare run forwards only 'install'"

echo "failure propagation"
cat > "$FIXTURE/install.sh" <<'STUB'
#!/bin/bash
echo "install.sh: exploding on purpose" >&2
exit 3
STUB
git -C "$FIXTURE" -c user.email=test@example.com -c user.name=test commit -qam "failing stub"
out="$(run_bootstrap --dir "$WORK/dest4" --skip-deps)"; rc=$?
assert_eq "$rc" "3" "install.sh exit status propagates"
case "$out" in
    *"Mazemaker is installed"*) fail "no success summary after a failed install" ;;
    *) pass "no success summary after a failed install" ;;
esac

echo ""
echo "passed: $PASS   failed: $FAIL"
[ "$FAIL" -eq 0 ]
