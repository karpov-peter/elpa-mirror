#!/bin/sh -e

mkdir -p m4/

test -n "$srcdir" || srcdir=`dirname "$0"`
test -n "$srcdir" || srcdir=.

$srcdir/generate_automake_test_programs.py > $srcdir/test_programs.am
autoreconf --force --install --verbose "$srcdir"

# Replace automake's mostlyclean-generic recipe with a find-based approach.
# The generated recipe uses "rm -f $(TEST_LOGS)" which expands to thousands of
# filenames and fails with "Argument list too long" on systems with large
# module environments.  find avoids the shell argument-list limit entirely.
python3 - "$srcdir/Makefile.in" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path) as f:
    data = f.read()
# Older automake uses "test -z ... || rm -f"; newer uses "$(am__rm_f)".
candidates = [
    (
        'mostlyclean-generic:\n'
        '\t-test -z "$(TEST_LOGS)" || rm -f $(TEST_LOGS)\n'
        '\t-test -z "$(TEST_LOGS:.log=.trs)" || rm -f $(TEST_LOGS:.log=.trs)\n'
        '\t-test -z "$(TEST_SUITE_LOG)" || rm -f $(TEST_SUITE_LOG)\n'
    ),
    (
        'mostlyclean-generic:\n'
        '\t-$(am__rm_f) $(TEST_LOGS)\n'
        '\t-$(am__rm_f) $(TEST_LOGS:.log=.trs)\n'
        '\t-$(am__rm_f) $(TEST_SUITE_LOG)\n'
    ),
]
new = (
    'mostlyclean-generic:\n'
    "\t-find . -maxdepth 1 \\( -name '*.log' -o -name '*.trs' \\) -delete\n"
)
for old in candidates:
    if old in data:
        with open(path, 'w') as f:
            f.write(data.replace(old, new))
        break
else:
    print(f'Warning: mostlyclean-generic patch not applied to {path} -- pattern not found', file=sys.stderr)
PYEOF
