"""Make `tests` a regular package.

test_prior_shift and test_real_arm both do `from tests.test_traveling_arm import ...`
to share the ACQ fixture and the exact-render helper. Without this file `tests` is only a
namespace package, and any environment carrying an unrelated regular `tests` package in
site-packages shadows it -- regular packages win over namespace packages whatever the
sys.path order -- so collection fails with ModuleNotFoundError on those two modules.
"""
