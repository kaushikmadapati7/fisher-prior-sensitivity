"""Every `physharm.*` import in the package must resolve. (Follow-up to #4/#21)

WHY THIS EXISTS. The physics core was ported from the archive module by module, and the port
was checked by grepping the imports of the files being copied -- but `dataset_physics.py` was
ported on a different branch and its OWN imports were never checked. Three modules were
missing (`data.preprocessing`, `manifest.build_manifest`, `manifest.contrast_rules`) and
nothing noticed, because every unit test either avoided the dataset or stopped before the
image path. It surfaced only when the first real cache build reached
`from physharm.data.dataset_physics import ...` and died on the cluster.

Static analysis, not importing: most of these modules need torch / nibabel / SimpleITK, which
a laptop does not have, so an import-based check would be skipped exactly where it is needed.
Parsing the AST works everywhere and would have caught this.
"""
from __future__ import annotations

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
PKG = ROOT / "physharm"


def _modules():
    """Every importable name the package provides, including package dirs."""
    have = {"physharm"}
    for f in PKG.rglob("*.py"):
        rel = f.relative_to(ROOT).with_suffix("")
        have.add(str(rel).replace("/", "."))
        if f.name == "__init__.py":
            have.add(str(rel.parent).replace("/", "."))
    for d in PKG.rglob("*"):
        if d.is_dir():
            have.add(str(d.relative_to(ROOT)).replace("/", "."))
    return have


def _required():
    """Every `physharm.*` module referenced anywhere in the package, with its source file."""
    need = {}
    for f in PKG.rglob("*.py"):
        tree = ast.parse(f.read_text())
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("physharm"):
                need.setdefault(n.module, set()).add(str(f.relative_to(ROOT)))
            elif isinstance(n, ast.Import):
                for a in n.names:
                    if a.name.startswith("physharm"):
                        need.setdefault(a.name, set()).add(str(f.relative_to(ROOT)))
    return need


def test_every_physharm_import_resolves():
    have, need = _modules(), _required()
    missing = {m: sorted(v) for m, v in need.items() if m not in have}
    assert not missing, "unresolvable physharm imports:\n" + "\n".join(
        f"  {m}  <- imported by {', '.join(v)}" for m, v in sorted(missing.items()))


def test_every_package_dir_has_an_init():
    """A missing __init__.py makes a subpackage invisible in some import modes."""
    bad = [str(d.relative_to(ROOT)) for d in PKG.rglob("*")
           if d.is_dir() and d.name != "__pycache__" and not (d / "__init__.py").exists()]
    assert not bad, f"package dirs without __init__.py: {bad}"


def test_the_modules_that_were_actually_missing_are_present():
    """Regression pin for the three that broke the first cache build."""
    for rel in ("data/preprocessing.py", "manifest/build_manifest.py",
                "manifest/contrast_rules.py", "data/dataset_physics.py"):
        assert (PKG / rel).exists(), f"physharm/{rel} is missing"


def test_analysis_path_stays_dependency_light():
    """The CRLB/report path must run on a login node: numpy and stdlib only.

    Enforced because it is easy to add a torch import to a shared helper and silently make
    the manifest-only analysis un-runnable without a GPU environment.
    """
    heavy = {"torch", "nibabel", "SimpleITK", "monai"}
    light = ["physharm/physics/fisher.py", "physharm/analysis/crlb_table.py",
             "physharm/analysis/stats.py", "physharm/data/exclusions.py",
             "physharm/data/splits.py", "physharm/data/units.py"]
    for rel in light:
        tree = ast.parse((ROOT / rel).read_text())
        for n in ast.walk(tree):
            mods = set()
            if isinstance(n, ast.Import):
                mods = {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.module:
                mods = {n.module.split(".")[0]}
            assert not (mods & heavy), f"{rel} imports {mods & heavy}; it must stay light"


def test_limit_zero_means_no_limit_everywhere():
    """`--limit 0` must mean "no limit" in every script, because they are used together.

    audit/physics_sanity.py sliced unconditionally, so `--limit 0` examined ZERO cases and
    still wrote an output file. Downstream that file is the mislabelled-b-band exclusion
    list, so an empty one silently disables the gate while looking like a clean pass -- and
    the command was copied from the other scripts, where 0 does mean no limit.
    """
    import re
    offenders = []
    for f in sorted(PKG.parent.glob("audit/*.py")) + sorted(PKG.rglob("*.py")):
        src = f.read_text()
        if '"--limit"' not in src:
            continue
        # an unguarded slice by args.limit is the bug
        for m in re.finditer(r"\[:\s*args\.limit\s*\]", src):
            line = src[:m.start()].count("\n") + 1
            guarded = re.search(r"if args\.limit:", src)
            if not guarded:
                offenders.append(f"{f.name}:{line}")
    assert not offenders, (
        "unguarded [:args.limit] slice -- `--limit 0` would read nothing: " + ", ".join(offenders))


if __name__ == "__main__":
    import traceback
    npass = nfail = 0
    for n, f in sorted(globals().items()):
        if n.startswith("test_") and callable(f):
            try:
                f(); npass += 1; print(f"  PASS {n}")
            except Exception:
                nfail += 1; print(f"  FAIL {n}"); traceback.print_exc(limit=2)
    print(f"\n{npass} passed, {nfail} failed")
    raise SystemExit(1 if nfail else 0)
