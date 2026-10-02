# ty migration receipt

This receipt records the evidence for replacing BasedPyright with ty as evaluatorq's required type checker.

Use it to audit the checker boundary and the accepted coverage differences at the migration commit. Do not use it as a claim that ty and BasedPyright implement identical diagnostics.

## Result

`uv run ty check` is the single type-check command for local development and continuous integration. The bare command reads `pyproject.toml`, checks `src`, `tests`, root Python files, and `docs/*.py`, excludes `examples`, `scripts`, and `.venv`, and analyzes against Python 3.10. CI runs it once on Ubuntu with Python 3.10; the workflow contract is enforced by `tests/test_developer_workflow.py`.

## Pinned inputs

| Input | Version or revision | Evidence |
|---|---:|---|
| ty | `0.0.84` | Exact development dependency in `pyproject.toml` and package entry in `uv.lock` |
| Replaced BasedPyright | `1.37.4`, commit `ecea0a681818fc8b90a1314b7d2f844fef8f53c0` | Package entry in `uv.lock` at `c12bdc240^` and the version's Git tag |
| Replaced checker defaults | `recommended` | BasedPyright's pinned [`BasedConfigOptions`](https://github.com/DetachHead/basedpyright/blob/ecea0a681818fc8b90a1314b7d2f844fef8f53c0/packages/pyright-internal/src/common/configOptions.ts#L2136-L2145) and [`getRecommendedDiagnosticRuleSet`](https://github.com/DetachHead/basedpyright/blob/ecea0a681818fc8b90a1314b7d2f844fef8f53c0/packages/pyright-internal/src/common/configOptions.ts#L947-L1061) |
| Python analysis target | `3.10` | `[tool.ty.environment]` in `pyproject.toml` |
| Benchmark pytest | `9.0.3` | Toolchain recorded with the raw samples |
| Benchmark uv | `0.11.29` | Host toolchain recorded with the raw samples |
| ty rule mapping | tag `0.0.84`, commit `8dd9a7f7fa35a18275d82117e6593ba45507065f` | Astral's version-matched [migration mapping](https://github.com/astral-sh/ty/blob/8dd9a7f7fa35a18275d82117e6593ba45507065f/docs/coming-from-mypy-or-pyright.md#mapping-pyrightmypy-rules-to-tyruff-rules) |

## Reproduce the checker evidence

Run the required checker and its repository contract from the repository root:

```bash
uv run ty --version
uv run ty check
uv run pytest tests/test_developer_workflow.py tests/test_reuse_guardrails.py -q
```

The contract suite above includes `test_ty_checks_the_supported_project_surface`, which asserts the four include patterns, three excludes, Python 3.10 target, and platform setting, plus `test_ty_checked_files_honor_configured_excludes`, which resolves the configured file set and rejects excluded paths. Run those two boundary checks alone with:

```bash
uv run pytest \
  tests/test_developer_workflow.py::test_ty_checks_the_supported_project_surface \
  tests/test_reuse_guardrails.py::test_ty_checked_files_honor_configured_excludes -q
```

Retrieve the exact official mapping used for the coverage review and print its missing or partial rows:

```bash
set -o pipefail
curl -L --fail --silent https://raw.githubusercontent.com/astral-sh/ty/8dd9a7f7fa35a18275d82117e6593ba45507065f/docs/coming-from-mypy-or-pyright.md \
  | rg '^\| .*?(None yet|partial coverage)'
```

Retrieve the replaced repository configuration and the replaced checker's effective default rule set with:

```bash
set -o pipefail
git show c12bdc240^:pyproject.toml | sed -n '/^\[tool.basedpyright\]/,/^\[tool.ty.src\]/p'
curl -L --fail --silent \
  https://raw.githubusercontent.com/DetachHead/basedpyright/ecea0a681818fc8b90a1314b7d2f844fef8f53c0/packages/pyright-internal/src/common/configOptions.ts \
  | sed -n '947,1061p'
```

Derive the retained diagnostics without a hand-maintained rule list:

```bash
uv run python - <<'PY'
import re
import subprocess
from urllib.request import urlopen

mapping_url = 'https://raw.githubusercontent.com/astral-sh/ty/8dd9a7f7fa35a18275d82117e6593ba45507065f/docs/coming-from-mypy-or-pyright.md'
defaults_url = 'https://raw.githubusercontent.com/DetachHead/basedpyright/ecea0a681818fc8b90a1314b7d2f844fef8f53c0/packages/pyright-internal/src/common/configOptions.ts'
mapping = urlopen(mapping_url).read().decode()
defaults = urlopen(defaults_url).read().decode()
old_config = subprocess.check_output(['git', 'show', 'c12bdc240^:pyproject.toml'], text=True)
recommended = defaults.split('getRecommendedDiagnosticRuleSet =', 1)[1].split('});', 1)[0]
levels = dict(re.findall(r"(report\w+): '(none|hint|warning|error)'", recommended))
global_overrides = old_config.split('[tool.basedpyright]', 1)[1].split('[[tool.basedpyright.executionEnvironments]]', 1)[0]
levels.update(dict(re.findall(r'(report\w+) = "(none|hint|warning|error)"', global_overrides)))
rows = [
    row
    for row in mapping.splitlines()
    if row.startswith('| ') and re.search(r'\[`report\w+`\]', row)
]
active = {diagnostic for diagnostic, level in levels.items() if level != 'none'}
mapped = {
    diagnostic
    for row in rows
    for diagnostic in re.findall(r'\[`(report\w+)`\]', row)
}
missing_or_partial = {
    diagnostic
    for row in rows
    if 'None yet' in row or 'partial coverage' in row
    for diagnostic in re.findall(r'\[`(report\w+)`\]', row)
    if diagnostic in active
}
unmapped = active - mapped
print('[mapped missing or partial]')
print('\n'.join(sorted(missing_or_partial)))
print('[absent from mapping]')
print('\n'.join(sorted(unmapped)))
PY
```

The named failure mode is an unpinned mapping: the same command against `main` can produce a different answer after a ty release, even though this repository still runs `0.0.84`. Changing the exact ty pin requires rerunning this mapping and gap audit and updating this receipt.

## Accepted coverage differences

BasedPyright `1.37.4` uses its `recommended` rules when `typeCheckingMode` is absent, as it was in the replaced configuration. The filtering rule is deterministic: apply the repository's explicit global overrides to the recommended rule set, retain every diagnostic whose resulting severity is not `none`, and compare that active set with every Pyright or BasedPyright diagnostic named in Astral's pinned mapping table. The accepted gaps are the active mapped rows marked `None yet` or `partial coverage` plus the active diagnostics absent from the table. The dashboard execution environments only loosened additional diagnostics, so they add no global gap.

| Replaced diagnostic | ty `0.0.84` mapping | Decision |
|---|---|---|
| `reportAbstractUsage` | No equivalent yet for abstract-class instantiation or abstract `super()` calls | Accepted with no automated replacement |
| `reportCallInDefaultInitializer` | Ruff `B006` and `B008` provide partial coverage; immutable annotations and calls are excluded | Accepted; Ruff covers only the mapped cases under `src`, while tests, root files, and documentation hooks have no automated replacement |
| `reportConstantRedefinition` | No equivalent yet | Accepted with no automated replacement |
| `reportDuplicateImport` | Ruff `F811` and `I001` provide partial coverage; separate import blocks may be missed | Accepted; Ruff covers only `src`, and separate import blocks may still be missed there |
| `reportIncompleteStub` | No equivalent yet | Accepted with no automated replacement |
| `reportImplicitAbstractClass` | Absent from the pinned mapping table | Accepted with no mapping-backed automated replacement |
| `reportImplicitRelativeImport` | Absent from the pinned mapping table | Accepted with no mapping-backed automated replacement |
| `reportInconsistentConstructor` | No equivalent yet | Accepted with no automated replacement |
| `reportInvalidAbstractMethod` | Absent from the pinned mapping table | Accepted with no mapping-backed automated replacement |
| `reportInvalidTypeVarUse` | No equivalent yet for the mapped cases | Accepted with no automated replacement |
| `reportMatchNotExhaustive` | No equivalent yet | Accepted with no automated replacement |
| `reportMissingModuleSource` | No equivalent yet | Accepted with no automated replacement |
| `reportMissingSuperCall` | No equivalent yet | Accepted with no automated replacement |
| `reportOverlappingOverload` | No equivalent yet | Accepted with no automated replacement |
| `reportPrivateImportUsage` | No equivalent yet | Accepted with no automated replacement |
| `reportPropertyTypeMismatch` | No equivalent yet | Accepted with no automated replacement |
| `reportSelfClsDefault` | Absent from the pinned mapping table | Accepted with no mapping-backed automated replacement |
| `reportTypedDictNotRequiredAccess` | No equivalent yet for reading non-required keys | Accepted with no automated replacement for reads; ty maps read-only mutations separately to `invalid-assignment` |
| `reportUnhashable` | No equivalent yet | Accepted with no automated replacement |
| `reportUninitializedInstanceVariable` | No equivalent yet | Accepted with no automated replacement |
| `reportUnsafeMultipleInheritance` | Absent from the pinned mapping table | Accepted with no mapping-backed automated replacement |
| `reportUntypedBaseClass`, `reportUntypedClassDecorator` | No equivalent yet | Accepted with no automated replacement |
| `reportUnusedClass` | No equivalent yet | Accepted with no automated replacement |

The old configuration already disabled its unknown-type, missing-stub, general private-use, most unused-code, import-cycle, and incompatible-variable-override diagnostics. It did not disable `reportPrivateImportUsage` or `reportUnusedClass`, so those remain explicit accepted gaps above. Ruff continues to own the lint families selected in `pyproject.toml`; the migration does not add a second home-grown diagnostic registry.

## Test-profile benchmark evidence

The checker migration shipped with a separate quick local pytest profile. Its evidence is preserved in [Local pytest profile benchmark](2026-10-01-local-test-profile-benchmark.md), including the identified host, collection output, all six raw `/usr/bin/time -lp` samples, and the failed pre-fix run that was excluded from the measurements.

The exact measured commands were:

```bash
/usr/bin/time -lp uv run pytest
/usr/bin/time -lp uv run pytest -m 'not integration'
```

Under the disclosed fixed-order protocol at `69fae17d4581730cea5125520f51eae45b693185`, the quick-profile median wall time was 88.85 seconds versus 127.37 seconds for the complete non-integration profile. This is faster local feedback from skipping tests that deliberately wait on real time, not a material pytest compute-load reduction: median user CPU time was 55.97 seconds versus 56.14 seconds, and median peak resident memory was only 2.42% lower. The broader resource rationale comes from replacing the checker and removing unnecessary global fixture setup; this pytest comparison only establishes iteration latency and the memory non-regression gate.

The quick command is for local iteration and excludes integration and deliberately slow tests. Before pushing, run the complete non-integration command; CI runs that complete selection, not the quick default.
