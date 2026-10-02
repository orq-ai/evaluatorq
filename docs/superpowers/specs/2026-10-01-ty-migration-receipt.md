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
| Replaced checker defaults | `recommended` | BasedPyright's immutable [`configOptions.ts`](https://raw.githubusercontent.com/DetachHead/basedpyright/ecea0a681818fc8b90a1314b7d2f844fef8f53c0/packages/pyright-internal/src/common/configOptions.ts), verified by the committed source hash |
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

Run the deterministic offline audit to load the committed snapshot, combine ty's pinned defaults with global rules and per-path overrides, evaluate Ruff selection and ignores against the actual `ruff check src` command, and validate all 65 receipt rows:

```bash
uv run python scripts/audit_ty_migration.py
```

The default command makes no network request. It asserts the exact membership of all four classes, their exhaustive partition, the exact ty package pin in `pyproject.toml` and `uv.lock`, the canonical Ruff CI command, and every table classification and effective surface.

Explicitly verify the committed snapshot against all three pinned primary sources when refreshing the receipt or upgrading ty:

```bash
uv run python scripts/audit_ty_migration.py --refresh
```

That command downloads the immutable ty mapping, ty rule reference, and BasedPyright configuration source, checks their SHA-256 hashes, and reconstructs the stored mapping, ty default levels, and active BasedPyright rules. The baseline is committed at `scripts/data/ty_migration_audit.json`, so the normal audit works after a squash merge and in a shallow clone with no historical migration commit.

The named failure mode is an unpinned mapping: running an equivalent comparison against either upstream `main` branch can change the answer while this repository still runs ty `0.0.84`. Changing the exact ty pin requires rerunning the audit and updating its expected sets and this table together.

## Accepted coverage differences

BasedPyright `1.37.4` uses its `recommended` rules when `typeCheckingMode` is absent, as it was in the replaced configuration. The audit applies the preserved global settings and execution-environment overrides, then evaluates each mapping against ty's default level, the current global ty rules, and every matching ty override. “Partial” includes disabled mapped rules, dashboard-only gaps, and semantic qualifiers such as Ruff covering duplicate exception handlers only; it does not depend on the literal phrase “partial coverage.”

Ruff-only is not whole-repository coverage. The audit evaluates Ruff's selected rules, global ignores, configured exclusions, and per-file ignores on the exact CI target, `src`. BasedPyright and ty also cover tests, root Python files, and documentation Python hooks, so every Ruff-only row names those surfaces as uncovered.

<!-- diagnostic-classification:start -->
| Diagnostic | Classification | Effective surface |
|---|---|---|
| `reportAbstractUsage` | partial | all applicable BasedPyright surfaces; partial semantics |
| `reportArgumentType` | ty-backed | all applicable BasedPyright surfaces |
| `reportAssertAlwaysTrue` | ruff-only | covered: src, src/dashboard; uncovered: tests, tests/dashboard, root, docs |
| `reportAssertTypeFailure` | ty-backed | all applicable BasedPyright surfaces |
| `reportAssignmentType` | partial | all applicable BasedPyright surfaces; mapped-rule gaps: src/dashboard, tests/dashboard |
| `reportAttributeAccessIssue` | partial | all applicable BasedPyright surfaces; partial semantics |
| `reportCallInDefaultInitializer` | partial | covered: src, src/dashboard; uncovered: tests, tests/dashboard, root, docs; partial semantics |
| `reportCallIssue` | ty-backed | all applicable BasedPyright surfaces |
| `reportConstantRedefinition` | absent | uncovered: src, src/dashboard, tests, tests/dashboard, root, docs |
| `reportDuplicateImport` | partial | covered: src, src/dashboard; uncovered: tests, tests/dashboard, root, docs; partial semantics |
| `reportFunctionMemberAccess` | partial | covered: src, tests, root, docs; uncovered: src/dashboard, tests/dashboard; mapped-rule gaps: src/dashboard, tests/dashboard |
| `reportGeneralTypeIssues` | ty-backed | all applicable BasedPyright surfaces |
| `reportIgnoreCommentWithoutRule` | ty-backed | all applicable BasedPyright surfaces |
| `reportImplicitAbstractClass` | absent | uncovered: src, src/dashboard, tests, tests/dashboard, root, docs |
| `reportImplicitRelativeImport` | absent | uncovered: src, src/dashboard, tests, tests/dashboard, root, docs |
| `reportIncompatibleMethodOverride` | ty-backed | all applicable BasedPyright surfaces |
| `reportIncompleteStub` | absent | uncovered: src, src/dashboard, tests, tests/dashboard, root, docs |
| `reportInconsistentConstructor` | absent | uncovered: src, src/dashboard, tests, tests/dashboard, root, docs |
| `reportInconsistentOverload` | ty-backed | all applicable BasedPyright surfaces |
| `reportIndexIssue` | ty-backed | all applicable BasedPyright surfaces |
| `reportInvalidAbstractMethod` | absent | uncovered: src, src/dashboard, tests, tests/dashboard, root, docs |
| `reportInvalidCast` | absent | uncovered: src, src/dashboard, tests, tests/dashboard, root, docs |
| `reportInvalidStringEscapeSequence` | ruff-only | covered: src, src/dashboard; uncovered: tests, tests/dashboard, root, docs |
| `reportInvalidStubStatement` | ruff-only | covered: src, src/dashboard; uncovered: tests, tests/dashboard, root, docs |
| `reportInvalidTypeArguments` | ty-backed | all applicable BasedPyright surfaces |
| `reportInvalidTypeForm` | partial | all applicable BasedPyright surfaces; mapped-rule gaps: src/dashboard |
| `reportInvalidTypeVarUse` | absent | uncovered: src, src/dashboard, tests, tests/dashboard, root, docs |
| `reportMatchNotExhaustive` | absent | uncovered: src, src/dashboard, tests, tests/dashboard, root, docs |
| `reportMissingModuleSource` | absent | uncovered: src, src/dashboard, tests, tests/dashboard, root, docs |
| `reportMissingSuperCall` | absent | uncovered: src, src/dashboard, tests, tests/dashboard, root, docs |
| `reportMissingTypeArgument` | ty-backed | all applicable BasedPyright surfaces |
| `reportNoOverloadImplementation` | ty-backed | all applicable BasedPyright surfaces |
| `reportOperatorIssue` | ty-backed | all applicable BasedPyright surfaces |
| `reportOptionalCall` | ty-backed | all applicable BasedPyright surfaces |
| `reportOptionalContextManager` | ty-backed | all applicable BasedPyright surfaces |
| `reportOptionalIterable` | partial | covered: src, src/dashboard, tests, root, docs; uncovered: tests/dashboard; mapped-rule gaps: tests/dashboard |
| `reportOptionalMemberAccess` | partial | covered: src, tests, root, docs; uncovered: src/dashboard, tests/dashboard; mapped-rule gaps: src/dashboard, tests/dashboard |
| `reportOptionalOperand` | ty-backed | all applicable BasedPyright surfaces |
| `reportOptionalSubscript` | ty-backed | all applicable BasedPyright surfaces |
| `reportOverlappingOverload` | absent | uncovered: src, src/dashboard, tests, tests/dashboard, root, docs |
| `reportPossiblyUnboundVariable` | ty-backed | all applicable BasedPyright surfaces |
| `reportPrivateImportUsage` | absent | uncovered: src, src/dashboard, tests, tests/dashboard, root, docs |
| `reportPropertyTypeMismatch` | absent | uncovered: src, src/dashboard, tests, tests/dashboard, root, docs |
| `reportRedeclaration` | ty-backed | all applicable BasedPyright surfaces |
| `reportReturnType` | ty-backed | all applicable BasedPyright surfaces |
| `reportSelfClsDefault` | absent | uncovered: src, src/dashboard, tests, tests/dashboard, root, docs |
| `reportSelfClsParameterName` | ruff-only | covered: src, src/dashboard; uncovered: tests, tests/dashboard, root, docs |
| `reportTypeCommentUsage` | ruff-only | covered: src, src/dashboard; uncovered: tests, tests/dashboard, root, docs |
| `reportTypedDictNotRequiredAccess` | partial | all applicable BasedPyright surfaces; partial semantics |
| `reportUnboundVariable` | ty-backed | all applicable BasedPyright surfaces |
| `reportUndefinedVariable` | ty-backed | all applicable BasedPyright surfaces |
| `reportUnhashable` | absent | uncovered: src, src/dashboard, tests, tests/dashboard, root, docs |
| `reportUninitializedInstanceVariable` | absent | uncovered: src, src/dashboard, tests, tests/dashboard, root, docs |
| `reportUnnecessaryCast` | ty-backed | all applicable BasedPyright surfaces |
| `reportUnnecessaryComparison` | partial | all applicable BasedPyright surfaces; mapped-rule gaps: src, src/dashboard, tests, tests/dashboard, root, docs |
| `reportUnnecessaryContains` | partial | all applicable BasedPyright surfaces; mapped-rule gaps: src, src/dashboard, tests, tests/dashboard, root, docs |
| `reportUnsafeMultipleInheritance` | absent | uncovered: src, src/dashboard, tests, tests/dashboard, root, docs |
| `reportUntypedBaseClass` | absent | uncovered: src, src/dashboard, tests, tests/dashboard, root, docs |
| `reportUntypedClassDecorator` | absent | uncovered: src, src/dashboard, tests, tests/dashboard, root, docs |
| `reportUntypedNamedTuple` | ruff-only | covered: src, src/dashboard; uncovered: tests, tests/dashboard, root, docs |
| `reportUnusedClass` | absent | uncovered: src, src/dashboard, tests, tests/dashboard, root, docs |
| `reportUnusedCoroutine` | partial | all applicable BasedPyright surfaces; partial semantics |
| `reportUnusedExcept` | partial | covered: src, src/dashboard; uncovered: tests, tests/dashboard, root, docs; partial semantics |
| `reportUnusedExpression` | ruff-only | covered: src, src/dashboard; uncovered: tests, tests/dashboard, root, docs |
| `reportWildcardImportFromLibrary` | ruff-only | covered: src, src/dashboard; uncovered: tests, tests/dashboard, root, docs |
<!-- diagnostic-classification:end -->

The 21 absent rows are accepted migration gaps. `reportInvalidCast` is among them because ty's mapped `disjoint-cast` rule is disabled by default; enabling it reports intentional casts that bridge broader library types, so this migration does not enable it. The partial rows name their effective surfaces, including `reportFunctionMemberAccess`, `reportOptionalIterable`, and `reportOptionalMemberAccess` gaps created by dashboard overrides. The eight Ruff-only rows remain uncovered outside `src`. This receipt does not imply that tests or review replace those diagnostics.

## Test-profile benchmark evidence

The checker migration shipped with a separate quick local pytest profile. Its evidence is preserved in [Local pytest profile benchmark](2026-10-01-local-test-profile-benchmark.md), including the identified host, collection output, all six raw `/usr/bin/time -lp` samples, and the failed pre-fix run that was excluded from the measurements.

The exact measured commands were:

```bash
/usr/bin/time -lp uv run pytest
/usr/bin/time -lp uv run pytest -m 'not integration'
```

Under the disclosed fixed-order protocol at `69fae17d4581730cea5125520f51eae45b693185`, the quick-profile median wall time was 88.85 seconds versus 127.37 seconds for the complete non-integration profile. This is faster local feedback from skipping tests that deliberately wait on real time, not a material pytest compute-load reduction: median user CPU time was 55.97 seconds versus 56.14 seconds, and median peak resident memory was only 2.42% lower.

The quick command is for local iteration and excludes integration and deliberately slow tests. Before pushing, run the complete non-integration command; CI runs that complete selection, not the quick default.
