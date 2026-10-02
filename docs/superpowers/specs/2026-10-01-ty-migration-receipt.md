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

Run the executable audit to fetch both pinned primary sources, reconstruct the active BasedPyright rules from the exact historical repository configuration, classify all 65 diagnostics, intersect Ruff mappings with the repository's selected rules and `src`-only lint surface, and validate the table below:

```bash
uv run python scripts/audit_ty_migration.py
```

The audit asserts the exact membership of all four classes, not only their counts. It also asserts that the classes partition the active diagnostics, every required Ruff rule is selected, Ruff's configured source root is exactly `src`, preview rules are enabled, and every table row has the expected classification and effective surface.

The named failure mode is an unpinned mapping: running an equivalent comparison against either upstream `main` branch can change the answer while this repository still runs ty `0.0.84`. Changing the exact ty pin requires rerunning the audit and updating its expected sets and this table together.

## Accepted coverage differences

BasedPyright `1.37.4` uses its `recommended` rules when `typeCheckingMode` is absent, as it was in the replaced configuration. The audit applies the repository's explicit global overrides, retains every diagnostic whose effective severity is not `none`, and classifies it against every matching row in Astral's pinned table. “Partial” includes semantic qualifiers such as Ruff covering duplicate exception handlers only; it does not depend on the literal phrase “partial coverage.” The dashboard execution environments only loosened diagnostics, so they add no global rule.

Ruff-only is not whole-repository coverage. CI runs Ruff over `src`, while BasedPyright and ty cover `src`, `tests`, root Python files, and documentation Python hooks. Every Ruff-only row therefore records the other three surfaces as uncovered.

<!-- diagnostic-classification:start -->
| Diagnostic | Classification | Effective surface |
|---|---|---|
| `reportArgumentType` | ty-backed | all ty roots |
| `reportAssertTypeFailure` | ty-backed | all ty roots |
| `reportAssignmentType` | ty-backed | all ty roots |
| `reportAttributeAccessIssue` | ty-backed | all ty roots |
| `reportCallIssue` | ty-backed | all ty roots |
| `reportFunctionMemberAccess` | ty-backed | all ty roots |
| `reportGeneralTypeIssues` | ty-backed | all ty roots |
| `reportIgnoreCommentWithoutRule` | ty-backed | all ty roots |
| `reportIncompatibleMethodOverride` | ty-backed | all ty roots |
| `reportInconsistentOverload` | ty-backed | all ty roots |
| `reportIndexIssue` | ty-backed | all ty roots |
| `reportInvalidCast` | ty-backed | all ty roots |
| `reportInvalidTypeArguments` | ty-backed | all ty roots |
| `reportInvalidTypeForm` | ty-backed | all ty roots |
| `reportMissingTypeArgument` | ty-backed | all ty roots |
| `reportNoOverloadImplementation` | ty-backed | all ty roots |
| `reportOperatorIssue` | ty-backed | all ty roots |
| `reportOptionalCall` | ty-backed | all ty roots |
| `reportOptionalContextManager` | ty-backed | all ty roots |
| `reportOptionalIterable` | ty-backed | all ty roots |
| `reportOptionalMemberAccess` | ty-backed | all ty roots |
| `reportOptionalOperand` | ty-backed | all ty roots |
| `reportOptionalSubscript` | ty-backed | all ty roots |
| `reportPossiblyUnboundVariable` | ty-backed | all ty roots |
| `reportRedeclaration` | ty-backed | all ty roots |
| `reportReturnType` | ty-backed | all ty roots |
| `reportUnboundVariable` | ty-backed | all ty roots |
| `reportUndefinedVariable` | ty-backed | all ty roots |
| `reportUnnecessaryCast` | ty-backed | all ty roots |
| `reportUnnecessaryComparison` | ty-backed | all ty roots |
| `reportUnnecessaryContains` | ty-backed | all ty roots |
| `reportUnusedCoroutine` | ty-backed | all ty roots |
| `reportAssertAlwaysTrue` | ruff-only | src only; tests/root/docs uncovered |
| `reportInvalidStringEscapeSequence` | ruff-only | src only; tests/root/docs uncovered |
| `reportInvalidStubStatement` | ruff-only | src only; tests/root/docs uncovered |
| `reportSelfClsParameterName` | ruff-only | src only; tests/root/docs uncovered |
| `reportTypeCommentUsage` | ruff-only | src only; tests/root/docs uncovered |
| `reportUntypedNamedTuple` | ruff-only | src only; tests/root/docs uncovered |
| `reportUnusedExpression` | ruff-only | src only; tests/root/docs uncovered |
| `reportWildcardImportFromLibrary` | ruff-only | src only; tests/root/docs uncovered |
| `reportAbstractUsage` | partial | all ty roots; partial semantics |
| `reportCallInDefaultInitializer` | partial | src partial; tests/root/docs uncovered |
| `reportDuplicateImport` | partial | src partial; tests/root/docs uncovered |
| `reportTypedDictNotRequiredAccess` | partial | all ty roots; partial semantics |
| `reportUnusedExcept` | partial | src partial; tests/root/docs uncovered |
| `reportConstantRedefinition` | absent | all ty roots uncovered |
| `reportImplicitAbstractClass` | absent | all ty roots uncovered |
| `reportImplicitRelativeImport` | absent | all ty roots uncovered |
| `reportIncompleteStub` | absent | all ty roots uncovered |
| `reportInconsistentConstructor` | absent | all ty roots uncovered |
| `reportInvalidAbstractMethod` | absent | all ty roots uncovered |
| `reportInvalidTypeVarUse` | absent | all ty roots uncovered |
| `reportMatchNotExhaustive` | absent | all ty roots uncovered |
| `reportMissingModuleSource` | absent | all ty roots uncovered |
| `reportMissingSuperCall` | absent | all ty roots uncovered |
| `reportOverlappingOverload` | absent | all ty roots uncovered |
| `reportPrivateImportUsage` | absent | all ty roots uncovered |
| `reportPropertyTypeMismatch` | absent | all ty roots uncovered |
| `reportSelfClsDefault` | absent | all ty roots uncovered |
| `reportUnhashable` | absent | all ty roots uncovered |
| `reportUninitializedInstanceVariable` | absent | all ty roots uncovered |
| `reportUnsafeMultipleInheritance` | absent | all ty roots uncovered |
| `reportUntypedBaseClass` | absent | all ty roots uncovered |
| `reportUntypedClassDecorator` | absent | all ty roots uncovered |
| `reportUnusedClass` | absent | all ty roots uncovered |
<!-- diagnostic-classification:end -->

The 20 absent rows are accepted migration gaps. The partial rows retain only the stated subset, and the eight Ruff-only rows remain uncovered outside `src`. This receipt does not imply that tests or review replace those diagnostics.

## Test-profile benchmark evidence

The checker migration shipped with a separate quick local pytest profile. Its evidence is preserved in [Local pytest profile benchmark](2026-10-01-local-test-profile-benchmark.md), including the identified host, collection output, all six raw `/usr/bin/time -lp` samples, and the failed pre-fix run that was excluded from the measurements.

The exact measured commands were:

```bash
/usr/bin/time -lp uv run pytest
/usr/bin/time -lp uv run pytest -m 'not integration'
```

Under the disclosed fixed-order protocol at `69fae17d4581730cea5125520f51eae45b693185`, the quick-profile median wall time was 88.85 seconds versus 127.37 seconds for the complete non-integration profile. This is faster local feedback from skipping tests that deliberately wait on real time, not a material pytest compute-load reduction: median user CPU time was 55.97 seconds versus 56.14 seconds, and median peak resident memory was only 2.42% lower. The broader resource rationale comes from replacing the checker and removing unnecessary global fixture setup; this pytest comparison only establishes iteration latency and the memory non-regression gate.

The quick command is for local iteration and excludes integration and deliberately slow tests. Before pushing, run the complete non-integration command; CI runs that complete selection, not the quick default.
