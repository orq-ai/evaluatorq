"""Audit the pinned BasedPyright-to-ty diagnostic coverage classification."""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
MAPPING_URL = 'https://raw.githubusercontent.com/astral-sh/ty/8dd9a7f7fa35a18275d82117e6593ba45507065f/docs/coming-from-mypy-or-pyright.md'
DEFAULTS_URL = 'https://raw.githubusercontent.com/DetachHead/basedpyright/ecea0a681818fc8b90a1314b7d2f844fef8f53c0/packages/pyright-internal/src/common/configOptions.ts'
OLD_CONFIG_COMMIT = 'c830b931c574fdab9483049e0c57e36b903ee802'

EXPECTED = {
    'ty-backed': {
        'reportArgumentType',
        'reportAssertTypeFailure',
        'reportAssignmentType',
        'reportAttributeAccessIssue',
        'reportCallIssue',
        'reportFunctionMemberAccess',
        'reportGeneralTypeIssues',
        'reportIgnoreCommentWithoutRule',
        'reportIncompatibleMethodOverride',
        'reportInconsistentOverload',
        'reportIndexIssue',
        'reportInvalidCast',
        'reportInvalidTypeArguments',
        'reportInvalidTypeForm',
        'reportMissingTypeArgument',
        'reportNoOverloadImplementation',
        'reportOperatorIssue',
        'reportOptionalCall',
        'reportOptionalContextManager',
        'reportOptionalIterable',
        'reportOptionalMemberAccess',
        'reportOptionalOperand',
        'reportOptionalSubscript',
        'reportPossiblyUnboundVariable',
        'reportRedeclaration',
        'reportReturnType',
        'reportUnboundVariable',
        'reportUndefinedVariable',
        'reportUnnecessaryCast',
        'reportUnnecessaryComparison',
        'reportUnnecessaryContains',
        'reportUnusedCoroutine',
    },
    'ruff-only': {
        'reportAssertAlwaysTrue',
        'reportInvalidStringEscapeSequence',
        'reportInvalidStubStatement',
        'reportSelfClsParameterName',
        'reportTypeCommentUsage',
        'reportUntypedNamedTuple',
        'reportUnusedExpression',
        'reportWildcardImportFromLibrary',
    },
    'partial': {
        'reportAbstractUsage',
        'reportCallInDefaultInitializer',
        'reportDuplicateImport',
        'reportTypedDictNotRequiredAccess',
        'reportUnusedExcept',
    },
    'absent': {
        'reportConstantRedefinition',
        'reportImplicitAbstractClass',
        'reportImplicitRelativeImport',
        'reportIncompleteStub',
        'reportInconsistentConstructor',
        'reportInvalidAbstractMethod',
        'reportInvalidTypeVarUse',
        'reportMatchNotExhaustive',
        'reportMissingModuleSource',
        'reportMissingSuperCall',
        'reportOverlappingOverload',
        'reportPrivateImportUsage',
        'reportPropertyTypeMismatch',
        'reportSelfClsDefault',
        'reportUnhashable',
        'reportUninitializedInstanceVariable',
        'reportUnsafeMultipleInheritance',
        'reportUntypedBaseClass',
        'reportUntypedClassDecorator',
        'reportUnusedClass',
    },
}


def _download(url: str) -> str:
    with urlopen(url, timeout=30) as response:
        return response.read().decode()


def _old_config() -> str:
    return subprocess.check_output(['git', 'show', f'{OLD_CONFIG_COMMIT}:pyproject.toml'], cwd=ROOT, text=True)


def _active_diagnostics(defaults: str, old_config: str) -> set[str]:
    recommended = defaults.split('getRecommendedDiagnosticRuleSet =', 1)[1].split('});', 1)[0]
    levels = dict(re.findall(r"(report\w+): '(none|hint|warning|error)'", recommended))
    global_overrides = old_config.split('[tool.basedpyright]', 1)[1].split(
        '[[tool.basedpyright.executionEnvironments]]', 1
    )[0]
    levels.update(re.findall(r'(report\w+) = "(none|hint|warning|error)"', global_overrides))
    return {diagnostic for diagnostic, level in levels.items() if level != 'none'}


def _mapping_cells(mapping: str, active: set[str]) -> dict[str, list[str]]:
    cells = {diagnostic: [] for diagnostic in active}
    for row in mapping.splitlines():
        if not row.startswith('| ') or not re.search(r'\[`report\w+`\]', row):
            continue
        replacement = row.split('|')[1]
        for diagnostic in re.findall(r'\[`(report\w+)`\]', row):
            if diagnostic in cells:
                cells[diagnostic].append(replacement)
    return cells


def _classify(cells: dict[str, list[str]]) -> dict[str, set[str]]:
    classified = {classification: set() for classification in EXPECTED}
    for diagnostic, replacements in cells.items():
        has_ty = any(re.match(r'\s*\[`[^`]+`\]\[ty-', replacement) is not None for replacement in replacements)
        has_ruff = any('Ruff `' in replacement for replacement in replacements)
        has_absent_case = not replacements or any('None yet' in replacement for replacement in replacements)
        has_limited_case = any(
            'partial coverage' in replacement or 'duplicate exception handlers only' in replacement
            for replacement in replacements
        )
        if (has_absent_case and (has_ty or has_ruff)) or has_limited_case:
            classification = 'partial'
        elif has_absent_case:
            classification = 'absent'
        elif has_ty:
            classification = 'ty-backed'
        elif has_ruff:
            classification = 'ruff-only'
        else:
            classification = 'absent'
        classified[classification].add(diagnostic)
    return classified


def _array(table: str, key: str) -> list[str]:
    match = re.search(rf'(?ms)^\s*{re.escape(key)}\s*=\s*(\[.*?\])', table)
    assert match is not None, f'missing {key}'
    value = ast.literal_eval(match.group(1))
    assert isinstance(value, list) and all(isinstance(item, str) for item in value)
    return value


def _assert_ruff_surface(mapping_cells: dict[str, list[str]]) -> None:
    pyproject = (ROOT / 'pyproject.toml').read_text()
    ruff = pyproject.split('[tool.ruff]', 1)[1].split('[tool.ty.', 1)[0]
    assert _array(ruff, 'src') == ['src']
    assert re.search(r'(?m)^preview\s*=\s*true$', ruff)
    selected = _array(ruff, 'select')
    ruff_backed = EXPECTED['ruff-only'] | {
        diagnostic for diagnostic in EXPECTED['partial'] if any('Ruff `' in cell for cell in mapping_cells[diagnostic])
    }
    codes = {
        code
        for diagnostic in ruff_backed
        for cell in mapping_cells[diagnostic]
        for code in re.findall(r'Ruff `([A-Z]+\d+)`', cell)
    }
    unselected = sorted(
        code for code in codes if not any(code == selector or code.startswith(selector) for selector in selected)
    )
    assert not unselected, f'pinned Ruff mappings are not selected: {unselected}'


def _surface(classification: str, replacements: list[str]) -> str:
    if classification == 'ty-backed':
        return 'all ty roots'
    if classification == 'ruff-only':
        return 'src only; tests/root/docs uncovered'
    if any('Ruff `' in replacement for replacement in replacements):
        return 'src partial; tests/root/docs uncovered'
    if classification == 'partial':
        return 'all ty roots; partial semantics'
    return 'all ty roots uncovered'


def _assert_receipt(receipt: Path, classified: dict[str, set[str]], cells: dict[str, list[str]]) -> None:
    text = receipt.read_text()
    table = text.split('<!-- diagnostic-classification:start -->', 1)[1].split(
        '<!-- diagnostic-classification:end -->', 1
    )[0]
    rows = re.findall(
        r'^\| `(?P<diagnostic>report\w+)` \| (?P<classification>[a-z-]+) \| (?P<surface>[^|]+?) \|', table, re.M
    )
    actual = {diagnostic: (classification, surface) for diagnostic, classification, surface in rows}
    expected = {
        diagnostic: (classification, _surface(classification, cells[diagnostic]))
        for classification, diagnostics in classified.items()
        for diagnostic in diagnostics
    }
    assert actual == expected, 'receipt decision table does not match the pinned classification and surfaces'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--receipt',
        type=Path,
        default=ROOT / 'docs/superpowers/specs/2026-10-01-ty-migration-receipt.md',
    )
    args = parser.parse_args()

    mapping = _download(MAPPING_URL)
    defaults = _download(DEFAULTS_URL)
    active = _active_diagnostics(defaults, _old_config())
    cells = _mapping_cells(mapping, active)
    classified = _classify(cells)
    assert classified == EXPECTED, 'pinned diagnostic classification changed'
    assert set().union(*classified.values()) == active
    assert sum(len(diagnostics) for diagnostics in classified.values()) == len(active)
    _assert_ruff_surface(cells)
    _assert_receipt(args.receipt, classified, cells)
    for classification in ('ty-backed', 'ruff-only', 'partial', 'absent'):
        print(f'{classification}: {len(classified[classification])}')


if __name__ == '__main__':
    main()
