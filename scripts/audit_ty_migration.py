"""Audit effective BasedPyright-to-ty coverage from a committed offline snapshot."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import tomllib
from pathlib import Path
from typing import Any
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / 'scripts/data/ty_migration_audit.json'
RECEIPT = ROOT / 'docs/superpowers/specs/2026-10-01-ty-migration-receipt.md'

SURFACES = {
    'src': 'src/evaluatorq/contracts.py',
    'src/dashboard': 'src/evaluatorq/dashboard/app.py',
    'tests': 'tests/unit/test_contracts.py',
    'tests/dashboard': 'tests/dashboard/test_app.py',
    'root': 'conftest.py',
    'docs': 'docs/hooks.py',
}

EXPECTED = {
    'ty-backed': {
        'reportArgumentType',
        'reportAssertTypeFailure',
        'reportCallIssue',
        'reportGeneralTypeIssues',
        'reportIgnoreCommentWithoutRule',
        'reportIncompatibleMethodOverride',
        'reportInconsistentOverload',
        'reportIndexIssue',
        'reportInvalidTypeArguments',
        'reportMissingTypeArgument',
        'reportNoOverloadImplementation',
        'reportOperatorIssue',
        'reportOptionalCall',
        'reportOptionalContextManager',
        'reportOptionalOperand',
        'reportOptionalSubscript',
        'reportPossiblyUnboundVariable',
        'reportRedeclaration',
        'reportReturnType',
        'reportUnboundVariable',
        'reportUndefinedVariable',
        'reportUnnecessaryCast',
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
        'reportAssignmentType',
        'reportAttributeAccessIssue',
        'reportCallInDefaultInitializer',
        'reportDuplicateImport',
        'reportFunctionMemberAccess',
        'reportInvalidTypeForm',
        'reportOptionalIterable',
        'reportOptionalMemberAccess',
        'reportTypedDictNotRequiredAccess',
        'reportUnnecessaryComparison',
        'reportUnnecessaryContains',
        'reportUnusedCoroutine',
        'reportUnusedExcept',
    },
    'absent': {
        'reportConstantRedefinition',
        'reportImplicitAbstractClass',
        'reportImplicitRelativeImport',
        'reportIncompleteStub',
        'reportInconsistentConstructor',
        'reportInvalidAbstractMethod',
        'reportInvalidCast',
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

LIMITED_MARKERS = (
    'None yet',
    'partial coverage',
    'duplicate exception handlers only',
    'function objects only',
    'incompatible method replacements only',
    'read-only TypedDict keys',
    'native coroutines only',
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    return value


def _project_config() -> dict[str, Any]:
    return tomllib.loads((ROOT / 'pyproject.toml').read_text())


def _download(url: str) -> str:
    with urlopen(url, timeout=30) as response:
        return response.read().decode()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _recommended_levels(defaults: str) -> dict[str, str]:
    recommended = defaults.split('getRecommendedDiagnosticRuleSet =', 1)[1].split('});', 1)[0]
    return dict(re.findall(r"(report\w+): '(none|hint|warning|error)'", recommended))


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


def _rule_defaults(rules: str, required: set[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    pattern = re.compile(
        r'^## `(?P<rule>[^`]+)`\n.*?Default level: .*?<code>(?P<level>ignore|warn|error)</code>',
        re.M | re.S,
    )
    for match in pattern.finditer(rules):
        if match['rule'] in required:
            parsed[match['rule']] = match['level']
    return parsed


def verify_upstream(baseline: dict[str, Any]) -> None:
    """Fetch pinned primary sources and prove the committed derivation still matches."""
    ty = baseline['ty']
    based = baseline['basedpyright']
    mapping = _download(ty['mapping_url'])
    rules = _download(ty['rules_url'])
    defaults = _download(based['defaults_url'])
    assert _sha256(mapping) == ty['mapping_sha256'], 'pinned ty mapping source changed'
    assert _sha256(rules) == ty['rules_sha256'], 'pinned ty rule source changed'
    assert _sha256(defaults) == based['defaults_sha256'], 'pinned BasedPyright source changed'

    levels = _recommended_levels(defaults)
    levels.update(based['global_rules'])
    active = {rule for rule, level in levels.items() if level != 'none'}
    assert active == set(based['active_diagnostics']), 'BasedPyright active-rule snapshot drifted'
    cells = _mapping_cells(mapping, active)
    assert cells == baseline['mapping_cells'], 'ty migration mapping snapshot drifted'
    required = {
        rule
        for replacements in cells.values()
        for replacement in replacements
        for rule in re.findall(r'\[`([^`]+)`\]\[ty-', replacement)
    }
    assert _rule_defaults(rules, required) == ty['rule_defaults'], 'ty default-level snapshot drifted'


def _selector_matches(code: str, selectors: list[str]) -> bool:
    return any(code == selector or code.startswith(selector) for selector in selectors)


def _ruff_enabled(code: str, path: str, config: dict[str, Any], *, requires_preview: bool) -> bool:
    ruff = config['tool']['ruff']
    lint = ruff['lint']
    if requires_preview and not ruff.get('preview', False):
        return False
    selected = [*lint.get('select', []), *lint.get('extend-select', [])]
    ignored = [*lint.get('ignore', []), *lint.get('extend-ignore', [])]
    if not _selector_matches(code, selected) or _selector_matches(code, ignored):
        return False
    excluded = [*ruff.get('exclude', []), *ruff.get('extend-exclude', [])]
    if any(fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch('/' + path, pattern) for pattern in excluded):
        return False
    for pattern, selectors in lint.get('per-file-ignores', {}).items():
        if (fnmatch.fnmatch(path, pattern) or Path(path).match(pattern)) and _selector_matches(code, selectors):
            return False
    return True


def _overridden_level(rule: str, surface: str, config: dict[str, Any], default: str) -> str:
    level = config['tool']['ty'].get('rules', {}).get(rule, default)
    path = SURFACES[surface]
    for override in config['tool']['ty'].get('overrides', []):
        if any(path == include or path.startswith(include.rstrip('/') + '/') for include in override['include']):
            level = override.get('rules', {}).get(rule, level)
    return level


def _based_active(diagnostic: str, surface: str, baseline: dict[str, Any]) -> bool:
    path = SURFACES[surface]
    for override in baseline['basedpyright']['overrides']:
        if path == override['include'] or path.startswith(override['include'].rstrip('/') + '/'):
            if override['rules'].get(diagnostic) == 'none':
                return False
    return True


def _classification_and_surface(
    diagnostic: str, replacements: list[str], baseline: dict[str, Any], config: dict[str, Any]
) -> tuple[str, str]:
    ty_rules = {rule for replacement in replacements for rule in re.findall(r'\[`([^`]+)`\]\[ty-', replacement)}
    ruff_codes = {code for replacement in replacements for code in re.findall(r'Ruff `([A-Z]+\d+)`', replacement)}
    applicable = [surface for surface in SURFACES if _based_active(diagnostic, surface, baseline)]
    covered: list[str] = []
    degraded: list[str] = []
    all_ty_rules_effective = True
    has_ty = False
    has_ruff = False
    for surface in applicable:
        effective_ty = {
            rule
            for rule in ty_rules
            if _overridden_level(rule, surface, config, baseline['ty']['rule_defaults'][rule]) != 'ignore'
        }
        effective_ruff = set()
        if surface in {'src', 'src/dashboard'}:
            for code in ruff_codes:
                code_cells = [replacement for replacement in replacements if f'Ruff `{code}`' in replacement]
                requires_preview = any('require preview' in replacement for replacement in code_cells)
                if _ruff_enabled(code, SURFACES[surface], config, requires_preview=requires_preview):
                    effective_ruff.add(code)
        has_ty |= bool(effective_ty)
        has_ruff |= bool(effective_ruff)
        if effective_ty or effective_ruff:
            covered.append(surface)
        if effective_ty != ty_rules:
            all_ty_rules_effective = False
            if ty_rules:
                degraded.append(surface)

    limited = any(marker in replacement for marker in LIMITED_MARKERS for replacement in replacements)
    if not covered:
        classification = 'absent'
    elif limited or (ty_rules and not all_ty_rules_effective):
        classification = 'partial'
    elif has_ty:
        classification = 'ty-backed'
    elif has_ruff:
        classification = 'ruff-only'
    else:
        classification = 'absent'

    uncovered = [surface for surface in applicable if surface not in covered]
    if not uncovered:
        surface_text = 'all applicable BasedPyright surfaces'
    elif not covered:
        surface_text = f'uncovered: {", ".join(applicable)}'
    else:
        surface_text = f'covered: {", ".join(covered)}; uncovered: {", ".join(uncovered)}'
    if classification == 'partial':
        if degraded:
            surface_text += f'; mapped-rule gaps: {", ".join(degraded)}'
        elif limited:
            surface_text += '; partial semantics'
    return classification, surface_text


def _receipt_rows(receipt: Path) -> dict[str, tuple[str, str]]:
    text = receipt.read_text()
    table = text.split('<!-- diagnostic-classification:start -->', 1)[1].split(
        '<!-- diagnostic-classification:end -->', 1
    )[0]
    rows = re.findall(
        r'^\| `(?P<diagnostic>report\w+)` \| (?P<classification>[a-z-]+) \| (?P<surface>[^|]+?) \|', table, re.M
    )
    return {diagnostic: (classification, surface) for diagnostic, classification, surface in rows}


def audit(*, baseline_path: Path = BASELINE, receipt: Path = RECEIPT) -> dict[str, set[str]]:
    baseline = _load(baseline_path)
    config = _project_config()
    workflow = (ROOT / '.github/workflows/ci.yml').read_text()
    assert baseline['schema'] == 1
    assert f'ty=={baseline["ty"]["version"]}' in config['dependency-groups']['dev']
    locked = tomllib.loads((ROOT / 'uv.lock').read_text())
    ty_packages = [package for package in locked['package'] if package['name'] == 'ty']
    assert [package['version'] for package in ty_packages] == [baseline['ty']['version']]
    assert len(re.findall(r'(?m)^\s*run:\s*uv run ruff check src\s*$', workflow)) == 1

    actual: dict[str, tuple[str, str]] = {}
    classified = {classification: set() for classification in EXPECTED}
    for diagnostic, replacements in baseline['mapping_cells'].items():
        classification, surface = _classification_and_surface(diagnostic, replacements, baseline, config)
        classified[classification].add(diagnostic)
        actual[diagnostic] = (classification, surface)
    assert classified == EXPECTED, 'effective diagnostic classification changed'
    assert set().union(*classified.values()) == set(baseline['basedpyright']['active_diagnostics'])
    assert sum(len(diagnostics) for diagnostics in classified.values()) == len(actual)
    assert _receipt_rows(receipt) == actual, 'receipt decision table does not match effective coverage'
    return classified


def _print_table(baseline_path: Path) -> None:
    baseline = _load(baseline_path)
    config = _project_config()
    for diagnostic, replacements in baseline['mapping_cells'].items():
        classification, surface = _classification_and_surface(diagnostic, replacements, baseline, config)
        print(f'| `{diagnostic}` | {classification} | {surface} |')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, default=BASELINE)
    parser.add_argument('--receipt', type=Path, default=RECEIPT)
    parser.add_argument('--refresh', action='store_true', help='verify the snapshot against pinned upstream sources')
    parser.add_argument('--print-table', action='store_true')
    args = parser.parse_args()
    if args.refresh:
        verify_upstream(_load(args.baseline))
        print('pinned upstream sources match the committed snapshot')
    if args.print_table:
        _print_table(args.baseline)
        return
    classified = audit(baseline_path=args.baseline, receipt=args.receipt)
    for classification in ('ty-backed', 'ruff-only', 'partial', 'absent'):
        print(f'{classification}: {len(classified[classification])}')


if __name__ == '__main__':
    main()
