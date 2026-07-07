"""내부 import 정적 해석 테스트 — 리팩터링으로 깨진 참조가 생기면 실패한다."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOTS = {'pipeline': ROOT, 'app': ROOT / 'backend'}


def _module_exists(dotted: str) -> bool:
    top = dotted.split('.')[0]
    if top not in PACKAGE_ROOTS:
        return True  # 외부 패키지는 검사 대상 아님
    current = PACKAGE_ROOTS[top]
    parts = dotted.split('.')
    for index, part in enumerate(parts):
        if (current / part).is_dir():
            current = current / part
        elif (current / f'{part}.py').is_file():
            return True
        else:
            return False
    return True


def _package_parts(pyfile: Path) -> list[str]:
    for top, base in PACKAGE_ROOTS.items():
        try:
            rel = pyfile.relative_to(base)
        except ValueError:
            continue
        if rel.parts[0] == top:
            return list(rel.parts[:-1])
    return []


def _collect_errors() -> list[str]:
    errors = []
    for top, base in PACKAGE_ROOTS.items():
        for pyfile in (base / top).rglob('*.py'):
            if '__pycache__' in pyfile.parts:
                continue
            tree = ast.parse(pyfile.read_text(encoding='utf-8'), filename=str(pyfile))
            pkg = _package_parts(pyfile)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if not _module_exists(alias.name):
                            errors.append(f'{pyfile.relative_to(ROOT)}:{node.lineno}: import {alias.name}')
                elif isinstance(node, ast.ImportFrom):
                    if node.level == 0:
                        if node.module and not _module_exists(node.module):
                            errors.append(f'{pyfile.relative_to(ROOT)}:{node.lineno}: from {node.module} import …')
                    elif node.module:
                        if len(pkg) < node.level - 1:
                            errors.append(f'{pyfile.relative_to(ROOT)}:{node.lineno}: 최상위를 벗어난 상대 import')
                            continue
                        anchor = pkg[: len(pkg) - (node.level - 1)] if node.level > 1 else pkg
                        dotted = '.'.join(anchor + [node.module])
                        if not _module_exists(dotted):
                            errors.append(
                                f'{pyfile.relative_to(ROOT)}:{node.lineno}: from {"." * node.level}{node.module} import … → {dotted}'
                            )
    return errors


def test_all_internal_imports_resolve():
    errors = _collect_errors()
    assert not errors, '깨진 내부 import:\n' + '\n'.join(errors)
