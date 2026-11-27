"""Check the exact environment needed by the archived prediction models."""
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    expected_python = (root / '.python-version').read_text().strip()
    errors = []
    if platform.python_implementation() != 'CPython' or platform.python_version() != expected_python:
        errors.append(f'Use CPython {expected_python}; found {platform.python_implementation()} {platform.python_version()}')
    for line in (root / 'requirements.lock').read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        name, expected = line.split('==', 1)
        try:
            actual = version(name)
        except PackageNotFoundError:
            actual = 'not installed'
        if actual != expected:
            errors.append(f'{name}: expected {expected}, found {actual}')
    if errors:
        print('\n'.join(errors))
        return 1
    print(f'Environment matches CPython {expected_python} and all locked dependencies.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
