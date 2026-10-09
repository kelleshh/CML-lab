"""Check and package distributable sources without local environments or user data."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import zipfile


ROOT_FILES = (
    'README.md', 'THIRD_PARTY_NOTICES.md', 'requirements.txt',
    'requirements-dev.txt', 'constraints.txt', 'run.py', 'start.sh',
    'start.command', 'start.bat', 'diagnose.bat', 'diagnose.command',
    'package.json', 'package-lock.json', 'vite.config.js', '.env.example', '.gitignore',
)
SOURCE_DIRECTORIES = ('linear_lab', 'web', 'docs', 'tests', 'tools')
EXCLUDED_DIRECTORIES = frozenset({
    '.venv', 'node_modules', '__pycache__', '.pytest_cache', '.git',
    '.sites-runtime', 'data', 'artifacts', 'experiments',
})
SOURCE_SUFFIXES = frozenset({
    '.py', '.js', '.mjs', '.css', '.html', '.json', '.md', '.txt',
    '.woff', '.woff2', '.ttf', '.svg',
})
EXECUTABLES = ('start.sh', 'start.command', 'diagnose.command')


def release_files(project: Path) -> list[Path]:
    """Return an explicit source manifest, rejecting links before reading them."""
    project = project.resolve()
    files = []
    for name in ROOT_FILES:
        path = project / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Required regular source file is missing: {name}')
        files.append(path)
    for name in SOURCE_DIRECTORIES:
        directory = project / name
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f'Required source directory is missing: {name}')
        for current, directories, filenames in os.walk(directory, followlinks=False):
            current = Path(current)
            directories[:] = [entry for entry in directories
                               if entry not in EXCLUDED_DIRECTORIES and not entry.startswith('.')]
            for entry in directories:
                if (current / entry).is_symlink():
                    raise ValueError(f'Symbolic link in source tree: {(current / entry).relative_to(project)}')
            for entry in filenames:
                path = current / entry
                if path.is_symlink():
                    raise ValueError(f'Symbolic link in source tree: {path.relative_to(project)}')
                if entry.startswith('.'):
                    continue
                # License files have no suffix; model exports and data tables are not source files.
                if path.suffix in SOURCE_SUFFIXES or entry == 'LICENSE' or entry.endswith('-LICENSE'):
                    if not path.is_file():
                        raise ValueError(f'Non-regular source entry: {path.relative_to(project)}')
                    files.append(path)
    for name in EXECUTABLES:
        if not (project / name).stat().st_mode & 0o111:
            raise ValueError(f'Launcher must have an executable bit before packaging: {name}')
    return sorted(files, key=lambda path: path.relative_to(project).as_posix())


def write_archive(project: Path, destination: Path) -> dict:
    """Create an atomic ZIP from the checked manifest; preserve launcher file modes."""
    project = project.resolve()
    destination = destination.resolve()
    if destination.suffix.lower() != '.zip':
        raise ValueError('Release output must have a .zip suffix')
    if destination == project or project in destination.parents:
        raise ValueError('Release output must be outside the source directory')
    files = release_files(project)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=destination.stem + '-', suffix='.tmp',
                                         dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in files:
                archive.write(path, 'linear-lab/' + path.relative_to(project).as_posix())
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError('Release ZIP failed its CRC check')
            for name in EXECUTABLES:
                if not (archive.getinfo('linear-lab/' + name).external_attr >> 16) & 0o111:
                    raise ValueError(f'Executable mode was not preserved: {name}')
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {'archive': str(destination), 'files': len(files), 'bytes': destination.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--check', action='store_true', help='Check source manifest without creating a ZIP')
    action.add_argument('--output', type=Path, help='Write a release ZIP outside the project')
    options = parser.parse_args()
    project = options.project.resolve()
    if options.check:
        files = release_files(project)
        report = {'project': str(project), 'files': len(files),
                  'source_bytes': sum(path.stat().st_size for path in files),
                  'launchers_executable': True}
    else:
        report = write_archive(project, options.output)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
