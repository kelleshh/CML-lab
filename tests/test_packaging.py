"""Release archives preserve source/licenses and exclude user data and environments."""
from pathlib import Path
import importlib.util
import zipfile

import pytest


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('linear_lab_package_release', PROJECT / 'tools/package_release.py')
packager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packager)


@pytest.fixture
def source_tree(tmp_path):
    project = tmp_path / 'project with spaces'
    project.mkdir()
    for name in packager.ROOT_FILES:
        path = project / name
        path.write_text('source', encoding='utf-8')
        if name in packager.EXECUTABLES:
            path.chmod(0o755)
    for name in packager.SOURCE_DIRECTORIES:
        (project / name).mkdir()
    return project


def test_archive_keeps_vendor_provenance_and_launcher_modes_but_excludes_data(source_tree, tmp_path):
    vendor = source_tree / 'linear_lab/_vendor'
    vendor.mkdir()
    for name in ('sampling_math.py', 'smogn-LICENSE', 'PROVENANCE.md', 'upstream-sha256.json'):
        (vendor / name).write_text('source')
    cache = source_tree / 'linear_lab/__pycache__'
    cache.mkdir()
    (cache / 'app.pyc').write_bytes(b'cache')
    (source_tree / '.env').write_text('USER_SECRET=value')
    (source_tree / 'linear_lab/.env').write_text('USER_SECRET=value')
    (source_tree / 'linear_lab/model.joblib').write_bytes(b'model artifact')
    (source_tree / 'web/users.csv').write_text('personal data')
    data = source_tree / 'data'
    data.mkdir()
    (data / 'dataset.json').write_text('user dataset')
    archive = tmp_path / 'release.zip'
    packager.write_archive(source_tree, archive)
    with zipfile.ZipFile(archive) as opened:
        names = opened.namelist()
        assert 'linear-lab/.env.example' in names
        assert all('USER_SECRET' not in opened.read(name).decode() for name in names)
        assert not any('__pycache__' in name or name.endswith(('.pyc', '.joblib', '.csv')) for name in names)
        assert not any(name.startswith('linear-lab/data/') for name in names)
        assert 'linear-lab/linear_lab/_vendor/smogn-LICENSE' in names
        assert 'linear-lab/linear_lab/_vendor/upstream-sha256.json' in names
        assert all((opened.getinfo('linear-lab/' + name).external_attr >> 16) & 0o111
                   for name in packager.EXECUTABLES)
        assert opened.testzip() is None


def test_symlink_cannot_copy_external_file_into_archive(source_tree, tmp_path):
    outside = tmp_path / 'private.py'
    outside.write_text('secret')
    (source_tree / 'linear_lab/alias.py').symlink_to(outside)
    with pytest.raises(ValueError, match='Symbolic link'):
        packager.release_files(source_tree)


def test_symlink_directory_cannot_pull_external_tree(source_tree, tmp_path):
    outside = tmp_path / 'private'
    outside.mkdir()
    (outside / 'secret.py').write_text('secret')
    (source_tree / 'web/alias').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='Symbolic link'):
        packager.release_files(source_tree)


def test_missing_executable_mode_and_inside_tree_output_are_rejected(source_tree):
    (source_tree / 'start.sh').chmod(0o644)
    with pytest.raises(ValueError, match='executable bit'):
        packager.release_files(source_tree)
    with pytest.raises(ValueError, match='outside'):
        packager.write_archive(source_tree, source_tree / 'release.zip')
