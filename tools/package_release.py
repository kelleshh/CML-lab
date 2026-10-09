"""Compatibility command for the CML-lab source packager."""
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.package_source import (ROOT_FILES, SOURCE_DIRECTORIES, EXCLUDED_DIRECTORIES,
                                 SOURCE_SUFFIXES, EXECUTABLES, release_files, write_archive, main)

if __name__ == '__main__':
    main()
