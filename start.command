#!/bin/sh
# macOS: двойной щелчок открывает Terminal.
LAB_DIRECTORY=$(CDPATH= cd -P "$(dirname "$0")" && pwd) || exit 1
exec /bin/sh "$LAB_DIRECTORY/start.sh" "$@"
