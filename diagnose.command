#!/bin/sh
LAB_DIRECTORY=$(CDPATH= cd -P "$(dirname "$0")" && pwd) || exit 1
/bin/sh "$LAB_DIRECTORY/start.sh" --doctor "$@"
LAB_STATUS=$?
if [ "$LAB_STATUS" -eq 0 ] && [ -t 0 ]; then
    printf '%s' 'Нажмите Enter, чтобы закрыть диагностику: '
    read -r LAB_REPLY
fi
exit "$LAB_STATUS"
