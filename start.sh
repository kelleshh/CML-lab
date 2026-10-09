#!/bin/sh
# Запуск из файлового менеджера (в терминале) или командой ./start.sh.
set -u
LAB_DIRECTORY=$(CDPATH= cd -P "$(dirname "$0")" && pwd) || exit 1
cd "$LAB_DIRECTORY" || exit 1
export PYTHONUTF8=1
export PYTHONUNBUFFERED=1
LAB_PYTHON=
for LAB_CANDIDATE in "$LAB_DIRECTORY/.venv/bin/python" python3.12 python3.13 python3.14 python3.11 python3 python; do
    if "$LAB_CANDIDATE" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; then
        LAB_PYTHON=$LAB_CANDIDATE
        break
    fi
done
if [ -z "$LAB_PYTHON" ]; then
    printf '%s\n' 'Нужен Python 3.11 или новее.' 'Скачайте его с официального сайта: https://www.python.org/downloads/' 'После установки запустите этот файл снова.'
    if [ -t 0 ]; then
        printf '%s' 'Нажмите Enter, чтобы закрыть окно: '
        read -r LAB_REPLY
    fi
    exit 1
fi
"$LAB_PYTHON" "$LAB_DIRECTORY/run.py" "$@"
LAB_STATUS=$?
if [ "$LAB_STATUS" -ne 0 ] && [ -t 0 ]; then
    printf '%s\n' 'Запуск завершился с ошибкой. Адрес журнала указан выше.'
    printf '%s' 'Нажмите Enter, чтобы закрыть окно: '
    read -r LAB_REPLY
fi
exit "$LAB_STATUS"
