"""Одна команда: python run.py. Первый запуск устанавливает зависимости в .venv."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import venv
import webbrowser
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description='Линейная лаборатория')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=8765, type=int)
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--system', action='store_true', help='Использовать текущее окружение Python')
    options = parser.parse_args()
    project = Path(__file__).resolve().parent
    if sys.version_info < (3, 11):
        raise SystemExit('Нужен Python 3.11 или новее.')
    os.chdir(project)
    envfile = project / '.env'
    if envfile.exists():
        for line in envfile.read_text().splitlines():
            if line.strip() and not line.lstrip().startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ.setdefault(key.strip(), value.strip().strip('\"\''))
    if not options.system:
        environment = project / '.venv'
        interpreter = environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        if not interpreter.exists():
            print('Подготовка окружения Python. Это требуется только при первом запуске.', flush=True)
            venv.create(environment, with_pip=True)
        stamp = environment / 'linear-lab-requirements.txt'
        requirements = (project / 'requirements.txt').read_text() + (project / 'constraints.txt').read_text()
        if not stamp.exists() or stamp.read_text() != requirements:
            subprocess.run([str(interpreter), '-m', 'pip', 'install', '-r', str(project / 'requirements.txt')], check=True)
            stamp.write_text(requirements)
        command = [str(interpreter), str(project / 'run.py'), '--system', '--host', options.host, '--port', str(options.port)]
        if options.no_browser:
            command.append('--no-browser')
        raise SystemExit(subprocess.call(command))
    import uvicorn
    from linear_lab.app import create_app
    print(f'Лаборатория: http://{options.host}:{options.port}', flush=True)
    if not options.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(f'http://127.0.0.1:{options.port}')).start()
    uvicorn.run(create_app(), host=options.host, port=options.port, log_level='warning')


if __name__ == '__main__':
    main()
