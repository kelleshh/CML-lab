"""Локальный запуск: подготовка Python, проверка готовности и окно браузера."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import venv
import webbrowser

PROJECT = Path(__file__).resolve().parent
APP_TITLE = 'CML-lab'
PYTHON_DOWNLOAD = 'https://www.python.org/downloads/'
MIN_PYTHON = (3, 12)
LOGGER = logging.getLogger('cml_lab.launcher')


class LauncherError(RuntimeError):
    """Ошибка запуска, которую можно объяснить без трассировки в окне пользователя."""


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=APP_TITLE)
    parser.add_argument('--host', default='127.0.0.1', help='Адрес сервера; по умолчанию только этот компьютер')
    parser.add_argument('--port', default=8765, type=int, help='Начальный порт; занятый порт заменяется свободным')
    parser.add_argument('--no-browser', action='store_true', help='Не открывать браузер автоматически')
    parser.add_argument('--system', action='store_true', help='Использовать текущее окружение Python')
    parser.add_argument('--doctor', '--check', dest='doctor', action='store_true', help='Проверить Python, библиотеки, модели и порт без запуска сервера')
    return parser


def load_env(project: Path) -> None:
    """Простой .env: значения не исполняются и не заменяют настройки процесса."""
    envfile = project / '.env'
    if not envfile.exists():
        return
    for line in envfile.read_text(encoding='utf-8-sig').splitlines():
        if line.strip() and not line.lstrip().startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            key = key.strip()
            if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key):
                os.environ.setdefault(key, value.strip().strip('"\''))


def data_directory() -> Path:
    """The new variable wins; an explicit old setting remains a migration alias."""
    configured = os.environ.get('CML_LAB_DATA') or os.environ.get('LINEAR_LAB_DATA')
    return Path(configured or str(Path.home() / '.cml-lab')).expanduser()


def configure_logging() -> Path:
    data = data_directory()
    data.mkdir(parents=True, exist_ok=True)
    logfile = data / 'launcher.log'
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    for handler in list(LOGGER.handlers):
        handler.close()
        LOGGER.removeHandler(handler)
    file_handler = RotatingFileHandler(logfile, maxBytes=2 * 1024 * 1024, backupCount=2, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter('%(message)s'))
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(console)
    return logfile


def browser_host(host: str) -> str:
    return {'0.0.0.0': '127.0.0.1', '::': '::1', 'localhost': '127.0.0.1'}.get(host, host)


def service_url(host: str, port: int) -> str:
    address = browser_host(host)
    if ':' in address and not address.startswith('['):
        address = f'[{address}]'
    return f'http://{address}:{port}'


def read_json_url(url: str, timeout: float = 0.5) -> dict | None:
    # Локальный запрос не должен идти через пользовательский HTTP_PROXY.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=timeout) as response:
            raw = response.read(2 * 1024 * 1024 + 1)
            if len(raw) > 2 * 1024 * 1024:
                return None
            document = json.loads(raw)
            return document if isinstance(document, dict) else None
    except (OSError, urllib.error.URLError, ValueError):
        return None


def probe_lab(host: str, port: int) -> bool:
    """Повторно открываем только локальную лабораторию с совпадающей идентичностью API."""
    if host not in {'localhost', '127.0.0.1', '::1', '0.0.0.0', '::'}:
        return False
    health = read_json_url(service_url(host, port) + '/api/health')
    if not health or health.get('status') != 'ok' or not isinstance(health.get('version'), str):
        return False
    api = read_json_url(service_url(host, port) + '/openapi.json')
    info = (api or {}).get('info', {})
    return info.get('title') == APP_TITLE and info.get('version') == health['version']


def reserve_port(host: str, start: int, attempts: int = 50) -> tuple[socket.socket, int]:
    """Удерживаем сокет до передачи Uvicorn, чтобы другой процесс не занял выбранный порт."""
    if not 0 <= start <= 65535:
        raise LauncherError('Порт должен быть от 1 до 65535. Значение 0 выбирает свободный автоматически.')
    try:
        bind_host = '127.0.0.1' if host == 'localhost' else host
        addresses = socket.getaddrinfo(bind_host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise LauncherError(f'Не удалось определить адрес сервера «{host}».') from exc
    for port in ([0] if start == 0 else range(start, min(start + attempts, 65536))):
        for family, kind, protocol, _, address in addresses:
            reserved = socket.socket(family, kind, protocol)
            try:
                # Windows SO_REUSEADDR позволяет перехватить занятый порт.
                if os.name == 'nt' and hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
                    reserved.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                reserved.bind((address[0], port, *address[2:]))
                reserved.listen(128)
                return reserved, int(reserved.getsockname()[1])
            except OSError:
                reserved.close()
    raise LauncherError(f'Не найден свободный порт рядом с {start}. Попробуйте --port 9000.')


def environment_python(project: Path) -> Path:
    return project / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def child_command(interpreter: Path, project: Path, options: argparse.Namespace) -> list[str]:
    command = [str(interpreter), str(project / 'run.py'), '--system', '--host', options.host, '--port', str(options.port)]
    if options.no_browser:
        command.append('--no-browser')
    if options.doctor:
        command.append('--doctor')
    return command


def call_child(command: list[str], project: Path, env: dict | None = None) -> int:
    """Терминал остается владельцем сервера; Ctrl+C не оставляет дочерний процесс."""
    child = subprocess.Popen(command, cwd=project, env=env)
    try:
        return child.wait()
    except KeyboardInterrupt:
        # Дочерний процесс получает Ctrl+C в той же группе консоли. Даем ему закрыть задачи.
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        raise


def requirements_signature(project: Path) -> str:
    files = [project / 'requirements.txt', project / 'constraints.txt']
    return '\n'.join(file.read_text(encoding='utf-8') for file in files if file.exists())


def install_dependencies(interpreter: Path, project: Path) -> None:
    command = [str(interpreter), '-m', 'pip', 'install', '--disable-pip-version-check', '-r', str(project / 'requirements.txt')]
    process = subprocess.Popen(command, cwd=project, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
    assert process.stdout is not None
    try:
        for line in process.stdout:
            LOGGER.info(line.rstrip())
        returncode = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    if returncode:
        raise LauncherError('Не удалось установить библиотеки. Проверьте интернет и строки ошибки выше. Затем запустите снова; завершенные загрузки используются повторно.')


def prepare_environment(project: Path) -> Path:
    interpreter = environment_python(project)
    if not interpreter.exists():
        LOGGER.info('[2/4] Создаем отдельное окружение Python. Системные библиотеки не меняются.')
        try:
            venv.create(project / '.venv', with_pip=True)
        except (OSError, subprocess.SubprocessError) as exc:
            raise LauncherError('Не удалось создать окружение. На Linux проверьте наличие пакета python3-venv. Убедитесь, что папка проекта доступна для записи. Можно использовать подготовленный Python с параметром --system.') from exc
    else:
        LOGGER.info('[2/4] Используем готовое окружение Python.')
        try:
            supported = subprocess.run(
                [str(interpreter), '-c', 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)'],
                cwd=project, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10
            ).returncode == 0
        except (OSError, subprocess.SubprocessError) as exc:
            raise LauncherError('Не удалось запустить Python из .venv. Переименуйте эту папку и запустите снова, чтобы создать новое окружение.') from exc
        if not supported:
            raise LauncherError('Окружение .venv использует Python старее 3.12. Переименуйте .venv и повторите запуск с Python 3.12; пользовательские данные хранятся отдельно.')
    stamp = project / '.venv' / 'cml-lab-requirements.txt'
    signature = requirements_signature(project)
    if not stamp.exists() or stamp.read_text(encoding='utf-8') != signature:
        LOGGER.info('[3/4] Устанавливаем библиотеки. Первый запуск может занять несколько минут; нужен интернет.')
        install_dependencies(interpreter, project)
        stamp.write_text(signature, encoding='utf-8')
    else:
        LOGGER.info('[3/4] Библиотеки уже установлены.')
    return interpreter


def direct_dependencies(project: Path, platform: str | None = None) -> list[tuple[str, str]]:
    """Read exact pins and the two platform markers used by this release.

    The launcher must work before pip/packaging are installed. Unsupported
    marker expressions are reported rather than silently skipped by diagnosis.
    """
    dependencies = []
    platform = sys.platform if platform is None else platform
    for line in (project / 'requirements.txt').read_text(encoding='utf-8').splitlines():
        match = re.fullmatch(r'\s*([A-Za-z0-9_.-]+)==([^\s;]+)\s*(?:;\s*([^#]+))?\s*(?:#.*)?', line)
        if match:
            if match[3]:
                marker = re.fullmatch(r"sys_platform\s*(==|!=)\s*['\"]([^'\"]+)['\"]\s*", match[3])
                if marker is None:
                    raise LauncherError('Неизвестное условие зависимости: ' + match[3])
                equal = platform == marker[2]
                if not (equal if marker[1] == '==' else not equal):
                    continue
            dependencies.append((match[1], match[2]))
    return dependencies


def doctor(project: Path, options: argparse.Namespace) -> int:
    LOGGER.info('Диагностика — сервер не запускается, библиотеки не устанавливаются.')
    LOGGER.info('Python %s · %s', sys.version.split()[0], sys.executable)
    failures = 0
    for distribution, expected in direct_dependencies(project):
        try:
            actual = importlib.metadata.version(distribution)
            if actual != expected:
                failures += 1
                LOGGER.error('✗ %s: установлена %s, требуется %s', distribution, actual, expected)
            else:
                LOGGER.info('✓ %s %s', distribution, actual)
        except importlib.metadata.PackageNotFoundError:
            failures += 1
            LOGGER.error('✗ %s: не установлена', distribution)
    try:
        from cml_lab.infrastructure.ml.catalogue import AlgorithmCatalogue
        from cml_lab.shared.domain import TaskKind
        catalogue = AlgorithmCatalogue().catalogue()
        unavailable = [item for item in catalogue if not item['available']]
        LOGGER.info('✓ Реестр: %s алгоритмов, доступно %s, задач %s', len(catalogue), len(catalogue) - len(unavailable), len(TaskKind))
        for task in TaskKind:
            count = sum(task.value in item['tasks'] and item['available'] for item in catalogue)
            LOGGER.info('  %s: %s доступных алгоритмов', task.value, count)
        for item in unavailable:
            failures += 1
            LOGGER.error('✗ %s: %s', item['name'], item['reason'])
    except Exception as exc:
        failures += 1
        LOGGER.error('✗ Реестр моделей не загружается: %s', exc)
        LOGGER.debug('Подробности ошибки реестра', exc_info=True)
    existing = running_instance(options)
    if existing:
        LOGGER.info('✓ Лаборатория уже работает: %s', existing)
    else:
        try:
            reserved, port = reserve_port(options.host, options.port)
            reserved.close()
            if port != options.port and options.port:
                LOGGER.info('✓ Порт %s занят; при запуске выберем %s', options.port, port)
            else:
                LOGGER.info('✓ Доступен порт %s', port)
        except LauncherError as exc:
            failures += 1
            LOGGER.error('✗ %s', exc)
    LOGGER.info('Итог: %s', 'все проверки пройдены.' if not failures else f'найдено проблем: {failures}. Обычный запуск подготовит окружение; журнал содержит подробности.')
    return 1 if failures else 0


def instance_file() -> Path:
    return data_directory() / 'running-server.json'


def running_instance(options: argparse.Namespace) -> str | None:
    try:
        saved = json.loads(instance_file().read_text(encoding='utf-8'))
        # Повторное открытие допустимо только для этой копии проекта и ее папки данных.
        if isinstance(saved, dict) and saved.get('project') == str(PROJECT) and saved.get('host') == options.host:
            port = int(saved['port'])
            if 0 < port <= 65535 and probe_lab(options.host, port):
                return service_url(options.host, port)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def open_browser(url: str) -> None:
    try:
        if not webbrowser.open(url):
            LOGGER.info('Браузер не открылся автоматически. Скопируйте адрес: %s', url)
    except webbrowser.Error:
        LOGGER.info('Браузер не открылся автоматически. Скопируйте адрес: %s', url)


def readiness_monitor(server, host: str, port: int, open_window: bool, stop: threading.Event) -> None:
    url = service_url(host, port)
    deadline = time.monotonic() + 30
    while not stop.wait(0.2):
        if server.should_exit:
            return
        if server.started:
            health = read_json_url(url + '/api/health', timeout=1)
            if health and health.get('status') == 'ok':
                LOGGER.info('Готово: %s', url)
                LOGGER.info('Остановка: Ctrl+C в этом окне. После остановки вычисления завершатся.')
                try:
                    instance_file().write_text(json.dumps({'project': str(PROJECT), 'host': host, 'port': port, 'pid': os.getpid()}, ensure_ascii=False), encoding='utf-8')
                except OSError as exc:
                    LOGGER.warning('Не удалось записать адрес запущенного сервера: %s', exc)
                if open_window:
                    open_browser(url)
                return
        if time.monotonic() > deadline:
            LOGGER.warning('Проверка готовности заняла больше 30 секунд. Сервер остается в этом окне; проверьте адрес %s и журнал запуска.', url)
            return


def serve(options: argparse.Namespace) -> int:
    import uvicorn
    from cml_lab.presentation.http.api import create_app
    from cml_lab.shared.domain import ConflictError

    reserved, port = reserve_port(options.host, options.port)
    url = service_url(options.host, port)
    if options.port and port != options.port:
        LOGGER.info('Порт %s занят другим приложением; используем %s.', options.port, port)
    LOGGER.info('[4/4] Запускаем лабораторию и проверяем готовность: %s', url)
    for name in ('uvicorn', 'uvicorn.error', 'uvicorn.access'):
        logger = logging.getLogger(name)
        logger.handlers = LOGGER.handlers[:]
        logger.propagate = False
    stop = threading.Event()
    server = None
    try:
        try:
            app = create_app()
        except ConflictError as exc:
            raise LauncherError(str(exc)) from exc
        config = uvicorn.Config(app, host=options.host, port=port, log_level='warning', log_config=None)
        server = uvicorn.Server(config)
        monitor = threading.Thread(target=readiness_monitor, args=(server, options.host, port, not options.no_browser, stop), daemon=True)
        monitor.start()
        server.run(sockets=[reserved])
        return 0 if server.started else 1
    finally:
        stop.set()
        reserved.close()
        try:
            saved = json.loads(instance_file().read_text(encoding='utf-8'))
            if isinstance(saved, dict) and saved.get('pid') == os.getpid():
                instance_file().unlink(missing_ok=True)
        except (OSError, ValueError):
            pass
        if server is not None and server.started:
            LOGGER.info('Лаборатория остановлена.')


def main(argv: list[str] | None = None) -> int:
    options = argument_parser().parse_args(argv)
    if sys.version_info < MIN_PYTHON:
        print(f'Нужен Python 3.12 или новее. Установите Python с {PYTHON_DOWNLOAD}', flush=True)
        return 1
    if not 0 <= options.port <= 65535:
        print('Порт должен быть от 1 до 65535, либо 0 для автоматического выбора.', flush=True)
        return 1
    os.chdir(PROJECT)
    load_env(PROJECT)
    try:
        logfile = configure_logging()
    except OSError as exc:
        print(f'Не удалось создать журнал запуска: {exc}. Проверьте доступ к папке данных.', flush=True)
        return 1
    LOGGER.info('Журнал запуска: %s', logfile)
    try:
        if options.doctor:
            interpreter = environment_python(PROJECT)
            if not options.system and interpreter.exists():
                return call_child(child_command(interpreter, PROJECT, options), PROJECT)
            if not options.system:
                LOGGER.warning('Отдельное окружение еще не создано. Проверяем текущий Python; обычный запуск подготовит .venv.')
            return doctor(PROJECT, options)
        existing = running_instance(options)
        if existing:
            LOGGER.info('Лаборатория уже работает: %s', existing)
            if not options.no_browser:
                open_browser(existing)
            return 0
        if not os.environ.get('CML_LAB_BOOTSTRAPPED'):
            LOGGER.info('[1/4] Python %s найден.', sys.version.split()[0])
        if not options.system:
            interpreter = prepare_environment(PROJECT)
            child_env = os.environ.copy()
            child_env['CML_LAB_BOOTSTRAPPED'] = '1'
            return call_child(child_command(interpreter, PROJECT, options), PROJECT, child_env)
        if not os.environ.get('CML_LAB_BOOTSTRAPPED'):
            LOGGER.info('[2/4] Используем текущее окружение (--system).')
            LOGGER.info('[3/4] Загружаем установленные библиотеки.')
        return serve(options)
    except KeyboardInterrupt:
        LOGGER.info('Запуск остановлен.')
        return 0
    except LauncherError as exc:
        LOGGER.error('Не удалось запустить лабораторию. %s', exc)
        LOGGER.error('Подробности сохранены: %s', logfile)
        return 1
    except Exception as exc:
        LOGGER.exception('Не удалось запустить лабораторию: %s', exc)
        LOGGER.error('Покажите этот журнал при обращении за помощью: %s', logfile)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
