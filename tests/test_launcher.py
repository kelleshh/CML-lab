"""Запускатель: безопасные аргументы, настоящий HTTP readiness и отсутствие второго сервера."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('linear_lab_run', PROJECT / 'run.py')
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


@pytest.fixture
def http_service():
    documents = {
        '/api/health': {'status': 'ok', 'version': '2.0.0'},
        '/openapi.json': {'info': {'title': launcher.APP_TITLE, 'version': '2.0.0'}},
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(documents.get(self.path)).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield server.server_port, documents
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def test_launcher_arguments_keep_paths_and_flags_separate():
    options = launcher.argument_parser().parse_args(['--check', '--no-browser', '--port', '9002'])
    assert options.doctor
    path = Path('/tmp/project with spaces & brackets')
    command = launcher.child_command(path / '.venv' / 'python', path, options)
    assert command == [str(path / '.venv' / 'python'), str(path / 'run.py'), '--system', '--host', '127.0.0.1', '--port', '9002', '--no-browser', '--doctor']


def test_dotenv_is_text_and_preserves_process_settings(tmp_path, monkeypatch):
    (tmp_path / '.env').write_text('LAB_TEST_FIRST="a path with spaces"\nLAB_TEST_KEEP=changed\nLAB_TEST_CODE=$(touch marker)\nBAD KEY=value\n', encoding='utf-8')
    monkeypatch.delenv('LAB_TEST_FIRST', raising=False)
    monkeypatch.setenv('LAB_TEST_KEEP', 'original')
    monkeypatch.delenv('LAB_TEST_CODE', raising=False)
    launcher.load_env(tmp_path)
    assert os.environ['LAB_TEST_FIRST'] == 'a path with spaces'
    assert os.environ['LAB_TEST_KEEP'] == 'original'
    assert os.environ['LAB_TEST_CODE'] == '$(touch marker)'
    assert not (tmp_path / 'marker').exists()


def test_occupied_port_is_skipped_and_selected_socket_is_reserved():
    busy = socket.socket()
    busy.bind(('127.0.0.1', 0))
    busy.listen()
    start = busy.getsockname()[1]
    try:
        if start == 65535:
            pytest.skip('Системный эфемерный порт не оставил места для следующего порта')
        reserved, port = launcher.reserve_port('127.0.0.1', start)
        try:
            assert port > start
            challenger = socket.socket()
            try:
                with pytest.raises(OSError):
                    challenger.bind(('127.0.0.1', port))
            finally:
                challenger.close()
        finally:
            reserved.close()
    finally:
        busy.close()


def test_ephemeral_port_selection_and_invalid_address():
    reserved, port = launcher.reserve_port('127.0.0.1', 0)
    assert 0 < port <= 65535
    reserved.close()
    with pytest.raises(launcher.LauncherError, match='Порт'):
        launcher.reserve_port('127.0.0.1', 65536)
    with pytest.raises(launcher.LauncherError, match='адрес'):
        launcher.reserve_port('invalid host name', 8765)


def test_service_urls_include_ipv6_brackets():
    assert launcher.service_url('0.0.0.0', 8765) == 'http://127.0.0.1:8765'
    assert launcher.service_url('::', 8765) == 'http://[::1]:8765'


def test_localhost_reservation_matches_browser_ipv4_address():
    reserved, port = launcher.reserve_port('localhost', 0)
    try:
        assert reserved.getsockname()[0] == '127.0.0.1'
        assert launcher.service_url('localhost', port) == f'http://127.0.0.1:{port}'
    finally:
        reserved.close()


def test_reuse_requires_matching_health_and_api_identity(http_service, monkeypatch):
    port, documents = http_service
    monkeypatch.setenv('HTTP_PROXY', 'http://127.0.0.1:1')
    assert launcher.probe_lab('127.0.0.1', port)
    documents['/openapi.json']['info']['title'] = 'Another application'
    assert not launcher.probe_lab('127.0.0.1', port)
    documents['/openapi.json']['info']['title'] = launcher.APP_TITLE
    documents['/openapi.json']['info']['version'] = 'different'
    assert not launcher.probe_lab('127.0.0.1', port)
    assert not launcher.probe_lab('example.com', port)


def test_autoselected_running_port_is_reused_from_manifest(http_service, tmp_path, monkeypatch):
    port, _ = http_service
    monkeypatch.setenv('LINEAR_LAB_DATA', str(tmp_path))
    launcher.instance_file().write_text(json.dumps({'project': str(launcher.PROJECT), 'host': '127.0.0.1', 'port': port, 'pid': 10}))
    options = launcher.argument_parser().parse_args(['--port', '0'])
    assert launcher.running_instance(options) == f'http://127.0.0.1:{port}'
    launcher.instance_file().write_text('not valid json')
    assert launcher.running_instance(options) is None


def test_requested_port_does_not_reuse_another_project_or_data_directory(http_service, tmp_path, monkeypatch):
    port, _ = http_service
    monkeypatch.setenv('CML_LAB_DATA', str(tmp_path))
    options = launcher.argument_parser().parse_args(['--port', str(port)])
    # An identical API on the requested port is insufficient: it may own other datasets.
    assert launcher.running_instance(options) is None


    launcher.instance_file().write_text(json.dumps({
        'project': str(tmp_path / 'another project'), 'host': options.host,
        'port': port, 'pid': 10,
    }))
    assert launcher.running_instance(options) is None
    launcher.instance_file().write_text(json.dumps({
        'project': str(launcher.PROJECT), 'host': options.host, 'port': port, 'pid': 10,
    }))
    assert launcher.running_instance(options) == f'http://127.0.0.1:{port}'
    monkeypatch.setenv('CML_LAB_DATA', str(tmp_path / 'different data directory'))
    assert launcher.running_instance(options) is None


@pytest.mark.parametrize('manifest', ['[]', 'null', '42', '"invalid manifest"'])
def test_non_object_server_manifest_does_not_prevent_startup(manifest, tmp_path, monkeypatch):
    monkeypatch.setenv('CML_LAB_DATA', str(tmp_path))
    launcher.instance_file().write_text(manifest)
    options = launcher.argument_parser().parse_args(['--port', '0'])
    assert launcher.running_instance(options) is None


def test_browser_waits_for_ready_health(tmp_path, monkeypatch):
    monkeypatch.setenv('LINEAR_LAB_DATA', str(tmp_path))
    events = []
    responses = iter([None, {'status': 'loading'}, {'status': 'ok'}])

    def read_health(url, timeout):
        value = next(responses)
        events.append(('health', value))
        return value

    monkeypatch.setattr(launcher, 'read_json_url', read_health)
    monkeypatch.setattr(launcher, 'open_browser', lambda url: events.append(('browser', url)))
    launcher.readiness_monitor(SimpleNamespace(started=True, should_exit=False), '127.0.0.1', 9000, True, threading.Event())
    assert [event[0] for event in events] == ['health', 'health', 'health', 'browser']
    assert events[-1][1] == 'http://127.0.0.1:9000'
    assert json.loads(launcher.instance_file().read_text())['port'] == 9000


def test_stopped_startup_does_not_open_browser(monkeypatch):
    monkeypatch.setattr(launcher, 'open_browser', lambda url: pytest.fail('Остановленный сервер не должен открывать браузер'))
    launcher.readiness_monitor(SimpleNamespace(started=False, should_exit=True), '127.0.0.1', 9000, True, threading.Event())


def test_shell_launchers_preserve_arguments_and_project_directory(tmp_path):
    # Реальный sh запускает копию скриптов из папки с пробелами; Python подменяет только run.py.
    project = tmp_path / 'project with spaces'
    project.mkdir()
    for name in ('start.sh', 'start.command'):
        (project / name).write_bytes((PROJECT / name).read_bytes())
    (project / 'run.py').write_text('import os, sys, json; print(json.dumps({"cwd": os.getcwd(), "args": sys.argv[1:]}))')
    output = subprocess.check_output(['/bin/sh', str(project / 'start.command'), '--no-browser', 'a value with spaces'], text=True)
    observed = json.loads(output)
    assert observed == {'cwd': str(project), 'args': ['--no-browser', 'a value with spaces']}


@pytest.mark.skipif(os.name == 'nt', reason='Сигналы настоящего локального сервера проверяем на POSIX')
def test_real_server_ready_reuse_and_foreground_shutdown(tmp_path):
    environment = os.environ.copy()
    environment['CML_LAB_DATA'] = str(tmp_path)
    interpreter = launcher.environment_python(PROJECT)
    if not interpreter.exists():
        interpreter = Path(sys.executable)
    command = [str(interpreter), str(PROJECT / 'run.py'), '--system', '--no-browser', '--port', '0']
    process = subprocess.Popen(command, cwd=PROJECT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    manifest = tmp_path / 'running-server.json'
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail('Сервер завершился до готовности: ' + process.stdout.read())
            if manifest.exists():
                break
            time.sleep(0.1)
        assert manifest.exists(), 'Сервер не достиг готовности за 30 секунд'
        running = json.loads(manifest.read_text())
        port = running['port']
        assert launcher.probe_lab('127.0.0.1', port)
        second = subprocess.run(command[:-1] + [str(port)], cwd=PROJECT, env=environment, capture_output=True, text=True, timeout=15)
        assert second.returncode == 0, second.stdout + second.stderr
        assert 'уже работает' in second.stdout
        assert json.loads(manifest.read_text())['pid'] == process.pid
        # Another launcher copy may use the same code, but must not recover this server's runs.
        other_project = tmp_path / 'other project with spaces'
        other_project.mkdir()
        shutil.copyfile(PROJECT / 'run.py', other_project / 'run.py')
        other_environment = environment.copy()
        other_environment['PYTHONPATH'] = str(PROJECT) + os.pathsep + environment.get('PYTHONPATH', '')
        other_command = [str(interpreter), str(other_project / 'run.py'), '--system',
                         '--no-browser', '--port', str(port)]
        rejected = subprocess.run(other_command, cwd=other_project, env=other_environment,
                                  capture_output=True, text=True, timeout=20)
        rejected_output = rejected.stdout + rejected.stderr
        assert rejected.returncode == 1, rejected_output
        assert 'Папка данных уже используется' in rejected_output
        assert 'Traceback' not in rejected_output
        assert 'Лаборатория уже работает' not in rejected_output
        assert 'Лаборатория остановлена' not in rejected_output
        assert json.loads(manifest.read_text()) == running
        assert launcher.probe_lab('127.0.0.1', port)
        # Diagnosis remains read-only and must not misidentify this copy as its own server.
        shutil.copyfile(PROJECT / 'requirements.txt', other_project / 'requirements.txt')
        diagnosis = subprocess.run(other_command + ['--doctor'], cwd=other_project,
                                   env=other_environment, capture_output=True, text=True, timeout=20)
        assert diagnosis.returncode == 0, diagnosis.stdout + diagnosis.stderr
        assert 'Лаборатория уже работает' not in diagnosis.stdout
        assert json.loads(manifest.read_text()) == running
        process.send_signal(signal.SIGINT)
        stdout, _ = process.communicate(timeout=15)
        assert process.returncode == 0, stdout
        assert 'Готово:' in stdout
        assert 'остановлена' in stdout
        assert not manifest.exists()
        assert not launcher.probe_lab('127.0.0.1', port)
        assert 'Готово:' in (tmp_path / 'launcher.log').read_text(encoding='utf-8')
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def test_cml_data_directory_precedence_and_explicit_legacy_alias(tmp_path, monkeypatch):
    monkeypatch.delenv('CML_LAB_DATA', raising=False)
    monkeypatch.delenv('LINEAR_LAB_DATA', raising=False)
    assert launcher.data_directory() == Path.home() / '.cml-lab'
    monkeypatch.setenv('LINEAR_LAB_DATA', str(tmp_path / 'legacy'))
    assert launcher.data_directory() == tmp_path / 'legacy'
    monkeypatch.setenv('CML_LAB_DATA', str(tmp_path / 'cml'))
    assert launcher.data_directory() == tmp_path / 'cml'
    assert launcher.instance_file().parent == tmp_path / 'cml'


def test_platform_dependency_markers_select_exactly_one_xgboost_distribution():
    linux = dict(launcher.direct_dependencies(PROJECT, platform='linux'))
    windows = dict(launcher.direct_dependencies(PROJECT, platform='win32'))
    mac = dict(launcher.direct_dependencies(PROJECT, platform='darwin'))
    assert linux['xgboost-cpu'] == '3.4.1' and 'xgboost' not in linux
    assert windows['xgboost'] == mac['xgboost'] == '3.4.1'
    assert 'xgboost-cpu' not in windows and 'xgboost-cpu' not in mac
    assert all(item['lightgbm'] == '4.7.0' for item in (linux, windows, mac))
    assert all('linearmodels' not in item for item in (linux, windows, mac))


def test_unknown_dependency_marker_is_reported(tmp_path):
    (tmp_path / 'requirements.txt').write_text('example==1.0; unknown_condition == "x"\n')
    with pytest.raises(launcher.LauncherError, match='условие зависимости'):
        launcher.direct_dependencies(tmp_path)


def test_launcher_rejects_python_311_before_installation_or_service(monkeypatch, capsys):
    monkeypatch.setattr(launcher.sys, 'version_info', (3, 11, 9))
    monkeypatch.setattr(launcher, 'prepare_environment', lambda *args: pytest.fail('Unsupported interpreter must not install dependencies'))
    monkeypatch.setattr(launcher, 'serve', lambda *args: pytest.fail('Unsupported interpreter must not start server'))
    assert launcher.main(['--system', '--no-browser']) == 1
    assert 'Python 3.12' in capsys.readouterr().out


def test_posix_start_script_preserves_arguments_in_folder_with_spaces(tmp_path):
    if os.name == 'nt':
        pytest.skip('POSIX shell scenario')
    project = tmp_path / 'CML project & spaces'
    project.mkdir()
    (project / 'start.sh').write_bytes((PROJECT / 'start.sh').read_bytes())
    (project / 'run.py').write_text('placeholder')
    interpreter = project / '.venv' / 'bin' / 'python'
    interpreter.parent.mkdir(parents=True)
    recorded = project / 'arguments.json'
    interpreter.write_text('#!/bin/sh\nif [ "$1" = "-c" ]; then exit 0; fi\nexec "' + sys.executable + '" -c \'import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps(sys.argv[2:]));\' "' + str(recorded) + '" "$@"\n')
    interpreter.chmod(0o755)
    process = subprocess.run(['/bin/sh', str(project / 'start.sh'), '--no-browser', '--port', '9030'], capture_output=True, text=True, timeout=10)
    assert process.returncode == 0, process.stdout + process.stderr
    assert json.loads(recorded.read_text()) == [str(project / 'run.py'), '--no-browser', '--port', '9030']


@pytest.mark.skipif(os.name == 'nt', reason='Interrupt process groups require POSIX')
def test_posix_launcher_waits_for_interrupt_cleanup(tmp_path):
    project = tmp_path / 'CML launch & spaces'
    project.mkdir()
    (project / 'start.sh').write_bytes((PROJECT / 'start.sh').read_bytes())
    interpreter = project / '.venv' / 'bin' / 'python'
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    (project / 'run.py').write_text(
        'from pathlib import Path\nimport signal, time\n'
        'def stop(signum, frame):\n'
        '    time.sleep(0.3)\n'
        '    Path("stopped").write_text("cleanup complete")\n'
        '    raise SystemExit(0)\n'
        'signal.signal(signal.SIGINT, stop)\n'
        'Path("ready").touch()\n'
        'while True: time.sleep(0.1)\n'
    )
    process = subprocess.Popen(['/bin/sh', str(project / 'start.sh')],
                               cwd=tmp_path, start_new_session=True)
    try:
        deadline = time.monotonic() + 10
        while not (project / 'ready').exists() and time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail('Launcher exited before the child was ready')
            time.sleep(0.05)
        assert (project / 'ready').exists()
        os.killpg(process.pid, signal.SIGINT)
        assert process.wait(timeout=5) == 0
        assert (project / 'stopped').read_text() == 'cleanup complete'
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
