"""Запускатель: безопасные аргументы, настоящий HTTP readiness и отсутствие второго сервера."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import signal
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
    environment['LINEAR_LAB_DATA'] = str(tmp_path)
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
