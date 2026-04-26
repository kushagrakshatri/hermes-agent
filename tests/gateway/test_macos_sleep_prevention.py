from unittest.mock import MagicMock

import gateway.run as gateway_run


def test_macos_sleep_prevention_starts_caffeinate(monkeypatch):
    popen_mock = MagicMock()
    proc = MagicMock()
    proc.pid = 123
    popen_mock.return_value = proc

    monkeypatch.setattr(gateway_run.sys, "platform", "darwin")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_PREVENT_SLEEP", raising=False)
    monkeypatch.setattr(gateway_run.shutil, "which", lambda name: "/usr/bin/caffeinate")
    monkeypatch.setattr(gateway_run.os, "getpid", lambda: 456)
    monkeypatch.setattr(gateway_run.subprocess, "Popen", popen_mock)

    assert gateway_run._start_macos_sleep_prevention() is proc
    popen_mock.assert_called_once()
    assert popen_mock.call_args.args[0] == ["/usr/bin/caffeinate", "-dimsu", "-w", "456"]


def test_macos_sleep_prevention_respects_opt_out(monkeypatch):
    popen_mock = MagicMock()

    monkeypatch.setattr(gateway_run.sys, "platform", "darwin")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("HERMES_GATEWAY_PREVENT_SLEEP", "false")
    monkeypatch.setattr(gateway_run.subprocess, "Popen", popen_mock)

    assert gateway_run._start_macos_sleep_prevention() is None
    popen_mock.assert_not_called()


def test_stop_macos_sleep_prevention_terminates_child():
    proc = MagicMock()
    proc.poll.return_value = None

    gateway_run._stop_macos_sleep_prevention(proc)

    proc.terminate.assert_called_once()
    proc.wait.assert_called_once_with(timeout=2)
