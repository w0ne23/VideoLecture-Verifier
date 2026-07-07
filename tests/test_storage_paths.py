"""storage_service 경로 해석 단위 테스트 — 호스트/컨테이너 이식성 회귀 방지."""
from pathlib import Path

import pytest

import app.services.storage_service as ss


@pytest.fixture
def storage(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, 'LOCAL_STORAGE_DIR', tmp_path)
    return tmp_path


def test_storage_relpath_from_absolute(storage):
    path = storage / 'inputs' / 'abc' / 'v.mp4'
    assert ss.storage_relpath(path) == 'inputs/abc/v.mp4'


def test_storage_relpath_passthrough_relative(storage):
    assert ss.storage_relpath('inputs/abc/v.mp4') == 'inputs/abc/v.mp4'


def test_resolve_relative_joins_storage_dir(storage):
    assert ss.resolve_storage_path('inputs/abc/v.mp4') == storage / 'inputs' / 'abc' / 'v.mp4'


def test_resolve_existing_absolute_kept(storage):
    file = storage / 'a.mp4'
    file.touch()
    assert ss.resolve_storage_path(file) == file


def test_resolve_legacy_absolute_rebased(storage):
    # 다른 호스트에서 저장된 절대경로 행 → 현재 스토리지로 rebase
    target = storage / 'results' / 'xyz'
    target.mkdir(parents=True)
    legacy = '/Users/someone-else/VeriLec/storage/results/xyz'
    assert ss.resolve_storage_path(legacy) == target


def test_resolve_missing_legacy_returns_original(storage):
    legacy = '/Users/someone-else/VeriLec/storage/results/none'
    assert ss.resolve_storage_path(legacy) == Path(legacy)


def test_resolve_none_is_none():
    assert ss.resolve_storage_path(None) is None
    assert ss.resolve_storage_path('') is None


def test_make_file_url_from_relative(storage):
    video = storage / 'inputs' / 'abc' / 'v.mp4'
    video.parent.mkdir(parents=True)
    video.touch()
    assert ss.make_file_url('inputs/abc/v.mp4') == '/files/inputs/abc/v.mp4'


def test_make_file_url_outside_storage_is_empty(storage):
    assert ss.make_file_url('/etc/hosts') == ''
