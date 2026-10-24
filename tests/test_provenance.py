import pytest
from claimreplay.provenance import digest,relative_path,verify,save


def test_portable_hash_binding(tmp_path):
    p=tmp_path/'input.txt';p.write_text('synthetic input')
    assert verify(tmp_path,{'path':'input.txt','sha256':digest(p)})==p
    with pytest.raises(ValueError,match='SHA-256'):
        verify(tmp_path,{'path':'input.txt','sha256':'invalid'})
    with pytest.raises(ValueError,match='Missing'):
        verify(tmp_path,{'path':'absent.txt','sha256':'invalid'})


def test_relative_path_rejects_traversal():
    for value in ('../input','/input',''):
        with pytest.raises(ValueError):relative_path(value)


def test_manifest_write(tmp_path):
    import json
    p=tmp_path/'manifest.json';save(p,{'version':1})
    assert json.loads(p.read_text())=={'version':1}
