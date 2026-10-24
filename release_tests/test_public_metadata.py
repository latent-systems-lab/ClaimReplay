import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def test_active_license():
    assert 'Version 2.0, January 2004' in (ROOT/'LICENSE').read_text()
    assert 'license = "Apache-2.0"' in (ROOT/'pyproject.toml').read_text()


def test_repository_metadata():
    for name in ('README.md','CITATION.cff','pyproject.toml'):
        assert 'https://github.com/latent-systems-lab/ClaimReplay' in (ROOT/name).read_text(encoding='utf-8')


def test_model_input_bindings():
    cfg=json.loads((ROOT/'configs/inference.json').read_text())
    for key in ('priority_prompt','judge_prompt','priority_protocol','judge_protocol','native_config'):
        r=cfg[key];assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256']


def test_only_consumer_directories_are_tracked():
    allowed={'claimreplay','tests','release_tests','examples','docs','configs','prompts','scripts','build_wheels','dist','.github'}
    for p in ROOT.iterdir():
        if p.is_dir() and not p.name.startswith('.'):
            assert p.name in allowed or p.name.endswith('.egg-info') or p.name=='build'
