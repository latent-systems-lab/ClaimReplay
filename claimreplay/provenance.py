"""Portable, hash-checked input bindings for optional inference."""
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8')


def relative_path(value):
    path=Path(value)
    if (PurePosixPath(value).is_absolute() or PureWindowsPath(value).anchor
            or '..' in PureWindowsPath(value).parts or not path.parts):
        raise ValueError('Input bindings must use safe relative paths')
    return path


def verify(root,row):
    path=Path(root)/relative_path(row['path'])
    if not path.is_file():raise ValueError(f"Missing input asset: {row['path']}")
    if digest(path)!=row['sha256']:raise ValueError(f"SHA-256 mismatch: {row['path']}")
    return path
