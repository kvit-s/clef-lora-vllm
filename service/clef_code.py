"""Find the downloaded Clef release and load Cloudflare's joint_schema_model.py from it.

The prompt layout (encode_record), the scoring network (JointSchemaHead) and the answer format
(systemone_answer) are Cloudflare's code, published next to Clef's weights. This repository does
not copy that file. Every script takes the release directory, from --release DIR or the
CLEF_RELEASE environment variable, and loads joint_schema_model.py from there, so the code that
runs is the code Clef was released with. fetch_release.py downloads the files needed.

Call use_release() before importing reuse, service or anything else that imports
joint_schema_model. The module is loaded from its file path and registered under its own name;
the release directory is not added to sys.path, so no other file in it can be imported by accident.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path

# Revisions this repository was built and tested against.
CLEF_REPO = "Cloudflare/clef"
CLEF_REVISION = "2f3de3dd85f379784083b0814d997ab627200f0c"
BASE_REPO = "Qwen/Qwen3.8-27B"
BASE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
# joint_schema_model.py at CLEF_REVISION
TESTED_CODE_SHA256 = "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3"


def release_dir(argv: list[str] | None = None) -> Path:
    """The release directory named by --release in argv (default sys.argv), or by CLEF_RELEASE."""
    argv = sys.argv[1:] if argv is None else argv
    value = None
    for index, arg in enumerate(argv):
        if arg == "--release" and index + 1 < len(argv):
            value = argv[index + 1]
        elif arg.startswith("--release="):
            value = arg.split("=", 1)[1]
    value = value or os.environ.get("CLEF_RELEASE")
    if not value:
        raise SystemExit("give the Clef release directory with --release DIR or CLEF_RELEASE=DIR "
                         "(service/fetch_release.py downloads the files it needs)")
    path = Path(value).expanduser().resolve()
    if not (path / "joint_schema_model.py").is_file():
        raise SystemExit(f"{path} has no joint_schema_model.py; download it with service/fetch_release.py")
    return path


def use_release(argv: list[str] | None = None) -> Path:
    """Load joint_schema_model.py from the release directory and return that directory."""
    path = release_dir(argv)
    if "joint_schema_model" in sys.modules:
        return path
    code = path / "joint_schema_model.py"
    digest = hashlib.sha256(code.read_bytes()).hexdigest()
    if digest != TESTED_CODE_SHA256:
        print(f"warning: {code} differs from the version this repository was tested with "
              f"({CLEF_REPO} at {CLEF_REVISION[:8]})", file=sys.stderr)
    spec = importlib.util.spec_from_file_location("joint_schema_model", code)
    module = importlib.util.module_from_spec(spec)
    sys.modules["joint_schema_model"] = module  # dataclasses look their module up while being defined
    spec.loader.exec_module(module)
    return path
