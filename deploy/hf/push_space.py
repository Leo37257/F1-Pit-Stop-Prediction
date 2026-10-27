"""Push the F1 Pit Predictor to Hugging Face Spaces as a Docker space.

Requires `huggingface_hub` and a logged-in HF account (run `hf auth login`).

    python deploy/hf/push_space.py                       # <your HF username>/f1-pit-predictor
    python deploy/hf/push_space.py --space you/some-name  # or HF_SPACE_ID=you/some-name
    python deploy/hf/push_space.py --recreate             # allow deleting a non-Docker space of that name first
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

import argparse
import os

ROOT = Path(__file__).resolve().parent.parent.parent

api = HfApi()
parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--space", default=os.environ.get("HF_SPACE_ID"),
                    help="owner/name of the Space (default: <logged-in user>/f1-pit-predictor)")
parser.add_argument("--recreate", action="store_true",
                    help="if a Space with this name exists but is not a Docker Space, delete and recreate it")
args = parser.parse_args()
# Never default to someone else's Space: deploy to the account that is logged in unless told otherwise.
REPO_ID = args.space or f"{api.whoami()['name']}/f1-pit-predictor"
print(f"Target Space: {REPO_ID}")

# Confirm the old space is static — if so, delete and recreate as docker.
# If it's already docker, just upload over it.
try:
    info = api.space_info(REPO_ID)
    current_sdk = getattr(info, "sdk", None)
except HfHubHTTPError:
    info = None
    current_sdk = None

if info is not None and current_sdk != "docker":
    if not args.recreate:
        sys.exit(f"{REPO_ID} exists with sdk={current_sdk!r}. Re-run with --recreate to delete and recreate it.")
    print(f"Existing space sdk={current_sdk!r}; deleting to recreate as docker…")
    api.delete_repo(repo_id=REPO_ID, repo_type="space")
    info = None

if info is None:
    print(f"Creating Docker space {REPO_ID}…")
    api.create_repo(repo_id=REPO_ID, repo_type="space", space_sdk="docker", exist_ok=True)
else:
    print(f"Space {REPO_ID} already exists as docker; uploading over it.")

with tempfile.TemporaryDirectory() as tmp:
    staging = Path(tmp)
    deploy_dir = ROOT / "deploy" / "hf"
    # Dockerfile and README must live at the space root; requirements.txt is
    # COPY'd at the path the Dockerfile expects (deploy/hf/requirements.txt).
    shutil.copy(deploy_dir / "Dockerfile", staging / "Dockerfile")
    shutil.copy(deploy_dir / "README.md", staging / "README.md")
    (staging / "deploy" / "hf").mkdir(parents=True, exist_ok=True)
    shutil.copy(deploy_dir / "requirements.txt", staging / "deploy" / "hf" / "requirements.txt")
    shutil.copytree(ROOT / "src", staging / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(ROOT / "models" / "champion", staging / "models" / "champion")
    shutil.copytree(ROOT / "dashboard", staging / "dashboard")

    print(f"Uploading {sum(1 for _ in staging.rglob('*') if _.is_file())} files from {staging}…")
    api.upload_folder(
        folder_path=str(staging),
        repo_id=REPO_ID,
        repo_type="space",
        commit_message="Deploy full-stack FastAPI + dashboard with CSV batch inference",
    )

print(f"\n✓ Done. Visit https://huggingface.co/spaces/{REPO_ID}")
sys.exit(0)
