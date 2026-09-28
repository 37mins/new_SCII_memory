#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
REPO_FILES = {
    REPO / "data/train.parquet": "7c8f6b07eacef5e06963b475c3c36e90196da870fae095b12c2ea0f7faf10912",
    REPO / "data/test.parquet": "4dbe5e3ff68df3fbf76e91284b28b06c52bf7495193013a4bf4ed7ca417a0cd7",
    REPO / "agent_system/environments/env_package/sc2_online_restored/SC2ArenaDyStrategy/knowledge/BO/replay_observations_Flat128_d4_TvT_Macro_g2_Result.Victory_twobase.json": "304b1f4ff12fe30c5e6d5f7b8eaef8ed6823ca65afed3a3706cae10e3639c494",
}
SC2_FILES = {
    "Versions/Base75689/SC2_x64": "115f65a49191c504f6336f7d5b7818b23dbba3d7a45cacf9142c86c959abaa41",
    "Maps/Melee/Flat128.SC2Map": "3916dbd2bb662a6e9c9bd27089906630c8b0a42fdbab51fc4401cf1df7b9b212",
}
RUSH_BO = REPO / "data/rush_bo.json"
RUSH_BO_SHA256 = "e21c5a8518bc7c16a408ba7e245d0fd26c1a8aa6ab9d2410cb4c41c599d92f14"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def model_files(model: Path, manifest: Path):
    for line in manifest.read_text().splitlines():
        if not line:
            continue
        expected, name = line.split(maxsplit=1)
        yield model / name, expected


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the immutable inputs used by the released run.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument(
        "--model-manifest",
        type=Path,
        default=REPO / "configs/qwen3-4b.sha256",
    )
    parser.add_argument("--sc2", required=True, type=Path)
    parser.add_argument("--bo-profile", choices=("canonical", "rush", "fivebo"), default="canonical")
    parser.add_argument("--bo-key", choices=("g1", "g2", "g3", "g4", "g5"))
    parser.add_argument("--bo", type=Path)
    args = parser.parse_args()

    expected_files = dict(REPO_FILES)
    expected_files.update(model_files(REPO, REPO / "configs/system-v1.sha256"))
    if args.bo_profile == "rush":
        expected_files[RUSH_BO] = RUSH_BO_SHA256
    if args.bo_profile == "fivebo":
        expected_files[args.bo] = json.loads((REPO / "configs/five-bo.json").read_text())[args.bo_key][1]
    expected_files.update(model_files(args.model, args.model_manifest))
    expected_files.update({args.sc2 / name: value for name, value in SC2_FILES.items()})

    for path, expected in expected_files.items():
        actual = digest(path)
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")
        print(f"ok  {path}")


if __name__ == "__main__":
    main()
