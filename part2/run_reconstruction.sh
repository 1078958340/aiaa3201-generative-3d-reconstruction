#!/usr/bin/env bash
set -e -o pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: bash run_reconstruction.sh SCENE full|sparse" >&2
  exit 2
fi
scene="$1"
mode="$2"
if [ "$mode" != full ] && [ "$mode" != sparse ]; then
  echo "Choose full or sparse" >&2
  exit 2
fi

package=$(cd -- "$(dirname -- "$0")" && pwd -P)
source "$package/use-part2.sh"
expected_i3dgs=cf4d5b9762359a1d6de76fb9abf7b3dc764c1a42
if [ -e "$P2_SRC/.git" ] && [ "$(git -C "$P2_SRC" rev-parse HEAD)" != "$expected_i3dgs" ]; then
  echo "Check out i3dgs $expected_i3dgs before reconstruction." >&2
  exit 2
fi
python "$package/fetch_weights.py" --check --stage reconstruct
python -B "$package/prepare_inputs.py" "$scene"
cd -- "$P2_SRC"

source_dir="$P2_WORK/datasets/part2-inputs-v1/$scene/$mode"
run_parent="$P2_WORK/runs/part2/$scene/$mode"
mkdir -p "$run_parent"
RUN=$(mktemp -d "$run_parent/run.XXXXXX")
echo "Run folder: $RUN"
run_started=$(date +%s)
finish_run() {
  run_status=$?
  run_ended=$(date +%s)
  printf 'exit_code=%s\nwall_seconds=%s\n' "$run_status" "$((run_ended - run_started))" >> "$RUN/run-info.txt"
}
trap finish_run EXIT

train_args=(
  -s "$source_dir" -m "$RUN/model"
  --viewer_mode none --test_hold -1 --downsampling 1
  --num_keyframes_miniba_bootstrap 4 --num_pts_miniba_bootstrap 1000
  --num_iterations 30 --da3_model_dir "$P2_WORK/weights/DA3-BASE"
)

cp -- "$package/input_plan.json" "$RUN/input_plan.json"
cp -- "$package/run_reconstruction.sh" "$RUN/runner.sh"
{
  printf 'scene=%s\nmode=%s\nseed=0\n' "$scene" "$mode"
  printf 'i3dgs_commit=%s\n' "$expected_i3dgs"
  printf 'bootstrap_compat=mask_unused_camera_slots_v1\n'
  date -u +%FT%TZ
  printf 'command='
  printf '%q ' python -u - "${train_args[@]}"
  printf '\n'
  python -c 'import platform, torch; print("Python", platform.python_version(), "PyTorch", torch.__version__, "CUDA", torch.version.cuda)'
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv
  sha256sum "$RUN/input_plan.json" "$RUN/runner.sh" "$package/weights.json"
} > "$RUN/run-info.txt"

python -u - "${train_args[@]}" 2>&1 <<'PY' | tee "$RUN/train.log"
import os
import runpy
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import huggingface_hub

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import torch

hub = Path(torch.hub.get_dir())
weights = Path(os.environ["P2_WORK"]) / "weights"
hf_download = huggingface_hub.hf_hub_download


def download_or_use_local(repo_id, filename, *args, **kwargs):
    if repo_id == "gberton/MegaLoc" and filename in ("model.safetensors", "config.json"):
        return str(weights / "MegaLoc" / filename)
    return hf_download(repo_id, filename, *args, **kwargs)


huggingface_hub.hf_hub_download = download_or_use_local
megaloc_source = hub / "gmberton_MegaLoc_main"
megaloc_code = (megaloc_source / "megaloc_model.py").read_text(encoding="utf-8")
original_einsum = 'torch.einsum("bdn,bkn->bdk", f, p)'
fp32_einsum = 'torch.einsum("bdn,bkn->bdk", f.float(), p.float())'
if megaloc_code.count(original_einsum) != 1:
    raise RuntimeError("Unexpected MegaLoc source in " + str(megaloc_source))

with TemporaryDirectory(prefix="megaloc-", dir=os.environ["P2_WORK"]) as directory:
    megaloc_repo = Path(directory)
    for name in ("hubconf.py", "LICENSE"):
        copyfile(megaloc_source / name, megaloc_repo / name)
    # Half-precision features meet FP32 transport probabilities here.
    (megaloc_repo / "megaloc_model.py").write_text(
        megaloc_code.replace(original_einsum, fp32_einsum), encoding="utf-8"
    )

    repositories = {
        "verlab/accelerated_features": Path(os.environ["P2_SRC"]) / "submodules/xfeat",
        "gmberton/MegaLoc": megaloc_repo,
    }
    hub_load = torch.hub.load

    def load_from_disk(repo, entry, *args, **kwargs):
        if repo in repositories:
            repo = str(repositories[repo])
            kwargs["source"] = "local"
        return hub_load(repo, entry, *args, **kwargs)

    torch.hub.load = load_from_disk
    print("MegaLoc: FP32 feature aggregation enabled", flush=True)

    from poses.pose_initializer import PoseInitializer
    from scene.scene_model import SceneModel

    original_bootstrap = PoseInitializer.initialize_bootstrap

    def bootstrap_with_available_views(self, desc_kpts_list, keyframe_dicts, ordered_bootstrap):
        camera_count = len(desc_kpts_list)
        solver = self.miniba_bootstrap
        unused_slots = solver.n_cams - camera_count
        if unused_slots == 0:
            return original_bootstrap(self, desc_kpts_list, keyframe_dicts, ordered_bootstrap)
        if camera_count < 2:
            print("Bootstrap needs at least two selected views.", flush=True)
            return None, None, None
        if unused_slots < 0:
            raise ValueError("Bootstrap selected more views than the solver supports")

        # The captured MiniBA solver expects the configured number of camera slots.
        # Its CUDA projection kernel skips negative camera indices.
        def solve(rts, focal, xyz, center, uv, camera_indices):
            identity = torch.eye(4, dtype=rts.dtype, device=rts.device)
            padded_rts = torch.cat((rts, identity.expand(unused_slots, -1, -1)), dim=0)
            padded_uv = torch.nn.functional.pad(uv, (0, 0, 0, unused_slots), value=-1)
            padded_indices = torch.nn.functional.pad(camera_indices, (0, unused_slots), value=-1)
            poses, *outputs = solver(padded_rts, focal, xyz, center, padded_uv, padded_indices)
            return poses[:camera_count], *outputs

        # build_problem divides the point budget by the actual number of selected views.
        original_ids = self._bp_aligned
        self._bp_aligned = torch.arange(
            self.num_pts_miniba_bootstrap // camera_count, device=original_ids.device, dtype=original_ids.dtype,
        )
        self.miniba_bootstrap = solve
        print(f"Bootstrap: {camera_count}/{solver.n_cams} selected camera slots; unused slots masked.", flush=True)
        try:
            return original_bootstrap(self, desc_kpts_list, keyframe_dicts, ordered_bootstrap)
        finally:
            self._bp_aligned = original_ids
            self.miniba_bootstrap = solver

    PoseInitializer.initialize_bootstrap = bootstrap_with_available_views

    def skip_lpips_during_reconstruction(self):
        pass

    SceneModel._ensure_lpips = skip_lpips_during_reconstruction
    runpy.run_path("train.py", run_name="__main__")
PY

python - "$RUN/model" <<'PY' 2>&1 | tee -a "$RUN/train.log"
from pathlib import Path
import sys

from plyfile import PlyData
from dataloaders.read_write_model import read_images_binary_poses

model = Path(sys.argv[1])
gaussians = PlyData.read(str(model / "hierarchy.ply"))["vertex"].count
poses = read_images_binary_poses(str(model / "sparse/0/images.bin"))
if gaussians <= 0 or len(poses) < 2:
    raise SystemExit("Reconstruction output is empty or has fewer than two poses")

print(f"Reconstruction OK | poses={len(poses)} | Gaussian nodes={gaussians}")
print(f"Model folder: {model}")
PY
