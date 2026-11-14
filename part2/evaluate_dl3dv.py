"""Compare a Part 2 Full/Sparse pair at the fixed held-out camera views."""

import argparse
import csv
from hashlib import file_digest
import json
import math
import os
from pathlib import Path
import sys
from tempfile import mkdtemp
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from fetch_weights import ensure_file


DEFAULT_SCENE = "DL3DV-2"
VGG_FILENAME = "vgg16-397923af.pth"


def sha256(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, document):
    path.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checked_pose(matrix):
    pose = np.asarray(matrix, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("Expected a finite 4x4 camera matrix")
    if not np.allclose(pose[3], [0, 0, 0, 1]):
        raise ValueError("Invalid homogeneous camera matrix")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or np.linalg.det(rotation) <= 0:
        raise ValueError("Camera orientation is not a proper rotation")
    return pose


def load_reference(root, resolution):
    cameras = {}
    for row in read_json(root / "cameras.json"):
        quat = np.asarray(row["cam_quat"], dtype=np.float64)
        norm = np.linalg.norm(quat)
        if quat.shape != (4,) or not np.isfinite(norm) or abs(norm - 1) > 1e-3:
            raise ValueError(f"Invalid quaternion for {row['image_name']}")
        x, y, z, w = quat / norm
        pose = np.eye(4)
        pose[:3, :3] = [
            [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
            [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
            [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
        ]
        pose[:3, 3] = row["cam_trans"]
        name = row["image_name"]
        if name in cameras:
            raise ValueError(f"Duplicate reference camera: {name}")
        cameras[name] = checked_pose(pose)
    global_intrinsics = read_json(root / "intrinsics.json")
    width, height = resolution
    fx, fy = global_intrinsics["fx"] * width, global_intrinsics["fy"] * height
    # The renderer indexes pixel centers from zero. The archive center is
    # expressed relative to the image edges (0.5 means the exact image center).
    cx, cy = global_intrinsics["cx"] * width - 0.5, global_intrinsics["cy"] * height - 0.5
    if not np.isfinite([fx, fy, cx, cy]).all() or min(fx, fy) <= 0:
        raise ValueError("Invalid reference intrinsics")
    return cameras, (fx, fy, cx, cy)


def load_waymo_reference(root, preprocessing_path, resolution, names):
    preprocessing = read_json(preprocessing_path)
    records = preprocessing["preprocessing"]
    if preprocessing["scene"] != "Waymo-405841" or len(records) != 1:
        raise ValueError("Expected the shared Waymo camera calibration from preprocessing.json")
    record = records[0]
    if record["final_size"] != resolution:
        raise ValueError("Waymo calibration does not match the evaluation image size")
    # Preprocessing already stores K in the renderer's zero-based pixel-center convention.
    k = np.asarray(record["final_K_opencv"], dtype=np.float64)
    if k.shape != (3, 3) or not np.isfinite(k).all():
        raise ValueError("Invalid Waymo camera intrinsics")
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    if min(fx, fy) <= 0 or not np.allclose(k, [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]):
        raise ValueError("Expected a pinhole camera calibration for the rectified Waymo images")

    cameras = {}
    reference_files = {"preprocessing.json": preprocessing_path}
    for name in names:
        key = f"gt/{Path(name).stem}.txt"
        pose_path = root / key
        cameras[name] = checked_pose(np.loadtxt(pose_path, dtype=np.float64))
        reference_files[key] = pose_path
    return cameras, (fx, fy, cx, cy), reference_files


def load_scene_reference(scene, work, spec, shared_root):
    names = spec["full"] + spec["test"]
    resolution = spec["resolution"]
    if scene == "Waymo-405841":
        preprocessing = work / spec["source"]["directory"] / "preprocessing.json"
        cameras, intrinsics, files = load_waymo_reference(
            shared_root / "405841" / "FRONT", preprocessing, resolution, names,
        )
    else:
        root = shared_root / scene
        cameras, intrinsics = load_reference(root, resolution)
        files = {name: root / name for name in ("cameras.json", "intrinsics.json")}
    expected = set(names)
    missing = expected.difference(cameras)
    if missing:
        raise ValueError(f"{scene}: missing reference cameras for {sorted(missing)}")
    if scene != "Re10k-1" and cameras.keys() != expected:
        raise ValueError(f"{scene}: the reference cameras do not match the fixed input plan")
    # Re10k has reference poses for the original timeline, including removed duplicates.
    selected = {name: cameras[name] for name in names}
    return selected, intrinsics, {name: sha256(path) for name, path in files.items()}


def fit_similarity(estimated, reference):
    """Fit X_ref = scale * rotation @ X_est + translation to camera centers."""
    x_mean, y_mean = estimated.mean(axis=0), reference.mean(axis=0)
    x, y = estimated - x_mean, reference - y_mean
    covariance = y.T @ x / len(x)
    u, singular_values, vt = np.linalg.svd(covariance)
    if np.linalg.matrix_rank(covariance) < 2:
        raise ValueError("Camera centers do not constrain a 3D similarity alignment")
    signs = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        signs[-1] = -1
    rotation = (u * signs) @ vt
    scale = (singular_values @ signs) / np.mean(np.sum(x * x, axis=1))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid estimated scale")
    return scale, rotation, y_mean - scale * (rotation @ x_mean)


def to_model_camera(reference_c2w, alignment):
    scale, rotation, translation = alignment
    model_c2w = np.eye(4)
    model_c2w[:3, :3] = rotation.T @ reference_c2w[:3, :3]
    model_c2w[:3, 3] = rotation.T @ (reference_c2w[:3, 3] - translation) / scale
    return model_c2w


def rmse(values):
    return float(np.sqrt(np.mean(np.square(values))))


def align_run(folder, scene, mode, spec, plan_hash, reference):
    run_info = (folder / "run-info.txt").read_text(encoding="utf-8").splitlines()
    if f"scene={scene}" not in run_info or f"mode={mode}" not in run_info or "exit_code=0" not in run_info:
        raise ValueError(f"Expected a completed {scene} {mode} run at {folder}")
    if sha256(folder / "input_plan.json") != plan_hash:
        raise ValueError(f"The run used a different image manifest: {folder}")

    metadata = read_json(folder / "model" / "metadata.json")
    config = metadata["config"]
    if [config["width"], config["height"]] != spec["resolution"]:
        raise ValueError(f"Unexpected model resolution: {folder}")
    predictions = {}
    training_names = set(spec[mode])
    for row in metadata["keyframes"]:
        name = row["info"]["name"]
        if row["info"]["is_test"] or name not in training_names or name in predictions:
            raise ValueError(f"Unexpected camera in the {mode} model: {name}")
        predictions[name] = np.linalg.inv(checked_pose(row["Rt"]))
    names = sorted(predictions)
    if len(names) < 3:
        raise ValueError(f"Only {len(names)} cameras were saved for {mode}")
    estimates = np.stack([predictions[name] for name in names])
    targets = np.stack([reference[name] for name in names])
    centers = estimates[:, :3, 3]
    reference_centers = targets[:, :3, 3]
    alignment = fit_similarity(centers, reference_centers)
    scale, rotation, translation = alignment
    errors = np.linalg.norm(scale * (centers @ rotation.T) + translation - reference_centers, axis=1)
    pose_delta = targets[:, :3, :3].transpose(0, 2, 1) @ rotation @ estimates[:, :3, :3]
    angles = np.degrees(np.arccos(np.clip((np.trace(pose_delta, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    spread = np.linalg.svd(reference_centers - reference_centers.mean(axis=0), compute_uv=False)

    rows = [
        {"mode": mode, "image": name, "translation_error": float(error), "rotation_error_deg": float(angle)}
        for name, error, angle in zip(names, errors, angles, strict=True)
    ]
    report = {
        "run": str(folder),
        "input_count": len(training_names),
        "saved_keyframes": len(names),
        "keyframe_fraction": len(names) / len(training_names),
        "reconstruction_seconds": float(metadata["time"]),
        "ate_rmse_reference_units": rmse(errors),
        "rotation_mean_deg": float(angles.mean()),
        "reference_center_svd_second_over_first": float(spread[1] / spread[0]),
        "model_to_reference": {
            "scale": float(scale), "rotation": rotation.tolist(), "translation": translation.tolist(),
        },
    }
    print(f"{mode} | {len(names)} aligned cameras | ATE RMSE {report['ate_rmse_reference_units']:.5f} reference units", flush=True)
    return alignment, rows, report


def ensure_vgg(checkpoint_folder):
    manifest = read_json(Path(__file__).with_name("weights.json"))
    checkpoint = checkpoint_folder / VGG_FILENAME
    record = next(item for item in manifest if item["stage"] == "evaluate")
    ensure_file(checkpoint, record, check_only=False)
    return sha256(checkpoint)


def set_reference_projection(scene, intrinsics, torch, get_projection, focal_to_fov):
    width, height = scene.width, scene.height
    fx, fy, cx, cy = intrinsics
    scene.update_f(torch.tensor(fx, dtype=torch.float32, device="cuda"))
    projection = get_projection(0.01, 10000.0, focal_to_fov(fx, width), focal_to_fov(fy, height))
    projection[0, 2] = 2 * (cx + 0.5) / width - 1
    projection[1, 2] = 2 * (cy + 0.5) / height - 1
    scene.projection_matrix = projection.T.contiguous().cuda()
    scene.tanfovx, scene.tanfovy = width / (2 * fx), height / (2 * fy)


def render_and_score(model, alignment, cameras, intrinsics, truth, names, output, perceptual):
    import torch
    from fused_ssim import fused_ssim
    from scene.scene_model import SceneModel
    from utils import focal2fov, getProjectionMatrix, psnr

    rows = []
    with torch.no_grad():
        scene = SceneModel.from_scene(str(model), SimpleNamespace())
        set_reference_projection(scene, intrinsics, torch, getProjectionMatrix, focal2fov)
        # Use the evaluator's synchronous hierarchy update at each requested view.
        scene.hierarchy.update_cut = scene.hierarchy.update_cut_for_test
        output.mkdir()
        for index, name in enumerate(names, start=1):
            view = np.linalg.inv(to_model_camera(cameras[name], alignment)).T.copy()
            view = torch.as_tensor(view, dtype=torch.float32, device="cuda")
            rendered = scene.render(scene.width, scene.height, view, 1.0)["render"]
            if not torch.isfinite(rendered).all().item():
                raise ValueError(f"Non-finite render for {name}")
            pixels = rendered.clamp(0, 1).mul(255).round().to(torch.uint8)
            Image.fromarray(pixels.permute(1, 2, 0).cpu().numpy()).save(output / name)
            predicted = (pixels[None].float() / 255).contiguous()
            expected = torch.from_numpy(truth[name]).to(device="cuda", dtype=torch.float32)
            expected = (expected.permute(2, 0, 1)[None] / 255).contiguous()
            psnr_value = psnr(predicted, expected)
            ssim = fused_ssim(predicted, expected, padding="same", train=False).item()
            lpips = perceptual(predicted, expected, normalize=True).item()
            if not np.isfinite([ssim, lpips]).all():
                raise ValueError(f"Non-finite image metric for {name}")
            rows.append({"image": name, "psnr_db": psnr_value, "ssim": ssim, "lpips_vgg": lpips})
            if index % 10 == 0 or index == len(names):
                print(f"{output.name}: rendered and scored {index}/{len(names)}", flush=True)
    return rows


def save_comparison(truth, names, output):
    selection = [names[i] for i in np.linspace(0, len(names) - 1, min(6, len(names)), dtype=int)]
    width, height = Image.fromarray(truth[selection[0]]).size
    margin, heading, label = 12, 24, 20
    row_height = height + label + margin
    sheet = Image.new(
        "RGB", (3 * width + 4 * margin, 2 * margin + heading + len(selection) * row_height), "#f4f4f4",
    )
    draw = ImageDraw.Draw(sheet)
    columns = [margin + i * (width + margin) for i in range(3)]
    for x, name in zip(columns, ("Test photo", "Full", "Sparse"), strict=True):
        draw.text((x, margin), name, fill="#202020")
    for index, name in enumerate(selection):
        top = margin + heading + index * row_height
        draw.text((margin, top), name, fill="#202020")
        sheet.paste(Image.fromarray(truth[name]), (columns[0], top + label))
        for x, mode in zip(columns[1:], ("full", "sparse"), strict=True):
            with Image.open(output / mode / name) as render:
                sheet.paste(render, (x, top + label))
    sheet.save(output / "comparison.png")


def main():
    plan_file = Path(__file__).with_name("input_plan.json")
    plan = read_json(plan_file)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=tuple(plan["scenes"]), default=DEFAULT_SCENE)
    parser.add_argument("--only", choices=("full", "sparse"), help="Evaluate one completed run when its counterpart failed")
    parser.add_argument("full_run", help="Full run folder name, for example run.ABC123")
    parser.add_argument("sparse_run", nargs="?", help="Sparse run folder name, for example run.DEF456")
    args = parser.parse_args()
    if args.only:
        if args.sparse_run is not None:
            parser.error("Specify a single run folder with --only")
        requested = {args.only: args.full_run}
    else:
        if args.sparse_run is None:
            parser.error("Specify Full and Sparse run folders, or choose --only")
        requested = {"full": args.full_run, "sparse": args.sparse_run}
    plan_hash = sha256(plan_file)
    work = Path(os.environ["P2_WORK"]).resolve(strict=True)
    spec = plan["scenes"][args.scene]
    names = spec["test"]
    data_root = Path(os.environ["P2_DATA_ROOT"]).resolve(strict=True)
    reference, intrinsics, reference_hashes = load_scene_reference(args.scene, work, spec, data_root)
    truth_dir = work / "datasets" / plan["version"] / args.scene / "test" / "images"
    if {path.name for path in truth_dir.iterdir()} != set(names):
        raise ValueError("The test image directory does not match the fixed list")
    truth = {}
    for name in names:
        image_path = truth_dir / name
        if sha256(image_path) != spec["image_sha256"][name]:
            raise ValueError(f"Unexpected test image bytes: {image_path}")
        with Image.open(image_path) as image:
            if list(image.size) != spec["resolution"]:
                raise ValueError(f"Unexpected test image dimensions: {image_path}")
            truth[name] = np.array(image.convert("RGB"), dtype=np.uint8)

    sys.path.insert(0, str(Path(os.environ["P2_SRC"]).resolve(strict=True)))
    scene_runs = work / "runs" / "part2" / args.scene
    pairs = {}
    for mode, name in requested.items():
        if name != Path(name).name or not name.startswith("run."):
            parser.error(f"Expected a run directory name for {mode}")
        pairs[mode] = (scene_runs / mode / name).resolve(strict=True)
    parent = scene_runs / "eval"
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(mkdtemp(prefix="run.", dir=parent))
    print(f"Evaluation folder: {output}", flush=True)
    aligned = {
        mode: align_run(run, args.scene, mode, spec, plan_hash, reference)
        for mode, run in pairs.items()
    }
    reports = {mode: record[2] for mode, record in aligned.items()}
    pose_errors = {mode: {row["image"]: row["translation_error"] for row in record[1]}
                   for mode, record in aligned.items()}
    common_names = sorted(pose_errors["full"].keys() & pose_errors["sparse"].keys()) if len(pairs) == 2 else []
    for mode in reports:
        reports[mode]["common_keyframe_ate_rmse_reference_units"] = (
            rmse([pose_errors[mode][name] for name in common_names]) if common_names else None
        )
    summary = {
        "scene": args.scene, "input_plan_sha256": plan_hash,
        "test_count": len(names), "alignment_common_keyframes": common_names,
        "pose_reference": "Course-supplied camera coordinates in reference units",
        "reference_sha256": reference_hashes,
        "evaluation_script_sha256": sha256(Path(__file__)),
        "metric_setup": "Saved 8-bit RGB, full image, fused SSIM same padding, LPIPS VGG v0.1",
        "runs": reports,
    }
    write_csv(output / "pose-errors.csv", [row for record in aligned.values() for row in record[1]])
    write_json(output / "alignment.json", summary)

    import torch
    import lpips

    if not torch.cuda.is_available():
        raise RuntimeError("The CUDA GPU is unavailable for rendering")
    summary["lpips_vgg16_sha256"] = ensure_vgg(Path(torch.hub.get_dir()) / "checkpoints")
    perceptual = lpips.LPIPS(net="vgg", version="0.1", verbose=False).to("cuda").eval()
    for mode, run in pairs.items():
        started = perf_counter()
        rows = render_and_score(run / "model", aligned[mode][0], reference, intrinsics, truth,
                                names, output / mode, perceptual)
        write_csv(output / f"{mode}-per-view.csv", rows)
        report = reports[mode]
        report["evaluation_seconds"] = perf_counter() - started
        for metric in ("psnr_db", "ssim", "lpips_vgg"):
            average = float(np.mean([row[metric] for row in rows]))
            report[metric] = average if math.isfinite(average) else "inf"
        print(f"{mode}: PSNR {report['psnr_db']}, SSIM {report['ssim']:.5f}, "
              f"LPIPS {report['lpips_vgg']:.5f}", flush=True)
    if len(pairs) == 2:
        save_comparison(truth, names, output)
    summary["evaluation_scope"] = args.only or "full_and_sparse"
    columns = (
        "input_count", "saved_keyframes", "keyframe_fraction", "reconstruction_seconds",
        "ate_rmse_reference_units", "rotation_mean_deg", "psnr_db", "ssim", "lpips_vgg",
    )
    write_csv(output / "summary.csv", [
        {"mode": mode, **{field: report[field] for field in columns}} for mode, report in reports.items()
    ])
    write_json(output / "summary.json", summary)
    if args.only:
        print(f"Evaluation OK: {len(names)} test views for {args.only} only", flush=True)
    else:
        print(f"Evaluation OK: {len(names)} common test views per model", flush=True)
    print(f"Evaluation folder: {output}", flush=True)


if __name__ == "__main__":
    main()
