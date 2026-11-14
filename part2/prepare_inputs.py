import argparse
from hashlib import file_digest
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory


def read_plan(path):
    return json.loads(path.read_text(encoding="utf-8"))


def image_groups(plan, scene):
    spec = plan["scenes"][scene]
    groups = {group: spec[group] for group in ("full", "sparse", "test")}
    timeline = sorted(groups["full"] + groups["test"])
    if len(timeline) != len(set(timeline)) or set(timeline) != set(spec["image_sha256"]):
        raise ValueError(f"{scene}: Full and Test lists disagree with the image manifest")
    if groups["full"] != sorted(groups["full"]) or groups["test"] != timeline[::8]:
        raise ValueError(f"{scene}: image ordering or the fixed test split changed")
    if groups["sparse"] != timeline[plan["start_index"] :: spec["stride"]]:
        raise ValueError(f"{scene}: the sparse list does not match the fixed sampling interval")
    if not set(groups["sparse"]).issubset(groups["full"]):
        raise ValueError(f"{scene}: Sparse must be a subset of Full")
    return groups


def source_directories(work, spec, data_root):
    source = spec["source"]
    if source["kind"] == "shared":
        rgb = data_root / Path(source["directory"]).relative_to("project4_3d")
        return {"full": rgb, "test": rgb}
    staged = work / source["directory"]
    return {"full": staged / "train/images", "test": staged / "test/images"}


def verify_source_images(groups, expected_hashes, directories):
    sources = {}
    for group in ("full", "test"):
        for name in groups[group]:
            if Path(name).name != name or Path(name).suffix.lower() != ".png":
                raise ValueError(f"Invalid image filename: {name}")
            image = (directories[group] / name).resolve(strict=True)
            with image.open("rb") as handle:
                digest = file_digest(handle, "sha256").hexdigest()
            if digest != expected_hashes[name]:
                raise ValueError(f"Image differs from the verified Part 1 version: {image}")
            sources[name] = image
    return sources


def stage_inputs(target, groups, sources):
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=f".{target.name}-", dir=target.parent) as temporary:
            staging = Path(temporary)
            for group, names in groups.items():
                images = staging / group / "images"
                images.mkdir(parents=True)
                for name in names:
                    (images / name).symlink_to(sources[name])
            staging.rename(target)

    for group, names in groups.items():
        case = target / group
        images = case / "images"
        if (case / "sparse").exists():
            raise ValueError(f"Unexpected camera data in {case}")
        if {path.name for path in images.iterdir()} != set(names):
            raise ValueError(f"Unexpected image set in {images}")
        for name in names:
            if not (images / name).samefile(sources[name]):
                raise ValueError(f"Incorrect image link: {images / name}")


def main():
    plan = read_plan(Path(__file__).with_name("input_plan.json"))
    parser = argparse.ArgumentParser(description="Check course images and prepare the fixed Part 2 inputs.")
    parser.add_argument("scene", choices=tuple(plan["scenes"]))
    args = parser.parse_args()
    work = Path(os.environ["P2_WORK"]).resolve(strict=True)
    spec = plan["scenes"][args.scene]
    groups = image_groups(plan, args.scene)
    data_root = Path(os.environ["P2_DATA_ROOT"]).resolve(strict=True)
    sources = verify_source_images(groups, spec["image_sha256"], source_directories(work, spec, data_root))
    target = work / "datasets" / plan["version"] / args.scene
    stage_inputs(target, groups, sources)
    counts = " | ".join(f"{group.title()} {len(names)}" for group, names in groups.items())
    print(f"Inputs ready: {args.scene} | {counts}")
    print(f"Input folder: {target}")


if __name__ == "__main__":
    main()
