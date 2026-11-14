"""CPU-only, reproducible audit and preparation of the course Part 1 RGB data.

Usage: python scripts/prepare_data.py [--root PROJECT_ROOT] [--archive FILE]
Raw bytes are preserved. No camera interpretation, undistortion, or resizing occurs.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import traceback
import uuid
import zipfile

from PIL import Image, ImageDraw, ImageFont, ImageOps, __version__ as pillow_version

SCENES = {
    'Waymo-405841': '405841/FRONT/rgb',
    'DL3DV-2': 'DL3DV-2/rgb',
    'Re10k-1': 'Re10k-1/images',
}
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
SPLIT_RULE = 'per scene: exact RGB pixel SHA256 deduplication keeps first natural-sorted frame; unique zero-based ordinal % 8 == 0 -> test; all other unique frames -> train; duplicates -> excluded_duplicate'
RESERVED = {'CON', 'PRN', 'AUX', 'NUL'} | {f'{stem}{n}' for stem in ('COM', 'LPT') for n in range(1, 10)}


def sha_file(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def natural_key(name: str):
    return tuple((1, int(part)) if part.isdigit() else (0, part.casefold())
                 for part in re.split(r'(\d+)', name))


def frame_id(name: str):
    matches = re.findall(r'\d+', Path(name).stem)
    return int(matches[-1]) if matches else None


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def validate_zip_members(archive: zipfile.ZipFile, raw: Path):
    """Reject Windows aliases, traversal, symlinks, special files, duplicate paths."""
    seen = set()
    total = 0
    validated = []
    for item in archive.infolist():
        name = item.filename
        if not name or '\\' in name or name.startswith('/') or '\x00' in name:
            raise ValueError(f'Unsafe ZIP member: {name!r}')
        parts = name.rstrip('/').split('/')
        if any(p in ('', '.', '..') or ':' in p or p.endswith((' ', '.'))
               or p.split('.')[0].upper() in RESERVED for p in parts):
            raise ValueError(f'Unsafe or Windows-aliased ZIP member: {name!r}')
        normalized = '/'.join(parts).casefold()
        if normalized in seen:
            raise ValueError(f'Duplicate ZIP path (case-insensitive): {name!r}')
        seen.add(normalized)
        kind = stat.S_IFMT(item.external_attr >> 16)
        if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise ValueError(f'Symlink/special ZIP member rejected: {name!r}')
        target = raw.joinpath(*PurePosixPath(name).parts)
        if not target.resolve().is_relative_to(raw.resolve()):
            raise ValueError(f'ZIP traversal rejected: {name!r}')
        total += item.file_size
        if total > 20 * 1024**3:
            raise ValueError('Archive exceeds the 20 GiB safety bound')
        validated.append((item, target))
    return validated, total


def extract_and_verify(root: Path, archive_path: Path, report: dict):
    if not archive_path.is_file():
        raise FileNotFoundError(f'Archive not downloaded: {archive_path}')
    raw = root / 'data/raw'
    raw.mkdir(parents=True, exist_ok=True)
    report['archive'] = {
        'path': rel(archive_path, root) if archive_path.is_relative_to(root) else str(archive_path),
        'bytes': archive_path.stat().st_size,
        'sha256': sha_file(archive_path),
        'publisher_checksum_available': False,
    }
    checked = 0
    extracted = 0
    with zipfile.ZipFile(archive_path) as archive:
        members, total = validate_zip_members(archive, raw)
        report['archive'].update({'members': len(members), 'uncompressed_bytes': total})
        for item, target in members:
            if item.is_dir():
                if target.exists() and not target.is_dir():
                    raise ValueError(f'Raw directory conflicts with file: {target}')
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                digest = hashlib.sha256()
                if target.exists():
                    if not target.is_file() or target.is_symlink():
                        raise ValueError(f'Unsafe existing raw target: {target}')
                    if target.stat().st_size != item.file_size:
                        raise ValueError(f'Existing raw bytes changed (will not overwrite): {target}')
                    with archive.open(item) as source:
                        while chunk := source.read(1024 * 1024):
                            digest.update(chunk)
                    if sha_file(target) != digest.hexdigest():
                        raise ValueError(f'Existing raw checksum changed (will not overwrite): {target}')
                else:
                    temporary = target.with_name(target.name + f'.extracting-{uuid.uuid4().hex}')
                    with archive.open(item) as source, temporary.open('xb') as output:
                        while chunk := source.read(1024 * 1024):
                            digest.update(chunk)
                            output.write(chunk)
                    if temporary.stat().st_size != item.file_size:
                        raise ValueError(f'Extracted byte length mismatch: {item.filename}')
                    if target.exists():
                        raise ValueError(f'Raw target appeared during extraction: {target}')
                    temporary.rename(target)
                    temporary = None
                    extracted += 1
                # Reading every member to EOF also verifies its ZIP CRC32.
                checked += 1
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()
    report['archive'].update({'crc_verified_files': checked, 'newly_extracted_files': extracted,
                              'crc_status': 'pass', 'raw_existing_files_unchanged': True})
    print(f'Archive: {checked} files read to EOF with valid CRC; {extracted} newly extracted', flush=True)


def audit_image(path: Path):
    """Verify container and independently decode every pixel; no source edits."""
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        mode = image.mode
        orientation = image.getexif().get(274, 1)
        exif = {str(k): str(v)[:400] for k, v in image.getexif().items()}
        frames = getattr(image, 'n_frames', 1)
        pixel_digest = hashlib.sha256()
        pixel_digest.update(f'RGB:{width}x{height}:'.encode('ascii'))
        pixel_digest.update(image.convert('RGB').tobytes())
        return {
            'width': width, 'height': height, 'mode': mode,
            'format': image.format, 'exif_orientation': orientation,
            'exif': exif, 'n_frames': frames,
            'sha256': sha_file(path), 'pixel_sha256': pixel_digest.hexdigest(),
            'bytes': path.stat().st_size,
        }


def direct_images(directory: Path):
    if not directory.is_dir():
        raise FileNotFoundError(f'Required image directory missing: {directory}')
    return sorted((p for p in directory.iterdir()
                   if p.is_file() and not p.name.startswith('.') and p.suffix.lower() in IMAGE_SUFFIXES),
                  key=lambda p: natural_key(p.name))


def audit_scenes(root: Path, report: dict):
    rows = []
    raw = root / 'data/raw'
    for scene, source_relative in SCENES.items():
        directory = raw / source_relative
        paths = direct_images(directory)
        if not paths:
            report['errors'].append(f'{scene}: no RGB images')
        summary = {'source_directory': rel(directory, root), 'source_rgb_images': len(paths),
                   'train_images': 0, 'test_images': 0, 'bad_images': [],
                   'dimensions': {}, 'modes': {}, 'exif_orientation_counts': {},
                   'exif_present_images': 0, 'missing_frame_ids': [], 'duplicate_frame_ids': [],
                   'metadata_source_files': []}
        report['scenes'][scene] = summary
        dimensions, modes, orientations = collections.Counter(), collections.Counter(), collections.Counter()
        ids = []
        for ordinal, path in enumerate(paths):
            partition = 'test' if ordinal % 8 == 0 else 'train'
            fid = frame_id(path.name)
            if fid is not None:
                ids.append(fid)
            row = {'scene': scene, 'ordinal': ordinal, 'raw_ordinal': ordinal, 'unique_ordinal': None, 'duplicate_of': '', 'frame_id': fid, 'filename': path.name,
                   'split': partition, 'raw_path': rel(path, root),
                   'prepared_path': f'data/prepared/part1/{scene}/{partition}/images/{path.name}'}
            try:
                details = audit_image(path)
                row.update(details)
                dimensions[f"{details['width']}x{details['height']}"] += 1
                modes[details['mode']] += 1
                orientations[str(details['exif_orientation'])] += 1
                summary['exif_present_images'] += bool(details['exif'])
                summary[f'{partition}_images'] += 1
                if details['exif_orientation'] != 1:
                    report['errors'].append(f'{scene}/{path.name}: non-identity EXIF orientation; no implicit transpose allowed')
                if details['n_frames'] != 1:
                    report['errors'].append(f'{scene}/{path.name}: multi-frame image')
                if details['mode'] != 'RGB':
                    report['warnings'].append(f'{scene}/{path.name}: source mode {details["mode"]}, bytes preserved; loader must handle explicitly')
            except Exception as error:
                row['error'] = f'{type(error).__name__}: {error}'
                summary['bad_images'].append({'filename': path.name, 'error': row['error']})
                report['errors'].append(f'{scene}/{path.name}: {row["error"]}')
            rows.append(row)
        if ids:
            summary['first_frame_id'], summary['last_frame_id'] = min(ids), max(ids)
            summary['missing_frame_ids'] = sorted(set(range(min(ids), max(ids) + 1)) - set(ids))
            summary['duplicate_frame_ids'] = [k for k, n in collections.Counter(ids).items() if n > 1]
        summary['dimensions'] = dict(dimensions)
        summary['modes'] = dict(modes)
        summary['exif_orientation_counts'] = dict(orientations)
        scene_root = raw / ('405841/FRONT' if scene == 'Waymo-405841' else scene)
        summary['metadata_source_files'] = sorted(rel(p, root) for p in scene_root.rglob('*')
                                                 if p.is_file() and p.suffix.lower() in {'.json', '.txt', '.md'}
                                                 and not any(part.startswith('.') for part in p.relative_to(scene_root).parts))
        if summary['missing_frame_ids']:
            report['warnings'].append(f'{scene}: missing numeric frame IDs {summary["missing_frame_ids"]}')
        if summary['duplicate_frame_ids']:
            report['errors'].append(f'{scene}: duplicate numeric frame IDs')
        if len(dimensions) > 1:
            report['warnings'].append(f'{scene}: multiple source resolutions; per-frame dimensions retained')
        print(f'{scene}: {len(paths)} source RGB decoded and audited', flush=True)
    return rows


def apply_unique_split(root: Path, rows: list, report: dict):
    excluded = []
    for scene in SCENES:
        seen = {}
        unique_ordinal = 0
        summary = report['scenes'][scene]
        summary['train_images'] = summary['test_images'] = 0
        for row in (r for r in rows if r['scene'] == scene):
            if 'pixel_sha256' not in row:
                continue
            digest = row['pixel_sha256']
            if digest in seen:
                retained = seen[digest]
                row.update({'split': 'excluded_duplicate', 'unique_ordinal': retained['unique_ordinal'],
                            'duplicate_of': retained['filename'], 'prepared_path': ''})
                excluded.append({'scene': scene, 'discarded_filename': row['filename'],
                                 'retained_filename': retained['filename'], 'raw_ordinal': row['raw_ordinal'],
                                 'retained_raw_ordinal': retained['raw_ordinal'],
                                 'raw_source_path': row['raw_path'], 'pixel_sha256': digest})
            else:
                partition = 'test' if unique_ordinal % 8 == 0 else 'train'
                row.update({'ordinal': unique_ordinal, 'unique_ordinal': unique_ordinal, 'split': partition,
                            'prepared_path': f'data/prepared/part1/{scene}/{partition}/images/{row["filename"]}'})
                summary[f'{partition}_images'] += 1
                seen[digest] = row
                unique_ordinal += 1
        summary['unique_rgb_images'] = unique_ordinal
        summary['excluded_duplicate_images'] = sum(r['scene'] == scene for r in excluded)
        print(f'{scene}: unique={unique_ordinal}, train={summary["train_images"]}, test={summary["test_images"]}, excluded={summary["excluded_duplicate_images"]}', flush=True)
    report['excluded_duplicate_images'] = len(excluded)
    report['deduplication_policy'] = 'Exact decoded RGB pixel SHA256 only; first natural-sorted filename retained per scene; all raw bytes preserved'
    report['excluded_duplicate_manifest'] = 'data/manifests/excluded_duplicates.csv'
    output = root / 'data/manifests/excluded_duplicates.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['scene', 'discarded_filename', 'retained_filename',
                                                    'raw_ordinal', 'retained_raw_ordinal', 'raw_source_path', 'pixel_sha256'])
        writer.writeheader()
        writer.writerows(excluded)
    if excluded:
        report['warnings'].append(f'{len(excluded)} exact RGB duplicate frames excluded from prepared inputs; every original retained in data/raw; see excluded_duplicates.csv')
    # Check reference metadata consistency without interpreting pose direction/units.
    cameras_path = root / 'data/raw/Re10k-1/cameras.json'
    if cameras_path.exists():
        cameras = {c['image_name']: c for c in json.loads(cameras_path.read_text(encoding='utf-8'))}
        disagree = []
        for pair in (r for r in excluded if r['scene'] == 'Re10k-1'):
            a, b = cameras.get(pair['retained_filename']), cameras.get(pair['discarded_filename'])
            if a and b and (a.get('cam_quat') != b.get('cam_quat') or a.get('cam_trans') != b.get('cam_trans')):
                disagree.append({'retained': pair['retained_filename'], 'excluded': pair['discarded_filename']})
        report['re10k_identical_image_distinct_reference_pose_pairs'] = disagree
        if disagree:
            report['warnings'].append(f'Re10k-1: {len(disagree)} identical-image pairs have different supplied reference poses. These metadata are reference estimates, not verified precise ground truth; pose accuracy claims must state this limitation.')


def check_duplicates(rows: list, report: dict):
    for key, label in [('sha256', 'file'), ('pixel_sha256', 'decoded_rgb_pixel')]:
        groups = collections.defaultdict(list)
        for row in rows:
            if key in row:
                groups[row[key]].append(row)
        duplicate_groups = []
        active_duplicate_groups = []
        cross_split_groups = []
        for digest, group in groups.items():
            if len(group) < 2:
                continue
            details = {'sha256': digest,
                       'frames': [{k: row[k] for k in ('scene', 'filename', 'split', 'duplicate_of')} for row in group]}
            duplicate_groups.append(details)
            active = [row for row in group if row['split'] in ('train', 'test')]
            if len(active) > 1:
                active_duplicate_groups.append(details)
                if len({row['split'] for row in active}) > 1:
                    cross_split_groups.append(details)
        report[f'{label}_duplicate_groups'] = duplicate_groups
        report[f'prepared_{label}_duplicate_groups'] = active_duplicate_groups
        report[f'cross_split_{label}_duplicate_groups'] = cross_split_groups
        if cross_split_groups:
            report['errors'].append(f'{len(cross_split_groups)} {label} duplicate groups cross train/test after deduplication')
        if active_duplicate_groups:
            report['warnings'].append(f'{len(active_duplicate_groups)} {label} duplicate groups remain in prepared inputs; see structured report')


def audit_depth(root: Path, report: dict):
    directory = root / 'data/raw/405841/FRONT/depth'
    paths = direct_images(directory)
    dimensions, modes = collections.Counter(), collections.Counter()
    errors = []
    for path in paths:
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.load()
                dimensions[f'{image.width}x{image.height}'] += 1
                modes[image.mode] += 1
        except Exception as error:
            errors.append({'filename': path.name, 'error': f'{type(error).__name__}: {error}'})
    report['auxiliary_waymo_depth'] = {'path': rel(directory, root), 'images': len(paths),
                                      'dimensions': dict(dimensions), 'modes': dict(modes),
                                      'bad_images': errors, 'used_as_part1_input': False}
    if errors:
        report['errors'].append(f'Waymo auxiliary depth: {len(errors)} bad images')
    rgb_names = {Path(r['filename']).stem for r in report.get('_rows', []) if r['scene'] == 'Waymo-405841'}
    depth_names = {p.stem for p in paths}
    report['auxiliary_waymo_depth']['missing_rgb_frame_stems'] = sorted(rgb_names - depth_names)
    report['auxiliary_waymo_depth']['extra_depth_frame_stems'] = sorted(depth_names - rgb_names)


def split_content(scene_rows: list, partition: str):
    return ''.join(row['filename'] + '\n' for row in scene_rows if row['split'] == partition)


def audit_output_state(root: Path, rows: list, report: dict):
    """Refuse stale/mutated prepared inputs before any copying."""
    for scene in SCENES:
        scene_rows = [r for r in rows if r['scene'] == scene]
        for partition in ('train', 'test'):
            target = root / f'data/prepared/part1/{scene}/{partition}/images'
            expected = {r['filename'] for r in scene_rows if r['split'] == partition}
            if target.exists():
                extras = {p.name for p in target.iterdir()} - expected
                if extras:
                    report['errors'].append(f'{rel(target, root)} contains stale/unexpected entries: {sorted(extras)}')
            for row in scene_rows:
                if row['split'] != partition or 'sha256' not in row:
                    continue
                path = root / row['prepared_path']
                if path.exists() and (not path.is_file() or sha_file(path) != row['sha256']):
                    report['errors'].append(f'Prepared file changed, will not overwrite: {rel(path, root)}')


def prepare(root: Path, rows: list, report: dict):
    configurations = {'protocol_version': 'part1-raw-rgb-v2', 'stage': 'raw RGB copy before any camera-dependent preprocessing', 'split_rule': SPLIT_RULE,
                      'preprocessing': 'identity: original RGB file bytes, dimensions, filenames and EXIF retained',
                      'reference_cameras_provided_to_initializers': False, 'scenes': {}}
    for scene in SCENES:
        scene_rows = [r for r in rows if r['scene'] == scene]
        scene_config = {}
        for partition in ('train', 'test'):
            target = root / f'data/prepared/part1/{scene}/{partition}/images'
            target.mkdir(parents=True, exist_ok=True)
            content = split_content(scene_rows, partition)
            manifest_path = root / f'splits/part1/{scene}/{partition}.txt'
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(content, encoding='utf-8', newline='\n')
            split_rows = [r for r in scene_rows if r['split'] == partition]
            for row in split_rows:
                output = root / row['prepared_path']
                if not output.exists():
                    shutil.copy2(root / row['raw_path'], output)
                if sha_file(output) != row['sha256']:
                    raise ValueError(f'Copy integrity mismatch: {output}')
            scene_config[partition] = {'images': rel(target, root), 'count': len(split_rows),
                                       'list': rel(manifest_path, root),
                                       'list_sha256': hashlib.sha256(content.encode('utf-8')).hexdigest()}
        configurations['scenes'][scene] = scene_config
        report['scenes'][scene]['split_sha256'] = {p: scene_config[p]['list_sha256'] for p in ('train', 'test')}
        contact_sheet(root, scene, scene_rows)
    write_json(root / 'configs/part1/data.json', configurations)
    report['prepared_copy_integrity'] = 'all prepared RGB SHA256 values equal the source values'
    report['prepared_rgb_files'] = len(rows)
    report['configuration'] = 'configs/part1/data.json'


def contact_sheet(root: Path, scene: str, rows: list):
    count = min(12, len(rows))
    if not count:
        return
    indices = sorted(set(round(i * (len(rows) - 1) / max(1, count - 1)) for i in range(count)))
    # Avoid periodic sampling accidentally showing only test frames in a 1/8 split.
    if len({rows[i]['split'] for i in indices}) == 1 and len({r['split'] for r in rows}) > 1:
        partition = rows[indices[0]]['split']
        for slot in range(1, len(indices), 2):
            candidate = indices[slot]
            neighbors = sorted(range(len(rows)), key=lambda i: abs(i - candidate))
            indices[slot] = next(i for i in neighbors if rows[i]['split'] != partition and i not in indices)
        indices.sort()
    cell_w, cell_h, label_h = 320, 228, 28
    canvas = Image.new('RGB', (cell_w * 4, cell_h * 3 + 42), (24, 28, 36))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 17)
    except OSError:
        font = ImageFont.load_default()
    draw.text((12, 9), f'{scene} | source RGB only | evenly sampled preview', fill='white', font=font)
    for slot, index in enumerate(indices):
        row = rows[index]
        x, y = (slot % 4) * cell_w, 42 + (slot // 4) * cell_h
        with Image.open(root / row['raw_path']) as image:
            thumbnail = ImageOps.contain(image.convert('RGB'), (cell_w - 10, cell_h - label_h - 8))
        canvas.paste(thumbnail, (x + (cell_w - thumbnail.width) // 2, y + 2))
        color = (255, 205, 103) if row['split'] == 'test' else (115, 225, 184)
        draw.text((x + 8, y + cell_h - label_h), f'{row["filename"]} [{row["split"]}]', fill=color, font=font)
    output = root / f'results/part1/data_preview_{scene}.jpg'
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=90)


def save_rows(root: Path, rows: list):
    fields = ['scene', 'raw_ordinal', 'ordinal', 'unique_ordinal', 'duplicate_of', 'frame_id', 'filename', 'split', 'raw_path', 'prepared_path',
              'width', 'height', 'mode', 'format', 'bytes', 'sha256', 'pixel_sha256',
              'exif_orientation', 'n_frames', 'exif', 'error']
    output = root / 'data/manifests/all_frames.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            copy = dict(row)
            copy['exif'] = json.dumps(copy.get('exif', {}), ensure_ascii=False, sort_keys=True)
            writer.writerow(copy)


def save_report(root: Path, report: dict):
    write_json(root / 'data/manifests/data_audit.json', report)
    lines = ['# Part 1 原始 RGB 准备阶段审核', '', f'- 状态：**{report["status"]}**',
             f'- 划分规则（团队约定，原包未提供划分）：{SPLIT_RULE}',
             '- 本报告覆盖原始 RGB 准备阶段：原始文件完整保留，当前阶段的 RGB 副本未缩放、裁剪或去畸变。后续相机相关预处理需另有 manifest，不在本报告校验范围。',
             '- 参考相机与深度不放入初始化输入目录；相机约定与畸变处理由实验协议单独说明。',
             f'- Python：{sys.version.split()[0]}；Pillow：{pillow_version}', '',
             '| 场景 | 原始 RGB | 独立图像 | 排除完全重复 | 训练 | 测试 | 尺寸 | 坏图 |', '|---|---:|---:|---:|---:|---:|---|---:|']
    for scene, summary in report.get('scenes', {}).items():
        lines.append(f'| {scene} | {summary["source_rgb_images"]} | {summary.get("unique_rgb_images", "未完成")} | {summary.get("excluded_duplicate_images", "未完成")} | {summary["train_images"]} | {summary["test_images"]} | {summary["dimensions"]} | {len(summary["bad_images"])} |')
    lines += ['', '## 完整性', '']
    archive = report.get('archive', {})
    if archive:
        lines += [f'- 压缩包：{archive.get("bytes")} 字节；SHA256 `{archive.get("sha256")}`。',
                  '- SHA256 为本地生成的复现校验值，未获得发布方校验值。',
                  f'- ZIP CRC：{archive.get("crc_status", "未完成")}；完整读取 {archive.get("crc_verified_files", 0)} 个文件。']
    lines += [f'- 文件重复组：{len(report.get("file_duplicate_groups", []))}；解码像素重复组：{len(report.get("decoded_rgb_pixel_duplicate_groups", []))}。',
              '- 每帧路径、raw/unique ordinal、duplicate_of、原始尺寸、模式、EXIF、SHA256、像素 SHA256、固定划分见 `data/manifests/all_frames.csv`。',
              '- 只按完全相同的 RGB 像素去重，每组保留自然排序最早帧；不删除相似帧。原件全保留，排除对应关系见 `data/manifests/excluded_duplicates.csv`。',
              f'- 去重后跨 train/test 文件重复组：{len(report.get("cross_split_file_duplicate_groups", []))}；像素重复组：{len(report.get("cross_split_decoded_rgb_pixel_duplicate_groups", []))}。',
              '- 机器可读审核及原始元数据清单见 `data/manifests/data_audit.json`。',
              '- 固定名单见 `splits/part1/<scene>/train.txt` 和 `test.txt`；名单哈希见 `configs/part1/data.json`。',
              '', '## 错误', '']
    lines += ['- ' + str(x) for x in report.get('errors', [])] or ['- 无。']
    lines += ['', '## 警告', '']
    lines += ['- ' + str(x) for x in report.get('warnings', [])] or ['- 无。']
    lines += ['', '## 范围与限制', '',
              '- 本审核只验证文件、像素和划分；不证明相机是传感器真值，也不推断四元数顺序或坐标系。',
              '- 未执行 COLMAP、VGGT、VGGT-Ω 或 3DGS，未产生重建实验结果。',
              '- 测试图像不允许进入初始化或训练。使用固定名单，勿在 3DGS 加载器中再次自动划分。', '']
    output = root / 'docs/data_audit.md'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    archive = args.archive.resolve() if args.archive else root / 'data/downloads/Project4.zip'
    report = {'schema_version': 1, 'status': 'running', 'scope': 'CPU raw RGB preparation-stage audit and deterministic Part 1 split before camera-dependent preprocessing',
              'split_rule': SPLIT_RULE, 'errors': [], 'warnings': [], 'scenes': {},
              'python_version': sys.version, 'pillow_version': pillow_version,
              'camera_conventions_verified_by_this_script': False}
    rows = []
    try:
        extract_and_verify(root, archive, report)
        rows = audit_scenes(root, report)
        apply_unique_split(root, rows, report)
        check_duplicates(rows, report)
        report['_rows'] = rows
        audit_depth(root, report)
        del report['_rows']
        audit_output_state(root, [r for r in rows if r['split'] in ('train', 'test')], report)
        save_rows(root, rows)
        if report['errors']:
            raise RuntimeError(f'Data audit found {len(report["errors"])} error(s); prepared input was not created')
        prepare(root, [r for r in rows if r['split'] in ('train', 'test')], report)
        report['status'] = 'pass'
    except Exception as error:
        report.pop('_rows', None)
        message = f'{type(error).__name__}: {error}'
        if message not in report['errors']:
            report['errors'].append(message)
        report['status'] = 'failed'
        report['exception_traceback'] = traceback.format_exc()
        if rows:
            save_rows(root, rows)
        save_report(root, report)
        print(message, file=sys.stderr, flush=True)
        return 1
    save_report(root, report)
    print(f'PASS: {len(rows)} source RGB images audited; {report["prepared_rgb_files"]} unique inputs copied and split; {report["excluded_duplicate_images"]} exact duplicate frames excluded.', flush=True)
    print(f'Report: {root / "docs/data_audit.md"}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
