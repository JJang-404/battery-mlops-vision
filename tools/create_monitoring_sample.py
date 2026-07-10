"""Create a reproducible 98% OK / 2% NG monitoring image sample.

The source JSON files contain a few malformed/non-UTF-8 metadata strings, so this
utility deliberately reads only the ASCII ``image_info.is_normal`` field with a
regular expression instead of parsing the complete JSON document.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


STATUS_PATTERN = re.compile(rb'"is_normal"\s*:\s*(true|false)')
BATTERY_ID_PATTERN = re.compile(r"^RGB_cell_cylindrical_(\d+)_")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = (
    PROJECT_ROOT.parent
    / "103.배터리 불량 이미지 데이터"
    / "3.개방데이터"
    / "1.데이터"
    / "Training"
    / "01. 원천데이터"
)
DEFAULT_DESTINATION = PROJECT_ROOT / "monitoring_sample_150"


@dataclass(frozen=True)
class Candidate:
    image_path: Path
    battery_id: str
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="라벨 기준으로 모니터링용 OK/NG 이미지를 선별해 복사합니다."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help=f"원천데이터 폴더 (기본값: {DEFAULT_SOURCE_ROOT})",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
        help=f"복사 대상 폴더 (기본값: {DEFAULT_DESTINATION})",
    )
    parser.add_argument("--total", type=int, default=150, help="전체 이미지 수")
    parser.add_argument(
        "--ng-ratio", type=float, default=0.02, help="불량 이미지 비율 (0~1)"
    )
    parser.add_argument("--seed", type=int, default=20260710, help="무작위 시드")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="파일을 복사하지 않고 선별 결과만 확인",
    )
    return parser.parse_args()


def read_status(label_path: Path) -> str | None:
    match = STATUS_PATTERN.search(label_path.read_bytes())
    if match is None:
        return None
    return "OK" if match.group(1) == b"true" else "NG"


def battery_id_from_name(stem: str) -> str:
    match = BATTERY_ID_PATTERN.match(stem)
    return match.group(1) if match else "unknown"


def collect_candidates(source_root: Path) -> tuple[list[Candidate], dict[str, int]]:
    label_dir = source_root / "TL_Exterior_Img_Datasets_label"
    image_dirs = sorted(
        path
        for path in source_root.glob("TS_Exterior_Img_Datasets_images*")
        if path.is_dir()
    )

    if not label_dir.is_dir():
        raise FileNotFoundError(f"라벨 폴더가 없습니다: {label_dir}")
    if not image_dirs:
        raise FileNotFoundError(f"이미지 폴더가 없습니다: {source_root}")

    candidates: list[Candidate] = []
    stats = {"images": 0, "ok": 0, "ng": 0, "missing_label": 0, "invalid": 0}
    seen_names: set[str] = set()

    print("라벨과 실제 이미지를 대조합니다. 데이터 크기에 따라 몇 분 걸릴 수 있습니다.")
    for image_dir in image_dirs:
        print(f"  검사 중: {image_dir.name}")
        for image_path in image_dir.iterdir():
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            stats["images"] += 1
            if image_path.name in seen_names:
                raise RuntimeError(f"중복 이미지 파일명입니다: {image_path.name}")
            seen_names.add(image_path.name)

            label_path = label_dir / f"{image_path.stem}.json"
            if not label_path.is_file():
                stats["missing_label"] += 1
                continue

            status = read_status(label_path)
            if status is None:
                stats["invalid"] += 1
                continue

            stats[status.lower()] += 1
            candidates.append(
                Candidate(
                    image_path=image_path,
                    battery_id=battery_id_from_name(image_path.stem),
                    status=status,
                )
            )

    return candidates, stats


def select_diverse(
    candidates: list[Candidate], count: int, rng: random.Random
) -> list[Candidate]:
    """Prefer one image per battery ID, then randomly fill the remainder."""
    if len(candidates) < count:
        raise ValueError(f"필요 수량 {count}장보다 후보 {len(candidates)}장이 적습니다.")

    groups: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate.battery_id].append(candidate)

    battery_ids = list(groups)
    rng.shuffle(battery_ids)
    for group in groups.values():
        rng.shuffle(group)

    selected = [groups[battery_id][0] for battery_id in battery_ids[:count]]
    if len(selected) == count:
        return selected

    selected_paths = {item.image_path for item in selected}
    remainder = [
        item for item in candidates if item.image_path not in selected_paths
    ]
    rng.shuffle(remainder)
    selected.extend(remainder[: count - len(selected)])
    return selected


def write_manifest(destination: Path, selected: list[Candidate]) -> None:
    for status in ("OK", "NG"):
        names = sorted(item.image_path.name for item in selected if item.status == status)
        (destination / f"{status}_images.txt").write_text(
            "\n".join(names) + "\n", encoding="utf-8"
        )

    with (destination / "sample_manifest.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["status", "battery_id", "file_name", "source_path"])
        for item in sorted(selected, key=lambda value: value.image_path.name):
            writer.writerow(
                [item.status, item.battery_id, item.image_path.name, item.image_path]
            )


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    destination = args.destination.resolve()

    if args.total <= 0:
        raise ValueError("--total은 1 이상이어야 합니다.")
    if not 0.0 <= args.ng_ratio <= 1.0:
        raise ValueError("--ng-ratio는 0 이상 1 이하여야 합니다.")

    ng_count = round(args.total * args.ng_ratio)
    ok_count = args.total - ng_count
    candidates, stats = collect_candidates(source_root)

    ok_candidates = [item for item in candidates if item.status == "OK"]
    ng_candidates = [item for item in candidates if item.status == "NG"]
    rng = random.Random(args.seed)
    selected_ok = select_diverse(ok_candidates, ok_count, rng)
    selected_ng = select_diverse(ng_candidates, ng_count, rng)
    selected = selected_ok + selected_ng

    print("\n원천데이터 확인 결과")
    print(f"  이미지: {stats['images']:,}장")
    print(f"  OK 후보: {stats['ok']:,}장")
    print(f"  NG 후보: {stats['ng']:,}장")
    print(f"  라벨 누락: {stats['missing_label']:,}장")
    print(f"  상태 판독 실패: {stats['invalid']:,}장")
    print("\n선별 결과")
    print(f"  OK: {len(selected_ok)}장")
    print(f"  NG: {len(selected_ng)}장")
    print(f"  전체: {len(selected)}장")
    print(f"  OK 배터리 ID: {len({item.battery_id for item in selected_ok})}종")
    print(f"  NG 배터리 ID: {len({item.battery_id for item in selected_ng})}종")

    if args.dry_run:
        print("\n--dry-run: 파일을 복사하지 않았습니다.")
        return 0

    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"대상 폴더가 비어 있지 않습니다: {destination}\n"
            "기존 결과를 보존하기 위해 중단했습니다. 다른 --destination을 지정하세요."
        )

    destination.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(selected, start=1):
        shutil.copy2(item.image_path, destination / item.image_path.name)
        if index % 25 == 0 or index == len(selected):
            print(f"  복사: {index}/{len(selected)}")

    write_manifest(destination, selected)
    copied_count = sum(
        1
        for path in destination.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if copied_count != args.total:
        raise RuntimeError(
            f"복사 검증 실패: 예상 {args.total}장, 실제 {copied_count}장"
        )

    print(f"\n완료: {destination}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        print(f"오류: {error}", file=sys.stderr)
        raise SystemExit(1)
