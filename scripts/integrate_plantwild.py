"""
Copies the 16 shared classes from a local PlantWild (Wei et al., 2024) download
into data/plantwild/, matching the folder layout produced by
download_real_data.py for the other two datasets.

PlantWild must be downloaded separately (see the README) because its release
requires accepting a license on Hugging Face. Its folder names are plain
lowercase with spaces (e.g. "apple rust"), which PLANTWILD_MAP below maps to
the canonical class names.

Usage (run from the repository root):
    python scripts/integrate_plantwild.py --source /path/to/plantwild/images

On Windows, quote the path, e.g.:
    python scripts/integrate_plantwild.py --source "C:\\path\\to\\plantwild\\images"
"""

import argparse
import os
import shutil

# canonical_name -> exact PlantWild folder name (checked against the real
# images/ directory listing). pepper_bell_bacterial_spot is not included; see
# download_real_data.py for why.
PLANTWILD_MAP = {
    "apple_rust": "apple rust",
    "apple_scab": "apple scab",
    "corn_common_rust": "corn rust",
    "corn_gray_leaf_spot": "corn gray leaf spot",
    "corn_northern_leaf_blight": "corn northern leaf blight",
    "grape_black_rot": "grape black rot",
    "potato_early_blight": "potato early blight",
    "potato_late_blight": "potato late blight",
    "squash_powdery_mildew": "squash powdery mildew",
    "tomato_bacterial_spot": "tomato bacterial leaf spot",
    "tomato_early_blight": "tomato early blight",
    "tomato_late_blight": "tomato late blight",
    "tomato_leaf_mold": "tomato leaf mold",
    "tomato_mosaic_virus": "tomato mosaic virus",
    "tomato_septoria_leaf_spot": "tomato septoria leaf spot",
    "tomato_yellow_leaf_curl_virus": "tomato yellow leaf curl virus",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True,
                         help="Path to PlantWild's 'images' folder (contains one subfolder per class)")
    args = parser.parse_args()

    source_root = args.source
    if not os.path.isdir(source_root):
        raise FileNotFoundError(f"Source folder not found: {source_root}")

    repo_root = os.path.join(os.path.dirname(__file__), "..")
    dest_root = os.path.join(repo_root, "data", "plantwild")
    os.makedirs(dest_root, exist_ok=True)

    print(f"{'Class':<32} {'PlantWild images':>16}")
    counts = {}
    missing = []
    for canonical, pw_folder in PLANTWILD_MAP.items():
        src = os.path.join(source_root, pw_folder)
        if not os.path.isdir(src):
            missing.append((canonical, pw_folder))
            continue
        dest = os.path.join(dest_root, canonical)
        os.makedirs(dest, exist_ok=True)
        files = [f for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
        for f in files:
            shutil.copy2(os.path.join(src, f), os.path.join(dest, f))
        counts[canonical] = len(files)
        print(f"{canonical:<32} {len(files):>16}")

    if missing:
        print("\nWARNING - these classes were NOT found and were skipped:")
        for canonical, pw_folder in missing:
            print(f"  {canonical} (looked for '{pw_folder}')")

    print(f"\nTotal PlantWild images copied: {sum(counts.values())} across {len(counts)} classes")
    print(f"Written to: {os.path.abspath(dest_root)}")


if __name__ == "__main__":
    main()
