import os
import random
import shutil
import argparse


def select_and_copy(src_dir, dst_dir, count):
    os.makedirs(dst_dir, exist_ok=True)
    patients = [
        d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))
    ]
    if count > len(patients):
        raise ValueError(f"Asked for {count}, but only {len(patients)} available.")
    selected = random.sample(patients, count)
    for p in selected:
        shutil.copytree(os.path.join(src_dir, p), os.path.join(dst_dir, p))
    print(f"Copied {len(selected)} folders from {src_dir} → {dst_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select N random patient dirs from test set"
    )
    parser.add_argument("--src", default="data/test", help="Path to source test folder")
    parser.add_argument(
        "--dst", default="subset_test", help="Path to target folder to create"
    )
    parser.add_argument(
        "--count", type=int, default=50, help="Number of patient folders to select"
    )
    args = parser.parse_args()
    select_and_copy(args.src, args.dst, args.count)
