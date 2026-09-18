#!/usr/bin/env python3
"""Generate all 600 AlienBody environments.

Usage:
    python scripts/generate_envs.py
    python scripts/generate_envs.py --family 1 --output data/envs
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from alienbody.env.generator import generate_all, generate_family


def main():
    parser = argparse.ArgumentParser(description="Generate AlienBody environments")
    parser.add_argument("--output", type=str, default="data/envs",
                        help="Output directory")
    parser.add_argument("--family", type=int, default=0,
                        help="Family to generate (1-4, 0=all)")
    args = parser.parse_args()

    output_dir = Path(__file__).parent.parent / args.output

    print("=" * 50)
    print("  AlienBody Environment Generator")
    print("=" * 50)

    if args.family:
        stats = generate_family(args.family, output_dir)
        print(f"\nFamily {args.family}: {stats['generated']} generated, "
              f"{stats['rejected']} rejected")
    else:
        results = generate_all(output_dir)
        total_gen = sum(r["generated"] for r in results)
        total_rej = sum(r["rejected"] for r in results)
        print(f"\nTotal: {total_gen} generated, {total_rej} rejected")

    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
