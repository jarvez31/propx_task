"""CLI: `python -m roofkit --location karlsplatz` -> outputs/roof_attributes.json + a figure."""
import argparse

from .config import LOCATIONS, Config
from .pipeline import run


def main(argv=None):
    ap = argparse.ArgumentParser(prog="roofkit", description="Vienna rooftop attribute extraction.")
    ap.add_argument("--location", choices=sorted(LOCATIONS), default="karlsplatz",
                    help="named study area (default: karlsplatz)")
    ap.add_argument("--n", type=int, default=10, help="number of buildings to sample")
    ap.add_argument("--seed", type=int, default=42, help="random seed for the selection")
    ap.add_argument("--ortho-year", type=int, default=2024, help="Vienna ortho layer year (lbYYYY)")
    ap.add_argument("--out", default="outputs/roof_attributes.json", help="output JSON path")
    args = ap.parse_args(argv)

    config = Config.at(args.location, n_buildings=args.n, seed=args.seed,
                       ortho_year=args.ortho_year, out_json=args.out)
    attrs = run(config)
    print(f"done: {len(attrs)} buildings -> {args.out}")


if __name__ == "__main__":
    main()
