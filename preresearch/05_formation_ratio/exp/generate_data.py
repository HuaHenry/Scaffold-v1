"""Retired frame-compression entry point: supervision sampling generates no data."""


def main(argv=None):
    raise SystemExit("[05] Frame compression has been retired. No trajectories should be replayed or generated. "
                     "Create a NEW run using: bash run_all.sh --tag supervision_ratio --stage prepare")


if __name__ == "__main__":
    main()
