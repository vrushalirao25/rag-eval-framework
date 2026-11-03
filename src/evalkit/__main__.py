"""CLI: python -m evalkit --config config/eval_config.yaml [--provider bedrock] [--limit 5] [--no-fail]

Exit codes: 0 = gate passed (or --no-fail), 1 = gate failed, 2 = preflight health check failed.
"""
import argparse
import logging
import sys

from .runner import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG evaluation pipeline.")
    parser.add_argument("--config", default="config/eval_config.yaml")
    parser.add_argument("--provider", choices=["mock", "bedrock", "openai"],
                        help="Override the provider for both generator and judge.")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N golden items.")
    parser.add_argument("--no-fail", action="store_true", help="Always exit 0 (report-only mode).")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level,
        stream=sys.stderr,
        format="%(asctime)s level=%(levelname)s logger=%(name)s msg=%(message)s",
    )
    sys.exit(run(args.config, args.provider, args.limit, fail=not args.no_fail))


if __name__ == "__main__":
    main()
