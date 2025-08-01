import argparse
import logging
import os
from pathlib import Path

import uvicorn
from tqdm import tqdm

from code_search_mcp.mcp_search import Indexer

# MCP-compliant logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_argparse():
    """Configure command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Codebase indexing tool for MCP code search server."
    )
    parser.add_argument(
        "--project-path",
        type=Path,
        default=Path(os.getenv("MCP_PROJECT_PATH", os.getcwd())),
        help="Path to the project directory to index (default: MCP_PROJECT_PATH or current directory).",
    )
    parser.add_argument(
        "--action",
        choices=["reindex", "index", "serve"],
        default="reindex",
        help="Action to perform: reindex (incremental), index (full indexing), or serve (run API server).",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind the server (used with --action serve)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to run the server on (used with --action serve)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging."
    )
    return parser


def main():
    """Main entry point for command-line indexing or serving."""
    parser = setup_argparse()
    args = parser.parse_args()

    # Adjust logging level if verbose
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.action == "serve":
        logger.info(f"Starting Uvicorn server at http://{args.host}:{args.port}")
        uvicorn.run("mcp_server.api:app", host=args.host, port=args.port, reload=True)
        return

    logger.info(f"Starting indexing for project at {args.project_path}")

    try:
        # Estimate total files for progress bar
        total_files = sum(1 for _ in args.project_path.rglob("*") if _.is_file())

        with tqdm(
            total=total_files, desc="Indexing files", disable=args.verbose
        ) as pbar:

            def progress_callback():
                pbar.update(1)

            if args.action == "reindex":
                Indexer(args.project_path).index_incremental(
                    progress_callback=progress_callback
                )
                logger.info("Incremental indexing completed successfully.")
            elif args.action == "index":
                Indexer(args.project_path).index_full(
                    progress_callback=progress_callback
                )
                logger.info("Full indexing completed successfully.")
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise


if __name__ == "__main__":
    main()
