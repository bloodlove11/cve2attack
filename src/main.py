"""Typer CLI entrypoint for interactive agent runs.

Usage::

    python -m src.main "query" [--no-trace]

Loads project-root ``.env`` (ExperientialLabs credentials) before dispatch.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv

# Load .env from project root if present
_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_ROOT / ".env")

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def ask(
    query: str = typer.Argument(..., help="User question for the agent"),
    no_trace: bool = typer.Option(False, "--no-trace", help="Disable writing traces/"),
    pretty: bool = typer.Option(True, "--pretty/--compact", help="Pretty-print JSON"),
) -> None:
    """Run the agent and print a validated JSON answer."""
    if not os.environ.get("EXPLABS_API_KEY"):
        typer.echo(
            "EXPLABS_API_KEY is not set. Copy .env.example to .env and add your key.",
            err=True,
        )
        raise typer.Exit(code=1)

    from src.agent.dispatch import run_chat_query

    answer = run_chat_query(query, trace=not no_trace)
    payload = answer.model_dump()
    if pretty:
        typer.echo(json.dumps(payload, indent=2))
    else:
        typer.echo(json.dumps(payload))


def main() -> None:
    """Invoke Typer; inject ``ask`` so ``python -m src.main "query"`` works."""
    # Typer treats first positional as command unless we invoke ask directly.
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-") and sys.argv[1] not in {
        "ask",
        "--help",
        "-h",
    }:
        # Inject default command name for ergonomic CLI
        sys.argv.insert(1, "ask")
    app()


if __name__ == "__main__":
    main()
