"""Allow ``python -m src`` / ``python -m src.main`` style entry."""

from src.main import main

if __name__ == "__main__":
    main()
