## Code Style
1. We use Ruff & Pyright for code quality
2. I had a strong preference for surfacing errors immediately instead of fallbacks. For example, use os.environ["variable"] instead of os.getenv("variable")