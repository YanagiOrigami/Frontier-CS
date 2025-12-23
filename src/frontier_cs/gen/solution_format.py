"""Solution file naming utilities.

Standard format: {problem}.{model}.{ext}

Examples:
    flash_attn.gpt5.py      -> problem=flash_attn, model=gpt5
    cross_entropy.claude.py -> problem=cross_entropy, model=claude
    1.gpt5.cpp              -> problem=1, model=gpt5
"""

import re
from pathlib import Path
from typing import Optional, Tuple


def validate_problem_name(problem: str) -> None:
    """Validate that a problem name doesn't contain dots."""
    if '.' in problem:
        raise ValueError(f"Problem name cannot contain '.': {problem}")


def parse_solution_filename(filename: str) -> Optional[Tuple[str, str, str]]:
    """Parse a solution filename into (problem, model, ext).

    Args:
        filename: Filename like "flash_attn.gpt5.py"

    Returns:
        Tuple of (problem, model, ext) or None if not parseable
    """
    parts = filename.rsplit('.', 2)
    if len(parts) != 3:
        return None
    problem, model, ext = parts
    if not problem or not model or not ext:
        return None
    return problem, model, ext


def format_solution_filename(problem: str, model: str, ext: str) -> str:
    """Format a solution filename.

    Args:
        problem: Problem ID (e.g., "flash_attn", "1")
        model: Model prefix (e.g., "gpt5", "claude")
        ext: File extension without dot (e.g., "py", "cpp")

    Returns:
        Filename like "flash_attn.gpt5.py"
    """
    validate_problem_name(problem)
    return f"{problem}.{model}.{ext}"


def get_solution_path(
    solutions_dir: Path,
    problem: str,
    model: str,
    ext: str,
) -> Path:
    """Get the path for a solution file.

    Args:
        solutions_dir: Directory containing solutions
        problem: Problem ID
        model: Model prefix
        ext: File extension without dot

    Returns:
        Path to the solution file
    """
    filename = format_solution_filename(problem, model, ext)
    return solutions_dir / filename


def scan_solutions_dir(solutions_dir: Path) -> list[Tuple[Path, str, str]]:
    """Scan a solutions directory for solution files.

    Args:
        solutions_dir: Directory to scan

    Returns:
        List of (path, problem, model) tuples
    """
    results = []
    if not solutions_dir.is_dir():
        return results

    for path in solutions_dir.iterdir():
        if not path.is_file():
            continue
        parsed = parse_solution_filename(path.name)
        if parsed:
            problem, model, _ = parsed
            results.append((path, problem, model))

    return results
