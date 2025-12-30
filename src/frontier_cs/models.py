"""
Model name utilities for Frontier-CS.

Provides consistent model prefix conversion used across:
- Solution generation (generate_solutions.py)
- Solution matrix checking (frontier-eval check)
- Batch evaluation (frontier-eval batch)
"""

import re
from typing import Dict, List, Optional, Tuple

# Model prefix aliases for backward compatibility
MODEL_PREFIX_ALIASES: Dict[str, str] = {
    "grokcodefast1_": "grok4fastreasoning_",
}


def get_model_prefix(model: str) -> str:
    """
    Convert model name to the prefix format used in solution folder names.

    This is the canonical function for model prefix conversion. All tools
    should use this to ensure consistent naming.

    Examples:
        >>> get_model_prefix("gpt-5")
        'gpt5'
        >>> get_model_prefix("gpt-5.1-preview")
        'gpt5.1'
        >>> get_model_prefix("gemini/gemini-2.5-pro")
        'gemini2.5pro'
        >>> get_model_prefix("claude-sonnet-4-5-20250929")
        'claude4.5sonnet'
        >>> get_model_prefix("grok-3-fast-reasoning")
        'grok3fastreasoning'

    Args:
        model: Model name (e.g., "gpt-5", "gemini/gemini-2.5-pro")

    Returns:
        Normalized prefix for solution directory names
    """
    original = model

    # Remove provider prefix if present (e.g., 'gemini/gemini-2.5-pro' -> 'gemini-2.5-pro')
    if "/" in model:
        model = model.split("/", 1)[1]

    model_lower = model.lower().strip()

    # Handle GPT-5 variants
    # Keep 'gpt-5.1', 'gpt-5.2' etc. distinct so their artifacts prefix correctly
    if model_lower.startswith("gpt-5.2") or model_lower.startswith("gpt5.2"):
        return "gpt5.2"
    if model_lower.startswith("gpt-5.1") or model_lower.startswith("gpt5.1"):
        return "gpt5.1"
    if model_lower.startswith("gpt-5") or model_lower.startswith("gpt5"):
        return "gpt5"

    # Handle Gemini 2.5 Pro variants
    if "gemini-2.5-pro" in model_lower or "gemini2.5pro" in model_lower:
        return "gemini2.5pro"

    # Handle other Gemini variants (e.g., gemini-1.5-pro -> gemini1.5pro)
    gemini_match = re.match(r"gemini-?(\d+\.?\d*)-?pro", model_lower)
    if gemini_match:
        version = gemini_match.group(1)
        return f"gemini{version}pro"

    # Handle Claude variants (e.g., claude-sonnet-4-5-20250929 -> claude4.5sonnet)
    claude_match = re.match(r"claude-([a-z]+)-(\d+)-(\d+)", model_lower)
    if claude_match:
        family = claude_match.group(1)
        major = claude_match.group(2)
        minor = claude_match.group(3)
        return f"claude{major}.{minor}{family}"

    # Handle Grok variants - keep 'fast' and 'reasoning' in the prefix
    if "grok" in model_lower:
        sanitized = re.sub(r"[^a-zA-Z0-9]+", "", model_lower)
        if sanitized:
            return sanitized

    # Default: sanitize by removing all non-alphanumeric characters
    sanitized = re.sub(r"[^a-zA-Z0-9]+", "", model_lower)
    if not sanitized:
        raise ValueError(f"Unable to derive model prefix from '{original}'")
    return sanitized


def normalize_solution_name(name: str) -> str:
    """
    Normalize a solution directory name by applying prefix aliases.

    Args:
        name: Solution directory name

    Returns:
        Normalized name with aliases applied
    """
    for old_prefix, new_prefix in MODEL_PREFIX_ALIASES.items():
        if name.startswith(old_prefix):
            return new_prefix + name[len(old_prefix):]
    return name


def sanitize_problem_name(problem: str) -> str:
    """
    Convert problem path to solution name suffix.

    Examples:
        >>> sanitize_problem_name("flash_attn")
        'flash_attn'
        >>> sanitize_problem_name("gemm_optimization/squares")
        'gemm_optimization_squares'

    Args:
        problem: Problem ID (may contain slashes)

    Returns:
        Sanitized problem name for use in solution directory names
    """
    return problem.replace("/", "_")


def build_problem_name_map(problems_dir: "Path") -> Dict[str, str]:
    """
    Build a mapping from sanitized problem names to actual problem paths.

    Scans the problems directory recursively to find all problem directories
    (those containing config.yaml or evaluate.sh).

    Args:
        problems_dir: Path to problems directory

    Returns:
        Dict mapping sanitized names to relative paths
        e.g., {"cant_be_late_high_availability": "cant_be_late/high_availability"}
    """
    from pathlib import Path

    mapping: Dict[str, str] = {}

    if not problems_dir.is_dir():
        return mapping

    def is_problem_dir(path: Path) -> bool:
        """Check if a directory is a problem (has config.yaml or evaluate.sh)."""
        return (path / "config.yaml").exists() or (path / "evaluate.sh").exists()

    def scan_recursive(current: Path, rel_path: str = "") -> None:
        if is_problem_dir(current):
            # This is a problem directory
            if rel_path:
                sanitized = sanitize_problem_name(rel_path)
                mapping[sanitized] = rel_path
            return

        # Scan subdirectories
        for child in current.iterdir():
            if child.is_dir() and not child.name.startswith(".") and child.name != "__pycache__":
                child_rel = f"{rel_path}/{child.name}" if rel_path else child.name
                scan_recursive(child, child_rel)

    scan_recursive(Path(problems_dir))
    return mapping


# Cached problem name map
_problem_name_map: Optional[Dict[str, str]] = None
_problem_name_map_dir: Optional[str] = None


def resolve_problem_name(sanitized_name: str, problems_dir: "Path") -> Optional[str]:
    """
    Resolve a sanitized problem name back to its actual path.

    Args:
        sanitized_name: Sanitized name like "cant_be_late_high_availability"
        problems_dir: Path to problems directory

    Returns:
        Actual problem path like "cant_be_late/high_availability" or None if not found
    """
    global _problem_name_map, _problem_name_map_dir

    problems_dir_str = str(problems_dir)

    # Rebuild cache if directory changed
    if _problem_name_map is None or _problem_name_map_dir != problems_dir_str:
        _problem_name_map = build_problem_name_map(problems_dir)
        _problem_name_map_dir = problems_dir_str

    return _problem_name_map.get(sanitized_name)


def parse_solution_name(solution_name: str) -> Tuple[str, str, int]:
    """
    Parse a solution directory name into components.

    Examples:
        >>> parse_solution_name("gpt5_flash_attn")
        ('gpt5', 'flash_attn', 0)
        >>> parse_solution_name("gpt5_flash_attn_1")
        ('gpt5', 'flash_attn', 1)
        >>> parse_solution_name("claude4.5sonnet_gemm_optimization_squares_2")
        ('claude4.5sonnet', 'gemm_optimization_squares', 2)

    Args:
        solution_name: Solution directory name

    Returns:
        Tuple of (model_prefix, problem_slug, variant_index)
    """
    # Check for variant suffix
    parts = solution_name.rsplit("_", 1)
    variant_index = 0
    base_name = solution_name

    if len(parts) == 2 and parts[1].isdigit():
        variant_index = int(parts[1])
        base_name = parts[0]

    # Split into model prefix and problem slug
    # Model prefix is the first part before underscore
    first_underscore = base_name.find("_")
    if first_underscore == -1:
        return (base_name, "", variant_index)

    model_prefix = base_name[:first_underscore]
    problem_slug = base_name[first_underscore + 1:]

    return (model_prefix, problem_slug, variant_index)


def build_solution_name(model: str, problem: str, variant_index: int = 0) -> str:
    """
    Build a solution directory name from components.

    Args:
        model: Model name (will be converted to prefix)
        problem: Problem ID
        variant_index: Variant index (0 = no suffix)

    Returns:
        Solution directory name
    """
    prefix = get_model_prefix(model)
    slug = sanitize_problem_name(problem)
    suffix = "" if variant_index == 0 else f"_{variant_index}"
    return f"{prefix}_{slug}{suffix}"


def detect_provider(model: str) -> str:
    """
    Detect the LLM provider from model name.

    Args:
        model: Model name

    Returns:
        Provider name: 'openai', 'google', 'anthropic', 'xai', 'deepseek', 'openrouter'
    """
    normalized = model.strip()
    if "/" in normalized:
        provider_hint, actual_model = normalized.split("/", 1)
    else:
        provider_hint, actual_model = "", normalized

    provider_hint = provider_hint.lower()
    actual_lower = actual_model.lower()

    if (provider_hint in {"", "openai", "azure", "azure_openai"}) and actual_lower.startswith("gpt"):
        return "openai"
    if provider_hint in {"gemini", "google"} or "gemini" in actual_lower:
        return "google"
    if provider_hint == "anthropic" or "claude" in actual_lower:
        return "anthropic"
    if provider_hint == "xai" or "grok" in actual_lower:
        return "xai"
    if provider_hint == "deepseek" or "deepseek" in actual_lower:
        return "deepseek"

    return provider_hint or "openai"


def is_reasoning_model(model: str, override: Optional[bool] = None) -> bool:
    """
    Determine if a model is a reasoning model.

    Args:
        model: Model name
        override: If set, use this value instead of auto-detection

    Returns:
        True if the model is a reasoning model
    """
    if override is not None:
        return override

    prefixes = ("gpt-5", "o1", "o3", "deepseek-reasoner")
    if any(model.startswith(p) for p in prefixes):
        return True

    normalized = model.lower()
    if "reasoning" in normalized and normalized.startswith("grok-"):
        return True

    return False
