prompting_code = '''"""
prompting.py - Prompting techniques for Boston Rideshare Agent
Formats prompts and parses LLM outputs following ReAct pattern.
"""

import re
import csv
from typing import List, Dict, Any, Optional, Tuple


SYSTEM_PREAMBLE = """You are a helpful rideshare decision agent. You help users choose between Uber and Lyft based on historical Boston rideshare data.

Available tools:
- search[query="<text>", k=<int>] : Searches historical trips and returns top-k similar results based on route, time, and conditions.
- finish[answer="<final answer>"] : Provides the final recommendation to the user.

Follow the exact step format:
Thought: <your reasoning about what to do next>
Action: <one of the tool calls above, or finish[...]>

IMPORTANT: Respond with EXACTLY two lines in this format:
Thought: <one concise sentence>
Action: <either search[...] or finish[answer=...]>

Do NOT include Observation - the system will provide that.""".strip()


def convert_value(raw: str) -> Any:
    """Convert raw string to Python type."""
    import ast
    raw = raw.strip()
    if raw.lower() == "true": return True
    if raw.lower() == "false": return False
    try:
        return ast.literal_eval(raw)
    except Exception:
        return raw.strip('"').strip("'")


def split_args(argstr: str) -> Dict[str, Any]:
    """Parse function arguments from string."""
    args: Dict[str, Any] = {}
    row = next(csv.reader([argstr], delimiter=',', skipinitialspace=True, quotechar='"'))
    for field in row:
        field = field.strip()
        if not field:
            continue
        if "=" in field:
            key, val = field.split("=", 1)
            args[key.strip()] = convert_value(val)
        else:
            args[field] = True
    return args


def parse_action(line: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Parse action from LLM output."""
    prefix = "Action:"
    if not line.strip().startswith(prefix):
        return None
    s = line[len(prefix):].strip()
    lb = s.find("[")
    rb = s.rfind("]")
    if lb == -1 or rb == -1 or rb < lb:
        return None
    name = s[:lb].strip()
    if not name or not all(c.isalpha() or c == "_" for c in name):
        return None
    inner = s[lb + 1 : rb].strip()
    args = split_args(inner) if inner else {}
    return name, args


def format_history(trajectory: List[Dict[str, str]]) -> str:
    """Format conversation history for prompt."""
    lines: List[str] = []
    for step in trajectory:
        lines.append(f"Thought: {step['thought']}")
        lines.append(f"Action: {step['action']}")
        lines.append(f"Observation: {step['observation']}")
    return "\\n".join(lines)


def make_prompt(user_query: str, trajectory: List[Dict[str, str]]) -> str:
    """Build complete prompt for LLM."""
    history_block = format_history(trajectory)
    return (
        f"{SYSTEM_PREAMBLE}\\n\\n"
        f"User Question: {user_query}\\n\\n"
        f"{history_block}\\n"
        f"Next step:\\n"
        f"Thought:"
    )
'''