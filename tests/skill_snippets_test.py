"""
Run the Python code blocks of the agent skill in ``skills/``, so that the skill cannot drift away
from the API.

The ```python blocks of each Markdown file are executed in order in one namespace, so a block can
use the names defined by the previous ones. A block directly preceded by the line
``<!-- skip-test -->`` is not executed.
"""

import re
from pathlib import Path

import pytest

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"
SKIP_MARKER = "<!-- skip-test -->"
BLOCK_RE = re.compile(r"(?P<before>[^\n]*)\n```python\n(?P<code>.*?)\n```", re.DOTALL)


def _code_blocks(path):
    text = "\n" + path.read_text()
    return [
        m.group("code")
        for m in BLOCK_RE.finditer(text)
        if m.group("before").strip() != SKIP_MARKER
    ]


SKILL_FILES = sorted(SKILLS_DIR.rglob("*.md"))


@pytest.mark.parametrize(
    "path", SKILL_FILES, ids=[str(p.relative_to(SKILLS_DIR)) for p in SKILL_FILES]
)
def test_skill_code_blocks_run(path):
    namespace = {"__name__": f"skill_snippet_{path.stem}"}
    for i, code in enumerate(_code_blocks(path)):
        try:
            exec(compile(code, f"{path.name}[block {i}]", "exec"), namespace)
        except Exception as e:
            raise AssertionError(
                f"Code block {i} of {path.relative_to(SKILLS_DIR)} failed:\n{code}"
            ) from e


def test_skill_files_found():
    assert (SKILLS_DIR / "jvmc" / "SKILL.md").exists()
