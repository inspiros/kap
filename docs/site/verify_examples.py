"""Run documentation examples and check the captured output used by mdBook."""
from __future__ import annotations

import __future__
from html.parser import HTMLParser
import os
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "docs/site"
EXAMPLES = SITE / "examples"


def run(command: list[str]) -> str:
    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=True,
        env={**os.environ, "CARGO_BUILD_JOBS": "1"},
    )
    if result.stderr:
        raise RuntimeError(result.stderr)
    return result.stdout


class CodeBlocks(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[tuple[str, str]] = []
        self.language: str | None = None
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "code":
            self.language = dict(attrs).get("class", "")
            self.parts = []

    def handle_data(self, data):
        if self.language is not None:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag == "code" and self.language is not None:
            self.blocks.append((self.language, "".join(self.parts)))
            self.language = None


def main() -> None:
    python_count = rust_count = 0
    for source in sorted(EXAMPLES.glob("*.py")):
        name = source.stem
        python_output = run([str(ROOT / ".venv/bin/python"), str(source)])
        assert python_output == source.with_suffix(".python.txt").read_text(), name
        python_count += 1
        if name == "exact":
            continue
        rust_output = run(["cargo", "run", "-q", "-p", "kap-solver", "--example", f"doc_{name}"])
        assert rust_output == source.with_suffix(".rust.txt").read_text(), name
        assert (python_output.startswith(rust_output) if name == "lap" else python_output == rust_output), name
        rust_count += 1
    print(f"Python examples: {python_count} ran; captured output matches")
    print(f"Rust examples: {rust_count} compiled and ran; stated output matches")

    # Check rendered examples against the included files. Signature declarations
    # are not calls; compile and execute their Python declarations separately.
    rendered = 0
    for html in sorted((SITE / "book").glob("*.html")):
        if html.name in ("index.html", "print.html"):
            continue
        parser = CodeBlocks()
        parser.feed(html.read_text())
        for language, code in parser.blocks:
            if language == "language-python" and code.startswith("def "):
                exec(compile(code, str(html), "exec", flags=__future__.annotations.compiler_flag), {})
            if language == "language-rust" and code.startswith("use kap::"):
                candidates = list((ROOT / "crates/kap-solver/examples").glob("doc_*.rs"))
                # mdBook omits the code block's final line ending in HTML.
                assert any(code == candidate.read_text().removesuffix("\n") for candidate in candidates), html
                rendered += 1
    assert rendered == rust_count, (rendered, rust_count)
    print(f"Rendered Rust examples: {rendered} match the included source files")

    # README code blocks are inline copies, so check them independently.
    with tempfile.TemporaryDirectory(prefix="kap-docs-") as directory:
        directory = Path(directory)
        for document in (ROOT / "README.md", ROOT / "crates/kap-solver/README.md"):
            for number, (language, code) in enumerate(re.findall(r"```(python|rust)\n(.*?)```", document.read_text(), re.S)):
                if language == "python":
                    run([str(ROOT / ".venv/bin/python"), "-c", code])
                else:
                    source = directory / f"example_{number}.rs"
                    binary = directory / f"example_{number}"
                    source.write_text(code)
                    library = max((ROOT / "target/debug/deps").glob("libkap-*.rlib"), key=lambda p: p.stat().st_mtime)
                    run(["rustc", "--edition=2024", str(source), "--extern", f"kap={library}", "-L", f"dependency={ROOT / 'target/debug/deps'}", "-o", str(binary)])
                    run([str(binary)])
    print("README Python/Rust examples: compiled and ran")


if __name__ == "__main__":
    main()
