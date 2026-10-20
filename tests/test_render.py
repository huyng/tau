"""Tests for the terminal presentation layer."""

from tau.render import MarkdownRenderer, render_tool_result
from tau.theme import Theme, color_enabled


class FakeTty:
    def isatty(self) -> bool:
        return True


def test_markdown_rendering_plain_mode():
    renderer = MarkdownRenderer(Theme(False), width=60)
    out = renderer.feed(
        "## Plan\n"
        "\n"
        "- first step\n"
        "- second step\n"
        "\n"
        "```python\n"
        "x = 1\n"
        "```\n"
    )
    out += renderer.flush()

    assert "Plan" in out
    assert "## " not in out
    assert "• first step" in out
    assert "╭─ python" in out
    assert "│ x = 1" in out
    assert out.endswith("\n")
    assert "\x1b[" not in out  # colors disabled


def test_markdown_rendering_survives_split_deltas():
    renderer = MarkdownRenderer(Theme(False), width=60)
    out = renderer.feed("**bo")
    out += renderer.feed("ld**\n")
    out += renderer.flush()

    assert "bold" in out
    assert "**" not in out


def test_markdown_fence_closes_on_flush():
    renderer = MarkdownRenderer(Theme(False), width=60)
    out = renderer.feed("```\nunfinished")
    out += renderer.flush()

    assert "│ unfinished" in out
    assert "╰─" in out
    assert out.endswith("\n")


def test_colors_are_emitted_when_enabled():
    out = MarkdownRenderer(Theme(True), width=60).feed("`code` and **bold**\n")
    assert "\x1b[" in out


def test_diff_tool_result_is_colorized_and_summarized():
    output = (
        "Edited src/app.py (1 change)\n\n"
        "--- a/src/app.py\n"
        "+++ b/src/app.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-old line\n"
        "+new line\n"
    )
    out = render_tool_result(Theme(False), "edit", output, False, 0.42)

    assert "✓" in out
    assert "+1" in out and "−1" in out
    assert "-old line" in out
    assert "+new line" in out


def test_bash_result_shows_preview_and_exit_code():
    out = render_tool_result(Theme(False), "bash", "hello\nworld\n[exit code: 0]", False, 1.2)
    assert "hello" in out
    assert "[exit code: 0]" in out


def test_read_result_is_summarized_not_dumped():
    out = render_tool_result(Theme(False), "read", "line\n" * 500, False, 0.01)
    assert "500 lines" in out
    assert "line\n" not in out


def test_error_result_is_red_and_bounded():
    out = render_tool_result(
        Theme(False), "bash", "Error: boom\n" + "\n".join(f"line {i}" for i in range(20)),
        True, 0.1,
    )
    assert "✗" in out
    assert "Error: boom" in out
    assert "more lines" in out


def test_color_enabled_respects_no_color(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    assert not color_enabled(FakeTty())


def test_color_enabled_requires_tty(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")

    class Pipe:
        def isatty(self) -> bool:
            return False

    assert not color_enabled(Pipe())
    assert color_enabled(FakeTty())
