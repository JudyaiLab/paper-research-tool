"""Tests for paper_tool CLI."""

import subprocess
import sys
from pathlib import Path

import pytest

# Resolve project root relative to this test file
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


class TestCLI:
    """Test CLI entry point."""

    def test_version(self):
        result = subprocess.run(
            [sys.executable, "paper_tool.py", "--version"],
            capture_output=True, text=True,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0
        assert "0.3.1" in result.stdout

    def test_help(self):
        result = subprocess.run(
            [sys.executable, "paper_tool.py", "--help"],
            capture_output=True, text=True,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0
        assert "summarize" in result.stdout
        assert "serve" in result.stdout

    def test_config_show(self):
        result = subprocess.run(
            [sys.executable, "paper_tool.py", "config", "--show"],
            capture_output=True, text=True,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0
        assert "OpenAI" in result.stdout

    def test_list_empty(self):
        result = subprocess.run(
            [sys.executable, "paper_tool.py", "list"],
            capture_output=True, text=True,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0

    def test_search_no_results(self):
        result = subprocess.run(
            [sys.executable, "paper_tool.py", "search", "nonexistent_query_xyz"],
            capture_output=True, text=True,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0

    def test_summarize_nonexistent_file(self):
        result = subprocess.run(
            [sys.executable, "paper_tool.py", "summarize", "/nonexistent/file.pdf"],
            capture_output=True, text=True,
            cwd=PROJECT_ROOT,
        )
        # Should fail gracefully
        assert result.returncode != 0 or "Error" in result.stdout or "not found" in result.stdout.lower()

    def test_lang_switch(self):
        result = subprocess.run(
            [sys.executable, "paper_tool.py", "--lang", "en", "--help"],
            capture_output=True, text=True,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0
        assert "Paper Research Tool" in result.stdout
