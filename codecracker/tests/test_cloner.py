"""Security / hygiene tests for the git cloner."""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codecracker.cloner import (
    _clone_multi_options,
    _hooks_null_config,
    _safe_git_env,
    enforce_clone_rate_limit,
    normalize_repo_url,
    safe_clone_dest,
    sanitize_slug,
)
from codecracker.config import Settings


class TestSanitizeSlug:
    def test_accepts_normal_names(self) -> None:
        assert sanitize_slug("wallstead", kind="owner") == "wallstead"
        assert sanitize_slug("SQLit", kind="repo") == "SQLit"
        assert sanitize_slug("my.repo_name-1") == "my.repo_name-1"

    def test_rejects_traversal(self) -> None:
        with pytest.raises(ValueError, match="traversal|Invalid"):
            sanitize_slug("../etc")
        with pytest.raises(ValueError, match="separators|Invalid"):
            sanitize_slug("foo/bar")
        with pytest.raises(ValueError, match="separators|Invalid"):
            sanitize_slug("foo\\bar")
        with pytest.raises(ValueError):
            sanitize_slug("..")


class TestNormalizeRepoUrl:
    def test_github_https(self) -> None:
        url, owner, name = normalize_repo_url("https://github.com/wallstead/SQLit")
        assert url == "https://github.com/wallstead/SQLit.git"
        assert owner == "wallstead"
        assert name == "SQLit"

    def test_rejects_non_github_host(self) -> None:
        with pytest.raises(ValueError, match="github.com"):
            normalize_repo_url("https://evil.example/owner/repo")

    def test_rejects_traversal_in_url(self) -> None:
        with pytest.raises(ValueError):
            normalize_repo_url("https://github.com/../evil/repo")
        with pytest.raises(ValueError):
            normalize_repo_url("https://github.com/owner/../../tmp")


class TestSafeCloneDest:
    def test_stays_under_work_dir(self, tmp_path: Path) -> None:
        dest = safe_clone_dest(tmp_path, "owner", "repo")
        assert dest == (tmp_path / "owner__repo").resolve()
        assert dest.parent == tmp_path.resolve()

    def test_rejects_unsafe_segments(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            safe_clone_dest(tmp_path, "..", "repo")


class TestHooksDisabled:
    def test_hooks_path_is_devnull(self) -> None:
        assert os.devnull in _hooks_null_config()
        env = _safe_git_env({"PATH": "/usr/bin"})
        assert env["GIT_CONFIG_KEY_0"] == "core.hooksPath"
        assert env["GIT_CONFIG_VALUE_0"] == os.devnull
        assert env["PATH"] == "/usr/bin"

    def test_clone_options_include_hooks_config(self) -> None:
        settings = Settings(shallow_clone=True, clone_filter_blob_none=True)
        opts = _clone_multi_options(settings)
        assert "-c" in opts
        assert any(o.startswith("core.hooksPath=") for o in opts)
        assert "--depth=1" in opts
        assert "--filter=blob:none" in opts


class TestRateLimit:
    def test_fresh_clone_respects_interval(self, tmp_path: Path) -> None:
        settings = Settings(
            work_dir=tmp_path,
            output_dir=tmp_path / "out",
            min_clone_interval_seconds=2,
        )
        settings.ensure_dirs()
        enforce_clone_rate_limit(settings, is_fresh_clone=True)
        stamp = tmp_path / ".clone_rate_limit"
        assert stamp.exists()

        with patch("codecracker.cloner.time.sleep") as sleep:
            # Pretend only 0.1s elapsed
            stamp.write_text(str(time.time() - 0.1), encoding="utf-8")
            enforce_clone_rate_limit(settings, is_fresh_clone=True)
            sleep.assert_called()
            assert sleep.call_args[0][0] > 0

    def test_reuse_skips_rate_limit(self, tmp_path: Path) -> None:
        settings = Settings(
            work_dir=tmp_path,
            output_dir=tmp_path / "out",
            min_clone_interval_seconds=60,
        )
        with patch("codecracker.cloner.time.sleep") as sleep:
            enforce_clone_rate_limit(settings, is_fresh_clone=False)
            sleep.assert_not_called()


class TestCloneFromUsesHardening:
    def test_clone_from_called_with_hooks_env(self, tmp_path: Path) -> None:
        settings = Settings(
            work_dir=tmp_path / "repos",
            output_dir=tmp_path / "out",
            min_clone_interval_seconds=0,
            shallow_clone=True,
        )
        settings.ensure_dirs()

        fake_repo = MagicMock()
        with patch("codecracker.cloner.Repo") as repo_cls, patch(
            "codecracker.cloner._disable_hooks_in_repo"
        ) as disable:
            repo_cls.clone_from.return_value = fake_repo
            repo_cls.return_value = fake_repo
            from codecracker.cloner import clone_repo

            dest = clone_repo("https://github.com/wallstead/SQLit", settings=settings)

            repo_cls.clone_from.assert_called_once()
            kwargs = repo_cls.clone_from.call_args.kwargs
            assert kwargs["env"]["GIT_CONFIG_KEY_0"] == "core.hooksPath"
            assert kwargs["env"]["GIT_CONFIG_VALUE_0"] == os.devnull
            assert kwargs["allow_unsafe_options"] is True
            assert any(
                str(o).startswith("core.hooksPath=") for o in kwargs["multi_options"]
            )
            disable.assert_called()
            assert dest.name == "wallstead__SQLit"
