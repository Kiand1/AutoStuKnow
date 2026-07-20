from pathlib import Path

import pytest

from app.catalog import (
    DirectoryCatalog,
    directory_ancestors,
    normalize_directory_path,
    path_is_within,
)


def test_directory_paths_are_normalized_and_nested() -> None:
    assert normalize_directory_path(" 投资研究 \\ 虚拟币 / BTC ") == "投资研究/虚拟币/BTC"
    assert directory_ancestors("投资研究/虚拟币/BTC") == [
        "投资研究",
        "投资研究/虚拟币",
        "投资研究/虚拟币/BTC",
    ]
    assert path_is_within("投资研究/虚拟币/BTC", "投资研究/虚拟币")
    assert not path_is_within("投资研究/股票", "投资研究/虚拟币")


@pytest.mark.parametrize("path", ["../秘密", "投资/../秘密", "a/./b"])
def test_directory_paths_reject_traversal(path: str) -> None:
    with pytest.raises(ValueError):
        normalize_directory_path(path)


def test_catalog_persists_and_deletes_only_selected_subtree(tmp_path: Path) -> None:
    catalog = DirectoryCatalog(tmp_path)
    catalog.create("research", "投资/虚拟币/BTC")
    catalog.create("research", "投资/股票")
    catalog.create("research", "软件开发/Python")

    restarted = DirectoryCatalog(tmp_path)
    assert restarted.list_paths("research") == [
        "投资",
        "投资/股票",
        "投资/虚拟币",
        "投资/虚拟币/BTC",
        "软件开发",
        "软件开发/Python",
    ]

    removed = restarted.delete("research", "投资/虚拟币")

    assert removed == ["投资/虚拟币", "投资/虚拟币/BTC"]
    assert restarted.list_paths("research") == [
        "投资",
        "投资/股票",
        "软件开发",
        "软件开发/Python",
    ]
