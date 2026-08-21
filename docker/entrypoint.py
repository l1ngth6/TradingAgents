"""Prepare Docker-mounted state, drop root privileges, and start the CLI."""

from __future__ import annotations

import os
import pwd
import sys
from pathlib import Path


RUNTIME_USER = "appuser"
WRITABLE_PATHS = (
    Path("/home/appuser/.tradingagents"),
    Path("/home/appuser/app/reports"),
)


def _chown_tree_if_needed(path: Path, uid: int, gid: int) -> None:
    """Own a mounted tree when its root belongs to Docker's root user."""
    path.mkdir(parents=True, exist_ok=True)
    stat = path.lstat()
    if stat.st_uid == uid and stat.st_gid == gid:
        return

    # lchown plus followlinks=False prevents a mounted report symlink from
    # changing ownership outside the writable tree.
    for root, directories, files in os.walk(path, topdown=False, followlinks=False):
        for name in (*directories, *files):
            os.lchown(Path(root) / name, uid, gid)
        os.lchown(root, uid, gid)


def _drop_privileges(user: pwd.struct_passwd) -> None:
    os.initgroups(user.pw_name, user.pw_gid)
    os.setgid(user.pw_gid)
    os.setuid(user.pw_uid)
    os.environ.update(
        HOME=user.pw_dir,
        USER=user.pw_name,
        LOGNAME=user.pw_name,
    )


def main() -> None:
    user = pwd.getpwnam(RUNTIME_USER)

    if os.geteuid() == 0:
        for path in WRITABLE_PATHS:
            _chown_tree_if_needed(path, user.pw_uid, user.pw_gid)
        _drop_privileges(user)

    os.execvp("tradingagents", ["tradingagents", *sys.argv[1:]])


if __name__ == "__main__":
    main()
