# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Implements the os.path operations.

You can import these APIs from the `os.path` package. For example:

```mojo
from os.path import isdir
```
"""

from stat import S_ISDIR, S_ISLNK, S_ISREG
from sys.info import has_neon, os_is_linux, os_is_macos, os_is_windows

from .. import PathLike
from .._linux_aarch64 import _lstat as _lstat_linux_arm
from .._linux_aarch64 import _stat as _stat_linux_arm
from .._linux_x86 import _lstat as _lstat_linux_x86
from .._linux_x86 import _stat as _stat_linux_x86
from .._macos import _lstat as _lstat_macos
from .._macos import _stat as _stat_macos


# ===----------------------------------------------------------------------=== #
# Utilities
# ===----------------------------------------------------------------------=== #
fn _constrain_unix():
    constrained[
        not os_is_windows(), "operating system must be Linux or macOS"
    ]()


@always_inline
fn _get_stat_st_mode(path: String) raises -> Int:
    @parameter
    if os_is_macos():
        return int(_stat_macos(path).st_mode)
    elif has_neon():
        return int(_stat_linux_arm(path).st_mode)
    else:
        return int(_stat_linux_x86(path).st_mode)


@always_inline
fn _get_lstat_st_mode(path: String) raises -> Int:
    @parameter
    if os_is_macos():
        return int(_lstat_macos(path).st_mode)
    elif has_neon():
        return int(_lstat_linux_arm(path).st_mode)
    else:
        return int(_lstat_linux_x86(path).st_mode)


# ===----------------------------------------------------------------------=== #
# isdir
# ===----------------------------------------------------------------------=== #
fn isdir(path: String) -> Bool:
    """Return True if path is an existing directory. This follows
    symbolic links, so both islink() and isdir() can be true for the same path.

    Args:
      path: The path to the directory.

    Returns:
      True if the path is a directory or a link to a directory and
      False otherwise.
    """
    _constrain_unix()
    try:
        var st_mode = _get_stat_st_mode(path)
        if S_ISDIR(st_mode):
            return True
        return S_ISLNK(st_mode) and S_ISDIR(_get_lstat_st_mode(path))
    except:
        return False


fn isdir[pathlike: os.PathLike](path: pathlike) -> Bool:
    """Return True if path is an existing directory. This follows
    symbolic links, so both islink() and isdir() can be true for the same path.

    Parameters:
      pathlike: The a type conforming to the os.PathLike trait.

    Args:
      path: The path to the directory.

    Returns:
      True if the path is a directory or a link to a directory and
      False otherwise.
    """
    return isdir(path.__fspath__())


# ===----------------------------------------------------------------------=== #
# isfile
# ===----------------------------------------------------------------------=== #


fn isfile(path: String) -> Bool:
    """Test whether a path is a regular file.

    Args:
      path: The path to the directory.

    Returns:
      Returns True if the path is a regular file.
    """
    _constrain_unix()
    try:
        var st_mode = _get_stat_st_mode(path)
        if S_ISREG(st_mode):
            return True
        return S_ISLNK(st_mode) and S_ISREG(_get_lstat_st_mode(path))
    except:
        return False


fn isfile[pathlike: os.PathLike](path: pathlike) -> Bool:
    """Test whether a path is a regular file.

    Parameters:
      pathlike: The a type conforming to the os.PathLike trait.

    Args:
      path: The path to the directory.

    Returns:
      Returns True if the path is a regular file.
    """
    return isfile(path.__fspath__())


# ===----------------------------------------------------------------------=== #
# islink
# ===----------------------------------------------------------------------=== #
fn islink(path: String) -> Bool:
    """Return True if path refers to an existing directory entry that is a
    symbolic link.

    Args:
      path: The path to the directory.

    Returns:
      True if the path is a link to a directory and False otherwise.
    """
    _constrain_unix()
    try:
        return S_ISLNK(_get_lstat_st_mode(path))
    except:
        return False


fn islink[pathlike: os.PathLike](path: pathlike) -> Bool:
    """Return True if path refers to an existing directory entry that is a
    symbolic link.

    Parameters:
      pathlike: The a type conforming to the os.PathLike trait.

    Args:
      path: The path to the directory.

    Returns:
      True if the path is a link to a directory and False otherwise.
    """
    return islink(path.__fspath__())


# ===----------------------------------------------------------------------=== #
# exists
# ===----------------------------------------------------------------------=== #


fn exists(path: String) -> Bool:
    """Return True if path exists.

    Args:
      path: The path to the directory.

    Returns:
      Returns True if the path exists and is not a broken symbolic link.
    """
    _constrain_unix()
    try:
        _ = _get_stat_st_mode(path)
        return True
    except:
        return False


fn exists[pathlike: os.PathLike](path: pathlike) -> Bool:
    """Return True if path exists.

    Parameters:
      pathlike: The a type conforming to the os.PathLike trait.

    Args:
      path: The path to the directory.

    Returns:
      Returns True if the path exists and is not a broken symbolic link.
    """
    return exists(path.__fspath__())


# ===----------------------------------------------------------------------=== #
# lexists
# ===----------------------------------------------------------------------=== #


fn lexists(path: String) -> Bool:
    """Return True if path exists or is a broken symlink.

    Args:
      path: The path to the directory.

    Returns:
      Returns True if the path exists or is a broken symbolic link.
    """
    _constrain_unix()
    try:
        _ = _get_lstat_st_mode(path)
        return True
    except:
        return False


fn lexists[pathlike: os.PathLike](path: pathlike) -> Bool:
    """Return True if path exists or is a broken symlink.

    Parameters:
      pathlike: The a type conforming to the os.PathLike trait.

    Args:
      path: The path to the directory.

    Returns:
      Returns True if the path exists or is a broken symbolic link.
    """
    return exists(path.__fspath__())
