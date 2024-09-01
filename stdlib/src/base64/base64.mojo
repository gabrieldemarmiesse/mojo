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
"""Provides functions for base64 encoding strings.

You can import these APIs from the `base64` package. For example:

```mojo
from base64 import b64encode
```
"""

from collections import List
from sys import simdwidthof
import bit

# ===----------------------------------------------------------------------===#
# Utilities
# ===----------------------------------------------------------------------===#


@always_inline
fn _ascii_to_value(char: String) -> Int:
    """Converts an ASCII character to its integer value for base64 decoding.

    Args:
        char: A single character string.

    Returns:
        The integer value of the character for base64 decoding, or -1 if invalid.
    """
    var char_val = ord(char)

    if char == "=":
        return 0
    elif ord("A") <= char_val <= ord("Z"):
        return char_val - ord("A")
    elif ord("a") <= char_val <= ord("z"):
        return char_val - ord("a") + 26
    elif ord("0") <= char_val <= ord("9"):
        return char_val - ord("0") + 52
    elif char == "+":
        return 62
    elif char == "/":
        return 63
    else:
        return -1


# ===----------------------------------------------------------------------===#
# b64encode
# ===----------------------------------------------------------------------===#


@always_inline
fn _subtract_with_saturation[
    simd_size: Int, //, b: Int
](a: SIMD[DType.uint8, simd_size]) -> SIMD[DType.uint8, simd_size]:
    """The equivalent of https://doc.rust-lang.org/core/arch/x86_64/fn._mm_subs_epu8.html .
    This can be a single instruction on some architectures.
    """
    alias b_as_vector = SIMD[DType.uint8, simd_size](b)
    return max(a, b_as_vector) - b_as_vector


"""
| 6-bit Value | ASCII Range | Target index | Offset (6-bit to ASCII) |
|-------------|-------------|--------------|-------------------------|
| 0 ... 25    | A ... Z     | 13           | 65                      |
| 26 ... 51   | a ... z     | 0            | 71                      |
| 52 ... 61   | 0 ... 9     | 1 ... 10     | -4                      |
| 62          | +           | 11           | -19                     |
| 63          | /           | 12           | -16                     |
"""
alias UNUSED = 0
alias TABLE_BASE64_OFFSETS = SIMD[DType.uint8, 16](
    71, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -19, -16, 65, UNUSED, UNUSED
)


fn _bitcast[
    new_dtype: DType, new_size: Int
](owned input: SIMD) -> SIMD[new_dtype, new_size]:
    var result = UnsafePointer.address_of(input).bitcast[
        SIMD[new_dtype, new_size]
    ]()[]
    return result


fn b64encode_simd(input_bytes: List[UInt8, _]) -> String:
    # +1 for the null terminator and +1 to be sure
    var result = List[UInt8, True](capacity=int(len(input_bytes) * (4 / 3)) + 2)
    b64encode_simd(input_bytes, result)
    return String(result^)


fn b64encode_simd(input_bytes: List[UInt8, _], inout result: List[UInt8, _]):
    """Performs base64 encoding on the input string using SIMD instructions.

    Args:
        input_bytes: The input string.

    Returns:
        Base64 encoding of the input string.
    """
    alias simd_width = 16  # TODO: Make this flexible
    alias input_simd_width = 12  # 16 * 0.75
    alias constant_13 = SIMD[DType.uint8, 16](13)

    # TODO: add condition on cpu flags
    var input_index = 0
    while input_index + input_simd_width <= len(input_bytes):
        # We don't want to read past the input buffer
        var start_of_input_chunk = input_bytes.unsafe_ptr() + input_index
        alias load_mask = SIMD[DType.bool, 16](
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
        )
        alias passthrough = SIMD[DType.uint8, 16](0)
        # TODO: Only do a masked_load at the end.
        var input_vector = sys.intrinsics.masked_load(
            start_of_input_chunk, mask=load_mask, passthrough=passthrough
        )

        # We reorder the bytes to fall in their correct 4 bytes chunks, 15 is a dummy value
        alias UNUSED_2 = 15
        alias shuffle_mask = SIMD[DType.uint8, 16](
            0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11
        )
        var shuffled_vector = input_vector._dynamic_shuffle(shuffle_mask)

        # We have 4 different masks to extract each group of 6 bits from the 4 bytes
        alias mask_1 = SIMD[DType.uint8, 16](
            0b11111100,
            0,
            0,
            0,
            0b11111100,
            0,
            0,
            0,
            0b11111100,
            0,
            0,
            0,
            0b11111100,
            0,
            0,
            0,
        )
        var masked_1 = shuffled_vector & mask_1
        var shifted_1 = masked_1 >> 2

        alias mask_2 = SIMD[DType.uint8, 16](
            0b00000011,
            0b11110000,
            0,
            0,
            0b00000011,
            0b11110000,
            0,
            0,
            0b00000011,
            0b11110000,
            0,
            0,
            0b00000011,
            0b11110000,
            0,
            0,
        )
        var masked_2 = shuffled_vector & mask_2
        var masked_2_as_uint16 = _bitcast[DType.uint16, 8](masked_2)
        var rotated_2 = bit.rotate_bits_right[4](masked_2_as_uint16)
        var shifted_2 = _bitcast[DType.uint8, 16](rotated_2)

        alias mask_3 = SIMD[DType.uint8, 16](
            0,
            0,
            0b00001111,
            0b11000000,
            0,
            0,
            0b00001111,
            0b11000000,
            0,
            0,
            0b00001111,
            0b11000000,
            0,
            0,
            0b00001111,
            0b11000000,
        )
        var masked_3 = shuffled_vector & mask_3
        var masked_3_as_uint16 = _bitcast[DType.uint16, 8](masked_3)
        var rotated_3 = bit.rotate_bits_left[2](masked_3_as_uint16)
        var shifted_3 = _bitcast[DType.uint8, 16](rotated_3)

        alias mask_4 = SIMD[DType.uint8, 16](
            0,
            0,
            0,
            0b00111111,
            0,
            0,
            0,
            0b00111111,
            0,
            0,
            0,
            0b00111111,
            0,
            0,
            0,
            0b00111111,
        )
        var shifted_4 = shuffled_vector & mask_4

        var ready_to_encode_per_byte = shifted_1 | shifted_2 | shifted_3 | shifted_4
        # See the table above for the offsets, we try to go from 6-bits values to target indexes.
        var saturated = _subtract_with_saturation[51](ready_to_encode_per_byte)

        var mask_below_25 = ready_to_encode_per_byte <= 25

        # Now are have the target indexes
        var indices = mask_below_25.select(constant_13, saturated)

        var offsets = TABLE_BASE64_OFFSETS._dynamic_shuffle(indices)

        var result_vector = ready_to_encode_per_byte + offsets

        # We write the result to the output buffer
        (result.unsafe_ptr() + len(result)).store(result_vector)
        result.size += simd_width
        input_index += input_simd_width

    # TODO: Handle the last pieces of the input buffer
    # null-terminate the result
    result.append(0)


fn b64encode(str: String) -> String:
    var out = String._buffer_type(capacity=str.byte_length() + 1)
    b64encode(str, out=out)
    return String(out^)


fn b64encode(str: String, inout out: List[UInt8, True]):
    """Performs base64 encoding on the input string.

    Args:
      str: The input string.

    Returns:
      Base64 encoding of the input string.
    """
    alias lookup = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    var b64chars = lookup.unsafe_ptr()

    var length = str.byte_length()

    @parameter
    @always_inline
    fn s(idx: Int) -> Int:
        return int(str.unsafe_ptr()[idx])

    # This algorithm is based on https://arxiv.org/abs/1704.00605
    var end = length - (length % 3)
    for i in range(0, end, 3):
        var si = s(i)
        var si_1 = s(i + 1)
        var si_2 = s(i + 2)
        out.append(b64chars[si // 4])
        out.append(b64chars[((si * 16) % 64) + si_1 // 16])
        out.append(b64chars[((si_1 * 4) % 64) + si_2 // 64])
        out.append(b64chars[si_2 % 64])

    if end < length:
        var si = s(end)
        out.append(b64chars[si // 4])
        if end == length - 1:
            out.append(b64chars[(si * 16) % 64])
            out.append(ord("="))
        elif end == length - 2:
            var si_1 = s(end + 1)
            out.append(b64chars[((si * 16) % 64) + si_1 // 16])
            out.append(b64chars[(si_1 * 4) % 64])
        out.append(ord("="))
    out.append(0)


# ===----------------------------------------------------------------------===#
# b64decode
# ===----------------------------------------------------------------------===#


@always_inline
fn b64decode(str: String) -> String:
    """Performs base64 decoding on the input string.

    Args:
      str: A base64 encoded string.

    Returns:
      The decoded string.
    """
    var n = str.byte_length()
    debug_assert(n % 4 == 0, "Input length must be divisible by 4")

    var p = String._buffer_type(capacity=n + 1)

    # This algorithm is based on https://arxiv.org/abs/1704.00605
    for i in range(0, n, 4):
        var a = _ascii_to_value(str[i])
        var b = _ascii_to_value(str[i + 1])
        var c = _ascii_to_value(str[i + 2])
        var d = _ascii_to_value(str[i + 3])

        debug_assert(
            a >= 0 and b >= 0 and c >= 0 and d >= 0,
            "Unexpected character encountered",
        )

        p.append((a << 2) | (b >> 4))
        if str[i + 2] == "=":
            break

        p.append(((b & 0x0F) << 4) | (c >> 2))

        if str[i + 3] == "=":
            break

        p.append(((c & 0x03) << 6) | d)

    p.append(0)
    return p


# ===----------------------------------------------------------------------===#
# b16encode
# ===----------------------------------------------------------------------===#


fn b16encode(str: String) -> String:
    """Performs base16 encoding on the input string.

    Args:
      str: The input string.

    Returns:
      Base16 encoding of the input string.
    """
    alias lookup = "0123456789ABCDEF"
    var b16chars = lookup.unsafe_ptr()

    var length = str.byte_length()
    var out = List[UInt8](capacity=length * 2 + 1)

    @parameter
    @always_inline
    fn str_bytes(idx: UInt8) -> UInt8:
        return str._buffer[int(idx)]

    for i in range(length):
        var str_byte = str_bytes(i)
        var hi = str_byte >> 4
        var lo = str_byte & 0b1111
        out.append(b16chars[int(hi)])
        out.append(b16chars[int(lo)])

    out.append(0)

    return String(out^)


# ===----------------------------------------------------------------------===#
# b16decode
# ===----------------------------------------------------------------------===#


@always_inline
fn b16decode(str: String) -> String:
    """Performs base16 decoding on the input string.

    Args:
      str: A base16 encoded string.

    Returns:
      The decoded string.
    """

    # TODO: Replace with dict literal when possible
    @parameter
    @always_inline
    fn decode(c: String) -> Int:
        var char_val = ord(c)

        if ord("A") <= char_val <= ord("Z"):
            return char_val - ord("A") + 10
        elif ord("a") <= char_val <= ord("z"):
            return char_val - ord("a") + 10
        elif ord("0") <= char_val <= ord("9"):
            return char_val - ord("0")

        return -1

    var n = str.byte_length()
    debug_assert(n % 2 == 0, "Input length must be divisible by 2")

    var p = List[UInt8](capacity=n // 2 + 1)

    for i in range(0, n, 2):
        var hi = str[i]
        var lo = str[i + 1]
        p.append(decode(hi) << 4 | decode(lo))

    p.append(0)
    return p
