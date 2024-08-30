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

from utils.string_slice import StringSlice
from .constants import std_size, maximum_int_as_str
from memory import memcpy, memcmp


fn standardize_string_slice(
    x: StringSlice,
) -> InlineArray[UInt8, size=std_size]:
    var standardized_x = InlineArray[UInt8, size=std_size](ord("0"))
    memcpy(
        dest_data=(standardized_x.unsafe_ptr() + std_size - len(x)).bitcast[
            Int8
        ](),
        src_data=x.unsafe_ptr().bitcast[Int8](),
        n=len(x),
    )
    return standardized_x


# The idea is to end up with a InlineArray of size
# 24, which is enough to store the largest integer
# that can be represented in 64 bits (size 19), and
# is also SIMD friendly because divisible by 8, 4, 2, 1.
# This 24 could be computed at compile time and adapted
# to the simd width and the base, but Mojo's compile-time
# computation is not yet powerful enough yet.
fn to_integer(x: String) raises -> Int:
    return to_integer(x.as_string_slice())


fn to_integer(x: StringSlice) raises -> Int:
    if len(x) > len(maximum_int_as_str):
        raise Error("The string size too big. '" + str(x) + "'")
    return to_integer(standardize_string_slice(x))


fn to_integer(standardized_x: InlineArray[UInt8, size=std_size]) raises -> Int:
    """Takes a inline array containing the ASCII representation of a number.
    It must be padded with "0" on the left. Using an InlineArray makes
    this SIMD friendly.

    The function returns the integer value represented by the input string.

    "000000000048642165487456" -> 48642165487456
    """

    # This could be done with simd if we see it's a bottleneck.
    for i in range(std_size):
        if not (UInt8(ord("0")) <= standardized_x[i] <= UInt8(ord("9"))):
            # We make a string out of this number. +1 for the null terminator.
            number_as_string = String._buffer_type(capacity=std_size + 1)
            for j in range(std_size):
                number_as_string.append(standardized_x[j])
            number_as_string.append(0)
            raise Error(
                "Invalid character(s) in the number: '"
                + String(number_as_string^)
                + "'"
            )

    # We assume there are no leading or trailing whitespaces, no leading zeros, no sign.
    # We could compute all those aliases at compile time, by knowing the size of int, simd width,
    # and the base of the number system. Here it only works for base 10.
    alias vector_with_exponents = InlineArray[
        SIMD[DType.index, 1], size=std_size
    ](
        0,
        0,
        0,
        0,
        0,
        1_000_000_000_000_000_000,
        100_000_000_000_000_000,
        10_000_000_000_000_000,
        1_000_000_000_000_000,
        100_000_000_000_000,
        10_000_000_000_000,
        1_000_000_000_000,
        100_000_000_000,
        10_000_000_000,
        1_000_000_000,
        100_000_000,
        10_000_000,
        1_000_000,
        100_000,
        10_000,
        1_000,
        100,
        10,
        1,
    )

    # 24 is not divisible by 16, so we stop at 8. Later on,
    # when we have better compile-time computation, we can
    # change 24 to be adapted to the simd width.
    alias simd_width = min(sys.simdwidthof[DType.index](), 8)

    var accumulator = SIMD[DType.index, simd_width](0)

    var max_standardized_x = "000009223372036854775807"
    if (
        memcmp(
            standardized_x.unsafe_ptr(),
            max_standardized_x.unsafe_ptr(),
            count=std_size,
        )
        == 1
    ):  # memcmp is pretty fast
        raise Error("The string is too large to be converted to an integer. '")
    _ = max_standardized_x

    @parameter
    for i in range(std_size // simd_width):
        var ascii_vector = (standardized_x.unsafe_ptr() + i * simd_width).load[
            width=simd_width
        ]()
        var as_digits = ascii_vector - SIMD[DType.uint8, simd_width](ord("0"))
        var as_digits_index = as_digits.cast[DType.index]()
        alias vector_slice = (
            vector_with_exponents.unsafe_ptr() + i * simd_width
        ).load[width=simd_width]()
        accumulator += as_digits_index * vector_slice
    return int(accumulator.reduce_add())
