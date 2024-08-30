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


fn ensure_valid_integers(x: SIMD[DType.uint8, _]) -> SIMD[DType.bool, x.size]:
    alias min_values = SIMD[DType.uint8, x.size](ord("0"))
    alias max_values = SIMD[DType.uint8, x.size](ord("9"))
    return (min_values <= x) & (x <= max_values)


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


fn to_integer(x: String) raises -> Int:
    return to_integer(x.as_string_slice())


fn to_integer(x: StringSlice) raises -> Int:
    if len(x) > len(maximum_int_as_str):
        raise Error("The string size too big. '" + str(x) + "'")
    return to_integer(standardize_string_slice(x))


fn to_integer(standardized_x: InlineArray[UInt8, size=std_size]) raises -> Int:
    for i in range(std_size):
        if not (UInt8(ord("0")) <= standardized_x[i] <= UInt8(ord("9"))):
            # We make a string out of this number.
            number_as_string = List[UInt8](std_size)
            for j in range(std_size):
                number_as_string.append(standardized_x[j])
            raise Error(
                "Invalid character(s) in the number: '"
                + String(number_as_string)
                + "'"
            )

    # We assume there are no leading or trailing whitespaces, no leading zeros, no sign.
    # We could compute all those aliases at compile time, by knowing the size of int, simd width,
    # and the base of the number system. Here it only works for base 10.
    alias index_simd_width = sys.simdwidthof[DType.index]()
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

    var accumulator = SIMD[DType.index, sys.simdwidthof[DType.index]()](0)

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
    for i in range(std_size // index_simd_width):
        ascii_vector = (
            standardized_x.unsafe_ptr() + i * index_simd_width
        ).load[width = sys.simdwidthof[DType.index]()]()
        as_digits = ascii_vector - SIMD[
            DType.uint8, sys.simdwidthof[DType.index]()
        ](ord("0"))
        as_digits_index = as_digits.cast[DType.index]()
        alias vector_slice = (
            vector_with_exponents.unsafe_ptr() + i * index_simd_width
        ).load[width = sys.simdwidthof[DType.index]()]()
        accumulator += as_digits_index * vector_slice
    return int(accumulator.reduce_add())
