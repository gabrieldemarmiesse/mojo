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

"""
We make use of the following papers for the implementation, note that there
are some small differences.

Wojciech Muła, Daniel Lemire, Base64 encoding and decoding at almost the
speed of a memory copy, Software: Practice and Experience 50 (2), 2020.
https://arxiv.org/abs/1910.05109

Wojciech Muła, Daniel Lemire, Faster Base64 Encoding and Decoding using AVX2
Instructions, ACM Transactions on the Web 12 (3), 2018.
https://arxiv.org/abs/1704.00605

The reference implementation can be found here:
https://github.com/simdutf/simdutf/blob/master/src/haswell/avx2_base64.cpp
"""
from memory import UnsafePointer
from collections import InlineArray
from memory.maybe_uninitialized import UnsafeMaybeUninitialized

# TODO: Replace all .load[width=1]() with [] when the compiler/lsp stops throwing errors

# fmt: off
# TODO: generate at compile-time.

alias INV = 255 # invalid
alias WSP = 64  # whitespace

alias to_base64_value = InlineArray[UInt8, 256](
    INV, INV, INV, INV, INV, INV, INV, INV, INV, WSP,
    WSP, INV, WSP, WSP, INV, INV, INV, INV, INV, INV,
    INV, INV, INV, INV, INV, INV, INV, INV, INV, INV,
    INV, INV, WSP, INV, INV, INV, INV, INV, INV, INV,
    INV, INV, INV, 62 , INV, INV, INV, 63 , 52 , 53 ,
    54 , 55 , 56 , 57 , 58 , 59 , 60 , 61 , INV, INV,
    INV, INV, INV, INV, INV, 0  , 1  , 2  , 3  , 4  ,
    5  , 6  , 7  , 8  , 9  , 10 , 11 , 12 , 13 , 14 ,
    15 , 16 , 17 , 18 , 19 , 20 , 21 , 22 , 23 , 24 ,
    25 , INV, INV, INV, INV, INV, INV, 26 , 27 , 28 ,
    29 , 30 , 31 , 32 , 33 , 34 , 35 , 36 , 37 , 38 ,
    39 , 40 , 41 , 42 , 43 , 44 , 45 , 46 , 47 , 48 ,
    49 , 50 , 51 , INV, INV, INV, INV, INV, INV, INV,
    INV, INV, INV, INV, INV, INV, INV, INV, INV, INV,
    INV, INV, INV, INV, INV, INV, INV, INV, INV, INV,
    INV, INV, INV, INV, INV, INV, INV, INV, INV, INV,
    INV, INV, INV, INV, INV, INV, INV, INV, INV, INV,
    INV, INV, INV, INV, INV, INV, INV, INV, INV, INV,
    INV, INV, INV, INV, INV, INV, INV, INV, INV, INV,
    INV, INV, INV, INV, INV, INV, INV, INV, INV, INV,
    INV, INV, INV, INV, INV, INV, INV, INV, INV, INV,
    INV, INV, INV, INV, INV, INV, INV, INV, INV, INV,
    INV, INV, INV, INV, INV, INV, INV, INV, INV, INV,
    INV, INV, INV, INV, INV, INV, INV, INV, INV, INV,
    INV, INV, INV, INV, INV, INV, INV, INV, INV, INV,
    INV, INV, INV, INV, INV, INV,
)
# fmt: on


fn is_whitespace(x: UInt8) -> Bool:
    return to_base64_value[int(x)] == WSP


fn is_valid_base64(x: UInt8) -> Bool:
    return to_base64_value[int(x)] != INV


@value
struct TrimmingInfo:
    var new_src_length: Int
    var equal_signs: Int
    var equal_location: Int


fn trim_src_end(src: UnsafePointer[UInt8], src_length: Int) -> TrimmingInfo:
    """Returns the new source_length after trimming spaces and and equal signs.
    """
    # location of the first padding character if any
    var equal_location = src_length
    var src_length_after_trimming = src_length
    # skip trailing spaces

    var last_element = src.load[width=1](src_length_after_trimming - 1)
    while src_length_after_trimming > 0 and is_whitespace(last_element):
        src_length_after_trimming -= 1
        last_element = src.load[width=1](src_length_after_trimming - 1)

    var equal_signs = 0
    alias ord_equal = ord("=")
    if (
        src_length_after_trimming > 0
        and src[src_length_after_trimming - 1] == ord_equal
    ):
        equal_location = src_length_after_trimming - 1
        src_length_after_trimming -= 1
        equal_signs = 1
        # skip trailing spaces
        last_element = src[src_length_after_trimming - 1]
        while src_length_after_trimming > 0 and is_whitespace(last_element):
            src_length_after_trimming -= 1
            last_element = src[src_length_after_trimming - 1]

        if (
            src_length_after_trimming > 0
            and src[src_length_after_trimming - 1] == ord_equal
        ):
            equal_location = src_length_after_trimming - 1
            src_length_after_trimming -= 1
            equal_signs = 2

    return TrimmingInfo(src_length_after_trimming, equal_signs, equal_location)


fn to_base64_mask(block: SIMD[DType.uint8, 64]) -> Tuple[UInt64, Bool]:
    return (UInt64(0), False)


fn compress_block(
    inout b: SIMD[DType.uint8, 64],
    mask: UInt64,
    output_buffer: UnsafePointer[UInt8],
) -> UInt64:
    return UInt64(0)


fn base64_decode_block_safe(
    dst_writer_position: UnsafePointer[UInt8], b: SIMD[DType.uint8, 64]
):
    pass


fn base64_decode_block(
    dst_writer_position: UnsafePointer[UInt8], b: SIMD[DType.uint8, 64]
):
    pass


fn compress_decode_base64(
    *, dst: UnsafePointer[UInt8], src: UnsafePointer[UInt8], src_length: Int
) raises:
    var trimming_info = trim_src_end(src, src_length)

    var end_of_safe_64byte_zone: UnsafePointer[UInt8]
    if (trimming_info.new_src_length + 3) // 4 * 3 >= 63:
        end_of_safe_64byte_zone = dst + (
            (trimming_info.new_src_length + 3) // 4 * 3 - 63
        )
    else:
        end_of_safe_64byte_zone = dst

    var src_end = src + trimming_info.new_src_length
    var src_reader_position = src
    var dst_writer_position = dst
    alias block_size = 6
    var buffer = InlineArray[UnsafeMaybeUninitialized[UInt8], block_size * 64](
        unsafe_uninitialized=True
    )
    var buffer_pointer = buffer.unsafe_ptr().bitcast[UInt8]()

    if trimming_info.new_src_length >= 64:
        var src_end_64 = src_reader_position + trimming_info.new_src_length - 64
        while src_reader_position < src_end_64:
            var b = src_reader_position.load[width=64]()
            src_reader_position += 64
            to_base64_mask_tuple = to_base64_mask(b)
            var badcharmask = to_base64_mask_tuple[0]
            var error = to_base64_mask_tuple[1]
            if error:
                # We rewind and check the characters one by one.
                src_reader_position -= 64
                while src_reader_position < src_end and is_valid_base64(
                    src_reader_position.load[width=1]()
                ):
                    src_reader_position += 1
                raise Error(
                    "Invalid base64 character, value is "
                    + str(src_reader_position.load[width=1]())
                    + " at position: "
                    + str(int(src_reader_position) - int(src))
                )
            if badcharmask != 0:
                # optimization opportunity: check for simple masks like those made of
                # continuous 1s followed by continuous 0s. And masks containing a
                # single bad character.
                buffer_pointer += int(
                    compress_block(b, badcharmask, buffer_pointer)
                )
            elif buffer_pointer != buffer.unsafe_ptr().bitcast[UInt8]():
                buffer_pointer.store(b)
                buffer_pointer += 64
            else:
                if dst_writer_position >= end_of_safe_64byte_zone:
                    base64_decode_block_safe(dst_writer_position, b)
                else:
                    base64_decode_block(dst_writer_position, b)
                dst_writer_position += 48

    _ = buffer
