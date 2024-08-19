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

from memory import memcpy
from sys._assembly import inlined_assembly
import sys
from testing import assert_true, assert_false
import benchmark

alias VECTOR_SIZE = 32
alias BytesVector = SIMD[DType.uint8, VECTOR_SIZE]
alias BoolsVector = SIMD[DType.bool, VECTOR_SIZE]


@value
struct ProcessedUtfBytes:
    var raw_bytes: BytesVector
    var high_nibbles: BytesVector
    var carried_continuations: BytesVector


fn pshuf(
    lookup_table: SIMD[DType.uint8, 16], indices: SIMD[DType.uint8, 16]
) -> SIMD[DType.uint8, 16]:
    return sys.llvm_intrinsic[
        "llvm.x86.ssse3.pshuf.b.128",
        SIMD[DType.uint8, 16],
        has_side_effect=False,
    ](lookup_table, indices)


@always_inline
fn simd_table_lookup[
    input_vector_size: Int
](
    lookup_table: SIMD[DType.uint8, 16],
    indices: SIMD[DType.uint8, input_vector_size],
) -> SIMD[DType.uint8, input_vector_size]:
    """The equivalent of https://doc.rust-lang.org/core/arch/x86_64/fn._mm_shuffle_epi8.html .

    For simplicity, we'll assume that unlike the rust version, the lookup table is always 16 bytes long
    and only the 4 least significant bits of the indices are set on each byte of the indices.
    Otherwise the behavior is undefined.
    """

    @parameter
    if sys.has_sse4() and input_vector_size == 16:
        # The compiler isn't very smart yet. Let's help it a bit and hope it
        # understands this is a no-op. So far there wasn't any performance
        # drop likely indicating that the compiler is doing the right thing.
        # we would need to look at the assembly to be sure.
        var copy_indices = indices.slice[16, offset=0]()  # no-op
        var result = pshuf(lookup_table, copy_indices)
        return result.slice[input_vector_size, offset=0]()  # no-op
    elif sys.has_sse4() and input_vector_size == 32:
        # We split it in two and call the 16 version twice.
        var first_indices_batch = indices.slice[16, offset=0]()
        var second_indices_batch = indices.slice[16, offset=16]()
        var first_result = simd_table_lookup(lookup_table, first_indices_batch)
        var second_result = simd_table_lookup(
            lookup_table, second_indices_batch
        )
        var result = first_result.join(second_result)
        # no-op but needed for the type checker
        return result.slice[input_vector_size, offset=0]()
    elif sys.has_sse4() and input_vector_size == 64:
        # We split it in two and call the 16 version 4 times.
        var first_indices_batch = indices.slice[16, offset=0]()
        var second_indices_batch = indices.slice[16, offset=16]()
        var third_indices_batch = indices.slice[16, offset=32]()
        var fourth_indices_batch = indices.slice[16, offset=48]()
        var first_result = simd_table_lookup(lookup_table, first_indices_batch)
        var second_result = simd_table_lookup(
            lookup_table, second_indices_batch
        )
        var third_result = simd_table_lookup(lookup_table, third_indices_batch)
        var fourth_result = simd_table_lookup(
            lookup_table, fourth_indices_batch
        )
        var first_half = first_result.join(second_result)
        var second_half = third_result.join(fourth_result)
        var result = first_half.join(second_half)
        # no-op but needed for the type checker
        return result.slice[input_vector_size, offset=0]()
    else:
        # Slow path, ~3x slower than pshuf for size 16
        var result = SIMD[DType.uint8, input_vector_size]()

        @parameter
        for i in range(0, input_vector_size):
            result[i] = lookup_table[int(indices[i])]
        return result


def test_simd_table_lookup():
    var lookup_table = SIMD[DType.uint8, 16](
        0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150
    )
    var indices = SIMD[DType.uint8, 16](
        3, 3, 5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15, 0, 1
    )
    result = simd_table_lookup(lookup_table, indices)
    expected_result = SIMD[DType.uint8, 16](
        30, 30, 50, 50, 70, 70, 90, 90, 110, 110, 130, 130, 150, 150, 0, 10
    )
    assert_true((result == expected_result).reduce_and())


def test_simd_table_lookup_size_8():
    var lookup_table = SIMD[DType.uint8, 16](
        0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150
    )

    # Let's use size 8
    indices = SIMD[DType.uint8, 8](3, 3, 5, 5, 7, 7, 9, 0)

    result = simd_table_lookup(lookup_table, indices)
    expected_result = SIMD[DType.uint8, 8](30, 30, 50, 50, 70, 70, 90, 0)
    assert_true((result == expected_result).reduce_and())


def test_simd_table_lookup_size_32():
    var table_lookup = SIMD[DType.uint8, 16](
        0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150
    )
    var indices = SIMD[DType.uint8, 32](
        3,
        3,
        5,
        5,
        7,
        7,
        9,
        9,
        11,
        11,
        13,
        13,
        15,
        15,
        0,
        1,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
    )
    result = simd_table_lookup(table_lookup, indices)

    expected_result = SIMD[DType.uint8, 32](
        30,
        30,
        50,
        50,
        70,
        70,
        90,
        90,
        110,
        110,
        130,
        130,
        150,
        150,
        0,
        10,
        0,
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
        110,
        120,
        130,
        140,
        150,
    )
    assert_true((result == expected_result).reduce_and())


@always_inline
fn _mm_alignr_epi8[count: Int](a: BytesVector, b: BytesVector) -> BytesVector:
    """The equivalent of https://doc.rust-lang.org/core/arch/x86_64/fn._mm_alignr_epi8.html .
    """
    # This can be one instruction with ssse3. We can maybe force the compiler to use it.
    var concatenated = a.join(b)
    var shifted = concatenated.rotate_left[count]()
    return shifted.slice[a.size, offset = a.size]()


@always_inline
fn subtract_with_saturation[b: Int](a: BytesVector) -> BytesVector:
    """The equivalent of https://doc.rust-lang.org/core/arch/x86_64/fn._mm_subs_epu8.html .
    """
    alias b_as_vector = BytesVector(b)
    return max(a, b_as_vector) - b_as_vector


@always_inline
fn count_nibbles(bytes: BytesVector, inout answer: ProcessedUtfBytes):
    answer.raw_bytes = bytes
    answer.high_nibbles = bytes >> 4


@always_inline
fn check_smaller_than_0xF4(
    current_bytes: BytesVector, inout has_error: BoolsVector
):
    var bigger_than_0xF4 = current_bytes > BytesVector(0xF4)
    has_error |= bigger_than_0xF4


@always_inline
fn continuation_lengths(high_nibbles: BytesVector) -> BytesVector:
    # The idea is to end up with this pattern:
    # Input:  0xxxxxxx, 110xxxxx, 10xxxxxx, 1110xxxx, 10xxxxxx, 10xxxxxx, 10xxxxxx, 1111xxxx,
    # Output: 1       , 2,      , 0       , 3,      , 0       , 0       , 0       , 4

    alias table_of_continuations = SIMD[DType.uint8, 16](
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,  # 0xxx (ASCII)
        0,
        0,
        0,
        0,  # 10xx (continuation)
        2,
        2,  # 110x
        3,  # 1110
        4,  # 1111, next should be 0 (not checked here)
    )
    # Use
    return simd_table_lookup(table_of_continuations, high_nibbles)


@always_inline
fn carry_continuations(
    initial_lengths: BytesVector, previous_carries: BytesVector
) -> BytesVector:
    var right1 = subtract_with_saturation[1](
        _mm_alignr_epi8[VECTOR_SIZE - 1](initial_lengths, previous_carries)
    )
    # Input:           0xxxxxxx, 110xxxxx, 10xxxxxx, 1110xxxx, 10xxxxxx, 10xxxxxx, 10xxxxxx, 1111xxxx,
    # initial_lengths: 1       , 2,      , 0       , 3,      , 0       , 0       , 0       , 4
    # right1           ?       , 0       , 1       , 0       , 2,      , 0       , 0       , 0
    var sum = initial_lengths + right1
    # sum              ?       , 2       , 1       , 3       , 2,      , 0       , 0       , 4

    var right2 = subtract_with_saturation[2](
        _mm_alignr_epi8[VECTOR_SIZE - 2](sum, previous_carries)
    )
    # right2           ?       , ?       , ?       , 0       , 0,      , 1       , 0       , 0
    return sum + right2
    # return           ?       , ?       , ?       , 3       , 2,      , 1       , 0       , 4


@always_inline
fn check_continuations(
    initial_lengths: BytesVector,
    carries: BytesVector,
    inout has_error: BoolsVector,
):
    var overunder = (initial_lengths < carries) == (
        BytesVector(0) < initial_lengths
    )
    has_error |= overunder


@always_inline
fn check_first_continuation_max(
    # When 0b11101101 is found, next byte must be no larger than 0b10011111
    # When 0b11110100 is found, next byte must be no larger than 0b10001111
    # Next byte must be continuation, ie sign bit is set, so signed < is ok
    current_bytes: BytesVector,
    off1_current_bytes: BytesVector,
    inout has_error: BoolsVector,
):
    var bad_follow_ED = (BytesVector(0b10011111) < current_bytes) & (
        off1_current_bytes == BytesVector(0b11101101)
    )
    var bad_follow_F4 = (BytesVector(0b10001111) < current_bytes) & (
        off1_current_bytes == BytesVector(0b11110100)
    )

    has_error |= bad_follow_ED | bad_follow_F4


@always_inline
fn check_overlong(
    current_bytes: BytesVector,
    off1_current_bytes: BytesVector,
    hibits: BytesVector,
    previous_hibits: BytesVector,
    inout has_error: BoolsVector,
):
    """Map off1_hibits => error condition.

    hibits     off1    cur
    C       => < C2 && true
    E       => < E1 && < A0
    F       => < F1 && < 90
    else      false && false
    """
    var off1_hibits = _mm_alignr_epi8[VECTOR_SIZE - 1](hibits, previous_hibits)
    alias table1 = SIMD[DType.uint8, 16](
        0x80,
        0x80,
        0x80,
        0x80,
        0x80,
        0x80,
        0x80,
        0x80,
        0x80,
        0x80,
        0x80,
        0x80,  # 10xx => false
        0xC2,
        0x80,  # 110x
        0xE1,  # 1110
        0xF1,
    )
    var initial_mins = simd_table_lookup(table1, off1_hibits).cast[DType.int8]()
    var initial_under = off1_current_bytes.cast[DType.int8]() < initial_mins
    alias table2 = SIMD[DType.uint8, 16](
        0x80,
        0x80,
        0x80,
        0x80,
        0x80,
        0x80,
        0x80,
        0x80,
        0x80,
        0x80,
        0x80,
        0x80,  # 10xx => False
        127,
        127,  # 110x => True
        0xA0,  # 1110
        0x90,
    )
    var second_mins = simd_table_lookup(table2, off1_hibits).cast[DType.int8]()
    var second_under = current_bytes.cast[DType.int8]() < second_mins
    has_error |= initial_under & second_under


@no_inline
fn check_utf8_bytes(
    current_bytes: BytesVector,
    previous: ProcessedUtfBytes,
    inout has_error: BoolsVector,
) -> ProcessedUtfBytes:
    var pb = ProcessedUtfBytes(BytesVector(), BytesVector(), BytesVector())
    count_nibbles(current_bytes, pb)
    check_smaller_than_0xF4(current_bytes, has_error)

    var initial_lengths = continuation_lengths(pb.high_nibbles)
    pb.carried_continuations = carry_continuations(
        initial_lengths, previous.carried_continuations
    )

    check_continuations(initial_lengths, pb.carried_continuations, has_error)
    var off1_current_bytes = _mm_alignr_epi8[VECTOR_SIZE - 1](
        pb.raw_bytes, previous.raw_bytes
    )
    check_first_continuation_max(current_bytes, off1_current_bytes, has_error)
    check_overlong(
        current_bytes,
        off1_current_bytes,
        pb.high_nibbles,
        previous.high_nibbles,
        has_error,
    )

    return pb


fn get_last_carried_continuation_check_vector() -> BytesVector:
    """Returns a vector (9, 9, 9, 9, ... 9, 9, 9, 1)."""
    result = BytesVector(9)
    result[VECTOR_SIZE - 1] = 1
    return result


@no_inline
fn validate_utf8_fast(source: UnsafePointer[UInt8], length: Int) -> Bool:
    var i: Int = 0
    var has_error = BoolsVector()
    var previous = ProcessedUtfBytes(
        BytesVector(), BytesVector(), BytesVector()
    )
    while i + VECTOR_SIZE <= length:
        var current_bytes = (source + i).load[width=VECTOR_SIZE]()
        previous = check_utf8_bytes(current_bytes, previous, has_error)
        if has_error.reduce_or():
            return False
        i += VECTOR_SIZE

    # last part
    if i != length:
        var buffer = BytesVector(0)
        for j in range(i, length):
            buffer[j - i] = (source + j)[]
        previous = check_utf8_bytes(buffer, previous, has_error)
    else:
        # Just check that the last carried_continuations is 1 or 0
        alias base = get_last_carried_continuation_check_vector()
        var comparison = previous.carried_continuations > base
        has_error |= comparison

    return not has_error.reduce_or()


@no_inline
fn validate_utf8_fast(string: String) -> Bool:
    return validate_utf8_fast(string.unsafe_ptr(), len(string))


fn ljust8(string: String, char: String = " ") -> String:
    return (char * (8 - len(string))) + string


fn display_as_hex(string: String):
    for i in range(0, len(string)):
        if i % 16 == 0:
            print("##", end="")
        print(ljust8(hex(string._buffer[i], prefix="")) + "|", end="")
    print("")


fn display_as_bin(string: String):
    for i in range(0, len(string)):
        if i % 16 == 0:
            print("##", end="")
        print(ljust8(bin(string._buffer[i], prefix=""), char="0") + "|", end="")
    print("")


fn display_as_number(string: String):
    for i in range(0, len(string)):
        if i % 16 == 0:
            print("##", end="")
        print(ljust8(str(string._buffer[i])) + "|", end="")
    print("")


fn display_all(string: String):
    display_as_hex(string)
    display_as_bin(string)
    display_as_number(string)


alias GOOD_SEQUENCES = List[String](
    "a",
    "\xc3\xb1",
    "\xe2\x82\xa1",
    "\xf0\x90\x8c\xbc",
    "안녕하세요, 세상",
    "\xc2\x80",  # 6.7.2
    "\xf0\x90\x80\x80",  # 6.7.4
    "\xee\x80\x80",  # 6.11.2
    "very very very long string 🔥🔥🔥",
)


alias BAD_SEQUENCES = List[String](
    "\xc3\x28",  # 0
    "\xa0\xa1",  # 1
    "\xe2\x28\xa1",  # 2
    "\xe2\x82\x28",  # 3
    "\xf0\x28\x8c\xbc",  # 4
    "\xf0\x90\x28\xbc",  # 5
    "\xf0\x28\x8c\x28",  # 6
    "\xc0\x9f",  # 7
    "\xf5\xff\xff\xff",  # 8
    "\xed\xa0\x81",  # 9
    "\xf8\x90\x80\x80\x80",  # 10
    "123456789012345\xed",  # 11
    "123456789012345\xf1",  # 12
    "123456789012345\xc2",  # 13
    "\xC2\x7F",  # 14
    "\xce",  # 6.6.1
    "\xce\xba\xe1",  # 6.6.3
    "\xce\xba\xe1\xbd",  # 6.6.4
    "\xce\xba\xe1\xbd\xb9\xcf",  # 6.6.6
    "\xce\xba\xe1\xbd\xb9\xcf\x83\xce",  # 6.6.8
    "\xce\xba\xe1\xbd\xb9\xcf\x83\xce\xbc\xce",  # 6.6.10
    "\xdf",  # 6.14.6
    "\xef\xbf",  # 6.14.7
)


def test_good_sequences():
    for sequence in GOOD_SEQUENCES:
        assert_true(validate_utf8_fast(sequence[]))


def test_bad_sequences():
    for sequence in BAD_SEQUENCES:
        assert_false(validate_utf8_fast(sequence[]))


def test_combinaison_good_sequences():
    # any combinaison of good sequences should be good
    for i in range(0, len(GOOD_SEQUENCES)):
        for j in range(i, len(GOOD_SEQUENCES)):
            var sequence = GOOD_SEQUENCES[i] + GOOD_SEQUENCES[j]
            assert_true(validate_utf8_fast(sequence))


def test_combinaison_bad_sequences():
    # any combinaison of bad sequences should be bad
    for i in range(0, len(BAD_SEQUENCES)):
        for j in range(i, len(BAD_SEQUENCES)):
            var sequence = BAD_SEQUENCES[i] + BAD_SEQUENCES[j]
            assert_false(validate_utf8_fast(sequence))


def test_combinaison_good_bad_sequences():
    # any combinaison of good and bad sequences should be bad
    for i in range(0, len(GOOD_SEQUENCES)):
        for j in range(0, len(BAD_SEQUENCES)):
            var sequence = GOOD_SEQUENCES[i] + BAD_SEQUENCES[j]
            assert_false(validate_utf8_fast(sequence))


def test_combinaison_10_good_sequences():
    # any 10 combinaison of good sequences should be good
    for i in range(0, len(GOOD_SEQUENCES)):
        for j in range(i, len(GOOD_SEQUENCES)):
            var sequence = GOOD_SEQUENCES[i] * 10 + GOOD_SEQUENCES[j] * 10
            assert_true(validate_utf8_fast(sequence))


def test_combinaison_10_good_10_bad_sequences():
    # any 10 combinaison of good and bad sequences should be bad
    for i in range(0, len(GOOD_SEQUENCES)):
        for j in range(0, len(BAD_SEQUENCES)):
            var sequence = GOOD_SEQUENCES[i] * 10 + BAD_SEQUENCES[j] * 10
            assert_false(validate_utf8_fast(sequence))


fn get_big_string() -> String:
    var string = str(
        "안녕하세요,세상 hello mojo! 🔥🔥hopefully this string is complicated enough :p"
        " éç__çè"
    )
    # The string is 100 bytes long.
    return string * 100_000  # 10MB


fn benchmark_big_string():
    var big_string = get_big_string()

    @parameter
    fn utf8_simd_validation_benchmark():
        # we want to validate ~1gb of data
        for _ in range(100):
            var result = validate_utf8_fast(big_string)
            benchmark.keep(result)

    var report = benchmark.run[utf8_simd_validation_benchmark](
        min_runtime_secs=5
    )
    report.print()
    print(1.0 / report.mean(), "GB/s")
    _ = big_string


def main():
    print(sys.has_avx(), "have sse4")
    print(sys.has_avx2(), "have avx2")
    print(sys.simdbytewidth(), "simd byte width")
    #
    # test_good_sequences()
    # test_bad_sequences()
    # test_combinaison_good_sequences()
    # test_combinaison_bad_sequences()
    # test_combinaison_good_bad_sequences()
    # test_combinaison_10_good_sequences()
    # test_combinaison_10_good_10_bad_sequences()
    # print("All tests passed")

    benchmark_big_string()
