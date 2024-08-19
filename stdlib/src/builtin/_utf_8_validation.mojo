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

import sys


fn pshuf(
    lookup_table: SIMD[DType.uint8, 16], indices: SIMD[DType.uint8, 16]
) -> SIMD[DType.uint8, 16]:
    """This calls the pshuf instruction which is basically a table lookup.

    Take a look at the intel documentation for information on the pshuf instruction:
    https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_shuffle_epi8&ig_expand=6003

    The rust documentation also has a good explanation:
    https://doc.rust-lang.org/core/arch/x86_64/fn._mm_shuffle_epi8.html
    """
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
    """This function performs a table lookup on simd vectors.

    This can also be known as shuffle.
    For simplicity, we'll assume that the lookup table is always 16 bytes long
    and only the 4 least significant bits of the indices are set on each byte of the indices.
    Otherwise the behavior is undefined.
    """

    # TODO: make it fast for more architectures, notably arm.
    # And make it more generic if possible.
    @parameter
    if sys.has_sse4() and input_vector_size == 16:
        # The compiler isn't very smart yet and can't narrow "input_vector_size".
        # Let's help it a bit and hope it
        # understands this is a no-op. So far there wasn't any performance
        # drop likely indicating that the compiler is doing the right thing.
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
        # We split it in 4 and call the 16  bytes version 4 times.
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
        # Slow path in case we, ~3x slower than pshuf for size 16
        var result = SIMD[DType.uint8, input_vector_size]()

        @parameter
        for i in range(0, input_vector_size):
            result[i] = lookup_table[int(indices[i])]
        return result
