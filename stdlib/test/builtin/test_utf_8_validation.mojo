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
# RUN: %mojo %s

from stdlib.builtin._utf_8_validation import simd_table_lookup
from testing import assert_true, assert_false, assert_equal


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
    assert_equal(result, expected_result)


def test_simd_table_lookup_size_8():
    var lookup_table = SIMD[DType.uint8, 16](
        0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150
    )

    # Let's use size 8
    indices = SIMD[DType.uint8, 8](3, 3, 5, 5, 7, 7, 9, 0)

    result = simd_table_lookup(lookup_table, indices)
    expected_result = SIMD[DType.uint8, 8](30, 30, 50, 50, 70, 70, 90, 0)
    assert_equal(result, expected_result)


def test_simd_table_lookup_size_32():
    var table_lookup = SIMD[DType.uint8, 16](
        0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150
    )
    # fmt: off
    var indices = SIMD[DType.uint8, 32](
        3 , 3 , 5 , 5 , 7 , 7 , 9 , 9 , 
        11, 11, 13, 13, 15, 15, 0 , 1 , 
        0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 
        8 , 9 , 10, 11, 12, 13, 14, 15,
    )
    result = simd_table_lookup(table_lookup, indices)
    
    expected_result = SIMD[DType.uint8, 32](
        30 , 30 , 50 , 50 , 70 , 70 , 90 , 90 ,
        110, 110, 130, 130, 150, 150, 0  , 10 ,
        0  , 10 , 20 , 30 , 40 , 50 , 60 , 70 ,
        80 , 90 , 100, 110, 120, 130, 140, 150,
    )
    # fmt: on
    assert_equal(result, expected_result)


def main():
    test_simd_table_lookup()
    test_simd_table_lookup_size_8()
    test_simd_table_lookup_size_32()
