import sys
from utils.string_slice import StringSlice
from math import ceil, log10
from collections import InlineArray
from memory import memcpy, memcmp
from testing import assert_equal
import bit
import random

from .constants import powers_of_10, get_power_of_5


@value
@register_passable
struct UInt128:
    var high: UInt64
    var low: UInt64

    fn most_significant_bit(self) -> UInt64:
        return self.high >> 63

alias std_size = 24
alias maximum_int_as_str = "9223372036854775807"
alias mantissa_explicit_bits = 52
alias smallest_power_of_five = -342


fn ensure_valid_integers(x: SIMD[DType.uint8, _]) -> SIMD[DType.bool, x.size]:
    alias min_values = SIMD[DType.uint8, x.size](ord("0"))
    alias max_values = SIMD[DType.uint8, x.size](ord("9"))
    return (min_values <= x) & (x <= max_values)

fn standardize_string_slice(x: StringSlice) -> InlineArray[UInt8, size=std_size]:
    var standardized_x = InlineArray[UInt8, size=std_size](ord("0"))
    memcpy(
        dest_data=(standardized_x.unsafe_ptr() + std_size - len(x)).bitcast[Int8](), 
        src_data=x.unsafe_ptr().bitcast[Int8](), 
        n=len(x)
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
            raise Error("Invalid character in the number. '" + String(number_as_string) + "'")

    # We assume there are no leading or trailing whitespaces, no leading zeros, no sign.
    # We could compute all those aliases at compile time, by knowing the size of int, simd width, 
    # and the base of the number system. Here it only works for base 10.
    alias index_simd_width = sys.simdwidthof[DType.index]()
    alias vector_with_exponents = InlineArray[
        SIMD[DType.index, 1],
        size=std_size
    ](
        0,0,0,0,
        0, 1_000_000_000_000_000_000, 100_000_000_000_000_000, 10_000_000_000_000_000,
        1_000_000_000_000_000, 100_000_000_000_000, 10_000_000_000_000, 1_000_000_000_000,
        100_000_000_000, 10_000_000_000, 1_000_000_000, 100_000_000,
        10_000_000, 1_000_000, 100_000, 10_000,
        1_000, 100, 10, 1,
    )

    var accumulator = SIMD[DType.index, sys.simdwidthof[DType.index]()](0)
    
    var max_standardized_x = "000009223372036854775807"
    if memcmp(standardized_x.unsafe_ptr(), max_standardized_x.unsafe_ptr(), count=std_size) == 1: # memcmp is pretty fast
        raise Error("The string is too large to be converted to an integer. '")
    _ = max_standardized_x

    @parameter
    for i in range(std_size // index_simd_width):
        ascii_vector = (standardized_x.unsafe_ptr() + i * index_simd_width).load[width=sys.simdwidthof[DType.index]()]()
        as_digits = ascii_vector - SIMD[DType.uint8, sys.simdwidthof[DType.index]()](ord("0"))
        as_digits_index = as_digits.cast[DType.index]()
        alias vector_slice = (vector_with_exponents.unsafe_ptr() + i * index_simd_width).load[width=sys.simdwidthof[DType.index]()]()
        accumulator += as_digits_index * vector_slice
    return int(accumulator.reduce_add())



fn _get_w_and_q_from_float_string(input_string: StringSlice) raises -> Tuple[Int, Int]:
    """We suppose the number is in the form '123.2481' or '123' or '123e-2' or '12.3e2'."""
    # We read the number from right to left.
    alias ord_0 = UInt8(ord("0"))
    alias ord_9 = UInt8(ord("9"))
    alias ord_dot = UInt8(ord("."))
    alias ord_minus = UInt8(ord("-"))
    alias ord_plus = UInt8(ord("+"))
    alias ord_e = UInt8(ord("e"))
    alias ord_E = UInt8(ord("E"))

    var additional_exponent = 0
    var exponent_multiplier = 1

    # We'll assume that we'll never go over 24 digit for each number.
    var exponent = InlineArray[UInt8, size=std_size](ord("0"))
    var significand = InlineArray[UInt8, size=std_size](ord("0"))

    var prt_to_array = UnsafePointer.address_of(exponent)
    var array_index = std_size
    var buffer = input_string.unsafe_ptr()

    var dot_or_e_found = False

    for i in range(len(input_string)-1, -1, -1):
        array_index -= 1
        if array_index < 0:
            raise Error("The number is too big. '" + String(input_string) + "'")
        if buffer[i] == ord_dot:
            dot_or_e_found = True
            if prt_to_array == UnsafePointer.address_of(exponent):
                # We thought we were writing the exponent, but we were writing the significand.
                significand = exponent
                exponent = InlineArray[UInt8, size=std_size](ord("0"))
                prt_to_array = UnsafePointer.address_of(significand)
            
            additional_exponent = std_size - array_index - 1
            # We don't want to progress in the significand array.
            array_index += 1
        elif buffer[i] == ord_minus:
            # Next should be the E letter (or e), so we'll just continue.
            exponent_multiplier = -1
        elif buffer[i] == ord_plus:
            # Next should be the E letter (or e), so we'll just continue.
            pass
        elif buffer[i] == ord_e or buffer[i] == ord_E:
            dot_or_e_found = True
            # We finished writing the exponent.
            prt_to_array = UnsafePointer.address_of(significand)
            array_index = std_size
        elif (ord_0 <= buffer[i]) and (buffer[i] <= ord_9):
            prt_to_array[][array_index] = buffer[i]
        else:
            raise Error("Invalid character in the number. '" + String(input_string) + "'")

    if not dot_or_e_found:
        # We were reading the significand
        significand = exponent
        exponent = InlineArray[UInt8, size=std_size](ord("0"))

    exponent_as_integer = exponent_multiplier * to_integer(exponent) - additional_exponent
    significand_as_integer = to_integer(significand)
    return (significand_as_integer, exponent_as_integer)


fn strip_unused_characters(x: String) -> String:
    result = x.strip()
    result = result.removesuffix("f")
    result = result.removesuffix("F")
    return result

fn get_sign(x: String) -> Tuple[Float64, String]:
    if x.startswith("-"):
        return (-1., x[1:])
    return (1., x)

# Powers of 10 and integers below 2**53 are exactly representable as Float64.
# Thus any operation done on them must be exact.
fn can_use_clinger_fast_path(w: Int, q: Int) -> Bool:
    return w <= 2**53 and (-22 <= q <= 22)

fn clinger_fast_path(w: Int, q: Int) -> Float64:
    if q >= 0:
            return Float64(w) * powers_of_10[q]
        else:
            return Float64(w) / powers_of_10[-q]

fn full_multiplication(x: UInt64, y: UInt64) -> UInt128:
    # Note that there are assembly instructions to 
    # do all that on some architectures.
    # That should speed things up.
    var x_low = x & 0xFFFFFFFF
    var x_high = x >> 32
    var y_low = y & 0xFFFFFFFF
    var y_high = y >> 32

    var low_low = x_low * y_low
    var low_high = x_low * y_high
    var high_low = x_high * y_low
    var high_high = x_high * y_high

    var carry = (low_low >> 32) + (low_high & 0xFFFFFFFF) + (high_low & 0xFFFFFFFF)

    var low = low_low + (low_high << 32) + (high_low << 32)
    var high = high_high + (low_high >> 32) + (high_low >> 32) + (carry >> 32)

    return UInt128(high, low)


fn get_128_bit_truncated_product(w: UInt64, q: Int64) -> UInt128:
    alias bit_precision = mantissa_explicit_bits + 3
    var index = 2 * (q - smallest_power_of_five)
    var first_product = full_multiplication(w, get_power_of_5(int(index)))

    var precision_mask = UInt64(0xFFFFFFFFFFFFFFFF) >> bit_precision
    if ((first_product.high & precision_mask) == precision_mask):
        second_product = full_multiplication(w, get_power_of_5(int(index + 1)))
        first_product.low = first_product.low + second_product.high
        if second_product.high > first_product.low:
            first_product.high = first_product.high + 1
    
    return first_product


fn create_float64(m: UInt64, p: Int64) -> Float64:
 
    var m_mask = UInt64(2 ** 52 - 1)
    var representation_as_int = (m & m_mask) | ((p + 1023).cast[DType.uint64]() << 52)

    return UnsafePointer.address_of(representation_as_int).bitcast[Float64]()[]



fn lemire_algorithm(owned w: UInt64, owned q: Int64) -> Float64:
    # This algorithm has 22 steps described 
    # in https://arxiv.org/pdf/2101.11408 (algorithm 1)
    # Step 1
    if w == 0 or q < -342:
        return 0.0
    
    # Step 2
    if q > 308:
        return FloatLiteral.infinity
    
    # Step 3
    l = bit.count_leading_zeros(w)

    # Step 4
    w <<= l

    # Step 5
    var product = get_128_bit_truncated_product(w, q)

    # Step 6
    # This step is skipped because it has been proven not necessary.
    # The proof can be found in the following paper by 
    # Noble Mushtak & Daniel Lemire:
    # Fast Number Parsing Without Fallback
    # https://arxiv.org/abs/2212.06644

    # Step 8
    # Comes before step 7 because we need the upper_bit
    var upper_bit = product.most_significant_bit()

    # Step 7
    var m: UInt64 = product.high >> (upper_bit + 9)

    # Step 9
    var p: Int64 = (((152170 + 65536) * q) >> 16) + 63 - l.cast[DType.int64]() + upper_bit.cast[DType.int64]()

    # Step 10
    if p <= (-1022 - 64):
        return 0.0
    
    # Step 11-15
    # Subnormal case
    if p <= -1022:
        s = -1022 - p + 1
        m >>= s.cast[DType.uint64]()

        return create_float64(m, p)

    # Step 16-18
    # Round ties to even
    if product.low <= 1 and (m & 3 == 1) and (Int64(-4) <= q <= Int64(23)):
        if bit.pop_count(product.high // m) == 1:
            m -= 2

    # step 19
    if m % 2 == 1:
        m += 1
    m //= 2

    # Step 20
    if m == 2 **53:
        print("in step 20")
        m //=2
        p = p + 1

    # step 21
    if p > 1023:
        return FloatLiteral.infinity

    # Step 22 
    return create_float64(m, p)



fn atof(owned x: String) raises -> Float64:
    """Parses the given string as a floating point and returns that value.

    For example, `atof("2.25")` returns `2.25`.

    Raises:
        If the given string cannot be parsed as an floating point value, for
        example in `atof("hi")`.

    Args:
        x: A string to be parsed as a floating point.

    Returns:
        An floating point value that represents the string, or otherwise raises.
    """
    x = strip_unused_characters(x)
    sign_and_x = get_sign(x)
    sign = sign_and_x[0]
    x = sign_and_x[1]

    if x == "nan":
        return FloatLiteral.nan
    if x == "in": # f was removed previously
        return FloatLiteral.infinity * sign

    
    var w_and_q = _get_w_and_q_from_float_string(x.as_string_slice())
    var w = w_and_q[0]
    var q = w_and_q[1]
    
    var result: Float64 = 0.0

    if can_use_clinger_fast_path(w, q):
        result = clinger_fast_path(w, q)
    else:
        result = lemire_algorithm(w, q)
    return result * sign
