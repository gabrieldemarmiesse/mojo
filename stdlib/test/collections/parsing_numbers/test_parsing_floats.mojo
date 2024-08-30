from testing import assert_equal

alias numbers_to_test_as_str = List[String](
    "1e-45",                 # subnormal, smallest value possible
    "3e-45",                 # subnormal
    "4e-45",                 # subnormal
    "3.4028235e38",          # largest value possible
    "15038927332917.156",     # triggers step 19
    "9000000000000000.5",    # tie to even
    "456.7891011e70",        # Lemire algorithm
    "inf",                   # infinity
    "5e-600",                # approximate to 0
    "5e1000",                # approximate to infinity
    "5484.2155e-38",         # Lemire algorithm
    "5e-35",                 # Lemire algorithm
    "5e30",                  # Lemire algorithm
    "47421763.54884",        # Clinger fast path
    "474217635486486e10",    # Clinger fast path
    "474217635486486e-10",   # Clinger fast path
    "474217635486486e-20",   # Clinger fast path
    "4e-22",                 # Clinger fast path
    "4.5e15",                # Clinger fast path
    "0.1",                   # Clinger fast path
    "0.2",                   # Clinger fast path
    "0.3",                   # Clinger fast path
)
# TODO add tests from issue
alias numbers_to_test = List[Float64](
    1e-45,
    3e-45,
    4e-45,
    3.4028235e38,
    15038927332917.156,
    9000000000000000.5,
    456.7891011e70,
    FloatLiteral.infinity,
    0.0,
    FloatLiteral.infinity,
    5484.2155e-38,
    5e-35,
    5e30,
    47421763.54884, 
    474217635486486e10,
    474217635486486e-10,
    474217635486486e-20,
    4e-22,
    4.5e15,
    0.1,
    0.2,
    0.3,
)


def test_atof_generate_cases():
    for i  in range(len(numbers_to_test)):
        for suffix in List("", "f", "F"):
            for exponent in List("e", "E"):
                for multiplier in List("", "-"):
                    var sign:Float64 = 1
                    if multiplier[] == "-":
                        sign = -1
                    final_string = numbers_to_test_as_str[i].replace("e", exponent[])
                    if final_string != "nan":
                        final_string = multiplier[] + final_string
                        if not final_string.endswith("inf"):
                            final_string += suffix[]
                    final_value =   sign * numbers_to_test[i]
            
                    assert_equal(
                        atof(final_string), final_value
                    )


def main():
    test_atof_generate_cases()