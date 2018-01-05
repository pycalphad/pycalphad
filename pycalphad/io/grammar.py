""""Common pyparsing grammar patterns."""

from pyparsing import alphas, nums
from pyparsing import Group, OneOrMore, Optional, Regex, Suppress, Word

pos_neg_int_number = Word('+-' + nums).setParseAction(lambda t: [int(t[0])])  # '+3' or '-2' are examples
# matching float w/ regex is ugly but is recommended by pyparsing
regex_after_decimal = r'([0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)'
float_number = Regex(r'[-+]?([0-9]+\.(?!([0-9]|[eE])))|{0}'.format(regex_after_decimal)) \
    .setParseAction(lambda t: [float(t[0])])

chemical_formula = Group(OneOrMore(Word(alphas, min=1, max=2) + Optional(float_number, default=1.0))) + \
                   Optional(Suppress('/') + pos_neg_int_number, default=0)
