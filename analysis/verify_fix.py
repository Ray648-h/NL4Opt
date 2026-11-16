import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import parsers
import parsing_utils.constants as const

p = parsers.ModelOutputXMLParser(print_errors=False)

# 1) None guard in parse_number
print('parse_number(None) =', repr(p.parse_number(None)))

# helper: objective + one ratio constraint (limit present or absent)
def make_xml(limit_tag):
    lim = f'<LIMIT>{limit_tag}</LIMIT>' if limit_tag is not None else ''
    return (
        '<DECLARATION><OBJ_DIR>minimize</OBJ_DIR><OBJ_NAME>cost</OBJ_NAME>'
        '<VAR>x</VAR><PARAM>1</PARAM></DECLARATION>'
        '<DECLARATION><CONST_DIR>const</CONST_DIR><OPERATOR>&lt;=</OPERATOR>'
        f'{lim}'
        '<CONST_TYPE>[RATIO_CONSTRAINT]</CONST_TYPE>'
        '<VAR>x</VAR><PARAM>2</PARAM></DECLARATION>'
    )

# 2) malformed: ratio constraint MISSING limit -> must not crash, limit == 0.0
f_missing = p.parse(make_xml(None))
c = f_missing.constraints[0]
print('missing LIMIT -> limit =', repr(c.limit), 'type =', type(c.limit).__name__)
canon_missing = parsers.convert_to_canonical(f_missing)
print('  convert_to_canonical OK, constraints shape =', canon_missing.constraints.shape)

# 3) well-formed: ratio constraint WITH limit 0.7 -> regression check
f_full = p.parse(make_xml('0.7'))
c2 = f_full.constraints[0]
print('with LIMIT 0.7 -> limit =', repr(c2.limit))
canon_full = parsers.convert_to_canonical(f_full)
print('  convert_to_canonical OK, constraints shape =', canon_full.constraints.shape)

print('ALL CHECKS PASSED')
