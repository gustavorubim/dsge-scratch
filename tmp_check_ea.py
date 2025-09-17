from sgelabs.io import parse_mod_file
m = parse_mod_file(r'model_base\EA_GNSS10\EA_GNSS10_rep\EA_GNSS10_rep.mod')
print('endo', len(m.endo))
print('eqs', len(m.equations))
