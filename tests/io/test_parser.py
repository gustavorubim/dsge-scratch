from pathlib import Path

from sgelabs.io import parse_mod_file


def test_parse_rbc_basic() -> None:
    model = parse_mod_file(Path('examples/rbc_basic/rbc.mod'))
    assert [v.name for v in model.endo] == ['y', 'c', 'k', 'i', 'a']
    assert [s.name for s in model.exo] == ['e']
    assert model.params['alpha'] == 0.36
    assert model.shocks['e'] == 0.01
    assert model.initvals['k'] == 10
    assert model.varobs == ['y', 'c', 'i']


def test_parse_smets_wouters() -> None:
    model = parse_mod_file(Path('model_base/US_SW07/US_SW07_rep/US_SW07_rep.mod'))
    assert len(model.endo) > 30
    assert 'cgamma' in model.params
    assert 'ea' in model.shocks
