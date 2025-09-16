import json
from pathlib import Path

from sgelabs.io import parse_mod_file
from sgelabs.ir import ModelIR


def test_model_ir_roundtrip(tmp_path: Path) -> None:
    model = parse_mod_file('examples/rbc_basic/rbc.mod')
    payload = model.to_json_dict()
    path = tmp_path / 'model.json'
    path.write_text(json.dumps(payload), encoding='utf-8')
    loaded = ModelIR.from_json_dict(json.loads(path.read_text(encoding='utf-8')))
    assert [v.name for v in loaded.endo] == [v.name for v in model.endo]
    assert loaded.params == model.params
    assert len(loaded.equations) == len(model.equations)
