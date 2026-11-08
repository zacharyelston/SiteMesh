import pycolmap
import pytest

from sitemesh.contracts import Survey
from sitemesh.dense import run_dense


@pytest.mark.skipif(pycolmap.has_cuda, reason="checks the no-CUDA path")
def test_dense_without_cuda_skips_and_writes_nothing(tmp_path):
    opts = pycolmap.SyntheticDatasetOptions()
    opts.num_rigs = 1
    opts.num_frames_per_rig = 4
    rec = pycolmap.synthesize_dataset(opts)
    survey = Survey(tmp_path)

    out = run_dense(survey, rec)

    assert out["status"] == "skipped"
    assert "CUDA" in out["reason"]
    lo, hi = out["depth_bounds_m_planned"]
    assert 0 < lo < hi
    assert not (survey.model / "dense").exists()
