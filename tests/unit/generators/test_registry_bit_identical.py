"""
Registry bit-identical regression guard (ADR-0006 / column-level deployment).

Adding optional `target_col_idx` support to the MNAR families (so diverse MNAR subtypes can
be placed per-column for the column-level experiment) must NOT change any generator's DEFAULT
behaviour — the registry has to stay bit-for-bit identical so every prior result (and the full
1260-test suite) remains attributable. Coding Bible Rule 6 (deterministic behaviour) makes this
checkable: a fixed input matrix + a fixed RNG seed pins each generator's `apply_to` output.

This test reconstructs the full `lacuna_tabular_110` registry, runs `apply_to` on the SAME fixed
z-scored matrix with the SAME seed used to capture the reference, and asserts every generator's
output mask hash equals the value captured BEFORE the targeting edits. Any drift in a default
code path — or a generator added/removed from the registry — fails loudly here.

The reference dict was captured from the pre-edit code (commit on `experiment/column-level-
missingness`). Do not regenerate it casually: its whole purpose is to be frozen.
"""

import hashlib

import pytest

from lacuna.core.rng import RNGState
from lacuna.core.types import CLASS_NAMES
from lacuna.data.semisynthetic import _zscore_columns
from lacuna.generators.families.registry_builder import load_registry_from_config

# Fixed predictor-view matrix + seed — identical to the capture script. Changing either
# invalidates the reference (which is the point: the reference pins behaviour, not the test).
_REF_MATRIX_SEED = 20260529
_REF_MATRIX_SHAPE = (400, 7)
_APPLY_SEED = 12345

# name -> (class_name, sha256(mask bytes)[:16]) captured from pre-targeting-edit code.
_REFERENCE = {
    'MAR-BinaryPred': ('MAR', '1bc4501d57a1b631'),
    'MAR-Branching': ('MAR', 'e2148b5736a42465'),
    'MAR-ColBlocks': ('MAR', '5cec65b49b7cea28'),
    'MAR-Conditional': ('MAR', '467b113f16ae0e4a'),
    'MAR-ContPred': ('MAR', '954fc84a007d520a'),
    'MAR-CrossClass': ('MAR', '2b2b353f5b19cb79'),
    'MAR-DemoGated-Sparse': ('MAR', '3e08ebac306edd5c'),
    'MAR-DiscPred': ('MAR', '57d4c0929ed0efb5'),
    'MAR-Distance': ('MAR', '4a4b76f844ad0c53'),
    'MAR-Interactive': ('MAR', '2984c1b5609f9512'),
    'MAR-Kernel': ('MAR', '047f647bda3d5fc3'),
    'MAR-Logistic': ('MAR', '7b69080edfd0757d'),
    'MAR-ManyPred': ('MAR', '20c5f7de1060570c'),
    'MAR-MixedPred': ('MAR', 'e8ffce3bd7e935d2'),
    'MAR-Moderate': ('MAR', 'f82eba9847075ed7'),
    'MAR-MultiCol': ('MAR', '5efd96d158799cc9'),
    'MAR-MultiPred': ('MAR', 'c9d55b797d0c7f71'),
    'MAR-Nested': ('MAR', '1770ded1432d5941'),
    'MAR-PartialResponse-Light': ('MAR', '22cf16dbe958307d'),
    'MAR-PolyMulti': ('MAR', '4f496bebabf085ca'),
    'MAR-Polynomial': ('MAR', '692832b136e68dd4'),
    'MAR-Probit': ('MAR', 'ce32384c5755711e'),
    'MAR-Quota': ('MAR', 'b16371b631cc8796'),
    'MAR-RealisticSingle-15': ('MAR', 'f937f111a2722aac'),
    'MAR-RealisticSingle-25': ('MAR', 'be838355826de178'),
    'MAR-ReqOpt': ('MAR', 'c51bb02f47e8e262'),
    'MAR-RowBlocks': ('MAR', '8bffbe6badaea0dd'),
    'MAR-Section': ('MAR', '4a621fb4689f78f9'),
    'MAR-SkipLogic': ('MAR', '33fff73a76a587cb'),
    'MAR-Spline': ('MAR', 'f0dabfc31e9f0fd7'),
    'MAR-SplineMulti': ('MAR', '43ae97934ee7f8f3'),
    'MAR-StepFunc': ('MAR', '3c8ffb3a7134b2e6'),
    'MAR-Strong': ('MAR', 'a643898599bfd29f'),
    'MAR-ThreePred': ('MAR', 'c06b251a8236a1ad'),
    'MAR-ThreeWay': ('MAR', '31018d308cf5f9ba'),
    'MAR-Threshold': ('MAR', '9d64989ffdadd087'),
    'MAR-TreeRules': ('MAR', '34262f2bd4ce6243'),
    'MAR-TwoPred': ('MAR', 'c9d55b797d0c7f71'),
    'MAR-Weak': ('MAR', '40fe5dc442d2dcd3'),
    'MAR-WeightPred': ('MAR', '4d1ab640403734ee'),
    'MCAR-BatchFx': ('MCAR', '546c830f0be900c9'),
    'MCAR-Bernoulli-10': ('MCAR', '0196eb22111c0a59'),
    'MCAR-Bernoulli-30': ('MCAR', 'fe466780579889ef'),
    'MCAR-Bernoulli-50': ('MCAR', '0669e4b27ca8d552'),
    'MCAR-Cauchy': ('MCAR', 'a7ee6a0920f01250'),
    'MCAR-Checkerboard': ('MCAR', '00f0e2ce0ff9f14c'),
    'MCAR-Clustered': ('MCAR', 'b932a4465d2f574f'),
    'MCAR-ColBeta': ('MCAR', '7cfb9fdb0f94e7b8'),
    'MCAR-ColClustered': ('MCAR', 'e252b9584980a6f2'),
    'MCAR-ColGamma': ('MCAR', 'a72fd3008110fc47'),
    'MCAR-ColGauss': ('MCAR', '4d064658a655045e'),
    'MCAR-ColMixture': ('MCAR', '605bb5d443ddf8cc'),
    'MCAR-ColOrdered': ('MCAR', '80272c004e0f02db'),
    'MCAR-CovDep': ('MCAR', '4fe51d086dd7c95c'),
    'MCAR-Diagonal': ('MCAR', 'a073376ffeb358cd'),
    'MCAR-LogNormal': ('MCAR', '0278caa0a079c6d3'),
    'MCAR-MixGauss': ('MCAR', '40dbcaf75de75903'),
    'MCAR-Nested': ('MCAR', '7bffa72a1e6a7470'),
    'MCAR-Pareto': ('MCAR', 'bf72d597f75e2169'),
    'MCAR-RandBlocks': ('MCAR', '99f46f2f73e9bb7e'),
    'MCAR-RowBeta': ('MCAR', '0a6238baf113c3f6'),
    'MCAR-RowColAdd': ('MCAR', 'e4255a5f7e13e83f'),
    'MCAR-RowColInt': ('MCAR', '6fefe18eff9d75a7'),
    'MCAR-RowDiscrete': ('MCAR', '637381d2cfd5f0e8'),
    'MCAR-RowExp': ('MCAR', 'd0da19ce26f783aa'),
    'MCAR-RowGamma': ('MCAR', 'cfb41f0295958b2f'),
    'MCAR-RowGauss': ('MCAR', '394d42f5ae62d00a'),
    'MCAR-RowMixture': ('MCAR', '7a7c789a8100eebf'),
    'MCAR-Scattered': ('MCAR', '12b7be0fa1ff9f03'),
    'MCAR-SparseMix': ('MCAR', '5f71779f0e5f1b84'),
    'MCAR-Subgroup': ('MCAR', '1629a0777f6fdc79'),
    'MCAR-TDist': ('MCAR', 'fc085053c08beda9'),
    'MNAR-AdaptSamp': ('MNAR', 'ecf2d627aeeb8c4b'),
    'MNAR-Berkson': ('MNAR', 'd318f7ee30d628ad'),
    'MNAR-ColSpecCensor': ('MNAR', '54fd476fc289e7f1'),
    'MNAR-ColSpecThresh': ('MNAR', '601cef0b93cca231'),
    'MNAR-CompEvents': ('MNAR', '5b9592304f6ebf5b'),
    'MNAR-Competitive': ('MNAR', 'c3618c9de66b4c01'),
    'MNAR-DemoDepend': ('MNAR', 'ec37c462b6faadb5'),
    'MNAR-DetectBoth': ('MNAR', 'c530ab240ee10c39'),
    'MNAR-DetectLower': ('MNAR', '636a057343c430d0'),
    'MNAR-DetectUpper': ('MNAR', 'd8965093b6021a84'),
    'MNAR-Gaming': ('MNAR', 'ced42a27096a5f4d'),
    'MNAR-LatentCorr': ('MNAR', 'b9b4e1d13f0a9123'),
    'MNAR-LatentHealth': ('MNAR', '05fe75e4cafad74d'),
    'MNAR-LatentMeasErr': ('MNAR', '05fe75e4cafad74d'),
    'MNAR-LatentMotiv': ('MNAR', '05fe75e4cafad74d'),
    'MNAR-LatentOrtho': ('MNAR', 'd8cb4959991bff4f'),
    'MNAR-LatentSES': ('MNAR', '05fe75e4cafad74d'),
    'MNAR-Logistic': ('MNAR', '8c74ba52ca2e381e'),
    'MNAR-MultiThresh': ('MNAR', '1b1e5ad5c8e3010e'),
    'MNAR-NonLinSocial': ('MNAR', '0636849ea8ddf64f'),
    'MNAR-OutcomeDep': ('MNAR', '81db545e7879e948'),
    'MNAR-OverReport': ('MNAR', '74f809ad95dd1fa2'),
    'MNAR-Privacy': ('MNAR', '586a6a2d7cda4887'),
    'MNAR-Q70': ('MNAR', '9045d6144cc17248'),
    'MNAR-Q80': ('MNAR', '7677a30830bc2cb8'),
    'MNAR-Q90': ('MNAR', 'd38cc384e39a8c89'),
    'MNAR-RiskMonitor': ('MNAR', '81e0cc8c84ad079f'),
    'MNAR-SelfCensor-Extreme': ('MNAR', '816bac3a77d4fba9'),
    'MNAR-SelfCensor-High': ('MNAR', '0b085862a926f1d9'),
    'MNAR-SelfCensor-Low': ('MNAR', 'fdfb929339c1dce7'),
    'MNAR-SelfCensor-Strong': ('MNAR', '6009f7f5d6dfa1aa'),
    'MNAR-SelfCensor-Weak': ('MNAR', 'c4e7212dd30a55d9'),
    'MNAR-SoftThresh': ('MNAR', 'adda37a40c3f3277'),
    'MNAR-SymptomTrig': ('MNAR', 'f1bc3cad3d2556ef'),
    'MNAR-ThreshLeft': ('MNAR', '9045d6144cc17248'),
    'MNAR-ThreshRight': ('MNAR', 'f86257393f579c93'),
    'MNAR-ThreshTwoSided': ('MNAR', '2b9b6ec29befc1f7'),
    'MNAR-Truncation': ('MNAR', 'ada763be16f083d5'),
    'MNAR-UnderReport': ('MNAR', '936133a5f5c50019'),
    'MNAR-ValDepStr': ('MNAR', '0fb3a5ae6a6114a8'),
    'MNAR-Volunteer': ('MNAR', '7d578d56e324d9b0'),
}


def _apply_hashes():
    """Reproduce the reference capture: registry × fixed matrix × fixed seed -> {name: (class, sha)}."""
    reg = load_registry_from_config("lacuna_tabular_110")
    Z = _zscore_columns(RNGState(seed=_REF_MATRIX_SEED).randn(*_REF_MATRIX_SHAPE))
    out = {}
    for g in reg:
        R = g.apply_to(Z, RNGState(seed=_APPLY_SEED))
        sha = hashlib.sha256(R.cpu().numpy().tobytes()).hexdigest()[:16]
        out[g.name] = (CLASS_NAMES[g.class_id], sha)
    return out


def test_registry_membership_unchanged():
    """No generator added to or removed from the registry by the targeting edits."""
    got = _apply_hashes()
    assert set(got) == set(_REFERENCE), {
        "added": sorted(set(got) - set(_REFERENCE)),
        "removed": sorted(set(_REFERENCE) - set(got)),
    }


def test_every_generator_default_output_bit_identical():
    """Every generator's DEFAULT apply_to output is byte-for-byte what it was pre-edit."""
    got = _apply_hashes()
    drift = {name: (_REFERENCE[name], got[name]) for name in _REFERENCE if got.get(name) != _REFERENCE[name]}
    assert not drift, f"default behaviour changed for {len(drift)} generators: {drift}"


@pytest.mark.parametrize("name", sorted(n for n, (c, _) in _REFERENCE.items() if c == "MNAR"))
def test_mnar_generator_default_unchanged(name):
    """Per-MNAR-generator guard (these are the families gaining target_col_idx)."""
    got = _apply_hashes()
    assert got[name] == _REFERENCE[name], f"{name}: {got[name]} != reference {_REFERENCE[name]}"
