import kap
import kap._core
import kap._reference


def test_native_version_matches_package_version():
    assert kap._core.__version__ == kap.__version__


def test_public_k_assignment_dispatches_to_reference_for_scipy(paper_costs):
    assert kap.k_assignment is not kap._reference.k_assignment
    options = dict(backend="scipy", return_free=True, random_state=42)
    assert kap.k_assignment(paper_costs, **options) == kap._reference.k_assignment(
        paper_costs, **options
    )
