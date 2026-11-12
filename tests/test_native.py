import kap
import kap._core
import kap._reference


def test_native_version_matches_package_version():
    assert kap._core.__version__ == kap.__version__


def test_public_k_assignment_is_reference_implementation_for_now():
    assert kap._reference.k_assignment is kap.k_assignment
