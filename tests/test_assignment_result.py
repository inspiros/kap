import pytest

from kap._reference.assignment import AssignmentResult


def test_fields_defaults_and_tuple_behavior():
    assert AssignmentResult._fields == ("cliques", "costs", "free")
    assert AssignmentResult() == (None, None, None)
    result = AssignmentResult([[1]], [2], [])
    assert tuple(result) == ([[1]], [2], [])
    assert result.cliques == [[1]]
    assert result.costs == [2]
    assert result.free == []
    assert not hasattr(result, "__dict__")


@pytest.mark.parametrize("costs, total, total_type", [([1, 2], 3, int), ([1.25, 2.5], 3.75, float), ([], 0, int)])
def test_total_cost(costs, total, total_type):
    result = AssignmentResult([], costs, None)
    assert result.cost == total
    assert type(result.cost) is total_type


def test_deprecated_aliases():
    result = AssignmentResult([[1]], [2], None)
    with pytest.warns(DeprecationWarning, match=r"matches is deprecated; use cliques\. It will be removed in 0\.4\."):
        assert result.matches is result.cliques
    with pytest.warns(DeprecationWarning, match=r"matching_costs is deprecated; use costs\. It will be removed in 0\.4\."):
        assert result.matching_costs is result.costs
