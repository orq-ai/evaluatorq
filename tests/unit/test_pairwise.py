from __future__ import annotations

from evaluatorq.pairwise import pairwise_consensus, reconcile_pair


def test_consistent_pick_counts_as_that_vote() -> None:
    """A judge picking the same side in both orderings votes for that side, no flip."""
    vote, flipped = reconcile_pair('A', 'A')

    assert vote == 'A'
    assert flipped is False


def test_flip_between_pick_and_other_pick_abstains() -> None:
    """A judge that picks A one way and B the other has no real preference; it abstains and flips."""
    vote, flipped = reconcile_pair('A', 'B')

    assert vote is None
    assert flipped is True


def test_consistent_tie_counts_as_tie_vote() -> None:
    """Tie in both orderings is a real tie vote, not a flip."""
    vote, flipped = reconcile_pair('tie', 'tie')

    assert vote == 'tie'
    assert flipped is False


def test_tie_versus_pick_is_a_flip() -> None:
    """Tie one way and a pick the other is inconsistent: abstain and flip."""
    vote, flipped = reconcile_pair('tie', 'A')

    assert vote is None
    assert flipped is True


def test_missing_ordering_abstains_without_flip() -> None:
    """If one ordering has no decisive verdict, the judge abstains but it is not counted as a flip."""
    vote, flipped = reconcile_pair('A', None)

    assert vote is None
    assert flipped is False


def test_consensus_is_the_majority_pick() -> None:
    """A clear plurality of decisive votes is the consensus winner."""
    assert pairwise_consensus(['A', 'A', 'B']) == 'A'


def test_consensus_ignores_abstained_judges() -> None:
    """Abstained (None) votes do not count toward the consensus."""
    assert pairwise_consensus(['A', 'A', None, None]) == 'A'


def test_consensus_tie_vote_can_win() -> None:
    """A plurality of tie votes makes the comparison a tie."""
    assert pairwise_consensus(['tie', 'tie', 'A']) == 'tie'


def test_even_split_is_inconclusive() -> None:
    """No plurality (A and B tied) yields inconclusive."""
    assert pairwise_consensus(['A', 'B']) == 'inconclusive'


def test_all_abstained_is_inconclusive() -> None:
    """When every judge abstains there is no consensus."""
    assert pairwise_consensus([None, None]) == 'inconclusive'
