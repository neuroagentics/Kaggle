"""Tests for the testable-hypothesis loop (Step 3).

Guards the core logic: distinguishing-action selection, evidence-based
strengthening/rejection, and that observed evidence (not text) decides survival.
"""
from agent.hypothesis_loop import Hypothesis, HypothesisLoop

# Two tiny 2x2 "worlds of prediction": one hypothesis flips the top-left cell,
# the other flips the bottom-right. Only one matches the observed outcome.
G0 = ((0, 0), (0, 0))
TL = ((1, 0), (0, 0))   # top-left changed
BR = ((0, 0), (0, 1))   # bottom-right changed


def _pred_tl(grid, aid, x, y):
    return TL


def _pred_br(grid, aid, x, y):
    return BR


def _make_loop():
    return HypothesisLoop([
        Hypothesis(name="tl", predict=_pred_tl),
        Hypothesis(name="br", predict=_pred_br),
    ])


def test_distinguishing_action_prefers_disagreement():
    loop = _make_loop()
    # both hypotheses ignore the action here, but they predict different grids,
    # so any action distinguishes them; the call must return a candidate.
    act = loop.distinguishing_action(G0, [(1, None, None), (2, None, None)])
    assert act in [(1, None, None), (2, None, None)]


def test_evidence_strengthens_correct_and_rejects_wrong():
    loop = _make_loop()
    # Observe TL repeatedly: 'tl' should strengthen, 'br' should be rejected.
    for _ in range(4):
        loop.update(G0, (1, None, None), TL)
    best = loop.best()
    assert best is not None and best.name == "tl"
    assert best.confidence > 0.5
    names_alive = [h.name for h in loop.alive()]
    assert "br" not in names_alive  # contradicted 4x -> rejected


def test_text_does_not_decide_only_evidence():
    """Renaming a hypothesis must not change which survives — evidence decides."""
    loop = HypothesisLoop([
        Hypothesis(name="sounds_right_but_wrong", predict=_pred_br),
        Hypothesis(name="plain", predict=_pred_tl),
    ])
    for _ in range(4):
        loop.update(G0, (1, None, None), TL)  # reality = TL
    best = loop.best()
    assert best is not None and best.name == "plain"  # the one matching evidence


def test_all_agree_action_is_not_distinguishing():
    # Two hypotheses that predict the SAME grid -> zero distinct predictions.
    loop = HypothesisLoop([
        Hypothesis(name="a", predict=_pred_tl),
        Hypothesis(name="b", predict=_pred_tl),
    ])
    # distinguishing_action still returns something (falls back to first candidate).
    act = loop.distinguishing_action(G0, [(1, None, None)])
    assert act == (1, None, None)


# ---------------------------------------------------------------------------
# Adapter: MechanicHypothesis -> testable Hypothesis (closes the Qwen boundary)
# ---------------------------------------------------------------------------

def test_mechanic_hypothesis_movement_adapter():
    from agent.model import MechanicHypothesis
    from agent.hypothesis_loop import hypothesis_from_mechanic

    mech = MechanicHypothesis(
        hypothesis_id="a" * 32,
        belief_id="b" * 32,
        mechanic_type="movement",
        description="avatar moves right",
        object_type="avatar",
        direction_vector=(1, 0),          # (dx, dy) -> move right one column
        affected_colors=(3,),
        precondition_tags=(),
        effect_tags=(),
        supporting_ids=(),
        confidence=0.7,
    )
    hyp = hypothesis_from_mechanic(mech)
    # A single agent (color 3) at (1,1) on a 3x3 grid should move to (1,2).
    grid = ((0, 0, 0), (0, 3, 0), (0, 0, 0))
    pred = hyp.predict(grid, 1, None, None)
    assert pred == ((0, 0, 0), (0, 0, 3), (0, 0, 0))
    assert abs(hyp.confidence - 0.7) < 1e-9


def test_mechanic_hypothesis_transformation_adapter():
    from agent.model import MechanicHypothesis
    from agent.hypothesis_loop import hypothesis_from_mechanic

    mech = MechanicHypothesis(
        hypothesis_id="c" * 32,
        belief_id="d" * 32,
        mechanic_type="transformation",
        description="target 8 becomes 9",
        object_type=None,
        direction_vector=None,
        affected_colors=(8,),
        precondition_tags=(),
        effect_tags=("to_color:9",),
        supporting_ids=(),
        confidence=0.6,
    )
    hyp = hypothesis_from_mechanic(mech)
    grid = ((8, 0), (0, 8))
    assert hyp.predict(grid, 6, 0, 0) == ((9, 0), (0, 9))
