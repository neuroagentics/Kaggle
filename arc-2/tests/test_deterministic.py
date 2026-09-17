from hyper_arc.deterministic import exact_candidates, prediction_pair


def test_infers_rotation_with_color_remap() -> None:
    train = [
        (
            [[1, 0], [1, 0], [1, 1]],
            [[2, 2, 2], [2, 0, 0]],
        )
    ]
    candidates = exact_candidates(train, [[[3, 3], [0, 3]]])
    assert candidates
    assert candidates[0].predict(train[0][0]) == train[0][1]


def test_infers_crop_using_modal_background() -> None:
    train = [
        (
            [[8, 8, 8, 8], [8, 2, 2, 8], [8, 2, 8, 8]],
            [[2, 2], [2, 8]],
        )
    ]
    candidates = exact_candidates(train)
    assert any(candidate.name.startswith("crop_mode+") for candidate in candidates)


def test_prediction_pair_prefers_distinct_exact_hypotheses() -> None:
    train = [([[1, 2], [2, 1]], [[1, 2], [2, 1]])]
    test = [[1, 1], [2, 2]]
    candidates = exact_candidates(train, [test])
    first, second = prediction_pair(candidates, test)
    assert first != second


def test_rejects_non_cellwise_recoloring() -> None:
    train = [([[1, 1], [1, 1]], [[1, 2], [1, 2]])]
    assert exact_candidates(train) == []
