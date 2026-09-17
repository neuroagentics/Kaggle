"""Small real-transition replay batch for game-local output calibration.

Only the decoder color bias is adapted initially (16 parameters). This is not
claimed as learned mechanic conditioning or full dynamics adaptation.
"""

def episode_batch(memory, *, limit=32):
    import torch
    from training.train_simulator import TrajectoryBatch
    episodes = list(reversed(memory.query_episodes(limit=limit)))
    if not episodes:
        return None
    latest = episodes[-1].before_grid()
    h, w = len(latest), len(latest[0])
    episodes = [e for e in episodes if all(
        len(g) == h and all(len(row) == w for row in g)
        for g in (e.before_grid(), e.observed_grid()))]
    if len(episodes) < 16:
        return None
    n = len(episodes)
    grids = torch.tensor([[e.before_grid(), e.observed_grid()] for e in episodes], dtype=torch.long)
    ids = torch.zeros((n, 2), dtype=torch.long)
    x = torch.zeros((n, 2)); y = torch.zeros((n, 2)); coords = torch.zeros((n, 2))
    progress = torch.full((n, 2), -1.0)
    for i, e in enumerate(episodes):
        ids[i, 0] = e.action_id
        if e.action_id == 6:
            x[i, 0], y[i, 0], coords[i, 0] = e.action_x / 63, e.action_y / 63, 1
        progress[i, 1] = float(e.actual_success)
    return TrajectoryBatch(
        grid_tokens=grids, validity_mask=torch.ones_like(grids, dtype=torch.bool),
        action_ids=ids, action_x=x, action_y=y, coord_valid=coords,
        progress_labels=progress, failure_labels=torch.full((n, 2), -1.0),
        avail_labels=torch.full((n, 2, 8), -1.0),
        game_ids=[memory.game_id] * n,
        level_ids=torch.tensor([[e.level_id, e.level_id] for e in episodes]),
    )
