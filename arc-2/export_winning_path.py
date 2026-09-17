import argparse
import json
import torch

# Project Imports
from hyper_arc.esb import ESB
from hyper_arc.mcts import MCTSEngine
from hyper_arc.hpm import GlobalMemoryBank, LocalTaskBuffer


def load_task_data(challenge_file: str, task_id: str):
    """Load task from aggregate JSON file."""
    with open(challenge_file, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and task_id in data:
        return data[task_id]
    if isinstance(data, list):
        for task in data:
            if task.get("task_id") == task_id:
                return task

    raise ValueError(f"Task ID '{task_id}' not found in {challenge_file}")


def grid_to_esb(grid_2d):
    """Convert 2D list grid to ESB object with shape (1, H, W)."""
    tensor_2d = torch.tensor(grid_2d, dtype=torch.int8)
    tensor_3d = tensor_2d.unsqueeze(0)  # Add channel dimension: (H,W) -> (1,H,W)
    return ESB(tensor_3d)


def esb_to_2d_list(esb_obj):
    """Extract 2D list from ESB object for Blender."""
    grid = esb_obj.data
    if grid.dim() == 3 and grid.shape[0] == 1:
        grid = grid.squeeze(0)
    return grid.tolist()


def main():
    # CRITICAL FIX: Renamed to 'cli_args' to completely prevent variable shadowing
    parser = argparse.ArgumentParser(
        description="Export MCTS winning path for Blender."
    )
    parser.add_argument(
        "--challenge-file",
        type=str,
        required=True,
        help="Path to aggregate challenges JSON",
    )
    parser.add_argument(
        "--task-id", type=str, required=True, help="The specific Task ID to solve"
    )
    parser.add_argument(
        "--output", type=str, default="winning_path.json", help="Output JSON file"
    )
    cli_args = parser.parse_args()

    print(f"[1/4] Loading task {cli_args.task_id} from {cli_args.challenge_file}...")
    task_data = load_task_data(cli_args.challenge_file, cli_args.task_id)

    train_pairs = task_data["train"]
    test_input = task_data["test"][0]["input"]

    print("[2/4] Converting grids to Eidetic Spatial Buffers...")
    test_esb = grid_to_esb(test_input)

    esb_pairs = []
    for pair in train_pairs:
        inp_esb = grid_to_esb(pair["input"])
        out_esb = grid_to_esb(pair["output"])
        esb_pairs.append((inp_esb, out_esb))

    print("[3/4] Running MCTS Search (this may take 1-5 minutes on CPU)...")
    global_mem = GlobalMemoryBank()
    local_mem = LocalTaskBuffer()

    engine = MCTSEngine(
        global_memory=global_mem,
        local_memory=local_mem,
        C=1.41,
        global_prior_weight=0.0,
    )

    # Solve returns the symbolic program!
    program = engine.solve(training_pairs=esb_pairs)

    print("[4/4] Applying program to test input to generate animation...")
    path_to_export = []

    # Record initial state
    current_esb = test_esb
    path_to_export.append(esb_to_2d_list(current_esb))

    # Extract operations from the DSLProgram
    ops = None
    for attr in ["ops", "operations", "actions", "sequence", "program", "steps"]:
        if hasattr(program, attr):
            ops = getattr(program, attr)
            break

    if ops is None and hasattr(program, "__iter__"):
        ops = list(program)

    if ops:
        print(f"   Found {len(ops)} operations in program.")
        for i, op in enumerate(ops):
            try:
                # Case 1: Operation is already a callable function
                if callable(op):
                    current_esb = op(current_esb)

                # Case 2: Operation is a tuple like ('translate', {'dx': -2, 'dy': -2})
                elif isinstance(op, tuple) and len(op) >= 1:
                    current_esb = engine._apply_action(current_esb, op)

                # Record the new state after applying the operation
                path_to_export.append(esb_to_2d_list(current_esb))

            except Exception as e:
                print(f"   Warning: Failed to apply operation {i} ({op}): {e}")
    else:
        print(
            "   Warning: No operations found in DSLProgram. Path only contains initial state."
        )

    # Save the path using the original cli_args namespace
    with open(cli_args.output, "w") as f:
        json.dump(path_to_export, f, indent=2)

    print(
        f"\n✓ Success! Created '{cli_args.output}' with {len(path_to_export)} frames."
    )
    print(
        "  Next: Copy to Blender folder, rename to 'mcts_path.json', and run visualizer."
    )


if __name__ == "__main__":
    main()
