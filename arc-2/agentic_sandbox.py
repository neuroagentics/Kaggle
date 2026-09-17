"""Pure-stdlib validator shared by the ARC code-agent controller and worker."""

from __future__ import annotations

import ast


ALLOWED_NODES = {
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign,
    ast.AnnAssign, ast.AugAssign, ast.For, ast.If, ast.IfExp, ast.Expr, ast.Pass,
    ast.Break, ast.Continue, ast.Name, ast.Load, ast.Store, ast.Constant,
    ast.List, ast.Tuple, ast.Dict, ast.Set, ast.Subscript, ast.Slice, ast.Starred,
    ast.Attribute,
    ast.Call, ast.keyword, ast.ListComp, ast.SetComp, ast.DictComp,
    ast.GeneratorExp, ast.comprehension, ast.BinOp, ast.UnaryOp, ast.BoolOp,
    ast.Compare, ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt,
    ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot,
}

SAFE_CALLS = {
    "abs", "all", "any", "bool", "dict", "enumerate", "int", "len",
    "list", "max", "min", "range", "reversed", "set", "sorted", "sum",
    "tuple", "zip",
    "background_color", "color_counts", "component_count", "components", "copy_grid",
    "crop_bbox", "largest_component", "make_grid", "paint",
    "reflect_horizontal", "reflect_vertical", "replace_color", "rotate90",
    "rotate180", "rotate270", "scale_grid", "smallest_component",
    "split_panels", "symmetries", "tile_grid", "transpose_grid", "unique_panel",
}

SAFE_METHODS = {
    "append", "copy", "count", "extend", "get", "index", "items", "keys",
    "pop", "reverse", "sort", "values",
}


def validate_code(code: str, *, max_nodes: int = 800) -> int:
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise ValueError(f"invalid Python: {exc.msg}") from exc
    nodes = list(ast.walk(tree))
    if len(nodes) > max_nodes:
        raise ValueError(f"program exceeds {max_nodes} AST nodes")
    invalid = [type(node).__name__ for node in nodes if type(node) not in ALLOWED_NODES]
    if invalid:
        raise ValueError(f"forbidden syntax: {invalid[0]}")
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    if "transform" not in functions:
        raise ValueError("program must define transform(grid)")
    if any(not isinstance(node, ast.FunctionDef) for node in tree.body):
        raise ValueError("module scope may contain only function definitions")
    for node in nodes:
        if isinstance(node, ast.Name) and "__" in node.id:
            raise ValueError("dunder names are forbidden")
        if isinstance(node, ast.FunctionDef):
            if "__" in node.name or node.decorator_list:
                raise ValueError("decorators and dunder functions are forbidden")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in SAFE_CALLS | functions:
                    raise ValueError(f"call is not allowed: {node.func.id}")
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr not in SAFE_METHODS or "__" in node.func.attr:
                    raise ValueError(f"method is not allowed: {node.func.attr}")
            else:
                raise ValueError("only safe direct calls and list methods are allowed")
        if isinstance(node, ast.Attribute):
            if node.attr not in SAFE_METHODS or "__" in node.attr:
                raise ValueError(f"attribute is not allowed: {node.attr}")
    transform = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "transform"
    )
    if len(transform.args.args) != 1 or transform.args.args[0].arg != "grid":
        raise ValueError("transform must take exactly one argument named grid")
    return len(nodes)
