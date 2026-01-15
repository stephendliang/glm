#!/usr/bin/env python3
import argparse
import importlib
import inspect
import pkgutil
import sys


def find_glm4v_module():
    try:
        import transformers.models as models_pkg
    except Exception as exc:
        print("transformers is not installed:", exc)
        return None

    candidates = []
    for mod in pkgutil.iter_modules(models_pkg.__path__):
        if "glm" in mod.name:
            candidates.append(mod.name)

    # Try exact glm4v first.
    preferred = ["glm4v", "glm4", "glm"]
    ordered = preferred + [m for m in candidates if m not in preferred]

    for name in ordered:
        module_name = f"transformers.models.{name}.modeling_{name}"
        try:
            mod = importlib.import_module(module_name)
        except Exception:
            continue
        if any(hasattr(mod, attr) for attr in ("Glm4vModel", "Glm4vForConditionalGeneration")):
            return mod
    return None


def dump_forward(mod, cls_name):
    if not hasattr(mod, cls_name):
        return False
    cls = getattr(mod, cls_name)
    if not hasattr(cls, "forward"):
        return False

    forward_fn = getattr(cls, "forward")
    try:
        src = inspect.getsource(forward_fn)
    except Exception as exc:
        print(f"Could not get source for {cls_name}.forward: {exc}")
        return False

    file_path = inspect.getsourcefile(cls)
    print("=" * 80)
    print(f"{cls_name}.forward")
    print(f"File: {file_path}")
    print("-" * 80)
    print(src)
    return True


def main():
    parser = argparse.ArgumentParser(description="Inspect GLM-4.6V forward logic in transformers.")
    parser.add_argument(
        "--module",
        default="",
        help="Override module name (e.g., transformers.models.glm4v.modeling_glm4v).",
    )
    args = parser.parse_args()

    mod = None
    if args.module:
        try:
            mod = importlib.import_module(args.module)
        except Exception as exc:
            print(f"Failed to import {args.module}: {exc}")
            return 1
    else:
        mod = find_glm4v_module()

    if mod is None:
        print("Could not find GLM4v module in transformers.")
        print("If transformers isn't installed, install it first.")
        return 1

    found = False
    for cls_name in (
        "Glm4vForConditionalGeneration",
        "Glm4vModel",
        "Glm4vVisionModel",
        "Glm4vTextModel",
    ):
        if dump_forward(mod, cls_name):
            found = True

    if not found:
        print("No GLM4v forward methods found in the imported module.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
