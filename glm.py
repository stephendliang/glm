from transformers import AutoProcessor, Glm4vForConditionalGeneration
import inspect

MODEL_PATH = "zai-org/GLM-4.6V-Flash"
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = Glm4vForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
)


def dump_forward(obj, label):
    cls = obj.__class__
    print("=" * 80)
    print(f"{label}: {cls.__name__}")
    try:
        src = inspect.getsource(cls.forward)
        path = inspect.getsourcefile(cls)
        print(f"File: {path}")
        print("-" * 80)
        print(src)
    except Exception as exc:
        print(f"Could not get source for {cls.__name__}.forward: {exc}")


dump_forward(model, "Glm4vForConditionalGeneration")
if hasattr(model, "model"):
    dump_forward(model.model, "Glm4vModel")
    if hasattr(model.model, "visual"):
        dump_forward(model.model.visual, "Glm4vVisionModel")
    if hasattr(model.model, "language_model"):
        dump_forward(model.model.language_model, "Glm4vTextModel")
