from typing import Any, Type, Union, get_origin, get_args
from pydantic import BaseModel


def _pretty(t: Any) -> str:
    origin = get_origin(t)
    if origin is Union:
        return " | ".join(_pretty(a) for a in get_args(t))
    return getattr(t, "__name__", str(t))

def snippet(model: Type[BaseModel]) -> str:
    lines = [f'{{\n intent: "{model.model_fields["intent"].default}",']
    for name, field in model.model_fields.items():
        if name == "intent":
            continue
        lines.append(f"  {name}: {_pretty(field.annotation)},")
    lines.append("}")
    return "\n".join(lines)