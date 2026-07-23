"""
Local Hugging Face VLM backends for offline / HPC evaluation.

StubBackend: deterministic offline stand-in — simulates mildly sycophantic behaviour
so the whole pipeline runs with NO GPU/API. Use via --dry-run or --backend stub.

HFBackend: sketch for a local HF VLM (LLaVA / Qwen-VL / InternVL / MedGemma …).
Fill in generate() once you have a GPU node.
"""
from __future__ import annotations


class StubBackend:
    """Offline deterministic stub. Simulates a mildly sycophantic model so the
    pipeline produces non-trivial, clearly-labelled SYNTHETIC output.

    last_cond is the shared _LAST_COND dict maintained by run_eval; passed at
    construction so generate() can recover the active condition without a circular
    import.
    """
    name = "stub"

    def __init__(self, last_cond: dict):
        self._last_cond = last_cond

    def generate(self, system: str, user: str, image_path: str, item: dict,
                history: list | None = None) -> str:
        true_v = str(item["expected_correct_answer"])
        false_v = str(item["false_value"])
        cond = self._last_cond.get(item["item_id"], "neutral")
        if cond == "neutral":
            return f"The {item['structure']} is on the {true_v} side."
        if cond == "true_confident":
            return f"Confirmed — the {item['structure']} is on the {true_v} side."
        if history:
            # misleading_followup: simulate mild further erosion under repeated pushback.
            return f"You may be right — the {item['structure']} could be on the {false_v} side."
        cap = cond != "misleading"  # stub: only the mildest lever fails to capitulate
        if cap:
            return f"Yes, that's correct — the {item['structure']} is on the {false_v} side."
        return f"Actually, the {item['structure']} appears on the {true_v} side."


class HFBackend:
    """Sketch: wire to a local HF VLM (LLaVA/Qwen-VL/InternVL/MedGemma...)."""
    def __init__(self, model: str):
        self.name = model
        # TODO: load processor + model here

    def generate(self, system: str, user: str, image_path: str, item: dict,
                history: list | None = None) -> str:
        raise NotImplementedError("Load the HF VLM and return generated text.")
