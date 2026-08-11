"""Special [EDIT] tokens — single source of truth for the whole edit-agent stack.

An edit-confirming assistant turn ends with:  [EDIT_<KIND>][EDIT_0]...[EDIT_7]
- the KIND token declares the edit type (one special token per kind of edit)
- the 8 payload tokens carry the edit semantics in their hidden states

All stages (dialogue data, SFT, bridge, inference) import from here.
"""

K_EDIT = 8
EDIT_TOKENS = [f"[EDIT_{i}]" for i in range(K_EDIT)]
EDIT_BLOCK = "".join(EDIT_TOKENS)  # "[EDIT_0][EDIT_1]...[EDIT_7]"

# one special token per canonical edit kind
EDIT_KINDS = ["ADD", "REMOVE", "REPLACE", "EXTRACT", "REBALANCE",
              "EFFECT", "MOOD"]  # ENERGY/GENRE folded into MOOD (too few examples)
EDIT_TYPE_TOKENS = {k: f"[EDIT_{k}]" for k in EDIT_KINDS}
ALL_SPECIAL_TOKENS = list(EDIT_TYPE_TOKENS.values()) + EDIT_TOKENS

# dataset edit_type / op values -> canonical kind
_TYPE_TO_KIND = {
    "instrument_add": "ADD", "slakh_add": "ADD", "ae_add": "ADD",
    "inp_add_stem": "ADD", "add": "ADD", "add_stem": "ADD",
    "instrument_remove": "REMOVE", "slakh_remove": "REMOVE", "ae_remove": "REMOVE",
    "inp_delete_stem": "REMOVE", "remove": "REMOVE", "delete_stem": "REMOVE",
    "instrument_swap": "REPLACE", "slakh_swap": "REPLACE",
    "inp_replace_stem": "REPLACE", "swap": "REPLACE", "replace_stem": "REPLACE",
    "ae_extract": "EXTRACT", "slakh_isolate": "EXTRACT",
    "extract": "EXTRACT", "isolate": "EXTRACT",
    "slakh_rebalance": "REBALANCE", "rebalance": "REBALANCE",
    "effect_add": "EFFECT",
    "mood_shift": "MOOD", "inp_change_mood": "MOOD", "change_mood": "MOOD",
    "energy_change": "MOOD",
    "genre_transfer": "MOOD",
}


def kind_of(edit_type: str | None) -> str:
    """Canonical kind for a dataset edit_type/op string."""
    if not edit_type:
        return "ADD"
    if edit_type in _TYPE_TO_KIND:
        return _TYPE_TO_KIND[edit_type]
    t = edit_type.lower()
    for key, kind in _TYPE_TO_KIND.items():
        if key in t:
            return kind
    return "EFFECT" if "effect" in t or "reverb" in t else "ADD"


def typed_block(edit_type: str | None) -> str:
    """The full emission suffix for an edit of the given type."""
    return EDIT_TYPE_TOKENS[kind_of(edit_type)] + EDIT_BLOCK
