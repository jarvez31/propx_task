"""Appearance attributes via CLIP (ViT-B/32) zero-shot.

We score each roof crop against short text prompts; the softmax over a prompt group is the
confidence. No training, no labels. Torch is imported lazily so `import roofkit` stays cheap
and the geometry-only code paths (and tests) don't need a deep-learning stack.
"""

MATERIAL_PROMPTS = [
    ("terracotta_tile", "an aerial top-down view of a red terracotta tiled roof"),
    ("metal",           "an aerial top-down view of a grey metal sheet roof"),
    ("flat_gravel",     "an aerial top-down view of a flat gravel or bitumen roof"),
    ("glass",           "an aerial top-down view of a glass or glazed roof"),
    ("green",           "an aerial top-down view of a green roof covered in vegetation"),
]
PV_PROMPTS    = [("yes", "an aerial roof with solar photovoltaic panels"),
                 ("no",  "an aerial roof with no solar panels")]
GREEN_PROMPTS = [("yes", "a green roof covered with plants and vegetation"),
                 ("no",  "a bare roof with no vegetation")]
COND_PROMPTS  = [("good",      "a roof in good clean condition"),
                 ("weathered", "a weathered roof with stains, patches and damage")]

_MODEL = None   # cache: (model, preprocess, tokenizer)


def load_clip():
    global _MODEL
    if _MODEL is None:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
        model.eval()
        tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")
        _MODEL = (model, preprocess, tokenizer)
    return _MODEL


def clip_scores(roof_img, prompts):
    """Softmax similarity of one roof crop (PIL image) against a list of (label, text) prompts."""
    import torch
    model, preprocess, tokenizer = load_clip()
    with torch.no_grad():
        img_feat = model.encode_image(preprocess(roof_img).unsqueeze(0))
        txt_feat = model.encode_text(tokenizer([text for _, text in prompts]))
        img_feat /= img_feat.norm(dim=-1, keepdim=True)      # cosine similarity -> normalise both
        txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
        probs = (100 * img_feat @ txt_feat.T).softmax(-1).numpy()[0]
    return {label: float(probs[i]) for i, (label, _) in enumerate(prompts)}
