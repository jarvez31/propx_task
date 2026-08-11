"""Appearance attributes via CLIP (ViT-B/32) zero-shot.

We score each roof crop against short text prompts; the softmax over a prompt group is the
confidence. No training, no labels. Torch is imported lazily so `import roofkit` stays cheap
and the geometry-only code paths (and tests) don't need a deep-learning stack.

Each class carries an ENSEMBLE of phrasings (the standard CLIP zero-shot trick): the class
embedding is the renormalised mean of its prompt embeddings, which is less sensitive to any
single wording than one sentence per class. The vocabulary includes slate/fibre-cement because
Vienna mansards are commonly dark slate — without that class, "metal" becomes a catch-all for
every dark pitched roof.
"""

MATERIAL_PROMPTS = [
    ("terracotta_tile", ["an aerial top-down view of a red terracotta tiled roof",
                         "an aerial photo of an old European building with an orange clay tile roof",
                         "rows of small rectangular red-brown roof tiles seen from above"]),
    ("slate",           ["an aerial top-down view of a dark slate roof",
                         "an aerial photo of a steep dark grey slate mansard roof",
                         "a roof of small dark grey slate or fibre-cement shingles seen from above"]),
    ("metal",           ["an aerial top-down view of a grey metal sheet roof",
                         "an aerial photo of a building with a light grey standing-seam metal roof",
                         "smooth pale grey metal roofing panels with long straight seams seen from above"]),
    ("flat_gravel",     ["an aerial top-down view of a flat gravel or bitumen roof",
                         "an aerial photo of a flat dark bitumen roof with gravel patches"]),
    ("glass",           ["an aerial top-down view of a glass or glazed roof",
                         "an aerial photo of a building with a glass skylight roof"]),
    ("green",           ["an aerial top-down view of a green roof covered in vegetation",
                         "an aerial photo of a rooftop covered in grass and plants"]),
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
    """Softmax similarity of one roof crop (PIL image) against (label, text-or-texts) prompts.

    A label with several phrasings is scored against the renormalised mean of their embeddings
    (prompt ensembling); a single string behaves exactly as before.
    """
    import torch
    model, preprocess, tokenizer = load_clip()
    with torch.no_grad():
        img_feat = model.encode_image(preprocess(roof_img).unsqueeze(0))
        img_feat /= img_feat.norm(dim=-1, keepdim=True)      # cosine similarity -> normalise both
        class_feats = []
        for _, texts in prompts:
            t = model.encode_text(tokenizer([texts] if isinstance(texts, str) else list(texts)))
            t /= t.norm(dim=-1, keepdim=True)
            t = t.mean(0)
            class_feats.append(t / t.norm())
        txt_feat = torch.stack(class_feats)
        probs = (100 * img_feat @ txt_feat.T).softmax(-1).numpy()[0]
    return {label: float(probs[i]) for i, (label, _) in enumerate(prompts)}
