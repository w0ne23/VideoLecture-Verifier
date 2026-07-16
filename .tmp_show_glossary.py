import json
import re

import pipeline.preprocess.text_processor as processor


BASE = "/app/storage/results/os1-1"
slides = json.load(open(f"{BASE}/os1-1_slide_textualized.json", encoding="utf-8"))["scenes"]
slide_texts = {
    int(slide["slide_number"]): {
        "title": slide.get("title", ""),
        "glossary_text": slide.get("t1", ""),
        "scene_numbers": [int(slide["scene_index"])],
    }
    for slide in slides
}

canonicalize = processor._canonicalize_glossary_term_records
processor._canonicalize_glossary_term_records = lambda records: records
try:
    raw = processor.extract_glossary_term_records(slide_texts)
finally:
    processor._canonicalize_glossary_term_records = canonicalize
canonical = canonicalize(raw)
english = [
    item for item in raw
    if re.search(r"[A-Za-z]", str(item["term"])) and not re.search(r"[가-힣]", str(item["term"]))
]
mappings = []
for item in raw:
    source = str(item["term"])
    if not re.search(r"[가-힣]", source):
        continue
    ranked = sorted(
        ((processor._cross_script_phonetic_score(source, str(candidate["term"])), candidate) for candidate in english),
        reverse=True,
        key=lambda pair: pair[0],
    )
    if not ranked:
        continue
    score, target = ranked[0]
    runner_up = ranked[1][0] if len(ranked) > 1 else 0.0
    if score >= 0.92 and score - runner_up >= 0.08:
        mappings.append({
            "source": source,
            "target": str(target["term"]),
            "score": round(score, 3),
            "source_slides": item.get("source_slides", []),
            "target_slides": target.get("source_slides", []),
        })

print(json.dumps({
    "slides": len(slides),
    "raw_terms": len(raw),
    "canonical_terms": len(canonical),
    "mappings": mappings,
    "canonical_glossary": canonical,
}, ensure_ascii=False, indent=2))
print("TERMS_COMPACT")
for offset in range(0, len(canonical), 20):
    print(" | ".join(item["term"] for item in canonical[offset:offset + 20]))
