from kiwipiepy import Kiwi

from pipeline.preprocess.text_processor import (
    _kiwi_prepass_text,
    build_batch_glossary,
    extract_glossary_term_records,
    extract_glossary_terms,
)


def test_kiwi_prepass_repairs_close_glossary_typos():
    result = _kiwi_prepass_text(
        "운영체재와 PuTi를 사용합니다.",
        ["운영체제", "PuTTY"],
        Kiwi(),
    )

    assert result["text"] == "운영체제와 PuTTY를 사용합니다."
    assert {edit["to_text"] for edit in result["edits"]} == {"운영체제", "PuTTY"}


def test_kiwi_prepass_does_not_add_unspoken_content():
    result = _kiwi_prepass_text(
        "어셈블리어로 개발을 합니다.",
        ["C++, 어셈블리어"],
        Kiwi(),
    )

    assert result["text"] == "어셈블리어로 개발을 합니다."
    assert result["edits"] == []


def test_kiwi_glossary_extractor_splits_sentence_like_slide_lines():
    terms = extract_glossary_terms(
        {
            1: {
                "title": "운영체제",
                "glossary_text": "컴퓨터 하드웨어나 응용소프트웨어 등 자원 관리",
            }
        }
    )

    assert "응용소프트웨어" in terms
    assert "컴퓨터 하드웨어나 응용소프트웨어 등 자원 관리" not in terms


def test_glossary_records_keep_source_scene_for_batch_prompt():
    slides = {
        3: {
            "title": "운영체제",
            "glossary_text": "CPU 메모리 관리",
            "scene_numbers": [12, 13],
        }
    }
    records = extract_glossary_term_records(slides)
    cpu = next(record for record in records if record["term"] == "CPU")
    assert cpu["source_slides"] == [3]
    assert cpu["source_scenes"] == [12, 13]

    glossary = build_batch_glossary(
        [(0, {"text": "CPU를 설명합니다."})],
        slides,
        3,
        [record["term"] for record in records],
        term_records=records,
        scene_indices={13},
    )
    assert "CPU (slide 3; scene 12/13)" in glossary
