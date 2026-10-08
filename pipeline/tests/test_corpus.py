from kgrag.corpus import build_verses, pack_passages


def test_ingest_keeps_only_whitelisted_fields_and_no_link_markup():
    raw_genesis_1_1 = {
        "id": "rec7mkRLwey2ntUG9",
        "fields": {
            "osisRef": "Gen.1.1",
            "verseNum": "1",
            "verseText": "In the beginning God created the heaven and the earth.",
            "people": ["reccZB8SVU5bEMcgo"],
            "peopleCount": 1,
            "mdText": "In the beginning [God]([/person/god_1324) created the heaven and the earth.",
            "richText": "In the beginning [God](/person/god_1324) created the heaven and the earth.\n",
            "verseID": "01001001",
            "event": ["recOIPcFaPCjMpQ6w"],
        },
    }

    assert build_verses([raw_genesis_1_1]) == [
        {
            "osis_ref": "Gen.1.1",
            "verse_id": "01001001",
            "book": "Gen",
            "chapter": 1,
            "verse": 1,
            "text": "In the beginning God created the heaven and the earth.",
        }
    ]


def test_passages_cover_every_verse_exactly_once_without_crossing_chapters():
    verses = [
        {
            "osis_ref": f"Gen.{chapter}.{verse}",
            "book": "Gen",
            "chapter": chapter,
            "text": "a b c",
        }
        for chapter, verse_count in [(1, 5), (2, 2)]
        for verse in range(1, verse_count + 1)
    ]

    passages = pack_passages(
        verses, count_tokens=lambda text: len(text.split()), max_tokens=12
    )

    assert [(passage["id"], passage["verse_refs"]) for passage in passages] == [
        ("Gen.1.p1", ["Gen.1.1", "Gen.1.2"]),
        ("Gen.1.p2", ["Gen.1.3", "Gen.1.4"]),
        ("Gen.1.p3", ["Gen.1.5"]),
        ("Gen.2.p1", ["Gen.2.1", "Gen.2.2"]),
    ]
    assert passages[0]["text"] == "[Gen.1.1] a b c\n[Gen.1.2] a b c"
