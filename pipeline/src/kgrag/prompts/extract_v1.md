You extract a knowledge graph from one passage of the King James Bible. The passage arrives as lines of the form `[OsisRef] verse text`. Return JSON that matches the provided schema.

Use only the passage. Do not add people, places or facts you know from elsewhere in the Bible.

## Entities

- **people**: every individual named in the passage. "God", "the LORD", "the Lord GOD" and "the Almighty" are one person named "God".
- **places**: cities, regions, landmarks, mountains, bodies of water, valleys and islands. Set `kind` to City, Region, Landmark, Mountain, Water, Valley or Island, or Unknown when the passage does not make it clear.
- **groups**: tribes, nations, peoples and sects, such as the Israelites, Philistines, Levites or Pharisees. A name used for a whole people ("the children of Israel", "the Egyptians") is a group. The same name used for one man (Israel meaning Jacob) is a person.
- List each entity once. Put the fullest name in `name`, and every other name or title the passage uses for the same entity in `aliases`. Example: when Abram is renamed, use name "Abraham" with aliases ["Abram"].
- `description` is one sentence about the entity, based only on this passage.

## named_in: names only

List a verse in `named_in` only when the entity is called by its name, an alias, or a title used as a name ("the LORD", "Pharaoh") in that verse. Pronouns ("he", "she", "him") and descriptions ("his wife", "the child") do not count.

## Relations

- `source` and `target` must be exact names from your entity lists.
- Family relations:
  - FATHER_OF and MOTHER_OF point from parent to child.
  - SIBLING_OF and SPOUSE_OF: list each pair once, in either direction.
- How KJV phrasing maps to family relations:

  | KJV phrasing | Relations |
  |---|---|
  | "X begat Y" | FATHER_OF X→Y |
  | "X knew his wife Y; and she conceived, and bare Z" | SPOUSE_OF X–Y, FATHER_OF X→Z, MOTHER_OF Y→Z |
  | "X took Y to wife" | SPOUSE_OF X–Y |
  | "the sons of X; A, and B" | FATHER_OF X→A and X→B |
  | "A his brother", "B his sister" | SIBLING_OF |

- Other relations:
  - MEMBER_OF: person → group.
  - BORN_IN and DIED_IN: person → place.
  - RULED: person → place or group.
- Narrative relations: SUCCEEDED, ANOINTED, FOUGHT, KILLED, BETRAYED, BLESSED, CURSED, HEALED, DISCIPLE_OF, SERVED, SPOKE_TO.
  - SUCCEEDED, ANOINTED, BETRAYED, HEALED and DISCIPLE_OF connect two people.
  - FOUGHT, KILLED, BLESSED, CURSED, SERVED and SPOKE_TO connect people or groups.
  - Use SPOKE_TO only for a significant address, command or promise, not for every "said unto".
- Anything else important uses type OTHER, with `other_label` as a short verb phrase such as "nursed" or "dreamed of". `other_label` is null for every other type.
- You may resolve pronouns within the passage to work out who did what. Every relation must list the verses that state it.

## Events

Notable happenings such as births, deaths, journeys, battles and covenants.
- `participants` are names from your people or groups lists.
- `place` is a name from your places list, or null.

## Example

Passage:
[Ruth.4.13] So Boaz took Ruth, and she was his wife: and when he went in unto her, the LORD gave her conception, and she bare a son.
[Ruth.4.14] And the women said unto Naomi, Blessed be the LORD, which hath not left thee this day without a kinsman, that his name may be famous in Israel.
[Ruth.4.15] And he shall be unto thee a restorer of thy life, and a nourisher of thine old age: for thy daughter in law, which loveth thee, which is better to thee than seven sons, hath born him.
[Ruth.4.16] And Naomi took the child, and laid it in her bosom, and became nurse unto it.
[Ruth.4.17] And the women her neighbours gave it a name, saying, There is a son born to Naomi; and they called his name Obed: he is the father of Jesse, the father of David.

Output:
{"people": [
  {"name": "Boaz", "aliases": [], "gender": "male", "description": "Boaz took Ruth as his wife and fathered her son.", "named_in": ["Ruth.4.13"]},
  {"name": "Ruth", "aliases": [], "gender": "female", "description": "Ruth became the wife of Boaz and bore a son.", "named_in": ["Ruth.4.13"]},
  {"name": "God", "aliases": ["the LORD"], "gender": "male", "description": "The LORD gave Ruth conception and was blessed by the women.", "named_in": ["Ruth.4.13", "Ruth.4.14"]},
  {"name": "Naomi", "aliases": [], "gender": "female", "description": "Naomi nursed the child, whom the women called a son born to her.", "named_in": ["Ruth.4.14", "Ruth.4.16", "Ruth.4.17"]},
  {"name": "Obed", "aliases": [], "gender": "male", "description": "Obed is the son born to Ruth, named by the women neighbours.", "named_in": ["Ruth.4.17"]},
  {"name": "Jesse", "aliases": [], "gender": "male", "description": "Jesse is the son of Obed and the father of David.", "named_in": ["Ruth.4.17"]},
  {"name": "David", "aliases": [], "gender": "male", "description": "David is the son of Jesse.", "named_in": ["Ruth.4.17"]}
 ],
 "places": [],
 "groups": [
  {"name": "Israel", "aliases": [], "description": "The people of Israel, among whom the child's name would be famous.", "named_in": ["Ruth.4.14"]}
 ],
 "events": [
  {"title": "Birth of Obed", "participants": ["Boaz", "Ruth", "Obed", "Naomi"], "place": null, "verses": ["Ruth.4.13", "Ruth.4.16", "Ruth.4.17"]}
 ],
 "relations": [
  {"type": "SPOUSE_OF", "source": "Boaz", "target": "Ruth", "description": "Boaz took Ruth, and she became his wife.", "verses": ["Ruth.4.13"], "other_label": null},
  {"type": "FATHER_OF", "source": "Boaz", "target": "Obed", "description": "Boaz fathered the son Ruth bore, who was named Obed.", "verses": ["Ruth.4.13", "Ruth.4.17"], "other_label": null},
  {"type": "MOTHER_OF", "source": "Ruth", "target": "Obed", "description": "Ruth bore the son who was named Obed.", "verses": ["Ruth.4.13", "Ruth.4.17"], "other_label": null},
  {"type": "FATHER_OF", "source": "Obed", "target": "Jesse", "description": "Obed is the father of Jesse.", "verses": ["Ruth.4.17"], "other_label": null},
  {"type": "FATHER_OF", "source": "Jesse", "target": "David", "description": "Jesse is the father of David.", "verses": ["Ruth.4.17"], "other_label": null},
  {"type": "OTHER", "source": "Naomi", "target": "Obed", "description": "Naomi laid the child in her bosom and became his nurse.", "verses": ["Ruth.4.16"], "other_label": "nursed"}
 ]}
