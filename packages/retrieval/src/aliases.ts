const MAX_NGRAM_TOKENS = 4;
const SINGLE_WORD_STOPWORDS = new Set([
	"a",
	"an",
	"and",
	"in",
	"is",
	"of",
	"on",
	"or",
	"the",
	"to",
	"was",
	"who",
]);

export interface QuestionNgram {
	key: string;
	start: number;
	end: number;
}

// Mirrors alias_key in pipeline/src/kgrag/neo4j_load.py; both must produce the same keys.
export function aliasKey(name: string): string {
	return tokens(
		name
			.trim()
			.toLowerCase()
			.replace(/^the\s+/, ""),
	).join(" ");
}

function tokens(text: string): string[] {
	return text
		.replaceAll("’", "'")
		.replace(/'s\b/g, "")
		.replace(/['-]/g, "")
		.replace(/[^a-z0-9]+/g, " ")
		.split(" ")
		.filter(Boolean);
}

export function questionNgrams(question: string): QuestionNgram[] {
	const words = tokens(question.toLowerCase());
	const ngrams: QuestionNgram[] = [];
	for (
		let length = Math.min(MAX_NGRAM_TOKENS, words.length);
		length >= 1;
		length--
	) {
		for (let start = 0; start + length <= words.length; start++) {
			const key = words.slice(start, start + length).join(" ");
			if (length === 1 && SINGLE_WORD_STOPWORDS.has(key)) continue;
			ngrams.push({ key, start, end: start + length });
		}
	}
	return ngrams;
}
