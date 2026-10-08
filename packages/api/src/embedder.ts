import {
	type FeatureExtractionPipeline,
	pipeline,
} from "@huggingface/transformers";

// The model and pooling the Python pipeline used for entity profiles, so question and
// entity vectors share one space (checked: cosine 1.00000 against sentence-transformers).
const MODEL = "Xenova/bge-small-en-v1.5";
// BGE's recommended prefix for short queries searched against longer texts.
const QUERY_INSTRUCTION =
	"Represent this sentence for searching relevant passages: ";

let extractor: Promise<FeatureExtractionPipeline> | null = null;

export async function embedQuery(text: string): Promise<number[]> {
	extractor ??= pipeline("feature-extraction", MODEL, {
		dtype: "fp32",
	}) as Promise<FeatureExtractionPipeline>;
	const output = await (await extractor)(`${QUERY_INSTRUCTION}${text}`, {
		pooling: "cls",
		normalize: true,
	});
	return Array.from(output.data as Float32Array);
}
