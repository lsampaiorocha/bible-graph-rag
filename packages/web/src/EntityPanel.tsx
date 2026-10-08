import { useEffect, useState } from "react";
import { KIND_COLORS } from "./GraphView.tsx";

interface EntityDetails {
	id: string;
	name: string;
	kind: string;
	aliases: string[];
	description: string;
	gender: string | null;
	placeKind: string | null;
	mentionCount: number;
	verses: { id: string; text: string }[];
	relations: {
		id: string;
		type: string;
		outgoing: boolean;
		neighbourId: string;
		neighbourName: string;
		description: string;
		otherLabel: string | null;
		verses: string[];
		supportCount: number;
	}[];
}

export function EntityPanel({
	entityId,
	onSelect,
}: {
	entityId: string;
	onSelect: (id: string) => void;
}) {
	const [details, setDetails] = useState<EntityDetails | null>(null);
	const [error, setError] = useState<string | null>(null);

	useEffect(() => {
		setDetails(null);
		setError(null);
		fetch(`/api/entity/${encodeURIComponent(entityId)}`)
			.then((response) =>
				response.ok
					? response.json()
					: Promise.reject(new Error(`HTTP ${response.status}`)),
			)
			.then(setDetails)
			.catch((reason: Error) => setError(reason.message));
	}, [entityId]);

	if (error)
		return (
			<p className="text-sm text-red-700">
				Could not load {entityId}: {error}
			</p>
		);
	if (!details) return <p className="text-sm text-gray-500">Loading…</p>;

	const facts = [details.gender, details.placeKind].filter(Boolean).join(" · ");
	return (
		<div className="flex flex-col gap-3 text-sm">
			<div>
				<div className="flex items-center gap-2">
					<span
						className="inline-block h-3 w-3 rounded-full"
						style={{ background: KIND_COLORS[details.kind] }}
					/>
					<h2 className="text-base font-semibold">{details.name}</h2>
					<span className="text-xs text-gray-500">
						{details.kind}
						{facts && ` · ${facts}`}
					</span>
				</div>
				{details.aliases.length > 0 && (
					<p className="text-xs text-gray-500">
						Also called: {details.aliases.join(", ")}
					</p>
				)}
				{details.description && <p className="mt-1">{details.description}</p>}
			</div>

			{details.relations.length > 0 && (
				<section>
					<h3 className="mb-1 text-xs font-semibold uppercase text-gray-500">
						Relationships ({details.relations.length})
					</h3>
					<ul className="flex flex-col gap-1">
						{details.relations.map((relation) => (
							<li key={relation.id} className="leading-snug">
								<span className="font-mono text-xs text-gray-600">
									{relation.outgoing ? "→" : "←"}{" "}
									{relation.otherLabel ?? relation.type}
								</span>{" "}
								<button
									type="button"
									onClick={() => onSelect(relation.neighbourId)}
									className="font-medium text-blue-700 hover:underline"
								>
									{relation.neighbourName}
								</button>
								<span className="text-gray-600"> · {relation.description}</span>
								<span className="text-xs text-gray-400">
									{" "}
									({relation.verses.slice(0, 3).join(", ")})
								</span>
							</li>
						))}
					</ul>
				</section>
			)}

			<section>
				<h3 className="mb-1 text-xs font-semibold uppercase text-gray-500">
					Named in {details.mentionCount}{" "}
					{details.mentionCount === 1 ? "verse" : "verses"}
					{details.mentionCount > details.verses.length &&
						` (first ${details.verses.length})`}
				</h3>
				<ul className="flex flex-col gap-1.5">
					{details.verses.map((verse) => (
						<li key={verse.id}>
							<span className="font-mono text-xs text-gray-500">
								{verse.id}
							</span>{" "}
							{verse.text}
						</li>
					))}
				</ul>
			</section>
		</div>
	);
}
