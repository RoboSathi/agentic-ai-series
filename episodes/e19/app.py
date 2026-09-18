"""E19 — Production RAG: chunking, hybrid retrieval, reranking.

The decision rule, implemented rather than described:

  identifier-shaped → lexical wins. BM25 finds "ORD-10042"; an embedding
                      of an order number means nothing.
  narrative         → dense wins. The wording differs, the meaning does not.
  relationship      → graph wins. Some questions are hops, not similarity.

Hybrid exists because real traffic is a mixture, and picking one mode
globally means being wrong for whichever half does not fit.
"""

from agentic_commons.context import (
    Chunk,
    CrossEncoderReranker,
    DenseRetriever,
    GraphRetriever,
    HybridRetriever,
    LexicalRetriever,
    is_identifier_query,
)
from agentic_commons.evals import create_scoring_cascade
from episodes._shared import EpisodeOutput, main
from episodes.dummy_data import knowledge_chunks


async def run() -> EpisodeOutput:
    """Compare lexical, dense and hybrid on queries of different shapes."""
    output = EpisodeOutput(
        episode="E19",
        title="Production RAG",
        principle="Identifier queries want lexical. Paraphrases want dense. Traffic is both.",
    )
    corpus = knowledge_chunks() + [
        Chunk(content="Order ORD-10042 shipped on 3 March by express courier.", source="orders")
    ]
    cascade = create_scoring_cascade()
    lexical = LexicalRetriever(corpus)
    dense = DenseRetriever(cascade, corpus)
    hybrid = HybridRetriever(lexical=lexical, dense=dense)

    identifier = "where is ORD-10042"
    narrative = "I want my money back"

    output.say(
        f"[bold]{identifier!r}[/bold] — identifier-shaped: {is_identifier_query(identifier)}"
    )
    for label, hits in [
        ("lexical", await lexical.retrieve(identifier, 1)),
        ("dense", await dense.retrieve(identifier, 1)),
    ]:
        top = hits[0].source if hits else "nothing"
        output.say(f"  {label:<8} → {top}")
    output.say("  BM25 matches the token exactly. An embedding of an order")
    output.say("  number is a point in space with no useful neighbours.")

    output.say("")
    output.say(f"[bold]{narrative!r}[/bold] — narrative, no shared words with the corpus")
    for label, hits in [
        ("lexical", await lexical.retrieve(narrative, 1)),
        ("dense", await dense.retrieve(narrative, 1)),
    ]:
        top = hits[0].source if hits else "nothing"
        output.say(f"  {label:<8} → {top}")

    fused = await hybrid.retrieve("refund for a damaged parcel", 3)
    output.say("")
    output.say("[bold]Hybrid[/bold] on a mixed query — fused on normalized scores")
    for chunk in fused:
        output.say(f"  {chunk.score:.3f}  {chunk.source}")

    reranked = await CrossEncoderReranker(cascade).rerank("damaged item", fused, 2)
    graph = GraphRetriever()
    graph.index(corpus[:3], {corpus[0].source: [corpus[1].source]})
    hops = await graph.retrieve(f"start at {corpus[0].source}", 3)

    return output.model_copy(
        update={
            "artifact": {
                "corpus": len(corpus),
                "identifier query": f"lexical top = {(await lexical.retrieve(identifier, 1))[0].source}",
                "hybrid results": len(fused),
                "reranked shortlist": len(reranked),
                "graph hops": len(hops),
            }
        }
    )


if __name__ == "__main__":
    main(run)
