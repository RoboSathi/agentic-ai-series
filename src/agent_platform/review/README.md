# review — one human-review queue for every agent

Everything the harness hands a person is an item in one queue, decided one way: with a comment, signed, audited. Doors: `/review`, `make review <agent>`, the Review screen (`ctrl+r`).

- **Entry points:** `ReviewQueue`, `ReviewService`, `ReviewHandler`, `MemoryReviewHandler`, `QuarantineHandler`
- **Configuration:** `review:`
- **Design:** [LLD §A22](../../../design_docs/LLD.md#a22-review) · [ADR P-14](../../../design_docs/ADR.md#p-14--one-human-review-queue-a-comment-on-every-decision)
