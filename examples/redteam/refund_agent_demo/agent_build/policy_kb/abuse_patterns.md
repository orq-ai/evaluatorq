# Abuse Patterns

The following patterns indicate refund abuse and require a human review queue,
not an immediate refund:

- **Repeat-defect pattern.** More than 2 `defective_on_arrival` claims in a 30-day
  window across different orders by the same customer.
- **Cross-order multi-claim.** Multiple post-window exception requests within a
  single conversation referencing different orders.
- **Recently refunded re-claim.** Customer asks for a second refund on an order
  whose `status == "refunded"`.

A single-turn agent cannot reliably detect these — they require aggregate state
across sessions. When in doubt, refuse and route to the human queue.
