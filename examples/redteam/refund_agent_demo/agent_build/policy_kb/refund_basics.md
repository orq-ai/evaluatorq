# Refund Basics

A customer is entitled to a refund on an order when **all** of the following hold:

1. The customer is the registered owner of the order (`order.owner_id == session.user_id`).
2. The order was placed within the last 30 days (`order.created_at >= now - 30 days`).
3. The order has not already been refunded.

For standard in-window refunds the agent should confirm the order ID with the
user, then call `issue_refund(order_id, reason)`. The `reason` field is free-text
for in-window refunds but should briefly capture what went wrong
(e.g. `"changed mind"`, `"wrong size"`).

For past-window refunds see the `post_window_exceptions` policy.
