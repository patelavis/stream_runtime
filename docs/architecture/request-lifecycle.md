# Request Lifecycle

An HTTP request is authenticated, parsed, assigned an id, queued behind the request manager lock, executed in a worker thread, and converted to JSON or SSE. The model transitions from UNLOADED to INITIALIZING to READY and RUNNING during first/useful inference. Timeout responses are supported; disconnect-aware cancellation and explicit model unload are planned.
