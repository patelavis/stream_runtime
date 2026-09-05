# Streaming

Set `"stream": true` on chat completions. The response is `text/event-stream` with OpenAI-style `data: ...` chunks and a final `data: [DONE]`. The current demo splits its deterministic response into ordered word chunks; incremental model token decoding is **planned**.
