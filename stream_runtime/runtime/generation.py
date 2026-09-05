class TokenGenerator:
    def __init__(self, engine):
        self.engine = engine

    async def generate(self, input_tokens, max_new_tokens=1, sampler=None):
        import torch

        tokens = torch.as_tensor(input_tokens)
        for _ in range(max_new_tokens):
            logits = self.engine.run(tokens)
            next_token = int(
                sampler(logits)
                if sampler
                else torch.argmax(logits, dim=-1).reshape(-1)[-1]
            )
            tokens = torch.cat(
                [tokens.reshape(-1), torch.tensor([next_token], dtype=tokens.dtype)]
            )
            yield next_token
