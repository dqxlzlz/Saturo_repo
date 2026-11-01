"""KV capacity claimed by RT requests still prefilling on P instances."""

import math


class DecodeReservations:
    def __init__(self):
        self.tokens = {}

    def reserve(self, reservation_id, prompt_tokens, block_size,
                lookahead_slots, total_blocks):
        prompt_tokens = int(prompt_tokens)
        if not reservation_id or prompt_tokens <= 0:
            raise ValueError("invalid decode reservation")
        blocks = math.ceil((prompt_tokens + 1 + lookahead_slots) / block_size)
        if blocks > total_blocks:
            raise ValueError("RT decode reservation exceeds GPU KV capacity")
        self.tokens[str(reservation_id)] = prompt_tokens
        return blocks

    def release(self, reservation_id):
        self.tokens.pop(str(reservation_id), None)

    def blocks(self, block_size, lookahead_slots):
        return sum(math.ceil((n + 1 + lookahead_slots) / block_size)
                   for n in self.tokens.values())

    def __len__(self):
        return len(self.tokens)
