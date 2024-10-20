import random


def generate_random_location(grid_size: int, disallowed_locations: list[tuple[int, int]] | None = None) -> tuple[int, int]:
    return generate_random_pair_numbers(grid_size-1, grid_size-1, disallowed_locations)


def generate_random_pair_numbers(min_val: int, max_val: int, disallowed_pairs: list[tuple[int, int]] | None = None) -> tuple[int, int]:
    if disallowed_pairs is None:
        disallowed_pairs = []
    
    while True:
        if (random_pair := (random.randint(min_val, max_val), random.randint(min_val, max_val))) in disallowed_pairs:
            continue
        return random_pair