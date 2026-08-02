import random

CHALLENGE_POOL = [
    "Smile",
    "Blink",
    "Turn Left",
    "Turn Right",
    "Look Up",
    "Look Down",
    "Raise Eyebrows",
    "Open Mouth"
]

def generate_challenge_sequence(length=3):
    """
    Randomly selects a sequence of distinct challenges from the pool.
    
    Args:
        length (int): Number of actions in the challenge (usually 3 or 4).
        
    Returns:
        list: A list of selected challenge action names.
    """
    # Clamp length to pool size
    length = max(1, min(length, len(CHALLENGE_POOL)))
    return random.sample(CHALLENGE_POOL, length)
