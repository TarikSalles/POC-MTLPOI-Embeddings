import torch

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# DEVICE = torch.device("cpu")

CATEGORIES_MAP: dict[int, str] = {
    0: 'Community',
    1: 'Entertainment',
    2: 'Food',
    3: 'Nightlife',
    4: 'Outdoors',
    5: 'Shopping',
    6: 'Travel',
    7: 'None'
}
