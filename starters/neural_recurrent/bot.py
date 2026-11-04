"""Arena entrypoint; reusable inference code lives in starters/shared."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
# Packaged submissions carry shared/ inside the bot directory.
SHARED = HERE / "shared" if (HERE / "shared").is_dir() else HERE.parent / "shared"
# The local move tracker uses the canonical referee rules at repository root.
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(SHARED))
import policy_bot

CHECKPOINT = HERE / "policy.pt"


def load_policy():
    return policy_bot.load_policy(CHECKPOINT)


if __name__ == "__main__":
    policy_bot.main(CHECKPOINT)
