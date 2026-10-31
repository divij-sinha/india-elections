import asyncio  # noqa: F401
import sys

from electoral_rolls.get_new import StateCode, run_full_state  # noqa: F401
# from electoral_rolls.parse import process_state  # noqa: F401  ## needs google-genai, not installed

DEFAULT_STATE = StateCode.JHARKHAND  ## used when no state is given: `uv run main.py`
EXIT_NOT_PUBLISHED = 3  ## scripts/run_states.sh treats this as "skip", not as a crash to retry


def main():
    ## `uv run main.py PUNJAB` (StateCode name, as scripts/run_states.sh passes it)
    state_code = StateCode[sys.argv[1].upper()] if len(sys.argv) > 1 else DEFAULT_STATE
    print("Hello from india-elections!")
    print(f"Running full state for {state_code}")
    result = asyncio.run(run_full_state(state_code))
    # process_state(state_code)
    if result == -1:
        print(f"Roll not published for {state_code}")
        sys.exit(EXIT_NOT_PUBLISHED)
    print(f"Finished running full state for {state_code}")


if __name__ == "__main__":
    main()
