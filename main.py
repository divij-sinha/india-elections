import asyncio  # noqa: F401

from electoral_rolls.get_new import StateCode, run_full_state  # noqa: F401
# from electoral_rolls.parse import process_state  # noqa: F401  ## needs google-genai, not installed


def main():
    state_code = StateCode.JHARKHAND
    print("Hello from india-elections!")
    print(f"Running full state for {state_code}")
    asyncio.run(run_full_state(state_code))
    # process_state(state_code)
    print(f"Finished running full state for {state_code}")


if __name__ == "__main__":
    main()
