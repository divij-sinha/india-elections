from electoral_rolls.get import StateCode
from electoral_rolls.parse import process_state


def main():
    state_code = StateCode.MIZORAM
    print("Hello from india-elections!")
    print(f"Running full state for {state_code}")
    # asyncio.run(run_full_state(state_code))
    process_state(state_code)
    print(f"Finished running full state for {state_code}")


if __name__ == "__main__":
    main()
