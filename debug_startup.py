import traceback
import sys

try:
    import comet.main
    # If main is just a module with functions, we might need to simulate execution if the import itself succeeds.
    # But usually import errors happen at import time.
    print("Import successful")
except Exception:
    with open("traceback.txt", "w") as f:
        f.write(traceback.format_exc())
    print("Error captured to traceback.txt")
