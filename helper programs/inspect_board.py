from sfm_app.calib.aprilboard_calib import load_aprilboards

at_coarseboard, at_fineboard = load_aprilboards()

for name, BOARD in [("coarse", at_coarseboard), ("fine", at_fineboard)]:
    print(f"\n=== BOARD ({name}) ===")
    print("type:", type(BOARD))
    print("len:", len(BOARD))

    if isinstance(BOARD, list) and len(BOARD) > 0:
        first = BOARD[0]
        print("\n  First element (BOARD[0]) type:", type(first))
        try:
            print("  Keys:", list(first.keys()))
        except Exception as e:
            print("  (cannot list keys:", e, ")")

        # Print a sample of key → value for debugging
        print("\n  Sample key/value pairs (truncated):")
        for k, v in list(first.items())[:10]:
            print(f"   - {k!r}: type={type(v)}, repr={str(v)[:120]}")

        # If there is an obvious tag id field, print all of them
        possible_id_keys = [k for k in first.keys() if 'id' in str(k).lower() or 'tag' in str(k).lower()]
        if possible_id_keys:
            id_key = possible_id_keys[0]
            print(f"\n  Using id_key={id_key!r} to list tag IDs:")
            try:
                ids = [entry[id_key] for entry in BOARD if isinstance(entry, dict) and id_key in entry]
                print(f"   Found {len(ids)} ids. First 10: {ids[:10]}")
            except Exception as e:
                print("   (error extracting ids:", e, ")")
    else:
        print("  BOARD is not a non-empty list; need to inspect differently.")