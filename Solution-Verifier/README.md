# 🧠 ANN Solution Verifier

The **ANN Solution Verifier** is a C++ tool for validating artificial neural network-based mining solutions. It simulates and verifies if a provided nonce yields a valid solution using a given mining seed and public identity.

---

## 🔧 Features

- Validates neural network solutions using simulation.
- Performs multiple mutations and rollback logic to verify outputs.
- Cross-platform compatible (Linux, Windows).

---

## 🛠 Build Instructions

To compile using CMake:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j8
```

> ✅ Can be compile as `Qiner.cpp` and rename it

---

## 🚀 Usage

Run the built binary:

```bash
./verifier
```

You’ll be prompted for:

- **Mining ID** – Computor Public Identity
- **Mining Seed** – 32 bytes hex string (64 characters)
- **Nonce** – 32 bytes hex string (64 characters)

### Example session

```
=== ANN Solution Verifier ===

Enter Mining ID (Computor Public Identity): COMPUTOR_PUBLIC_ID
Enter Mining Seed (32 bytes hex, 64 characters): XXXX
Enter Nonce (32 bytes hex, 64 characters): XXXX

Verifying solution for:
  MiningID: COMPUTOR_PUBLIC_ID
  Mining Seed: XXXX
  Nonce: XXXX

Score: 62/64 (threshold: 45)

✓ VALID SOLUTION
```

---

## ⚙️ ANN Parameters

Hardcoded in the source code:

```cpp
static constexpr unsigned long long NUMBER_OF_INPUT_NEURONS = 0000;
static constexpr unsigned long long NUMBER_OF_OUTPUT_NEURONS = 0000;
static constexpr unsigned long long NUMBER_OF_TICKS = 0000;
static constexpr unsigned long long MAX_NEIGHBOR_NEURONS = 0000;
static constexpr unsigned long long NUMBER_OF_MUTATIONS = 0000;
static constexpr unsigned long long POPULATION_THRESHOLD = 0000;
static constexpr unsigned int SOLUTION_THRESHOLD = 0000;
```

These define ANN complexity and solution acceptance criteria.

---

## 📁 Dependencies

Make sure these files are included:

- `K12AndKeyUtil.h`
- `keyUtils.h`

---

## ✅ Output

After verification, the program outputs:

```
Score: 46/64 (threshold: 45)

✓ VALID SOLUTION
```

or

```
Score: 40/64 (threshold: 45)

✗ INVALID SOLUTION
```

---


