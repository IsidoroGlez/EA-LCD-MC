# Monte Carlo Simulations of the 3D Edwards–Anderson Spin Glass in Tubular Geometry

This project provides a **GPU-accelerated Monte Carlo simulation** of the 3D Edwards–Anderson spin glass in a tubular geometry, using Parallel Tempering and (optionally) Houdayer cluster updates.

The system size is defined by:

- $L \times L$ in the transverse directions
- $L_Z$ in the longitudinal direction

## Compilation

Check the values of the following macros in the `Makefile`:

- `L`: lattice size in X and Y directions (default: 16)
- `LZ`: lattice size in Z direction (default: 48)
- `NBETAS`: number of temperatures in Parallel Tempering (default: 40)
- `OBC`: Open Boundary Conditions flag (default: 1)
- `WEAK` / `STRONG`: type of open boundary conditions (default: INDEPENDENT)
- `NO_CLUSTER`: turn off Houdayer updates (default: 0 → active)

To compile the program:

```bash
make
```

This generates an executable of the form:

```bash
bin/EA_CUBE_L16_LZ48_NB40_...
```

## Execution

If the program is executed without arguments, or with -h:

```bash
./bin/EA_CUBE_L16_LZ48_...
```

it prints the usage information:

```bash
Usage: progname isample nbits beta.dat input.in LUT device [max_time [list_samples]]
```

### Arguments

| Argument       | Type                            | Description                                          |
| -------------- | ------------------------------- | ---------------------------------------------------- |
| `isample`      | `int`                           | Sample index                                         |
| `nbits`        | `int`                           | Number of subsamples                                 |
| `beta.dat`     | `string`                        | File with inverse temperature list                   |
| `input.in`     | `string`                        | Main simulation input file                           |
| `LUT`          | `string`                        | Lookup table (binary file)                           |
| `device`       | `int`                           | GPU device ID                                        |
| `max_time`     | `unsigned long long` (optional) | Maximum execution time in seconds (0 disables limit) |
| `list_samples` | `string` (optional)             | File with a list of samples                          |

### Example

```bash
./bin/EA_CUBE_L16_LZ48_... \
    0 1 input/beta.dat input/input.in \
    input/LUT.bin 0 0
```

### Notes on execution

- Each run creates a directory structure:

```bash
output/  
 I###  
 BIT###  
 R##
```

- All configurations, energies, and logs are stored inside each replica folder.
- The program can be safely stopped and restarted using backup mode (controlled via `input.in`).

### Input files

The simulation requires three main inputs:

#### 1. `input.in`

Controls simulation parameters (system size, Monte Carlo steps, seeds, etc.).

#### 2. `beta.dat`

List of inverse temperature values used in Parallel Tempering.

#### 3. LUT file

Binary lookup table used to accelerate random number generation.

The LUT must be generated **before running the simulation**.

### Lookup Table (LUT)

The LUT is created separately using:

./compile_and_generate_LUT.sh nbits num_k beta.dat

Important:

- `nbits` must match the value used at compile time (`Makefile`)
- `num_k` must match the number of temperatures in `beta.dat`

Output:

LUT_for_PRNG_nbitsX_NBYY.bin

---

### Example workflow

1. Compile:

```bash 
make
```

2. Generate temperature list beta.dat

3. Build LUT:

```bash
./compile_and_generate_LUT.sh 10 40 beta.dat
```

4. Run simulation:

```bash
./bin/EA_CUBE_L16_LZ48_... 0 1 beta.dat input.in LUT.bin 0
```

## License and citation

This project is licensed under the **MIT License**. See the `LICENSE` file for the full license text.

If you use this code, data, or derived results in a publication, please cite the associated article and/or this repository.  
Citation metadata is provided in the `CITATION.cff` file.

---

## Contact

Developed by M. Bernaschi, L.A. Fernandez, I. González-Adalid Pemartín, V. Martín-Mayor, G. Parisi, and F. Ricci-Tersenghi.

For questions, reach out at [isiglezadalid@gmail.com](mailto:isiglezadalid@gmail.com) or via GitHub issues.
