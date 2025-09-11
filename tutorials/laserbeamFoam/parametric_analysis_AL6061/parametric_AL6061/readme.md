# OpenFOAM Parametric Study Tool

Automates generation and submission of multiple OpenFOAM cases with varying parameters.

## Quick Start

```bash
# 1. Create example config files
python parametric_study.py --create-examples

# 2. Run parametric study
python parametric_study.py --csv my_config.csv --output my_cases

# 3. Submit jobs to SLURM
python parametric_study.py --csv my_config.csv --output my_cases --submit
```

## Configuration Options

### CSV Format (Recommended)

Create a CSV file with parameter columns. Missing columns use default values.

**Available Parameters:**
- **Laser**: `laser_radius`, `laser_e_num_density`, `laser_radius_flavour`, `powder_sim`
- **Phase**: `phase_sigma`, `phase_dsigmadT`, `phase_p_amb`, `phase_Tvap`
- **Gas**: `gas_nu`, `gas_rho`, `gas_beta`, `gas_poly_kappa`, `gas_poly_cp`
- **Metal**: `metal_elec_resistivity`
- **Mesh**: `mesh_width`, `mesh_track_length`, `mesh_patching_height`, `mesh_size`
- **Time**: `time_total_time`, `time_scan_speed`, `time_laser_power`

**Example CSV - Vary only surface tension:**
```csv
phase_sigma
0.18
0.20
0.22
0.24
```

**Example CSV - Multiple parameters:**
```csv
laser_radius,time_laser_power,mesh_size
40e-6,400,8e-6
50e-6,500,10e-6
60e-6,600,12e-6
```

### JSON Format

```json
[
    {
        "laser": {"radius": "40e-6", "e_num_density": "1.5e29"},
        "time": {"total_time": 2000e-6, "scan_speed": 0.8, "laser_power": 400},
        "mesh": {"mesh_size": 8e-6}
    },
    {
        "laser": {"radius": "50e-6", "e_num_density": "1.8e29"},
        "time": {"total_time": 2500e-6, "scan_speed": 0.7, "laser_power": 500},
        "mesh": {"mesh_size": 10e-6}
    }
]
```

## Default Values

Parameters not specified in your config will use these defaults:

| Parameter | Default | Unit |
|-----------|---------|------|
| laser_radius | 50e-6 | m |
| laser_e_num_density | 1.8e29 | /m³ |
| phase_sigma | 0.21 | N/m |
| time_total_time | 2500e-6 | s |
| time_laser_power | 500 | W |
| mesh_size | 10e-6 | m |

## Usage Examples

```bash
# Create example files
python parametric_study.py --create-examples

# Run with CSV
python parametric_study.py --csv sigma_study.csv --output sigma_cases

# Run with JSON
python parametric_study.py --config params.json --output json_cases

# Submit jobs immediately
python parametric_study.py --csv params.csv --submit

# Custom output directory
python parametric_study.py --csv params.csv --output my_study_cases
```

## Output Structure

```
cases/
├── case_000/
│   ├── job.sh          # SLURM job script
│   ├── system/         # Modified OpenFOAM system files
│   ├── constant/       # Modified material properties
│   └── ...
├── case_001/
└── case_002/
```

## Job Submission

Each case gets a SLURM job script (`job.sh`) configured for:
- 64 cores, 30-hour time limit
- OpenFOAM-10 with custom OpenMPI
- Automatic mesh generation and decomposition
- Background monitoring and final reconstruction

Submit manually: `cd cases/case_000 && sbatch job.sh`

## Requirements

- Python 3.6+ with pandas
- OpenFOAM base case at hardcoded path
- SLURM cluster environment
