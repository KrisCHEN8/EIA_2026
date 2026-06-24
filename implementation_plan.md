# Implementation Plan: Hysteretic TES Control Logic

Modify the simulation's Rule-Based Control (RBC) for the Thermal Energy Storage (TES) scenarios. Instead of using the weather compensation curve to determine the primary flow rate, we will implement a hysteresis control loop based directly on the mean temperature of the TES.

## User Review Required

> [!NOTE]
> The initial state of the hysteresis loop is set to `charging_state = True` (meaning maximum primary flow `M_DOT_PRI_MAX` is used to charge the tank at the start of the simulation). This is appropriate since the initial tank temperature is 50.0 °C, which is below the maximum charging limit `T_MAX_TES = 65.0` °C.

## Proposed Changes

### Simulation Controller

#### [MODIFY] [run_simulation.py](file:///home/yangzhe/myprojects/EIA_2026/simulation/run_simulation.py)

- Initialize `charging_state = True` at the start of `run_scenario` for each scenario.
- In the hourly loop of `run_scenario`, for TES scenarios:
  - Check the mean water temperature: `T_water_mean = tank.mean_temperature`.
  - Update `charging_state`:
    - Set to `False` if `T_water_mean >= T_MAX_TES` (reaches maximum temperature).
    - Set to `True` if `T_water_mean <= T_MIN_TES` (reaches minimum temperature).
  - Set the primary flow rate `m_dot_primary`:
    - `M_DOT_PRI_MAX` (0.1 kg/s) if `charging_state` is `True`.
    - `M_DOT_PRI_MIN` (0.01 kg/s) if `charging_state` is `False`.
  - Set the heat exchanger target temperature `T_target_HEX`:
    - `75.0` °C (maximum safe heating limit) if `charging_state` is `True`.
    - `T_sec_supply_target` (weather compensation target) if `charging_state` is `False`.

## Verification Plan

### Automated Tests
We will execute the simulation and compare the results to verify that:
1. The baseline scenario still operates exactly as before (following the weather compensation curve).
2. The Water Tank and PCM scenarios correctly oscillate between `T_MIN_TES` and `T_MAX_TES` with the primary flow rate matching `M_DOT_PRI_MAX` and `M_DOT_PRI_MIN` according to the hysteresis state.
3. The comparison summary and figures are successfully generated and saved.

Run the simulation script:
```bash
python simulation/run_simulation.py
```

Run the scenario comparison script to generate updated figures:
```bash
python simulation/compare_scenarios.py
```
