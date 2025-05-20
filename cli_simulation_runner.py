import numpy as np
from core.models import System, TFBlock, OpAmpBlock

def run_simulation():
    """
    Sets up a sample system, runs a frequency analysis, and prints results.
    """
    # 1. Setup a Sample System
    system = System()

    # Instantiate blocks
    # Low-pass filter: 1 / (0.001s + 1) -> pole at 1/(0.001) = 1000 rad/s = 159 Hz
    lpf1 = TFBlock(name="LPF1", numerator=[1], denominator=[0.001, 1])

    # Op-Amp: Aol=100 (40dB), GBW=100kHz.
    # Dominant pole wp = GBW / Aol = 100kHz / 100 = 1kHz = 2*pi*1000 rad/s.
    # TF for series: (Aol*wp) / (s + wp)
    op_amp1 = OpAmpBlock(name="OpAmp1", open_loop_gain=100, gbw=1e5)

    # Gain stage: Simple gain of 10 (20dB)
    gain_stage1 = TFBlock(name="Gain1", numerator=[10], denominator=[1])

    # Add blocks to the system
    system.add_block(lpf1)
    system.add_block(op_amp1)
    system.add_block(gain_stage1)

    # Connect them in series: LPF1 -> OpAmp1 -> Gain1
    # Assuming output pin 0 connects to input pin 0 for series.
    # OpAmpBlock by default has 2 inputs; we connect to input 0 (e.g., non-inverting).
    system.connect(source_block_name="LPF1", source_output_idx=0,
                   dest_block_name="OpAmp1", dest_input_idx=0)
    system.connect(source_block_name="OpAmp1", source_output_idx=0,
                   dest_block_name="Gain1", dest_input_idx=0)

    print("\n--- System Configuration ---")
    print(system)

    # 2. Define Analysis Parameters
    input_block = "LPF1"
    output_block = "Gain1"
    # Frequencies from 1 Hz to 1 MHz, 100 points logarithmically spaced
    frequencies = np.logspace(0, 6, 100)

    print(f"\n--- Running Simulation: {input_block} to {output_block} ---")
    print(f"Frequencies from {frequencies[0]:.1f} Hz to {frequencies[-1]:.1e} Hz")

    # 3. Run Simulation
    try:
        magnitudes_db, phases_deg = system.get_system_tf(
            input_block_name=input_block,
            output_block_name=output_block,
            frequencies=frequencies
        )

        # 4. Display Results
        print("\n--- Simulation Results ---")
        print(f"{'Frequency (Hz)':<15} | {'Magnitude (dB)':<15} | {'Phase (deg)':<15}")
        print("-" * 50)
        for i in range(len(frequencies)):
            print(f"{frequencies[i]:<15.2e} | {magnitudes_db[i]:<15.2f} | {phases_deg[i]:<15.2f}")

    except ValueError as ve:
        print(f"\nError during simulation: {ve}")
    except NotImplementedError as nie:
        print(f"\nError during simulation: {nie}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == '__main__':
    run_simulation()
    # To make it executable and potentially add command-line arguments later:
    # import argparse
    # parser = argparse.ArgumentParser(description="Run a simple circuit simulation.")
    # Add arguments here if needed
    # args = parser.parse_args()
    # run_simulation(args)
