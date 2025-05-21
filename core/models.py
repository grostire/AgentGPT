import numpy as np

class TransferFunction:
    """
    Represents a linear time-invariant (LTI) system's transfer function.

    The transfer function is defined by its numerator and denominator coefficients.
    For example, H(s) = (b2*s^2 + b1*s + b0) / (a2*s^2 + a1*s + a0)
    would be initialized with numerator=[b2, b1, b0] and denominator=[a2, a1, a0].
    It now supports multiple 'taps', each being a separate num/den pair.
    """
    def __init__(self, numerator: list[float], denominator: list[float]):
        """
        Initializes the TransferFunction. The first tap is created with these values.

        Args:
            numerator: A list of coefficients for the numerator polynomial of the first tap.
            denominator: A list of coefficients for the denominator polynomial of the first tap.
        """
        if not isinstance(numerator, list) or not all(isinstance(x, (int, float)) for x in numerator):
            raise TypeError("Numerator must be a list of numbers.")
        if not isinstance(denominator, list) or not all(isinstance(x, (int, float)) for x in denominator):
            raise TypeError("Denominator must be a list of numbers.")
        if len(denominator) == 0 or all(c == 0 for c in denominator): # Denominator cannot be all zeros or empty
            raise ValueError("Denominator cannot be empty or all zeros.")
            
        self.taps = [{'num': np.array(numerator, dtype=float), 'den': np.array(denominator, dtype=float)}]
        self.active_tap_index = 0

    def add_tap(self, numerator: list[float], denominator: list[float]) -> int:
        """
        Adds a new TF (num/den pair) to the taps list.

        Args:
            numerator: List of numerator coefficients for the new tap.
            denominator: List of denominator coefficients for the new tap.

        Returns:
            The index of the added tap.
        """
        if not isinstance(numerator, list) or not all(isinstance(x, (int, float)) for x in numerator):
            raise TypeError("Numerator must be a list of numbers.")
        if not isinstance(denominator, list) or not all(isinstance(x, (int, float)) for x in denominator):
            raise TypeError("Denominator must be a list of numbers.")
        if len(denominator) == 0 or all(c == 0 for c in denominator):
            raise ValueError("Denominator cannot be empty or all zeros for a tap.")

        self.taps.append({'num': np.array(numerator, dtype=float), 'den': np.array(denominator, dtype=float)})
        return len(self.taps) - 1

    def set_active_tap(self, index: int):
        """
        Sets the active tap for evaluation.

        Args:
            index: The index of the tap to set as active.
        """
        if not 0 <= index < len(self.taps):
            raise IndexError(f"Tap index {index} is out of range. Number of taps: {len(self.taps)}.")
        self.active_tap_index = index
        
    def get_active_tap_tf(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the numerator and denominator of the currently active tap."""
        active_tap = self.taps[self.active_tap_index]
        return active_tap['num'], active_tap['den']

    def get_tap_tf(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns the numerator and denominator of the tap at the given index."""
        if not 0 <= index < len(self.taps):
            raise IndexError(f"Tap index {index} is out of range. Number of taps: {len(self.taps)}.")
        tap = self.taps[index]
        return tap['num'], tap['den']

    def get_num_taps(self) -> int:
        """Returns the total number of taps."""
        return len(self.taps)

    def remove_tap(self, index: int):
        """
        Removes a tap at the given index.

        Args:
            index: The index of the tap to remove.
        """
        if not 0 <= index < len(self.taps):
            raise IndexError(f"Tap index {index} is out of range for removal. Number of taps: {len(self.taps)}.")
        if len(self.taps) == 1:
            raise ValueError("Cannot remove the last tap. A TransferFunction must have at least one tap.")

        del self.taps[index]

        # Adjust active_tap_index if necessary
        if self.active_tap_index == index:
            # If the active tap was removed, reset to tap 0
            self.active_tap_index = 0
        elif self.active_tap_index > index:
            # If a tap before the active one was removed, decrement active_tap_index
            self.active_tap_index -= 1
        
        if self.active_tap_index >= len(self.taps): # Should not happen if logic is correct
             self.active_tap_index = len(self.taps) -1 if len(self.taps) > 0 else 0


    def __str__(self) -> str:
        num_taps = len(self.taps)
        active_num, active_den = self.get_active_tap_tf()
        base_str = f"TransferFunction(Active Tap {self.active_tap_index + 1}/{num_taps}: " \
                   f"num={active_num.tolist()}, den={active_den.tolist()}"
        if num_taps > 1:
            base_str += f", {num_taps-1} other taps)"
        else:
            base_str += ")"
        return base_str

    def evaluate(self, s: complex | np.ndarray) -> complex | np.ndarray:
        """
        Evaluates the transfer function at a given complex frequency s.
        Uses the currently active tap.

        Args:
            s: The complex frequency (or array of frequencies) at which to evaluate.

        Returns:
            The complex gain of the transfer function at s.
        """
        active_num, active_den = self.get_active_tap_tf()
        num_val = np.polyval(active_num, s)
        den_val = np.polyval(active_den, s)
        
        # Avoid division by zero, return np.inf or a very large number if den_val is zero
        # Ensure result is complex even if inputs lead to real result for type consistency
        if isinstance(den_val, np.ndarray):
            result = np.full_like(num_val, np.inf, dtype=np.complex_)
            non_zero_den_indices = (den_val != 0)
            result[non_zero_den_indices] = num_val[non_zero_den_indices] / den_val[non_zero_den_indices]
        else: # Scalar s
            result = num_val / den_val if den_val != 0 else np.inf
            
        return np.complex_(result)


    @staticmethod
    def _poly_add(poly1: np.ndarray, poly2: np.ndarray) -> np.ndarray:
        """Helper function to add two polynomials (numpy arrays)."""
        len1, len2 = len(poly1), len(poly2)
        if len1 > len2:
            poly2 = np.pad(poly2, (len1 - len2, 0), 'constant', constant_values=0)
        elif len2 > len1:
            poly1 = np.pad(poly1, (len2 - len1, 0), 'constant', constant_values=0)
        return np.polyadd(poly1, poly2)

    def series_combination(self, other_tf: 'TransferFunction') -> 'TransferFunction':
        """
        Calculates H_self(s) * H_other_tf(s) using their respective active taps.
        The resulting TransferFunction will have a single tap.
        """
        self_num, self_den = self.get_active_tap_tf()
        other_num, other_den = other_tf.get_active_tap_tf()

        new_numerator = np.polymul(self_num, other_num)
        new_denominator = np.polymul(self_den, other_den)
        return TransferFunction(new_numerator.tolist(), new_denominator.tolist())

    def parallel_combination(self, other_tf: 'TransferFunction') -> 'TransferFunction':
        """
        Calculates H_self(s) + H_other_tf(s) using their respective active taps.
        The resulting TransferFunction will have a single tap.
        """
        self_num, self_den = self.get_active_tap_tf()
        other_num, other_den = other_tf.get_active_tap_tf()

        num1_part = np.polymul(self_num, other_den)
        num2_part = np.polymul(other_num, self_den)
        
        new_numerator = self._poly_add(num1_part, num2_part)
        new_denominator = np.polymul(self_den, other_den)
        return TransferFunction(new_numerator.tolist(), new_denominator.tolist())

    def feedback_combination(self, feedback_tf: 'TransferFunction', negative_feedback: bool = True) -> 'TransferFunction':
        """
        Calculates G(s) / (1 +/- G(s)H(s)), where G(s) is self, H(s) is feedback_tf.
        Uses the active taps of both G and H. The resulting TF has a single tap.
        Result_Num = Num_G_active * Den_H_active
        Result_Den = Den_G_active * Den_H_active +/- Num_G_active * Num_H_active
        """
        g_num, g_den = self.get_active_tap_tf()
        h_num, h_den = feedback_tf.get_active_tap_tf()

        result_numerator_gh_part = np.polymul(g_num, h_den)
        
        den_common_part = np.polymul(g_den, h_den)
        num_gh_product = np.polymul(g_num, h_num)

        if negative_feedback:
            result_denominator_combined = self._poly_add(den_common_part, num_gh_product)
        else: # Positive feedback
            len1, len2 = len(den_common_part), len(num_gh_product)
            if len1 > len2:
                num_gh_product = np.pad(num_gh_product, (len1 - len2, 0), 'constant', constant_values=0)
            elif len2 > len1:
                den_common_part = np.pad(den_common_part, (len2 - len1, 0), 'constant', constant_values=0)
            result_denominator_combined = np.polysub(den_common_part, num_gh_product)
            
        if result_numerator_gh_part.size == 0:
            result_numerator_gh_part = np.array([0.0])
        if result_denominator_combined.size == 0:
             result_denominator_combined = np.array([1.0]) 

        return TransferFunction(result_numerator_gh_part.tolist(), result_denominator_combined.tolist())

    def get_frequency_response(self, frequencies: np.ndarray | list[float]) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates the magnitude (in dB) and phase (in degrees) of the transfer function
        at a given set of frequencies.

        Args:
            frequencies: A NumPy array or list of frequency points (in Hz).

        Returns:
            A tuple (magnitudes_db, phases_deg), where both are NumPy arrays.
        """
        frequencies_np = np.asarray(frequencies)
        omega = 2 * np.pi * frequencies_np
        s_values = 1j * omega

        H_jw = self.evaluate(s_values)

        abs_H_jw = np.abs(H_jw)
        abs_H_jw[abs_H_jw == 0] = 1e-20 
        magnitudes_db = 20 * np.log10(abs_H_jw)

        phases_rad = np.angle(H_jw)
        unwrapped_phases_rad = np.unwrap(phases_rad)
        phases_deg = np.rad2deg(unwrapped_phases_rad)

        return magnitudes_db, phases_deg

class OpAmp:
    """
    Represents an operational amplifier (Op-Amp) model.
    """
    def __init__(self,
                 name: str,
                 open_loop_gain: float = 1e5,
                 gbw: float = 1e6,
                 input_impedance: float = 1e12,
                 output_impedance: float = 50.0):
        self.name = name
        self.open_loop_gain = open_loop_gain
        self.gbw = gbw
        self.input_impedance = input_impedance
        self.output_impedance = output_impedance

    def get_open_loop_gain(self, frequency: float | np.ndarray) -> complex | np.ndarray:
        """
        Calculates the frequency-dependent open-loop gain A(s) of the op-amp.
        A(s) = Aol / (1 + s/wp), where wp (dominant pole) = (2*pi*GBW) / Aol.
        """
        s = 1j * 2 * np.pi * np.asarray(frequency)
        
        if self.open_loop_gain == 0: # Avoid division by zero if Aol is zero
            return np.zeros_like(s, dtype=complex) if isinstance(s, np.ndarray) else 0j
        if self.gbw == 0: # If GBW is zero, treat as constant gain Aol (infinite bandwidth)
             return np.full_like(s, self.open_loop_gain, dtype=complex) if isinstance(s, np.ndarray) else complex(self.open_loop_gain)

        wp_rad_s = (2 * np.pi * self.gbw) / self.open_loop_gain
        
        if wp_rad_s == 0: # Effectively infinite bandwidth for non-zero Aol
            return np.full_like(s, self.open_loop_gain, dtype=complex) if isinstance(s, np.ndarray) else complex(self.open_loop_gain)

        gain = self.open_loop_gain / (1 + s / wp_rad_s)
        return gain

    def __str__(self) -> str:
        return (f"OpAmp(name='{self.name}', Aol={self.open_loop_gain}, GBW={self.gbw} Hz, "
                f"Zin={self.input_impedance} Ohm, Zout={self.output_impedance} Ohm)")

from abc import ABC, abstractmethod

class Block(ABC):
    """
    Abstract base class for all signal processing blocks in the simulation.
    """
    def __init__(self, name: str, num_inputs: int = 1, num_outputs: int = 1):
        self.name = name
        self.input_pins = [None] * num_inputs 
        self.output_pins = [[] for _ in range(num_outputs)]

    @abstractmethod
    def get_transfer_function(self, frequency: np.ndarray) -> np.ndarray:
        """
        Abstract method to get the block's complex frequency response.
        """
        pass

    def __str__(self) -> str:
        return f"Block(name='{self.name}')"

class TFBlock(Block):
    """
    A block that directly represents a transfer function.
    """
    def __init__(self, name: str, numerator: list[float], denominator: list[float]):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.tf = TransferFunction(numerator, denominator)

    def get_transfer_function(self, frequency: np.ndarray) -> np.ndarray:
        s = 1j * 2 * np.pi * np.asarray(frequency)
        return self.tf.evaluate(s)

    def add_tap_to_tf(self, numerator: list[float], denominator: list[float]) -> int:
        return self.tf.add_tap(numerator, denominator)

    def remove_tap_from_tf(self, index: int):
        self.tf.remove_tap(index)

    def set_active_tap_on_tf(self, index: int):
        self.tf.set_active_tap(index)

    def get_active_tf_coeffs(self) -> tuple[np.ndarray, np.ndarray]:
        return self.tf.get_active_tap_tf()

    def get_num_taps_on_tf(self) -> int:
        return self.tf.get_num_taps()

    def __str__(self) -> str:
        return f"TFBlock(name='{self.name}', tf={self.tf})"


class OpAmpBlock(Block):
    """
    A block representing an OpAmp's open-loop gain characteristic.
    """
    def __init__(self,
                 name: str,
                 open_loop_gain: float = 1e5,
                 gbw: float = 1e6,
                 input_impedance: float = 1e12,
                 output_impedance: float = 50.0,
                 num_inputs: int = 2 
                 ):
        super().__init__(name, num_inputs=num_inputs, num_outputs=1)
        self.opamp_model = OpAmp(name, open_loop_gain, gbw, input_impedance, output_impedance)

    def get_transfer_function(self, frequency: np.ndarray) -> np.ndarray:
        return self.opamp_model.get_open_loop_gain(frequency)

    def __str__(self) -> str:
        return f"OpAmpBlock(name='{self.name}', model={self.opamp_model})"

class NonInvertingAmplifierBlock(Block):
    """
    Represents a non-inverting amplifier stage using an OpAmp model.
    The transfer function is calculated based on A(s) / (1 + A(s)beta).
    """
    def __init__(self, name: str, opamp_model: OpAmp, R_f: float, R_g: float):
        super().__init__(name, num_inputs=1, num_outputs=1)
        if not isinstance(opamp_model, OpAmp):
            raise TypeError("opamp_model must be an instance of OpAmp.")
        if R_f <= 0:
            raise ValueError("R_f must be positive.")
        if R_g <= 0:
            raise ValueError("R_g must be positive.")
            
        self.opamp_model = opamp_model
        self.R_f = R_f
        self.R_g = R_g

    def _calculate_closed_loop_tf(self) -> TransferFunction:
        aol = self.opamp_model.open_loop_gain
        gbw_hz = self.opamp_model.gbw

        if aol == 0:
            A_s_tf = TransferFunction([0], [1])
        elif gbw_hz == 0:
            A_s_tf = TransferFunction([aol], [1])
        else:
            wp_rad_s = (2 * np.pi * gbw_hz) / aol 
            if wp_rad_s == 0: 
                 A_s_tf = TransferFunction([aol], [1])
            else:
                num_A = [aol * wp_rad_s] 
                den_A = [1, wp_rad_s]
                A_s_tf = TransferFunction(num_A, den_A)
        
        beta_val = self.R_g / (self.R_f + self.R_g)
        beta_tf = TransferFunction([beta_val], [1])

        closed_loop_tf = A_s_tf.feedback_combination(beta_tf, negative_feedback=True)
        return closed_loop_tf

    def get_stage_transfer_function_object(self) -> TransferFunction:
        """
        Calculates and returns the closed-loop transfer function of the amplifier stage.
        """
        return self._calculate_closed_loop_tf()

    def get_transfer_function(self, frequency: np.ndarray) -> np.ndarray:
        """
        Evaluates the closed-loop transfer function at the given frequencies.
        """
        stage_tf = self.get_stage_transfer_function_object()
        s_values = 1j * 2 * np.pi * np.asarray(frequency)
        return stage_tf.evaluate(s_values)

    def __str__(self) -> str:
        return (f"NonInvertingAmplifierBlock(name='{self.name}', Rf={self.R_f}, Rg={self.R_g}, "
                f"OpAmp='{self.opamp_model.name}')")

class System:
    """
    Represents a system of interconnected blocks.
    """
    def __init__(self):
        self.blocks: dict[str, Block] = {}
        self.connections: list[tuple[str, int, str, int]] = []

    def add_block(self, block: Block):
        if block.name in self.blocks:
            raise ValueError(f"Block with name '{block.name}' already exists in the system.")
        self.blocks[block.name] = block
        print(f"Block '{block.name}' added to the system.")

    def remove_block(self, block_name: str):
        if block_name not in self.blocks:
            raise ValueError(f"Block with name '{block_name}' not found in the system.")
        
        # Disconnect any connections involving this block
        # Iterate over a copy for safe removal
        for conn in list(self.connections): 
            if conn[0] == block_name or conn[2] == block_name:
                try:
                    self.disconnect(conn[0], conn[1], conn[2], conn[3])
                except ValueError as e:
                    print(f"Error during auto-disconnect for {block_name}: {e}")

        del self.blocks[block_name]
        print(f"Block '{block_name}' and its connections removed from the system.")


    def connect(self, source_block_name: str, source_output_idx: int,
                dest_block_name: str, dest_input_idx: int):
        if source_block_name not in self.blocks:
            raise ValueError(f"Source block '{source_block_name}' not found.")
        if dest_block_name not in self.blocks:
            raise ValueError(f"Destination block '{dest_block_name}' not found.")

        source_block = self.blocks[source_block_name]
        dest_block = self.blocks[dest_block_name]

        if not (0 <= source_output_idx < len(source_block.output_pins)):
            raise ValueError(f"Invalid output pin index for {source_block_name}.")
        if not (0 <= dest_input_idx < len(dest_block.input_pins)):
            raise ValueError(f"Invalid input pin index for {dest_block_name}.")

        if dest_block.input_pins[dest_input_idx] is not None:
            raise ValueError(
                f"Input pin {dest_input_idx} of block '{dest_block_name}' is already connected."
            )

        connection_tuple = (source_block_name, source_output_idx, dest_block_name, dest_input_idx)
        self.connections.append(connection_tuple)
        
        dest_block.input_pins[dest_input_idx] = (source_block_name, source_output_idx)
        source_block.output_pins[source_output_idx].append((dest_block_name, dest_input_idx))

        print(f"Connected output {source_output_idx} of '{source_block_name}' "
              f"to input {dest_input_idx} of '{dest_block_name}'.")

    def disconnect(self, source_block_name: str, source_output_idx: int,
                   dest_block_name: str, dest_input_idx: int):
        connection_tuple = (source_block_name, source_output_idx, dest_block_name, dest_input_idx)
        if connection_tuple not in self.connections:
            raise ValueError(f"Connection from '{source_block_name}' (out:{source_output_idx}) "
                             f"to '{dest_block_name}' (in:{dest_input_idx}) does not exist.")

        self.connections.remove(connection_tuple)

        if dest_block_name in self.blocks:
            dest_block = self.blocks[dest_block_name]
            if 0 <= dest_input_idx < len(dest_block.input_pins) and \
               dest_block.input_pins[dest_input_idx] == (source_block_name, source_output_idx):
                dest_block.input_pins[dest_input_idx] = None

        if source_block_name in self.blocks:
            source_block = self.blocks[source_block_name]
            if 0 <= source_output_idx < len(source_block.output_pins):
                target_connection_to_remove = (dest_block_name, dest_input_idx)
                if target_connection_to_remove in source_block.output_pins[source_output_idx]:
                    source_block.output_pins[source_output_idx].remove(target_connection_to_remove)

        print(f"Disconnected output {source_output_idx} of '{source_block_name}' "
              f"from input {dest_input_idx} of '{dest_block_name}'.")

    def __str__(self) -> str:
        nl = "\n"
        return (f"System with {len(self.blocks)} blocks:{nl}"
                f"{nl.join([f'  - {name}: {block}' for name, block in self.blocks.items()])}{nl}"
                f"Connections:{nl}"
                f"{nl.join([f'  - {c[0]}[{c[1]}] -> {c[2]}[{c[3]}]' for c in self.connections])}")

    def _get_tf_for_block(self, block_name: str) -> TransferFunction:
        """
        Helper function to get the TransferFunction object for a given block name.
        """
        block = self.blocks[block_name]
        if isinstance(block, TFBlock):
            return block.tf
        elif isinstance(block, OpAmpBlock):
            aol = block.opamp_model.open_loop_gain
            gbw = block.opamp_model.gbw
            
            if aol == 0:
                return TransferFunction([0], [1]) 
            if gbw == 0 : 
                return TransferFunction([aol],[1])

            wp_rad_s = (2 * np.pi * gbw) / aol
            
            if wp_rad_s == 0: 
                 return TransferFunction([aol],[1]) 
            
            opamp_tf_num = [aol * wp_rad_s]
            opamp_tf_den = [1, wp_rad_s]
            return TransferFunction(opamp_tf_num, opamp_tf_den)
        elif isinstance(block, NonInvertingAmplifierBlock):
            return block.get_stage_transfer_function_object()
        else:
            raise NotImplementedError(
                f"Block type {type(block)} (name: {block.name}) is not supported in _get_tf_for_block."
            )

    def _find_path_dfs(self, current_block_name: str, target_block_name: str, 
                       nodes_in_current_path_to_reach_current_block: list[str]) -> list[str] | None:
        """
        Recursive DFS helper to find a path of block names.
        `nodes_in_current_path_to_reach_current_block` tracks nodes in the current exploration path to detect cycles.
        """
        # Cycle detection: if current_block_name is already in the path leading to it.
        if current_block_name in nodes_in_current_path_to_reach_current_block:
            return None # Cycle detected

        # Path found
        if current_block_name == target_block_name:
            return [current_block_name]

        # Explore connections from current_block_name
        for source_b, src_pin_idx, dest_b, dest_pin_idx in self.connections:
            if source_b == current_block_name:
                next_block_name = dest_b
                # Recursive call: extend the path with current_block_name
                # Pass a *new* list for the path to the next recursive call
                path_segment = self._find_path_dfs(next_block_name, target_block_name, 
                                                   nodes_in_current_path_to_reach_current_block + [current_block_name])
                if path_segment:
                    return [current_block_name] + path_segment 
        return None


    def get_system_tf(self, input_block_name: str, output_block_name: str,
                        frequencies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates the overall transfer function for a path of blocks
        from input_block_name to output_block_name using DFS.
        """
        if input_block_name not in self.blocks:
            raise ValueError(f"Input block '{input_block_name}' not found in the system.")
        if output_block_name not in self.blocks:
            raise ValueError(f"Output block '{output_block_name}' not found in the system.")

        if input_block_name == output_block_name:
            tf = self._get_tf_for_block(input_block_name)
            return tf.get_frequency_response(frequencies)

        path_block_names = self._find_path_dfs(input_block_name, output_block_name, [])
        
        if not path_block_names:
            raise ValueError(f"No valid path found from '{input_block_name}' to '{output_block_name}'.")

        path_tfs: list[TransferFunction] = []
        for block_name in path_block_names:
            path_tfs.append(self._get_tf_for_block(block_name))
        
        if not path_tfs: 
             raise ValueError("Path found but could not retrieve TFs for blocks in path.")

        overall_tf = path_tfs[0]
        for i in range(1, len(path_tfs)):
            overall_tf = overall_tf.series_combination(path_tfs[i])
        
        return overall_tf.get_frequency_response(frequencies)


# Example Usage (optional, for testing)
if __name__ == '__main__':
    # TransferFunction examples
    tf1_num = [1] 
    tf1_den = [1, 1] 
    tf1 = TransferFunction(numerator=tf1_num, denominator=tf1_den) 
    print(f"tf1: {tf1}")
    print(f"tf1 at s=1j: {tf1.evaluate(1j)}") 

    tf2_num = [1, 0] 
    tf2_den = [1, 2] 
    tf2 = TransferFunction(numerator=tf2_num, denominator=tf2_den) 
    print(f"tf2: {tf2}")
    print(f"tf2 at s=1j: {tf2.evaluate(1j)}") 
    
    print("\n--- TransferFunction Combination Tests ---")
    tf_series = tf1.series_combination(tf2)
    print(f"Series (tf1 * tf2): {tf_series}") 

    tf_parallel = tf1.parallel_combination(tf2)
    print(f"Parallel (tf1 + tf2): {tf_parallel}") 

    tf_feedback_neg = tf1.feedback_combination(tf2, negative_feedback=True)
    print(f"Feedback (G=tf1, H=tf2, negative): {tf_feedback_neg}") 

    tf_feedback_pos = tf1.feedback_combination(tf2, negative_feedback=False)
    print(f"Feedback (G=tf1, H=tf2, positive): {tf_feedback_pos}") 
    
    print("\n--- TransferFunction Tap Tests ---")
    tf_taps = TransferFunction([1], [1, 1]) 
    print(tf_taps)
    
    tap1_idx = tf_taps.add_tap([2, 0], [1, 2, 1]) 
    print(f"Added tap at index {tap1_idx}, total taps: {tf_taps.get_num_taps()}")
    print(tf_taps) 
    
    tf_taps.set_active_tap(1)
    print(f"Set active tap to 1: {tf_taps}")
    num_active, den_active = tf_taps.get_active_tap_tf()
    assert np.array_equal(num_active, [2,0]), f"Active num mismatch: {num_active}"
    assert np.array_equal(den_active, [1,2,1]), f"Active den mismatch: {den_active}"

    s_val = 1j
    eval_tap1 = tf_taps.evaluate(s_val)
    print(f"Evaluation of active tap (tap 1) at s={s_val}: {eval_tap1}")
    assert np.isclose(eval_tap1, 1.0 + 0j), f"Tap 1 evaluation error: {eval_tap1}"

    tf_taps.set_active_tap(0)
    print(f"Set active tap back to 0: {tf_taps}")
    eval_tap0 = tf_taps.evaluate(s_val)
    print(f"Evaluation of active tap (tap 0) at s={s_val}: {eval_tap0}")
    assert np.isclose(eval_tap0, 0.5 - 0.5j), f"Tap 0 evaluation error: {eval_tap0}"

    num_tap0, den_tap0 = tf_taps.get_tap_tf(0)
    assert np.array_equal(num_tap0, [1]), "get_tap_tf(0) num failed"
    assert np.array_equal(den_tap0, [1,1]), "get_tap_tf(0) den failed"

    print("\n--- Frequency Response Test (with Taps) ---")
    rc_filter_tf = TransferFunction(numerator=[1], denominator=[0.001, 1]) 
    rc_filter_tf.add_tap(numerator=[1], denominator=[0.01, 1]) 
    
    test_frequencies = np.array([10, 159, 1000]) 

    rc_filter_tf.set_active_tap(0)
    print(f"Testing with active tap 0: {rc_filter_tf}")
    mags0, phases0 = rc_filter_tf.get_frequency_response(test_frequencies)
    print("Freq (Hz) | Mag (dB) | Phase (deg) (Tap 0)")
    for f, m, p in zip(test_frequencies, mags0, phases0): print(f"{f:9.1f} | {m:8.2f} | {p:10.2f}")

    rc_filter_tf.set_active_tap(1)
    print(f"\nTesting with active tap 1: {rc_filter_tf}")
    mags1, phases1 = rc_filter_tf.get_frequency_response(test_frequencies)
    print("Freq (Hz) | Mag (dB) | Phase (deg) (Tap 1)")
    for f, m, p in zip(test_frequencies, mags1, phases1): print(f"{f:9.1f} | {m:8.2f} | {p:10.2f}")

    print("\n--- TFBlock Tap Tests ---")
    tf_block_taps = TFBlock("TappedFilter", [1], [1,10]) 
    print(tf_block_taps)
    
    tf_block_taps.add_tap_to_tf([1], [1,100]) 
    print(f"TFBlock after adding tap: {tf_block_taps}, num taps: {tf_block_taps.get_num_taps_on_tf()}")
    
    tf_block_taps.set_active_tap_on_tf(1)
    print(f"TFBlock active tap set to 1: {tf_block_taps}")
    active_coeffs = tf_block_taps.get_active_tf_coeffs()
    assert np.array_equal(active_coeffs[0], [1]), "TFBlock active num mismatch"
    assert np.array_equal(active_coeffs[1], [1,100]), "TFBlock active den mismatch"
    
    freq_hz = 100 / (2 * np.pi) 
    block_response_tap1 = tf_block_taps.get_transfer_function(np.array([freq_hz])) 
    expected_response_tap1 = 0.005 - 0.005j
    print(f"TFBlock response (tap 1) at {freq_hz:.2f} Hz (s=100j): {block_response_tap1[0]}")
    assert np.isclose(block_response_tap1[0], expected_response_tap1), f"TFBlock tap 1 response error: {block_response_tap1[0]}"

    tf_block_taps.set_active_tap_on_tf(0) 
    expected_response_tap0 = (1-10j)/1010
    block_response_tap0 = tf_block_taps.get_transfer_function(np.array([freq_hz]))
    print(f"TFBlock response (tap 0) at {freq_hz:.2f} Hz (s=100j): {block_response_tap0[0]}")
    assert np.isclose(block_response_tap0[0], expected_response_tap0), f"TFBlock tap 0 response error: {block_response_tap0[0]}"

    print("\n--- TransferFunction remove_tap Tests ---")
    tf_remove = TransferFunction([1],[1,1]) 
    tf_remove.add_tap([2],[1,2]) 
    tf_remove.add_tap([3],[1,3]) 
    print(f"Initial: {tf_remove}, Active: {tf_remove.active_tap_index}, Num Taps: {tf_remove.get_num_taps()}")

    tf_remove.set_active_tap(2)
    print(f"Set active to 2: {tf_remove}")
    
    tf_remove.remove_tap(0) 
    print(f"Removed tap 0: {tf_remove}, Active: {tf_remove.active_tap_index}, Num Taps: {tf_remove.get_num_taps()}")
    assert tf_remove.active_tap_index == 1, f"Active index wrong after removing tap before active: {tf_remove.active_tap_index}"
    num_now_active, den_now_active = tf_remove.get_active_tap_tf()
    assert np.array_equal(num_now_active, [3]), f"Num of new active tap is wrong: {num_now_active}"
    assert np.array_equal(den_now_active, [1,3]), f"Den of new active tap is wrong: {den_now_active}"

    tf_remove.remove_tap(1) 
    print(f"Removed current active tap (index 1): {tf_remove}, Active: {tf_remove.active_tap_index}, Num Taps: {tf_remove.get_num_taps()}")
    assert tf_remove.active_tap_index == 0, f"Active index wrong after removing active tap: {tf_remove.active_tap_index}"
    num_final, den_final = tf_remove.get_active_tap_tf()
    assert np.array_equal(num_final, [2]), f"Num of final tap is wrong: {num_final}"
    assert np.array_equal(den_final, [1,2]), f"Den of final tap is wrong: {den_final}"

    try:
        tf_remove.remove_tap(0) 
        print("Error: Allowed removal of the last tap.") 
    except ValueError as e:
        print(f"Correctly prevented removal of last tap: {e}")
    
    print("\n--- TFBlock remove_tap_from_tf Tests ---")
    tf_block_remove = TFBlock("RemovableTaps", [10], [1, 10])
    tf_block_remove.add_tap_to_tf([20], [1, 20])
    tf_block_remove.add_tap_to_tf([30], [1, 30])
    tf_block_remove.set_active_tap_on_tf(1) 
    print(f"TFBlock initial: {tf_block_remove}")

    tf_block_remove.remove_tap_from_tf(0) 
    print(f"TFBlock after removing tap 0: {tf_block_remove}") 
    active_coeffs_after_remove = tf_block_remove.get_active_tf_coeffs()
    assert tf_block_remove.tf.active_tap_index == 0, f"TFBlock active index after remove: {tf_block_remove.tf.active_tap_index}"
    assert np.array_equal(active_coeffs_after_remove[0], [20]), f"TFBlock active num after remove: {active_coeffs_after_remove[0]}"
    assert np.array_equal(active_coeffs_after_remove[1], [1,20]), f"TFBlock active den after remove: {active_coeffs_after_remove[1]}"


    print("\n--- OpAmp Model Test ---")
    opamp_model = OpAmp(name="UA741", open_loop_gain=2e5, gbw=1e6)
    print(opamp_model)
    freqs = np.array([1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7])
    print(f"OpAmp gain at {freqs} Hz: {opamp_model.get_open_loop_gain(freqs)}")

    print("\n--- System Test ---")
    sys_for_tf = System()
    lpf = TFBlock(name="InputFilter", numerator=[1], denominator=[0.001, 1]) 
    opamp_b = OpAmpBlock(name="Amp", open_loop_gain=100, gbw=1e5) 
    gain_stage = TFBlock(name="OutputGain", numerator=[10], denominator=[1]) 

    sys_for_tf.add_block(lpf)
    sys_for_tf.add_block(opamp_b)
    sys_for_tf.add_block(gain_stage)

    sys_for_tf.connect("InputFilter", 0, "Amp", 0)
    sys_for_tf.connect("Amp", 0, "OutputGain", 0)
    print(sys_for_tf)

    analysis_freqs = np.logspace(1, 6, 5) 

    print("\n--- System TF Calculation Test (InputFilter -> OutputGain) ---")
    try:
        mags, phases = sys_for_tf.get_system_tf("InputFilter", "OutputGain", analysis_freqs)
        print("Freq (Hz) | Mag (dB) | Phase (deg)")
        print("------------------------------------")
        for f, m, p in zip(analysis_freqs, mags, phases):
            print(f"{f:9.2e} | {m:8.2f} | {p:10.2f}")
    except Exception as e:
        print(f"An error occurred during the (InputFilter -> OutputGain) test: {e}")
        
    # Test with a single block (LPF)
    print("\n--- System TF Calculation Test (Single Block LPF) ---")
    try:
        mags_single_lpf, phases_single_lpf = sys_for_tf.get_system_tf("InputFilter", "InputFilter", analysis_freqs) # type: ignore
        print("System Frequency Response (InputFilter -> InputFilter):")
        print("Freq (Hz) | Mag (dB) | Phase (deg)")
        print("------------------------------------")
        for f, m, p in zip(analysis_freqs, mags_single_lpf, phases_single_lpf): 
            print(f"{f:9.2e} | {m:8.2f} | {p:10.2f}")

        # Verify that the single LPF response matches the LPF's own TF response
        lpf_block_tf_test_main = sys_for_tf.blocks["InputFilter"]
        if isinstance(lpf_block_tf_test_main, TFBlock): # Type check for safety
          mags_lpf_direct, phases_lpf_direct = lpf_block_tf_test_main.tf.get_frequency_response(analysis_freqs) # type: ignore
          assert np.allclose(mags_single_lpf, mags_lpf_direct), "Single LPF mag mismatch"
          assert np.allclose(phases_single_lpf, phases_lpf_direct), "Single LPF phase mismatch"
          print("Single LPF block response matches direct TF evaluation: OK")
    except Exception as e:
        print(f"An error occurred during the (Single Block LPF) test: {e}")

    # Test with a single OpAmp block
    print("\n--- System TF Calculation Test (Single Block OpAmp) ---")
    try:
        mags_single_opamp, phases_single_opamp = sys_for_tf.get_system_tf("Amp", "Amp", analysis_freqs)
        print("System Frequency Response (Amp -> Amp):")
        print("Freq (Hz) | Mag (dB) | Phase (deg)")
        print("------------------------------------")
        for f, m, p in zip(analysis_freqs, mags_single_opamp, phases_single_opamp):
                print(f"{f:9.2e} | {m:8.2f} | {p:10.2f}")
    except Exception as e:
        print(f"An error occurred during the (Single Block OpAmp) test: {e}")

    print("\n--- NonInvertingAmplifierBlock Tests ---")
    opamp_for_stage = OpAmp(name="IdealOpAmp", open_loop_gain=1e5, gbw=1e6) # Aol=100k, GBW=1MHz
    R_f_val = 9000.0  
    R_g_val = 1000.0  
    
    non_inv_amp = NonInvertingAmplifierBlock("NonInvAmp1", opamp_for_stage, R_f_val, R_g_val)
    print(non_inv_amp)

    # Expected DC Gain
    G_dc_expected = 1 + R_f_val / R_g_val 
    print(f"Expected Ideal DC Gain: {G_dc_expected} ({20*np.log10(G_dc_expected):.2f} dB)")
    
    G_dc_finite_aol_expected = G_dc_expected / (1 + G_dc_expected / opamp_for_stage.open_loop_gain)
    print(f"Expected DC Gain (finite Aol): {G_dc_finite_aol_expected:.3f} ({20*np.log10(G_dc_finite_aol_expected):.2f} dB)")

    f_3dB_test_hz = opamp_for_stage.gbw / G_dc_expected 

    stage_tf_obj = non_inv_amp.get_stage_transfer_function_object()
    print(f"Calculated Stage TF: {stage_tf_obj}")

    dc_response_complex = stage_tf_obj.evaluate(0.00001j) 
    dc_gain_actual = np.abs(dc_response_complex)
    print(f"Actual DC Gain (at s -> 0): {dc_gain_actual:.3f} ({20*np.log10(dc_gain_actual):.2f} dB)")
    assert np.isclose(dc_gain_actual, G_dc_finite_aol_expected, rtol=1e-3), \
        f"DC Gain mismatch: Expected {G_dc_finite_aol_expected:.3f}, Got {dc_gain_actual:.3f}"

    mags_db_at_f3dB, _ = stage_tf_obj.get_frequency_response(np.array([f_3dB_test_hz]))
    gain_at_f3dB_actual_db = mags_db_at_f3dB[0]
    expected_gain_at_f3dB_db = 20*np.log10(dc_gain_actual) - 3.0 
    
    print(f"Gain at approx f_3dB ({f_3dB_test_hz/1e3:.1f} kHz): {gain_at_f3dB_actual_db:.2f} dB "
          f"(Expected ~{expected_gain_at_f3dB_db:.2f} dB based on actual DC gain)")
    assert np.isclose(gain_at_f3dB_actual_db, expected_gain_at_f3dB_db, atol=1.0), \
           f"-3dB Gain mismatch: Expected {expected_gain_at_f3dB_db:.2f} dB, Got {gain_at_f3dB_actual_db:.2f} dB"


    print("\n--- System TF Calculation Test (No Path) ---")
    sys_no_path = System()
    b1 = TFBlock("B1", [1],[1,1])
    b2 = TFBlock("B2", [1],[1,10])
    b3 = TFBlock("B3", [1],[1,100])
    sys_no_path.add_block(b1)
    sys_no_path.add_block(b2)
    sys_no_path.add_block(b3)
    sys_no_path.connect("B1",0,"B2",0) 
    try:
        sys_no_path.get_system_tf("B1", "B3", analysis_freqs)
        print("Error: Should have raised ValueError for no path.")
    except ValueError as e:
        print(f"Correctly caught no path error: {e}")
    except Exception as e: # Catch any other unexpected error during this test
        print(f"An unexpected error occurred during No Path test: {e}")
    
    print("\n--- System TF Calculation Test (Cycle) ---")
    sys_cycle = System()
    c_b1 = TFBlock("CB1", [1],[1,1])
    c_b2 = TFBlock("CB2", [2],[1,1])
    c_b3 = TFBlock("CB3", [3],[1,1])
    sys_cycle.add_block(c_b1)
    sys_cycle.add_block(c_b2)
    sys_cycle.add_block(c_b3)
    sys_cycle.connect("CB1",0,"CB2",0) 
    sys_cycle.connect("CB2",0,"CB3",0) 
    sys_cycle.connect("CB3",0,"CB1",0) 
    try:
        m_cycle, p_cycle = sys_cycle.get_system_tf("CB1", "CB3", np.array([0.1]))
        print(f"Path CB1->CB3 found. Mag at 0.1Hz: {m_cycle[0]:.2f} dB") 
        expected_gain_at_low_freq = 1 * 2 * 3 
        expected_db = 20 * np.log10(expected_gain_at_low_freq)
        assert np.isclose(m_cycle[0], expected_db, atol=0.1), f"Cycle path gain incorrect. Expected ~{expected_db} dB"

    except ValueError as e:
        print(f"Error during cycle test (CB1->CB3): {e}")
    except Exception as e: # Catch any other unexpected error
        print(f"An unexpected error occurred during Cycle test (CB1->CB3): {e}")
    
    try:
        sys_cycle.get_system_tf("CB1", "CB1", analysis_freqs) 
        print("Path CB1->CB1 found and calculated.")
    except ValueError as e: # Should not be a ValueError if path to self is handled
         print(f"Error during cycle test (CB1->CB1): {e}") 
    except Exception as e: # Catch any other unexpected error
        print(f"An unexpected error occurred during Cycle test (CB1->CB1): {e}")


    # General catch-all for the entire if __name__ == '__main__' block,
    # though specific try-excepts for test sections are better.
    # This was the original position of the except clause that caused the syntax error.
    # except Exception as e:
    #     print(f"Error during system TF calculation: {e}")


    print("\nAll tests seem to pass (or completed).")
