from typing import NamedTuple

class ProcessStateIndices(NamedTuple):
    # Tensor indices
    DONOR_REDUCED_TENSOR: int = 0
    DONOR_OXIDIZED_TENSOR: int = 1
    DONOR_DIFFALPHA_TENSOR: int = 2
    ACCEPTOR_OXIDIZED_TENSOR: int = 3
    ACCEPTOR_REDUCED_TENSOR: int = 4
    ACCEPTOR_DIFFALPHA_TENSOR: int = 5
    
    # Diffalpha component indices
    DONOR_DIFFALPHA_START: int = 6
    DONOR_DIFFALPHA_END: int = 12
    ACCEPTOR_DIFFALPHA_START: int = 12
    ACCEPTOR_DIFFALPHA_END: int = 18
    
    # Field and energy indices
    DONOR_FIELD_START: int = 18
    DONOR_FIELD_END: int = 21
    DONOR_POLARIZATION: int = 21
    ACCEPTOR_FIELD_START: int = 22
    ACCEPTOR_FIELD_END: int = 25
    ACCEPTOR_POLARIZATION: int = 25
    TOTAL_POLARIZATION: int = 26
    COULOMBIC_ENERGY: int = 27
    TOTAL_ENERGY: int = 28
    
    # Energy statistics indices
    AVG_COULOMBIC: int = 29
    VAR_COULOMBIC: int = 30
    AVG_TOTAL: int = 31
    VAR_TOTAL: int = 32
    
    # Field statistics indices
    DONOR_FIELD_MAG: int = 33
    ACCEPTOR_FIELD_MAG: int = 34
    DONOR_FIELD_AVG: int = 35
    DONOR_FIELD_STD: int = 36
    ACCEPTOR_FIELD_AVG: int = 37
    ACCEPTOR_FIELD_STD: int = 38

    @classmethod
    def validate_tuple_length(cls, result_tuple: tuple) -> None:
        """Validate that the result tuple has the expected length."""
        expected_length = 39  # Max index + 1
        if len(result_tuple) != expected_length:
            raise ValueError(f"Process state result tuple has length {len(result_tuple)}, "
                           f"expected {expected_length}")

class FinalResultIndices(NamedTuple):
    # Field information
    REACTANT_DONOR_MAG: int = 0
    REACTANT_ACCEPTOR_MAG: int = 1
    PRODUCT_DONOR_MAG: int = 2
    PRODUCT_ACCEPTOR_MAG: int = 3
    REACTANT_FRAMES: int = 4
    PRODUCT_FRAMES: int = 5
    
    # Field statistics
    REACTANT_DONOR_MEAN: int = 6
    REACTANT_DONOR_STD: int = 7
    REACTANT_ACCEPTOR_MEAN: int = 8
    REACTANT_ACCEPTOR_STD: int = 9
    PRODUCT_DONOR_MEAN: int = 10
    PRODUCT_DONOR_STD: int = 11
    
    # Energy ranges
    REACTANT_START: int = 12
    REACTANT_END: int = 19
    PRODUCT_START: int = 20
    PRODUCT_END: int = 27
    
    # Reorganization energies
    REORG_UNPOLARIZED: int = 28
    REORG_POLARIZED: int = 29

    @classmethod
    def validate_tuple_length(cls, result_tuple: tuple) -> None:
        """Validate that the final result tuple has the expected length."""
        expected_length = 30  # Max index + 1
        if len(result_tuple) != expected_length:
            raise ValueError(f"Final result tuple has length {len(result_tuple)}, "
                           f"expected {expected_length}")

# Field component slices
class FieldSlices:
    REACTANT_DONOR = slice(0, 3)
    REACTANT_ACCEPTOR = slice(3, 6)
    PRODUCT_DONOR = slice(0, 3)
    PRODUCT_ACCEPTOR = slice(3, 6)
