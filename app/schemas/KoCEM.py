from typing import Literal

KOCEM_CONFIGS = [
    'Architectural_Planning', 'Building_System', 'Comprehensive_Understanding', 'Construction_Management',
    'Drawing_Interpretation', 'Domain_Reasoning', 'Interior', 'Industry_Jargon', 'Materials',
    'Safety_Management', 'Standard_Nomenclature', 'Structural_Engineering'
]
T_KoCEM_Config = Literal[
    "Architectural_Planning",
    "Building_System",
    "Comprehensive_Understanding",
    "Construction_Management",
    "Drawing_Interpretation",
    "Domain_Reasoning",
    "Interior",
    "Industry_Jargon",
    "Materials",
    "Safety_Management",
    "Standard_Nomenclature",
    "Structural_Engineering"
]
T_Split = Literal["dev", "extra", "test", "val"]

__all__ = ["KOCEM_CONFIGS", "T_KoCEM_Config", "T_Split"]