from typing import Literal, List

from pydantic import BaseModel, Field


KOCEM_CONFIGS = [
    'Architectural_Planning', 'Building_System', 'Comprehensive_Understanding', 'Construction_Management',
    'Drawing_Interpretation', 'Domain_Reasoning', 'Interior', 'Industry_Jargon', 'Materials',
    'Safety_Management', 'Standard_Nomenclature', 'Structural_Engineering'
]
KoCEMConfigType = Literal[
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
KoCEMDataSplitType = Literal["dev", "extra", "test", "val"]
LabelType = Literal["A", "B", "C", "D", "E", "F", "G"] | None



class Option(BaseModel):
    """
    Represents a single option in the translation task.
    """
    label: LabelType = Field(
        ...,
        description="Alpha-numeric identifier for the option.",
        example=[
            "A",
            "B",
            "C"
        ]
    )
    value: str = Field(
        ...,
        description="Description of the option.",
        example=[
            "Description of option A",
            "Description of option B",
            "Description of option C"
        ]
    )



class Quiz(BaseModel):
    question: str = Field(
        ...,
        description="The question to be translated.",
        example=[
            "What is the purpose of this document?",
            "How do you interpret this drawing?"
        ]
    )
    options: List[Option] = Field(
        ...,
        description="Options for the question.",
        example=[
            [
                {"label": "A", "value": "This document is for planning purposes."},
                {"label": "B", "value": "This document is for construction purposes."}
            ],
            [
                {"label": "A", "value": "The drawing provides detailed architectural plans."},
                {"label": "B", "value": "The drawing provides detailed construction instructions."}
            ]
        ]
    )
    answer: Option = Field(
        ...,
        description="The correct answer to the question.",
        example= [
            {"label": "A", "value": "This document is for planning purposes."},
            {"label": "B", "value": "The drawing provides detailed architectural plans."}
        ]
    )
    explanation: str = Field(
        "",
        description="Explanation of the answer.",
        example=[
            "The document outlines the architectural plans and specifications.",
            "The drawing provides detailed construction instructions."
        ]
    )



__all__ = ["KOCEM_CONFIGS", "KoCEMConfigType", "KoCEMDataSplitType", "Quiz", "Option"]