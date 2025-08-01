from typing import List

from pydantic import BaseModel, Field



class OptionParseType(BaseModel):
    output: List[str] = Field(
        ...,
        description="Parsed options from the input string, split by commas or semicolons."
    )



__all__ = ['OptionParseType']