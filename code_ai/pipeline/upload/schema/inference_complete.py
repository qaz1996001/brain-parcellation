from typing import Optional, Union, Literal, Annotated
from pydantic import BaseModel, Field, ConfigDict

class InferenceFailedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")  # 加這行

    studyInstanceUid : str = Field(...)
    modelName        : Literal["cmb_model", "aneurysm_model"] = Field(...)
    result           : Literal["failed"]  = Field(default="failed")


class InferenceSuccessRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")  # 加這行

    studyInstanceUid : str                                    = Field(...)
    modelName        : Literal["cmb_model", "aneurysm_model"] = Field(...)
    result           : Literal["success"]                     = Field(default="success")
    inferenceId      : str                                    = Field(...)


InferenceCompleteRequest = Annotated[
    Union[InferenceSuccessRequest, InferenceFailedRequest],
    Field(discriminator="result")
]
