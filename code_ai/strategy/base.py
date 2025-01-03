from typing import List, Optional

from .config import RequestIn
from typing import Optional,Union
from io import BytesIO


class ProcessingStrategy:
    def process(self,request: RequestIn,model):
        raise NotImplementedError("Subclasses should implement this!")
