# server.py
import enum
from typing import Any, Dict, Union, Optional, List
import litserve as ls

from utils_synthseg import SynthSeg
from strategy import RequestIn, InferenceEnum, TemplateProcessingStrategy, NoTemplateProcessingStrategy


class SynthSegLitAPI(ls.LitAPI):
    def setup(self, device):
        self.synthseg = SynthSeg()
        self.template_strategy = TemplateProcessingStrategy()
        self.no_plate_strategy = NoTemplateProcessingStrategy()

    @staticmethod
    def get_(request) -> Dict[str, Any]:

        return request

    def decode_request(self, request):
        # Convert the request payload to model input.
        intput_data = RequestIn(**request)

        if intput_data.template_file is not None:
            print('intput_data.template_file is not None')
        else:
            print('intput_data.template_file is None')

        return request

    def predict(self, x):

        return {"output": 'output',
                # "model_net_unet2" : str(self.synthseg.net_unet2),
                # "model_net_parcellation": str(self.synthseg.net_parcellation),
                }

    def encode_response(self, output):
        # Convert the model output to a response payload.
        return {"output": output}


# (STEP 2) - START THE SERVER
if __name__ == "__main__":
    # scale with advanced features (batching, GPUs, etc...)
    server = ls.LitServer(SynthSegLitAPI(), accelerator="auto", max_batch_size=1)
    server.run(port=8008)