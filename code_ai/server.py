# server.py
import asyncio
import enum
import os.path
import tempfile
import time
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

    # def decode_request(self, request):
    #     # Convert the request payload to model input.
    #     intput_data :RequestIn = RequestIn(**request)
    #
    #     if intput_data.template_file is not None:
    #         print('intput_data.template_file is not None')
    #     else:
    #         print('intput_data.template_file is None')
    #         print(intput_data.model_dump())
    #         input_file_byte = intput_data.input_file.read()
    #         with tempfile.TemporaryDirectory() as temp_output_dir:
    #             with tempfile.NamedTemporaryFile() as input_file:
    #                 input_file.write(input_file_byte)
    #     #         self.no_plate_strategy.process(intput_data, self.synthseg, temp_output_dir)
    #
    #
    #     return request

    def decode_request(self, request):
        # Convert the request payload to model input.

        input_data: RequestIn = RequestIn(**request)
        print('*****************************')
        input_file_byte = asyncio.run(input_data.input_file.read())
        # input_file_byte = await input_data.input_file.read()
        if input_data.template_file is not None:
            print('input_data.template_file is not None')
        else:
            print('input_data.template_file is None')
            print(input_data.model_dump())

            temp_output_dir =  tempfile.TemporaryDirectory()
            input_file = tempfile.NamedTemporaryFile(dir=temp_output_dir.name,delete=False)
            input_file.write(input_file_byte)
            input_file.flush()
            input_file.seek(0)
            # Correct way to get the absolute path of the file
            file_path = os.path.abspath(input_file.name)
            print('temp_output_dir:', temp_output_dir)
            print('Absolute path:', file_path)

        return {'input_data': input_data, 'temp_output_dir': temp_output_dir}

    def predict(self, x):
        input_data = x['input_data']
        temp_output_dir =  x['temp_output_dir']
        print('predict, input_data:', input_data)
        self.no_plate_strategy.process(input_data, self.synthseg, temp_output_dir)
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
    server = ls.LitServer(SynthSegLitAPI(), accelerator="auto", max_batch_size=1,track_requests=True)
    server.run(port=8008)