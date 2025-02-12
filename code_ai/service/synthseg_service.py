import gc
import bentoml



@bentoml.service(
    name="SynthSeg-Service",
    traffic={
        "timeout": 300,
        "concurrency": 1,
    },
    resources={
        "gpu": 1,
    },
)
class SynthSegService:
    def __new__(cls):
        import tensorflow as tf
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            try:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
            except:
                pass
        return super().__new__(cls)



    def __init__(self) -> None:
        from code_ai.utils_synthseg import SynthSeg
        self.synth_seg_model = SynthSeg()

    @bentoml.api
    async def synthseg_classify(self,
                                path_images:str,
                                path_segmentations:str,
                                path_segmentations33:str):
        print('synthseg_classify path_images',path_images)
        print('synthseg_classify path_segmentations', path_segmentations)
        print('synthseg_classify path_segmentations33', path_segmentations)
        self.synth_seg_model.run(path_images=path_images, path_segmentations=path_segmentations,
                                 path_segmentations33=path_segmentations33)

        gc.collect()