# import bentoml
# from .cmb_service import CMBService
# from .synthseg_service import SynthSegService

#
# @bentoml.service(
#     name="Service",
#     traffic={
#         "timeout": 300,
#         "concurrency": 1,
#     },
#     resources={
#         "gpu": 1,
#     },
# )
# class Service:
#     cmb_service = bentoml.depends(CMBService)
#     synthseg_service = bentoml.depends(SynthSegService)
#
