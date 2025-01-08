import pathlib

from code_ai.utils_resample import resampleSynthSEG2original

if __name__ == '__main__':
    raw_file = pathlib.Path()
    #
    raw_file = pathlib.Path('d:/00_Chen/Task04_git/data/20151006_02794664_31097_P002_out/Sag_FSPGR_BRAVO.nii')
    resample_image_file = pathlib.Path('d:/00_Chen/Task04_git/data_0106/20151006_02794664_31097_P002_out/Sag_FSPGR_BRAVO_resample.nii.gz')
    resample_seg_file   = pathlib.Path('d:/00_Chen/Task04_git/data_0106/20151006_02794664_31097_P002_out/Sag_FSPGR_BRAVO_resample_synthseg.nii.gz')
    resampleSynthSEG2original(raw_file,
                              resample_image_file,
                              resample_seg_file)

    resample_seg_file = pathlib.Path(
        'd:/00_Chen/Task04_git/data_0106/20151006_02794664_31097_P002_out/Sag_FSPGR_BRAVO_resample_synthseg.nii.gz')
    resampleSynthSEG2original(raw_file,
                              resample_image_file,
                              resample_seg_file)
    print(10000)

    resample_seg_file = pathlib.Path(
        'd:/00_Chen/Task04_git/data_0106/20151006_02794664_31097_P002_out/Sag_FSPGR_BRAVO_resample_synthseg33.nii.gz')
    resampleSynthSEG2original(raw_file,
                              resample_image_file,
                              resample_seg_file)

    resample_seg_file = pathlib.Path(
        'd:/00_Chen/Task04_git/data_0106/20151006_02794664_31097_P002_out/Sag_FSPGR_BRAVO_resample_david.nii.gz')
    resampleSynthSEG2original(raw_file,
                              resample_image_file,
                              resample_seg_file)
