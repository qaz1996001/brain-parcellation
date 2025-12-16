"""集中管理 sync router 使用的 URL 片段，避免硬編碼。"""

prefix = "/sync"
SYNC_PROT_STUDY = f"{prefix}/study"  # 接收新的 study 任務
SYNC_PROT_OPE_NO = f"{prefix}/ope_no"  # ope_no 回報入口
SYNC_PROT_STUDY_TRANSFER_COMPLETE = f"{prefix}/study/transfer/complete"
SYNC_PROT_STUDY_NIFTI_TOOL = f"{prefix}/nifti_tool"
SYNC_PROT_STUDY_CONVERSION_COMPLETE_UID = f"{prefix}/study/conversion/complete/by-uid"
SYNC_PROT_STUDY_CONVERSION_COMPLETE_RENAME_ID = (
    f"{prefix}/study/conversion/complete/by-rename_id"
)
