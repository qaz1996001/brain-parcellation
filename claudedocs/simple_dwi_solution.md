# 简单 DWI 判断方案

**Linus 哲学**: "Don't trust the database, trust the filesystem"

## 核心思路

在 `validate_series_ready()` 中：
1. 对每个 `series_uid`，读取一张 DICOM 文件
2. 用 `ConvertManager.rename_dicom_path()` 判断它是 DWI0/DWI1000
3. 如果是 DWI，扩展参数；否则保持原样

## 代码实现

```python
# backend/app/inference/service.py

def _detect_dwi_from_filesystem(
        self, study_uid: str, series_uid: str
) -> Optional[str]:
    """从文件系统读取 DICOM 判断 DWI 类型

    Linus: "文件系统才是真相"

    Returns:
        "DWI0", "DWI1000", 或 None (非 DWI)
    """
    import os
    from pydicom import dcmread
    from code_ai.dicom2nii.convert.dicom_rename_mr import ConvertManager

    # 1. 推断 raw_dicom 路径
    raw_dicom_path = self._infer_raw_dicom_path(study_uid, series_uid)
    if not raw_dicom_path or not os.path.exists(raw_dicom_path):
        return None

    # 2. 读取第一张 DICOM
    try:
        dicom_files = [
            f for f in os.listdir(raw_dicom_path)
            if f.endswith('.dcm')
        ]
        if not dicom_files:
            return None

        first_dicom = os.path.join(raw_dicom_path, dicom_files[0])
        dicom_ds = dcmread(first_dicom, stop_before_pixels=True, force=True)

        # 3. 用 ConvertManager 判断
        convert_mgr = ConvertManager()
        rename_result = convert_mgr.rename_dicom_path(dicom_ds)

        # 4. 返回 DWI0/DWI1000 或 None
        if rename_result in ("DWI0", "DWI1000"):
            return rename_result
        else:
            return None

    except Exception as e:
        logger.warning(f"Failed to detect DWI for {series_uid}: {e}")
        return None


async def validate_series_ready(
        self, study_uid: str, series_uids: List[str]
) -> Tuple[...]:
    """验证系列准备状态 - 简化版"""

    accepted_direct: List[str] = []
    accepted_convert: List[str] = []
    rejected: List[Dict[str, str]] = []
    nifti_paths: List[str] = []
    raw_dicom_paths: List[str] = []
    rename_dicom_paths: List[Optional[str]] = []

    # 对每个 series_uid 验证
    for series_uid in series_uids:
        # 检查是否 DWI
        dwi_type = self._detect_from_filesystem(study_uid, series_uid)

        if dwi_type:
            # 是 DWI - 需要处理 DWI0 和 DWI1000
            for target_id in ["DWI0", "DWI1000"]:
                # 检查 NIfTI 是否存在
                nifti_path = self._validate_nifti_exists(
                    study_uid, series_uid, target_id
                )

                if nifti_path:
                    # Direct mode
                    accepted_direct.append(target_id)
                    nifti_paths.append(nifti_path)
                    rename_dicom_paths.append(
                        self._infer_rename_dicom_path(study_uid, series_uid, target_id)
                    )
                else:
                    # Conversion mode
                    raw_path = self._infer_raw_dicom_path(study_uid, series_uid)
                    if raw_path and os.path.exists(raw_path):
                        accepted_convert.append(target_id)
                        raw_dicom_paths.append(raw_path)
                    else:
                        rejected.append({
                            "series_uid": target_id,
                            "reason": "Raw DICOM not found"
                        })
        else:
            # 非 DWI - 正常处理
            target_id = series_uid

            nifti_path = self._validate_nifti_exists(
                study_uid, series_uid, target_id
            )

            if nifti_path:
                # Direct mode
                accepted_direct.append(target_id)
                nifti_paths.append(nifti_path)
                rename_dicom_paths.append(
                    self._infer_rename_dicom_path(study_uid, series_uid, target_id)
                )
            else:
                # Conversion mode
                raw_path = self._infer_raw_dicom_path(study_uid, series_uid)
                if raw_path and os.path.exists(raw_path):
                    accepted_convert.append(target_id)
                    raw_dicom_paths.append(raw_path)
                else:
                    rejected.append({
                        "series_uid": target_id,
                        "reason": "Neither NIfTI nor raw DICOM found"
                    })

    return (accepted_direct, accepted_convert, rejected,
            nifti_paths, raw_dicom_paths, rename_dicom_paths)
```

## 优点

✅ **重用现有代码**: 直接用 `ConvertManager.rename_dicom_path()`
✅ **文件系统真相**: 读实际 DICOM 文件判断
✅ **简单直接**: 不需要新模块、新数据结构
✅ **移除事件依赖**: 不再依赖 CONVERSION_COMPLETE 事件

## 改进前后对比

| 项目 | 改进前 | 改进后 |
|------|--------|--------|
| DWI 判断方法 | 事件 series_desc 字串匹配 | 读 DICOM 用 ConvertManager |
| 代码重用 | ❌ 重复逻辑 | ✅ 重用 Worker 代码 |
| 准确性 | ⚠️ 近似判断 | ✅ 精确判断（b-value tag） |
| 依赖 | CONVERSION_COMPLETE 事件 | 文件系统 |

完成！
