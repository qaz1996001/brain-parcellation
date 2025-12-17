"""
Series 模組的依賴注入配置。

此模組管理所有與 DICOM 序列分析相關的依賴注入。
使用 FastAPI 的依賴注入系統和 LRU 快取機制。

核心概念
--------
依賴注入提供者（DIP）模式：
- 所有依賴統一通過依賴注入函數提供
- 使用 LRU 快取避免重複創建實例
- 提高效能和資源利用率

支援的依賴
-----------
1. **ConvertManager**
   - DICOM 序列類型識別管理器
   - 用於分析 DICOM 檔案並識別序列類型

2. **ImageOrientationProcessingStrategy**
   - 影像方向處理策略
   - 用於分析影像的空間方向（軸位、矢狀位、冠狀位）

快取機制
--------
所有依賴提供者都使用 `@lru_cache` 裝飾器：
- 避免重複創建相同的實例
- 提高請求處理效能
- 減少記憶體使用

Examples
--------
在路由中使用依賴注入：

    >>> @router.post("/analyze")
    ... async def analyze_dicom(
    ...     convert_manager: ConvertManager = Depends(get_rename_dicom_manager),
    ...     dicom_orientation: ImageOrientationProcessingStrategy = Depends(
    ...         get_dicom_orientation
    ...     ),
    ... ):
    ...     # 使用注入的依賴
    ...     series_type = convert_manager.rename_dicom_path(dcm_ds)
    ...     orientation = dicom_orientation.process(dcm_ds)

See Also
--------
backend.app.series.routers : 使用這些依賴的路由端點
code_ai.dicom2nii.convert : DICOM 轉換管理器實現
"""

from functools import lru_cache
from code_ai.dicom2nii.convert import ConvertManager
from code_ai.dicom2nii.convert.base import ImageOrientationProcessingStrategy


@lru_cache
def get_rename_dicom_manager() -> ConvertManager:
    """
    取得 DICOM 序列類型識別管理器。
    
    此函數提供 ConvertManager 實例，用於識別 DICOM 序列類型。
    使用 LRU 快取確保整個應用程式生命週期中只創建一個實例。
    
    Returns
    -------
    ConvertManager
        DICOM 轉換管理器實例。
        用於分析 DICOM 檔案並識別序列類型（如 T1_AXI, DWI0 等）。
    
    Notes
    -----
    - 使用 `@lru_cache` 裝飾器進行快取
    - 實例在首次調用時創建，後續調用返回快取的實例
    - input_path 和 output_path 設為空字串（僅用於識別，不進行轉換）
    
    See Also
    --------
    code_ai.dicom2nii.convert.ConvertManager : 轉換管理器類別
    """
    return ConvertManager(input_path="", output_path="")


@lru_cache
def get_dicom_orientation() -> ImageOrientationProcessingStrategy:
    """
    取得 DICOM 影像方向處理策略。
    
    此函數提供 ImageOrientationProcessingStrategy 實例，用於分析
    DICOM 影像的空間方向（軸位、矢狀位、冠狀位等）。
    使用 LRU 快取確保整個應用程式生命週期中只創建一個實例。
    
    Returns
    -------
    ImageOrientationProcessingStrategy
        影像方向處理策略實例。
        用於分析 DICOM 影像的空間方向資訊。
    
    Notes
    -----
    - 使用 `@lru_cache` 裝飾器進行快取
    - 實例在首次調用時創建，後續調用返回快取的實例
    - 處理策略是無狀態的，可以安全地共享
    
    See Also
    --------
    code_ai.dicom2nii.convert.base.ImageOrientationProcessingStrategy : 方向處理策略類別
    """
    return ImageOrientationProcessingStrategy()
