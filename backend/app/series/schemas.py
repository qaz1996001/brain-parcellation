"""
Series 模組的資料序列化方案和序列類型定義。

此模組定義了：
1. API 響應模型（SeriesResponse）
2. 序列類型識別的正則表達式模式
3. 序列類型的排序字典（用於優先級排序）

序列類型分類
-----------
系統支援四類序列類型：

1. **結構影像序列** (series_structure_sort)
   - T1, T2, DWI, ADC 等基礎序列
   - 各種方向和對比增強變體
   - 約 80+ 種變體

2. **特殊序列** (series_special_sort)
   - MRA（磁共振血管造影）
   - SWAN, eSWAN（磁敏感加權成像）
   - 約 10+ 種變體

3. **灌注序列** (series_perfusion_sort)
   - DSC（動態磁敏感對比）
   - ASL（動脈自旋標記）
   - 約 10+ 種變體

4. **功能序列** (series_functional_sort)
   - RESTING（靜息態）
   - CVR（腦血管反應性）
   - DTI（擴散張量成像）
   - 約 10+ 種變體

排序機制
--------
每個序列類型都有一個排序值（數字），用於：
- 確定序列的優先級
- 在列表中排序顯示
- 決定處理順序

See Also
--------
backend.app.series.routers : 使用這些模型的 API 端點
"""

import re

from pydantic import BaseModel


class SeriesResponse(BaseModel):
    """
    DICOM 序列分析響應模型。
    
    此模型表示單個 DICOM 檔案的分析結果，包含檔案名稱、
    識別的序列類型和影像方向。
    
    Attributes
    ----------
    file_name : str
        檔案名稱。
        對於檔案路徑分析，為檔案的基本名稱（不含路徑）。
        對於上傳分析，為原始上傳的檔案名稱。
    series_type : str
        識別的序列類型。
        可能的值包括：
        - 結構序列：T1_AXI, T2_COR, DWI0, ADC 等
        - 特殊序列：MRA_BRAIN, SWAN, eSWAN 等
        - 灌注序列：DSC, ASLSEQ 等
        - 功能序列：RESTING, CVR2000, DTI64D 等
        - 未知序列：'unknown'（無法識別時）
    series_orientation : str
        影像的空間方向。
        可能的值：
        - 'AXI': 軸位（Axial）
        - 'COR': 冠狀位（Coronal）
        - 'SAG': 矢狀位（Sagittal）
        - 其他方向標識符
    
    Examples
    --------
    >>> response = SeriesResponse(
    ...     file_name="scan1.dcm",
    ...     series_type="T1_AXI",
    ...     series_orientation="AXI"
    ... )
    >>> print(response.series_type)
    T1_AXI
    
    >>> response = SeriesResponse(
    ...     file_name="unknown_scan.dcm",
    ...     series_type="unknown",
    ...     series_orientation="AXI"
    ... )
    >>> print(response.series_type)
    unknown
    
    Notes
    -----
    - series_type 為空字串時會自動轉換為 'unknown'
    - 所有欄位都是必填的
    - 使用 Pydantic 進行自動驗證和序列化
    """
    file_name: str
    series_type: str
    series_orientation: str


# ========== 序列類型識別正則表達式 ==========

# 結構影像序列模式：匹配 T1, T2, DWI, ADC 等基礎序列
series_structure_pattern = re.compile(("T1|T2|DWI|ADC"))

# 特殊序列模式：匹配 MRA, SWAN, eSWAN 等特殊序列
series_special_pattern = re.compile(("MRA|SWAN|eSWAN"))

# 灌注序列模式：匹配 DSC, ASL 等灌注序列
series_perfusion_pattern = re.compile(("DSC|ASL"))

# 功能序列模式：匹配 RESTING, CVR, DTI 等功能序列
series_functional_pattern = re.compile(("RESTING|CVR|DTI"))

# ========== 序列類型排序字典 ==========
# 這些字典定義了每個序列類型的排序值（優先級）
# 排序值越小，優先級越高
# 用於在列表中排序和確定處理順序

# 結構影像序列排序字典
# 包含 T1, T2, DWI, ADC 等基礎序列及其各種變體
# 排序值範圍：100-507
# 分類：
#   - ADC/DWI: 100-120
#   - T1 系列: 300-407
#   - T2 系列: 410-507
series_structure_sort = {
    "ADC": 100,
    "DWI0": 110,
    "DWI1000": 120,
    "T1_AXI": 300,
    "T1_COR": 311,
    "T1_SAG": 312,
    "T1CE_AXI": 320,
    "T1CE_COR": 321,
    "T1CE_SAG": 323,
    "T1FLAIR_AXI": 331,
    "T1FLAIR_COR": 332,
    "T1FLAIR_SAG": 333,
    "T1FLAIRCE_AXI": 341,
    "T1FLAIRCE_COR": 342,
    "T1FLAIRCE_SAG": 343,
    "T1CUBE_AXI": 350,
    "T1CUBE_COR": 351,
    "T1CUBE_SAG": 352,
    "T1CUBE_AXIr": 355,
    "T1CUBE_CORr": 356,
    "T1CUBE_SAGr": 357,
    "T1CUBECE_AXI": 361,
    "T1CUBECE_COR": 362,
    "T1CUBECE_SAG": 363,
    "T1CUBECE_AXIr": 365,
    "T1CUBECE_CORr": 366,
    "T1CUBECE_SAGr": 367,
    "T1FLAIRCUBE_AXI": 370,
    "T1FLAIRCUBE_COR": 371,
    "T1FLAIRCUBE_SAG": 372,
    "T1FLAIRCUBE_AXIr": 375,
    "T1FLAIRCUBE_CORr": 376,
    "T1FLAIRCUBE_SAGr": 377,
    "T1FLAIRCUBECE_AXI": 380,
    "T1FLAIRCUBECE_COR": 381,
    "T1FLAIRCUBECE_SAG": 382,
    "T1FLAIRCUBECE_AXIr": 385,
    "T1FLAIRCUBECE_CORr": 386,
    "T1FLAIRCUBECE_SAGr": 387,
    "T1BRAVO_AXI": 390,
    "T1BRAVO_COR": 391,
    "T1BRAVO_SAG": 392,
    "T1BRAVO_AXIr": 395,
    "T1BRAVO_CORr": 396,
    "T1BRAVO_SAGr": 397,
    "T1BRAVOCE_AXI": 400,
    "T1BRAVOCE_COR": 401,
    "T1BRAVOCE_SAG": 402,
    "T1BRAVOCE_AXIr": 405,
    "T1BRAVOCE_CORr": 406,
    "T1BRAVOCE_SAGr": 407,
    "T2_AXI": 410,
    "T2_COR": 411,
    "T2_SAG": 412,
    "T2CE_AXI": 420,
    "T2CE_COR": 421,
    "T2CE_SAG": 423,
    "T2FLAIR_AXI": 431,
    "T2FLAIR_COR": 432,
    "T2FLAIR_SAG": 433,
    "T2FLAIRCE_AXI": 441,
    "T2FLAIRCE_COR": 442,
    "T2FLAIRCE_SAG": 443,
    "T2CUBE_AXI": 450,
    "T2CUBE_COR": 451,
    "T2CUBE_SAG": 452,
    "T2CUBE_AXIr": 455,
    "T2CUBE_CORr": 456,
    "T2CUBE_SAGr": 457,
    "T2CUBECE_AXI": 461,
    "T2CUBECE_COR": 462,
    "T2CUBECE_SAG": 463,
    "T2CUBECE_AXIr": 465,
    "T2CUBECE_CORr": 466,
    "T2CUBECE_SAGr": 467,
    "T2FLAIRCUBE_AXI": 470,
    "T2FLAIRCUBE_COR": 471,
    "T2FLAIRCUBE_SAG": 472,
    "T2FLAIRCUBE_AXIr": 475,
    "T2FLAIRCUBE_CORr": 476,
    "T2FLAIRCUBE_SAGr": 477,
    "T2FLAIRCUBECE_AXI": 480,
    "T2FLAIRCUBECE_COR": 481,
    "T2FLAIRCUBECE_SAG": 482,
    "T2FLAIRCUBECE_AXIr": 485,
    "T2FLAIRCUBECE_CORr": 486,
    "T2FLAIRCUBECE_SAGr": 487,
    "T2BRAVO_AXI": 490,
    "T2BRAVO_COR": 491,
    "T2BRAVO_SAG": 492,
    "T2BRAVO_AXIr": 495,
    "T2BRAVO_CORr": 496,
    "T2BRAVO_SAGr": 497,
    "T2BRAVOCE_AXI": 500,
    "T2BRAVOCE_COR": 501,
    "T2BRAVOCE_SAG": 502,
    "T2BRAVOCE_AXIr": 505,
    "T2BRAVOCE_CORr": 506,
    "T2BRAVOCE_SAGr": 507,
}
# 特殊序列排序字典
# 包含 MRA, SWAN, eSWAN 等特殊序列
# 排序值範圍：100-330
# 分類：
#   - MRA: 100-130
#   - SWAN: 200-210
#   - eSWAN: 300-330
series_special_sort = {
    "MRA_BRAIN": 100,
    "MRA_NECK": 110,
    "MRAVR_BRAIN": 120,
    "MRAVR_NECK": 130,
    "SWAN": 200,
    "SWANmIP": 210,
    "SWANPHASE": 210,
    "eSWAN": 300,
    "eSWANmIP": 310,
    "eSWANPHASE": 330,
}
# 灌注序列排序字典
# 包含 DSC, ASL 等灌注序列
# 排序值範圍：100-230
# 分類：
#   - ASL: 100-150
#   - DSC: 200-230
series_perfusion_sort = {
    "ASLSEQ": 100,
    "ASLSEQATT": 110,
    "ASLSEQATT_COLOR": 111,
    "ASLSEQCBF": 120,
    "ASLSEQCBF_COLOR": 121,
    "ASLPROD": 130,
    "ASLPRODCBF": 140,
    "ASLPRODCBF_COLOR": 141,
    "ASLSEQPW": 150,
    "DSC": 200,
    "DSCCBF_COLOR": 210,
    "DSCCBV_COLOR": 220,
    "DSCMTT_COLOR": 230,
}
# 功能序列排序字典
# 包含 RESTING, CVR, DTI 等功能序列
# 排序值範圍：100-330
# 分類：
#   - RESTING: 100-101
#   - CVR: 200-204
#   - DTI: 300-330
series_functional_sort = {
    "RESTING": 100,
    "RESTING2000": 101,
    "CVR": 200,
    "CVR1000": 201,
    "CVR2000": 202,
    "CVR2000_EAR": 203,
    "CVR2000_EYE": 204,
    "DTI32D": 300,
    "DTI64D": 330,
}
