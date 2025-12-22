#!/usr/bin/env python3
"""
測試 Series 狀態檢查邏輯
驗證修復後的集合操作是否正確
"""


def test_series_status_check():
    """測試各種 series 狀態組合"""
    
    # 定義必要的狀態集合（直接使用字串值）
    REQUIRED_STATUSES = {
        '100.025',  # SERIES_NEW
        '100.055',  # SERIES_TRANSFERRING
        '100.095',  # SERIES_TRANSFER_COMPLETE
        '200.155',  # SERIES_CONVERTING
    }
    
    COMPLETE_STATUSES = {
        '200.195',  # SERIES_CONVERSION_COMPLETE
        '200.190',  # SERIES_CONVERSION_SKIP
    }
    
    # 測試案例
    test_cases = [
        {
            'name': '正常完成流程',
            'ope_no': ['100.025', '100.055', '100.095', '200.155', '200.195'],
            'expected': True,
        },
        {
            'name': '跳過轉換（SKIP）',
            'ope_no': ['100.025', '100.055', '100.095', '200.155', '200.190'],
            'expected': True,
        },
        {
            'name': '亂序但完整',
            'ope_no': ['200.195', '200.155', '100.095', '100.055', '100.025'],
            'expected': True,
        },
        {
            'name': '缺少 100.095 (TRANSFER_COMPLETE)',
            'ope_no': ['100.025', '100.055', '200.155', '200.195'],
            'expected': False,
        },
        {
            'name': '缺少 200.155 (CONVERTING)',
            'ope_no': ['100.025', '100.055', '100.095', '200.195'],
            'expected': False,
        },
        {
            'name': '缺少完成狀態',
            'ope_no': ['100.025', '100.055', '100.095', '200.155'],
            'expected': False,
        },
        {
            'name': '實際問題案例（資料庫數據）',
            'ope_no': ['100.025', '100.055', '100.095', '200.155', '200.195'],
            'expected': True,
        },
        {
            'name': '包含額外狀態',
            'ope_no': ['100.020', '100.025', '100.055', '100.095', '200.155', '200.195', '200.200'],
            'expected': True,
        },
    ]
    
    print("=" * 80)
    print("Series 狀態檢查測試")
    print("=" * 80)
    print()
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        ope_no_set = set(test_case['ope_no'])
        
        # 執行檢查
        has_required = REQUIRED_STATUSES.issubset(ope_no_set)
        has_complete = bool(COMPLETE_STATUSES & ope_no_set)
        can_inference = has_required and has_complete
        
        # 驗證結果
        passed = can_inference == test_case['expected']
        status = "✅ PASS" if passed else "❌ FAIL"
        
        if not passed:
            all_passed = False
        
        print(f"測試 {i}: {test_case['name']}")
        print(f"  狀態: {sorted(test_case['ope_no'])}")
        print(f"  has_required: {has_required}")
        print(f"  has_complete: {has_complete}")
        print(f"  can_inference: {can_inference}")
        print(f"  expected: {test_case['expected']}")
        print(f"  結果: {status}")
        print()
    
    print("=" * 80)
    if all_passed:
        print("✅ 所有測試通過！")
    else:
        print("❌ 部分測試失敗！")
    print("=" * 80)
    
    return all_passed


def test_old_regex_vs_new_set():
    """對比舊的正則表達式方法和新的集合方法"""
    import re
    
    print("\n" + "=" * 80)
    print("對比測試：舊正則表達式 vs 新集合操作")
    print("=" * 80)
    print()
    
    # 舊的正則表達式（有 bug）
    old_pattern_str = '(100.025),(100.055),(100.095),(200.155),(200.195|200.190)'
    old_pattern = re.compile(old_pattern_str)
    
    # 新的集合檢查
    REQUIRED_STATUSES = {'100.025', '100.055', '100.095', '200.155'}
    COMPLETE_STATUSES = {'200.195', '200.190'}
    
    test_cases = [
        ['100.025', '100.055', '100.095', '200.155', '200.195'],  # 正常順序
        ['200.195', '200.155', '100.095', '100.055', '100.025'],  # 反序
        ['100.025', '100.055', '100.095', '200.155', '200.190'],  # SKIP
        ['100.055', '100.025', '200.155', '100.095', '200.195'],  # 隨機順序
    ]
    
    for i, ope_no_list in enumerate(test_cases, 1):
        # 舊方法
        test_str = ','.join(ope_no_list)
        old_result = bool(old_pattern.match(test_str))
        
        # 新方法
        ope_no_set = set(ope_no_list)
        has_required = REQUIRED_STATUSES.issubset(ope_no_set)
        has_complete = bool(COMPLETE_STATUSES & ope_no_set)
        new_result = has_required and has_complete
        
        print(f"測試案例 {i}:")
        print(f"  輸入: {ope_no_list}")
        print(f"  轉換字串: {test_str}")
        print(f"  舊方法（正則）: {old_result}")
        print(f"  新方法（集合）: {new_result}")
        
        if old_result != new_result:
            print("  ⚠️  結果不同！新方法修復了順序問題")
        else:
            print("  ✓ 結果一致")
        print()
    
    print("=" * 80)
    print("\n說明：")
    print("- 舊方法使用正則表達式，要求固定順序")
    print("- 新方法使用集合操作，不受順序影響")
    print("- PostgreSQL array_agg(DISTINCT ...) 的順序是不確定的")
    print("- 因此新方法能正確處理所有情況")
    print("=" * 80)


if __name__ == '__main__':
    # 執行測試
    test_passed = test_series_status_check()
    print()
    test_old_regex_vs_new_set()
    
    exit(0 if test_passed else 1)
