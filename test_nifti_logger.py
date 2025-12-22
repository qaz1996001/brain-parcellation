#!/usr/bin/env python3
"""
驗證 nifti_tool_logger 和 nifti_tool_get_series_info 方法的測試腳本
"""
import sys

# 添加專案路徑
sys.path.insert(0, '/mnt/d/00_Chen/Task04_git')

def test_logger_setup():
    """測試日誌記錄器設置"""
    print("=" * 80)
    print("[TEST] 開始測試日誌記錄器設置")
    print("=" * 80)
    
    try:
        # 導入日誌設置函數
        from backend.app.sync.service import setup_nifti_tool_logger, nifti_tool_logger
        
        print("\n[✓] 成功導入 setup_nifti_tool_logger 和 nifti_tool_logger")
        
        # 驗證 logger 配置
        print(f"\n[LOG_LEVEL] Logger level: {nifti_tool_logger.level}")
        print(f"[HANDLERS] Logger handlers: {nifti_tool_logger.handlers}")
        
        for idx, handler in enumerate(nifti_tool_logger.handlers, 1):
            print(f"\n  [HANDLER {idx}]")
            print(f"    - Type: {type(handler).__name__}")
            print(f"    - Level: {handler.level}")
            if hasattr(handler, 'baseFilename'):
                print(f"    - File: {handler.baseFilename}")
            print(f"    - Formatter: {handler.formatter._fmt if handler.formatter else 'None'}")
        
        # 測試日誌輸出
        print("\n[TEST] 測試日誌輸出...")
        nifti_tool_logger.info("✅ 日誌記錄器工作正常")
        nifti_tool_logger.debug("🔍 調試信息")
        nifti_tool_logger.warning("⚠️  警告信息")
        
        print("\n[✓] 日誌記錄器測試通過")
        
    except ImportError as e:
        print(f"\n[✗] 導入錯誤: {e}")
        return False
    except Exception as e:
        print(f"\n[✗] 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_multi_output_config():
    """測試 multi-output 配置邏輯"""
    print("\n" + "=" * 80)
    print("[TEST] 測試 multi-output 配置邏輯")
    print("=" * 80)
    
    try:
        # 測試配置
        MULTI_OUTPUT_CONFIG = {
            'DWI': ['DWI0', 'DWI1000'],
        }
        
        def is_multi_output_series(series_desc: str):
            """判斷是否為 multi-output series"""
            if not series_desc:
                return False, None
            
            series_desc_upper = series_desc.upper()
            for key, required_outputs in MULTI_OUTPUT_CONFIG.items():
                if key in series_desc_upper:
                    return True, required_outputs
            return False, None
        
        # 測試用例
        test_cases = [
            ("DWI", True, ['DWI0', 'DWI1000']),
            ("DWI LS B0 1200 WITH AP", True, ['DWI0', 'DWI1000']),
            ("T2FLAIR_AXI", False, None),
            ("SWAN MAG", False, None),
            (None, False, None),
            ("", False, None),
        ]
        
        all_passed = True
        for desc, expected_is_multi, expected_outputs in test_cases:
            is_multi, outputs = is_multi_output_series(desc)
            status = "✓" if (is_multi == expected_is_multi and outputs == expected_outputs) else "✗"
            print(f"  [{status}] series_desc: {desc!r}")
            print(f"       -> is_multi: {is_multi}, outputs: {outputs}")
            if is_multi != expected_is_multi or outputs != expected_outputs:
                all_passed = False
        
        if all_passed:
            print("\n[✓] 所有配置邏輯測試通過")
        else:
            print("\n[✗] 某些配置邏輯測試失敗")
            return False
            
    except Exception as e:
        print(f"\n[✗] 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_duplicate_path_removal():
    """測試重複路徑移除邏輯"""
    print("\n" + "=" * 80)
    print("[TEST] 測試重複路徑移除邏輯")
    print("=" * 80)
    
    try:
        # 模擬 result_data
        result_data = [
            {'rename_dicom_path': '/path/to/DWI0'},
            {'rename_dicom_path': '/path/to/DWI0'},  # 重複
            {'rename_dicom_path': '/path/to/DWI1000'},
            {'rename_dicom_path': '/path/to/DWI0'},  # 重複
            {'other_field': 'value'},  # 沒有 rename_dicom_path
            None,  # None
            {'rename_dicom_path': None},  # None 值
            {'rename_dicom_path': '/path/to/DWI1000'},  # 重複
        ]
        
        # 提取唯一路徑
        rename_paths = []
        rename_paths_set = set()
        
        for rd_idx, result_item in enumerate(result_data):
            if result_item and isinstance(result_item, dict):
                if 'rename_dicom_path' in result_item:
                    path = result_item['rename_dicom_path']
                    if path and path not in rename_paths_set:
                        rename_paths.append(path)
                        rename_paths_set.add(path)
                        print(f"  [✓] 提取路徑[{len(rename_paths)}]: {path}")
                    elif path and path in rename_paths_set:
                        print(f"  [⊘] result_data[{rd_idx}] 跳過重複: {path}")
                    else:
                        print(f"  [⊘] result_data[{rd_idx}] 路徑為 None")
        
        expected_paths = ['/path/to/DWI0', '/path/to/DWI1000']
        if rename_paths == expected_paths:
            print("\n[✓] 重複移除邏輯正確")
            print(f"   提取了 {len(rename_paths)} 個唯一路徑: {rename_paths}")
        else:
            print("\n[✗] 重複移除邏輯失敗")
            print(f"   預期: {expected_paths}")
            print(f"   實際: {rename_paths}")
            return False
            
    except Exception as e:
        print(f"\n[✗] 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_syntax_check():
    """檢查 service.py 語法"""
    print("\n" + "=" * 80)
    print("[TEST] 檢查 service.py 語法")
    print("=" * 80)
    
    try:
        import py_compile
        service_path = '/mnt/d/00_Chen/Task04_git/backend/app/sync/service.py'
        py_compile.compile(service_path, doraise=True)
        print(f"[✓] {service_path} 語法檢查通過")
        return True
    except py_compile.PyCompileError as e:
        print(f"[✗] 語法檢查失敗: {e}")
        return False
    except Exception as e:
        print(f"[✗] 測試失敗: {e}")
        return False


def main():
    """主測試函數"""
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  NIFTI_TOOL_GET_SERIES_INFO 診斷測試".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝\n")
    
    tests = [
        ("語法檢查", test_syntax_check),
        ("日誌記錄器設置", test_logger_setup),
        ("Multi-output 配置邏輯", test_multi_output_config),
        ("重複路徑移除邏輯", test_duplicate_path_removal),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"\n[✗] {test_name} 測試拋出未捕獲異常: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # 總結
    print("\n" + "=" * 80)
    print("[SUMMARY] 測試結果總結")
    print("=" * 80)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ 所有測試都通過了！")
    else:
        print("❌ 某些測試失敗了")
    print("=" * 80 + "\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())

