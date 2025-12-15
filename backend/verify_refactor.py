#!/usr/bin/env python3
"""
簡單的代碼分析腳本：驗證重構完成度
無需導入模組，直接分析源代碼
"""
import re
import sys

def analyze_service_py():
    """分析 service.py 的重構情況"""
    print("\n" + "="*70)
    print("🔍 Good Taste 重構代碼分析")
    print("="*70)
    
    with open('/var/www/brain-parcellation/backend/app/sync/service.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 統計關鍵指標
    stats = {
        'httpx_async_client': content.count('httpx.AsyncClient'),
        'client_post': content.count('client.post'),
        'early_return_comments': content.count('# ✅ Early return'),
        'good_taste_comments': content.count('# ✅ Good Taste'),
        'deprecated_markers': content.count('DEPRECATED'),
        'internal_methods': 0,
    }
    
    # 檢查新方法是否存在
    new_methods = [
        'async def _process_transfer_complete_internal',
        'async def _process_conversion_complete_internal',
        'async def _initiate_conversion_process_internal',
    ]
    
    print("\n📋 重構完成度檢查:")
    print("-" * 70)
    
    for method in new_methods:
        if method in content:
            print(f"   ✅ {method.replace('async def ', '')}() - 已添加")
            stats['internal_methods'] += 1
        else:
            print(f"   ❌ {method.replace('async def ', '')}() - 缺失")
    
    # 檢查關鍵重構點
    print("\n🔧 關鍵重構點:")
    print("-" * 70)
    
    # 1. post_ope_no_task 檢查
    post_ope_no_task_pattern = r'async def post_ope_no_task.*?(?=\n    async def|\nclass|\Z)'
    post_ope_no_match = re.search(post_ope_no_task_pattern, content, re.DOTALL)
    
    if post_ope_no_match:
        post_ope_no_code = post_ope_no_match.group(0)
        if 'httpx.AsyncClient' not in post_ope_no_code:
            print("   ✅ post_ope_no_task - 無 httpx 呼叫")
        else:
            print("   ❌ post_ope_no_task - 仍有 httpx 呼叫")
        
        if '_process_transfer_complete_internal' in post_ope_no_code:
            print("   ✅ post_ope_no_task - 調用內部方法")
        else:
            print("   ⚠️  post_ope_no_task - 未調用內部方法")
    
    # 2. check_study_series_transfer_complete 檢查
    check_transfer_pattern = r'async def check_study_series_transfer_complete.*?(?=\n    async def|\nclass|\Z)'
    check_transfer_match = re.search(check_transfer_pattern, content, re.DOTALL)
    
    if check_transfer_match:
        check_transfer_code = check_transfer_match.group(0)
        if '_send_events' not in check_transfer_code or 'await self._send_events' not in check_transfer_code:
            print("   ✅ check_study_series_transfer_complete - 無 _send_events 呼叫")
        else:
            print("   ❌ check_study_series_transfer_complete - 仍調用 _send_events")
    
    # 3. study_series_nifti_tool 檢查
    nifti_tool_pattern = r'async def study_series_nifti_tool.*?(?=\n    async def|\nclass|\Z)'
    nifti_tool_match = re.search(nifti_tool_pattern, content, re.DOTALL)
    
    if nifti_tool_match:
        nifti_tool_code = nifti_tool_match.group(0)
        # 檢查是否移除了 httpx POST
        httpx_count_in_method = nifti_tool_code.count('httpx.AsyncClient')
        if httpx_count_in_method == 0:
            print("   ✅ study_series_nifti_tool - 無 httpx 呼叫")
        else:
            print(f"   ❌ study_series_nifti_tool - 有 {httpx_count_in_method} 個 httpx 呼叫")
        
        # 檢查是否改為直接調用
        if 'await self.check_study_series_conversion_complete' in nifti_tool_code:
            print("   ✅ study_series_nifti_tool - 改為直接方法調用")
        else:
            print("   ⚠️  study_series_nifti_tool - 未找到直接方法調用")
    
    # 4. check_study_series_conversion_complete 檢查
    check_conversion_pattern = r'async def check_study_series_conversion_complete.*?(?=\n    async def|\nclass|\Z)'
    check_conversion_match = re.search(check_conversion_pattern, content, re.DOTALL)
    
    if check_conversion_match:
        check_conversion_code = check_conversion_match.group(0)
        # 檢查是否移除了多餘的 _send_events
        send_events_count = check_conversion_code.count('await self._send_events')
        if send_events_count == 0:
            print("   ✅ check_study_series_conversion_complete - 無 _send_events 呼叫")
        else:
            print(f"   ⚠️  check_study_series_conversion_complete - 有 {send_events_count} 個 _send_events 呼叫")
    
    # 統計報告
    print("\n📊 統計報告:")
    print("-" * 70)
    print(f"   httpx.AsyncClient 總使用次數: {stats['httpx_async_client']}")
    print(f"   client.post 總使用次數: {stats['client_post']}")
    print(f"   Early Return 註解: {stats['early_return_comments']}")
    print(f"   Good Taste 註解: {stats['good_taste_comments']}")
    print(f"   DEPRECATED 標記: {stats['deprecated_markers']}")
    print(f"   新增內部方法: {stats['internal_methods']}/3")
    
    # 評分
    print("\n🎖️ Linus 品味評級:")
    print("-" * 70)
    
    score = 0
    max_score = 6
    
    # 評分標準
    if stats['httpx_async_client'] <= 1:
        score += 1
        print("   ✅ httpx 使用最小化 (1分)")
    
    if stats['internal_methods'] == 3:
        score += 1
        print("   ✅ 所有內部方法已添加 (1分)")
    
    if stats['early_return_comments'] >= 5:
        score += 1
        print("   ✅ Early Return 充分應用 (1分)")
    
    if stats['good_taste_comments'] >= 3:
        score += 1
        print("   ✅ Good Taste 原則標註清晰 (1分)")
    
    if stats['deprecated_markers'] >= 1:
        score += 1
        print("   ✅ 舊方法已標記 DEPRECATED (1分)")
    
    if stats['client_post'] <= 1:
        score += 1
        print("   ✅ HTTP POST 呼叫最小化 (1分)")
    
    print(f"\n   總分: {score}/{max_score}")
    
    if score == max_score:
        print("\n   🏆 評級: 🟢 Good Taste - 完美重構！")
    elif score >= 4:
        print("\n   🥈 評級: 🟡 接近 Good Taste - 大部分重構完成")
    else:
        print("\n   🔴 評級: 🔴 需要改進")
    
    # 最終建議
    print("\n💡 建議:")
    print("-" * 70)
    
    if stats['httpx_async_client'] > 1:
        print("   ⚠️  httpx 使用仍然過多，建議進一步減少")
    else:
        print("   ✅ httpx 使用已優化")
    
    if stats['internal_methods'] < 3:
        print(f"   ⚠️  仍缺少 {3 - stats['internal_methods']} 個內部方法")
    else:
        print("   ✅ 所有必要的內部方法已完成")
    
    print("\n" + "="*70)
    print("✨ 分析完成")
    print("="*70 + "\n")
    
    return score == max_score

if __name__ == "__main__":
    success = analyze_service_py()
    sys.exit(0 if success else 1)

