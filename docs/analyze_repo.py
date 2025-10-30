#!/usr/bin/env python3
"""
Repository Structure Analyzer
分析 Task04_git 的結構，幫助識別可提取的共用代碼
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
import re
from typing import Dict, List, Set
import json


class CodeAnalyzer:
    """代碼結構分析工具"""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.python_files: List[Path] = []
        self.imports: Dict[str, Set[str]] = defaultdict(set)
        self.classes: Dict[str, List[str]] = defaultdict(list)
        self.functions: Dict[str, List[str]] = defaultdict(list)
        
    def find_python_files(self):
        """找出所有 Python 文件"""
        self.python_files = list(self.repo_path.rglob("*.py"))
        print(f"✓ 找到 {len(self.python_files)} 個 Python 文件")
        return self.python_files
    
    def analyze_imports(self):
        """分析所有導入"""
        for py_file in self.python_files:
            if "__pycache__" in str(py_file):
                continue
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                from_imports = re.findall(r'^from\s+([\w.]+)\s+import', content, re.MULTILINE)
                direct_imports = re.findall(r'^import\s+([\w.]+)', content, re.MULTILINE)
                relative_path = py_file.relative_to(self.repo_path)
                self.imports[str(relative_path)].update(from_imports + direct_imports)
            except Exception as e:
                print(f"⚠ 無法讀取 {py_file}: {e}")
        return self.imports
    
    def analyze_classes(self):
        """分析所有類定義"""
        for py_file in self.python_files:
            if "__pycache__" in str(py_file):
                continue
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                classes = re.findall(r'^class\s+(\w+)(?:\(([^)]*)\))?:', content, re.MULTILINE)
                relative_path = py_file.relative_to(self.repo_path)
                for class_name, parent_classes in classes:
                    self.classes[str(relative_path)].append({
                        'name': class_name,
                        'parents': parent_classes.split(',') if parent_classes else []
                    })
            except Exception as e:
                print(f"⚠ 無法分析 {py_file}: {e}")
        return self.classes
    
    def find_duplicate_functions(self) -> Dict[str, List[str]]:
        """找出重複的函數名"""
        all_functions = defaultdict(list)
        for file_path, funcs in self.functions.items():
            for func in funcs:
                all_functions[func].append(file_path)
        duplicates = {k: v for k, v in all_functions.items() if len(v) > 1}
        return duplicates
    
    def print_summary(self):
        """打印分析摘要"""
        print("\n" + "="*60)
        print("代碼結構分析摘要")
        print("="*60)
        print(f"\n📊 基本統計:")
        print(f"  - Python 文件數: {len(self.python_files)}")
        print(f"  - 導入分析: {len(self.imports)} 個文件")
        print(f"  - 類定義: {sum(len(v) for v in self.classes.values())} 個")


def main():
    if len(sys.argv) < 2:
        print("用法: python analyze_repo.py <path_to_Task04_git>")
        sys.exit(1)
    
    repo_path = sys.argv[1]
    if not Path(repo_path).exists():
        print(f"錯誤: 路徑不存在: {repo_path}")
        sys.exit(1)
    
    print(f"分析倉庫: {repo_path}\n")
    analyzer = CodeAnalyzer(repo_path)
    print("📂 尋找 Python 文件...")
    analyzer.find_python_files()
    print("\n📋 分析導入...")
    analyzer.analyze_imports()
    print("\n🏗️ 分析類定義...")
    analyzer.analyze_classes()
    analyzer.print_summary()
    print("\n✓ 分析完成")


if __name__ == "__main__":
    main()
