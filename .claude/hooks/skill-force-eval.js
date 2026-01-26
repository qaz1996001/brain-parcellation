#!/usr/bin/env node
// .claude/hooks/skill-force-eval.js
// 強制 AI 在執行任務前評估技能

const fs = require('fs');
const path = require('path');

// ===== 配置區 =====
const CONFIG = {
  // 冷卻時間（毫秒），同一對話中避免重複提醒
  cooldownMs: 120000, // 2 分鐘

  // 鎖定檔位置
  lockFile: '/tmp/claude-skill-eval-lock',

  // 技能目錄
  skillsDir: '.claude/skills',

  // 需要強制評估的工具（正則匹配）
  criticalTools: [
    'Bash',
    'Write',
    'Edit',
    'MultiEdit',
    'TaskCreate',
    'EnterPlanMode'
  ],

  // 排除的簡單操作（不需要評估）
  excludePatterns: [
    /^Read$/,
    /^Glob$/,
    /^Grep$/,
    /^LS$/
  ]
};

// ===== 主邏輯 =====
function main() {
  try {
    // 1. 解析輸入
    const input = parseInput();
    const toolName = input.tool_name || input.toolName || 'Unknown';

    // 2. 檢查是否需要評估
    if (!shouldEvaluate(toolName)) {
      process.exit(0);
    }

    // 3. 檢查冷卻期
    if (isInCooldown()) {
      process.exit(0);
    }

    // 4. 更新鎖定時間
    updateLock();

    // 5. 掃描可用技能
    const skills = scanSkills();

    // 6. 輸出評估提示
    const message = buildMessage(toolName, skills);
    process.stderr.write(message);

    // exit 0 = 繼續執行，但 AI 會看到 stderr 的訊息
    // exit 2 = 強制停止，要求 AI 重新思考
    process.exit(0);

  } catch (err) {
    // 出錯時靜默通過，不阻礙正常工作
    process.stderr.write(`[skill-eval warning] ${err.message}\n`);
    process.exit(0);
  }
}

// ===== 輔助函數 =====

function parseInput() {
  // Claude Code 會將 JSON 作為第一個參數傳入
  const inputString = process.argv[2] || '{}';
  try {
    return JSON.parse(inputString);
  } catch {
    return {};
  }
}

function shouldEvaluate(toolName) {
  // 檢查是否在排除列表
  for (const pattern of CONFIG.excludePatterns) {
    if (pattern.test(toolName)) {
      return false;
    }
  }

  // 檢查是否為關鍵工具
  return CONFIG.criticalTools.some(tool =>
    toolName.toLowerCase().includes(tool.toLowerCase())
  );
}

function isInCooldown() {
  if (!fs.existsSync(CONFIG.lockFile)) {
    return false;
  }

  try {
    const lastTime = parseInt(fs.readFileSync(CONFIG.lockFile, 'utf8'));
    return (Date.now() - lastTime) < CONFIG.cooldownMs;
  } catch {
    return false;
  }
}

function updateLock() {
  fs.writeFileSync(CONFIG.lockFile, Date.now().toString());
}

function scanSkills() {
  const skills = [];

  // 嘗試多個可能的路徑
  const possiblePaths = [
    path.join(process.cwd(), CONFIG.skillsDir),
    path.join(process.env.HOME || '', 'project', CONFIG.skillsDir),
    CONFIG.skillsDir
  ];

  for (const skillPath of possiblePaths) {
    if (fs.existsSync(skillPath)) {
      try {
        const files = fs.readdirSync(skillPath);
        for (const file of files) {
          if (file.endsWith('.md')) {
            const name = file.replace('.md', '');
            // 提取 sc: 前綴的技能
            if (name.startsWith('sc-') || name.startsWith('sc_')) {
              skills.push(`sc:${name.substring(3)}`);
            } else {
              skills.push(name);
            }
          }
        }
        break; // 找到就停止
      } catch {
        continue;
      }
    }
  }

  return skills;
}

function buildMessage(toolName, skills) {
  const skillList = skills.length > 0
    ? skills.map(s => `  - ${s}`).join('\n')
    : '  (未找到技能文檔，請檢查 .claude/skills/ 目錄)';

  return `
╔══════════════════════════════════════════════════════════════╗
║  [技能評估檢查點] 工具: ${toolName.padEnd(35)}║
╠══════════════════════════════════════════════════════════════╣
║  在繼續之前，請確認：                                        ║
║  1. 你已評估此任務需要哪些技能                               ║
║  2. 請在回應中明確聲明：「我將使用 [技能] 處理此任務」       ║
╚══════════════════════════════════════════════════════════════╝

可用技能清單：
${skillList}

技能選擇指南：
  - sc:pm / sc:analyze  → 任務初期，釐清範圍
  - sc:design           → 架構設計階段
  - sc:implement        → 代碼實作（需配合 uv + ruff）
  - sc:test             → 測試驗證
  - sc:cleanup          → 收尾與品質把關

`;
}

// 執行
main();