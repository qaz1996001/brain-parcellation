const { execSync } = require('child_process');
const inputString = process.argv[2];
if (!inputString) process.exit(0);

const input = JSON.parse(inputString);
const filePath = input.path || "";

if (!filePath) process.exit(0);

try {
  if (filePath.endsWith('.py')) {
    process.stdout.write(`\n✨ 自動執行 Ruff 校驗: ${filePath}\n`);
    execSync(`uv run ruff check --fix ${filePath}`, { stdio: 'inherit' });
    execSync(`uv run ruff format ${filePath}`, { stdio: 'inherit' });
    process.stdout.write(`\n✨ 自動執行 ty 校驗: ${filePath}\n`);
    execSync(`uv run ty check ${filePath}`, { stdio: 'inherit' });
  } else if (filePath.endsWith('.js') || filePath.endsWith('.ts')) {
    process.stdout.write(`\n✨ 自動執行 ESLint 校驗: ${filePath}\n`);
    execSync(`npx eslint --fix ${filePath}`, { stdio: 'inherit' });
  }
} catch (e) {
  // Lint 報錯不一定要中斷流程，可以只打印警告
  process.stderr.write(`\n⚠️  Lint 檢查發現無法自動修復的問題，請手動檢查。\n`);
}

process.exit(0);