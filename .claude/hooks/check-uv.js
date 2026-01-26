const inputString = process.argv[2];
if (!inputString) process.exit(0);

const input = JSON.parse(inputString);
const command = input.command || "";

// 邏輯：如果是 python/pip 但沒有使用 uv，則攔截
const isPython = /python\s+|pip\s+/.test(command);
const hasUv = command.toLowerCase().includes("uv");

if (isPython && !hasUv) {
  process.stderr.write("\n🛑 [規範攔截] 請使用 'uv run' 或 'uv pip'，嚴禁直接使用原生 Python。\n");
  process.exit(2); // 退出碼 2 會阻斷執行並通知 AI
}

process.exit(0);