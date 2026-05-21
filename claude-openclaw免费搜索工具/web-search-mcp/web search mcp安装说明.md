# Web Search MCP 安装说明

一个免费的 web 搜索 MCP 工具，使用 TypeScript 编写，零运行时依赖，仅使用 Node.js 内置模块。通过 Model Context Protocol (MCP) 与 Claude Code 通信。

## 特点

- **零运行时依赖** — 仅使用 Node.js 内置模块 (`node:https`, `node:readline` 等)
- **双后端支持**：
  - `bing`（默认）：Bing RSS 搜索，全球可用，返回结构化 XML
  - `duckduckgo`：DuckDuckGo HTML 搜索，更多结果
- **自动故障转移**：Bing 失败时自动切换到 DuckDuckGo
- **TypeScript 编写**，类型安全

## 项目结构

```
web-search-mcp/
├── package.json          # 项目配置
├── tsconfig.json         # TypeScript 配置
├── src/
│   └── index.ts          # MCP 服务器源代码
└── dist/
    ├── index.js           # 编译后的 JS 文件
    └── index.d.ts         # 类型声明
```

---

## 安装步骤

### 步骤 1: 确认 Node.js 版本

```bash
node --version   # 需要 >= 18.0.0
```

如果你的 Node.js 版本过低，请先升级：

```bash
# 使用 nvm 安装最新 LTS 版本
nvm install --lts
nvm use --lts
```

### 步骤 2: 安装依赖并编译

```bash
cd /home/fish/xing_claude/web-search-mcp

# 安装 devDependencies（仅 TypeScript 编译器）
npm install

# 编译 TypeScript
npx tsc
```

### 步骤 3: 测试服务器是否正常工作

**测试 initialize 握手：**

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize"}' | timeout 3 node /home/fish/xing_claude/web-search-mcp/dist/index.js
```

预期输出类似：
```json
{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05","capabilities":{"tools":{}},"serverInfo":{"name":"web-search","version":"1.0.0"}}}
```

**测试实际搜索：**

```bash
echo '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"web_search","arguments":{"query":"TypeScript tutorial","count":5}}}' | timeout 15 node /home/fish/xing_claude/web-search-mcp/dist/index.js 2>/dev/null
```

预期输出包含搜索结果的 JSON 响应。

### 步骤 4: 在 Claude Code 中配置 MCP 服务器

有三种配置方式：

**方式 A — 通过 `/update-config` 命令（推荐）**

在 Claude Code 会话中执行：

```
/update-config
```

然后按提示添加 MCP 服务器配置。

**方式 B — 直接编辑配置文件**

编辑 `~/.claude/settings.json`（全局配置，所有项目生效）：

```json
{
  "mcpServers": {
    "web-search": {
      "command": "node",
      "args": ["/home/fish/xing_claude/web-search-mcp/dist/index.js"],
      "env": {
        "WEB_SEARCH_BACKEND": "bing"
      }
    }
  }
}
```

或编辑 `<项目根>/.claude/settings.local.json`（仅当前项目生效），内容同上。

**方式 C — 使用 `mcp.json` 文件（项目级）**

在需要使用搜索功能的项目根目录下创建 `.mcp.json` 文件：

```json
{
  "mcpServers": {
    "web-search": {
      "command": "node",
      "args": ["/home/fish/xing_claude/web-search-mcp/dist/index.js"],
      "env": {
        "WEB_SEARCH_BACKEND": "bing"
      }
    }
  }
}
```

### 步骤 5: 重启 Claude Code

配置完成后，重启 Claude Code 会话。MCP 服务器会在 Claude Code 启动时自动启动。

### 步骤 6: 验证配置

在 Claude Code 中输入：

```
search the web for TypeScript tutorial
```

或中文：

```
帮我搜索一下 React 19 的新特性
```

Claude 会自动调用 `web_search` 工具获取结果。

### 步骤 7: 切换搜索后端（可选）

通过环境变量 `WEB_SEARCH_BACKEND` 切换后端：

| 值 | 说明 |
|---|---|
| `bing` | Bing RSS（**默认**，全球可用，推荐） |
| `duckduckgo` / `ddg` | DuckDuckGo HTML（更多结果，部分地区可能受限） |

修改配置文件中的 `env.WEB_SEARCH_BACKEND` 字段即可。

---

## 工具参数说明

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `query` | string | 是 | 搜索关键词 |
| `count` | number | 否 | 返回结果数（默认 10，最大 30） |

---

## 故障排除

**1. 搜索返回空结果**

- 检查网络连接是否正常
- 尝试切换后端：`WEB_SEARCH_BACKEND=bing`
- 确认搜索引擎在当前网络环境下可访问：
  ```bash
  curl -sL --connect-timeout 10 "https://www.bing.com/search?format=rss&q=test" | head -c 500
  ```

**2. MCP 服务器启动失败**

- 确认 Node.js 版本 >= 18：`node --version`
- 确认编译产物存在：`ls dist/index.js`
- 手动运行服务器查看错误日志：
  ```bash
  node /home/fish/xing_claude/web-search-mcp/dist/index.js
  ```

**3. Claude Code 中看不到 web_search 工具**

- 确认配置文件路径正确（全局：`~/.claude/settings.json`，项目：`.claude/settings.local.json`）
- 重启 Claude Code 会话
- 检查 MCP 日志（stderr 输出）

**4. 被搜索引擎限流**

- 减少搜索频率
- 等待几分钟后重试
- 自动故障转移会切换到备用后端

---

## 许可

MIT
