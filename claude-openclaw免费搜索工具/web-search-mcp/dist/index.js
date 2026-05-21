#!/usr/bin/env node
/**
 * MCP Web Search Server
 *
 * 免费的 web 搜索 MCP 工具，支持多种搜索后端。
 * 零运行时依赖 —— 仅使用 Node.js 内置模块。
 * 通过 Model Context Protocol (MCP) 与 Claude Code 通信。
 *
 * 后端支持（通过 WEB_SEARCH_BACKEND 环境变量切换）：
 *   - bing (默认): Bing RSS 搜索，全球可用，返回结构化 XML，免费无需 API Key
 *   - duckduckgo: DuckDuckGo HTML 搜索，更多结果，部分区域可能受限
 */
import { createInterface } from "node:readline";
import https from "node:https";
import http from "node:http";
// ============================================================
// 文本处理工具函数
// ============================================================
function decodeHtml(text) {
    return text
        .replace(/&amp;/g, "&")
        .replace(/&lt;/g, "<")
        .replace(/&gt;/g, ">")
        .replace(/&quot;/g, '"')
        .replace(/&#x27;/g, "'")
        .replace(/&#39;/g, "'")
        .replace(/&#(\d+);/g, (_, d) => String.fromCharCode(parseInt(d, 10)))
        .replace(/&#x([0-9a-fA-F]+);/g, (_, h) => String.fromCharCode(parseInt(h, 16)));
}
function decodeXmlEntities(text) {
    return text
        .replace(/&amp;/g, "&")
        .replace(/&lt;/g, "<")
        .replace(/&gt;/g, ">")
        .replace(/&quot;/g, '"')
        .replace(/&apos;/g, "'")
        .replace(/&#(\d+);/g, (_, d) => String.fromCharCode(parseInt(d, 10)))
        .replace(/&#x([0-9a-fA-F]+);/g, (_, h) => String.fromCharCode(parseInt(h, 16)))
        .replace(/<!\[CDATA\[([\s\S]*?)\]\]>/g, "$1");
}
function stripTags(html) {
    return html.replace(/<[^>]*>/g, "");
}
function truncate(text, maxLen) {
    if (text.length <= maxLen)
        return text;
    return text.slice(0, maxLen - 3) + "...";
}
// ============================================================
// HTTP 请求
// ============================================================
function fetchUrl(url, redirectCount = 0) {
    return new Promise((resolve, reject) => {
        if (redirectCount > 3) {
            reject(new Error("Too many redirects"));
            return;
        }
        const mod = url.startsWith("https://") ? https : http;
        const req = mod.get(url, {
            headers: {
                "User-Agent": "Mozilla/5.0 (compatible; MCPWebSearch/1.0)",
                Accept: "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
                "Accept-Encoding": "identity",
            },
            timeout: 15000,
        }, (res) => {
            const status = res.statusCode ?? 0;
            if ([301, 302, 303, 307, 308].includes(status)) {
                const location = res.headers.location;
                if (location) {
                    res.destroy();
                    const redirectUrl = location.startsWith("http")
                        ? location
                        : new URL(location, url).href;
                    fetchUrl(redirectUrl, redirectCount + 1).then(resolve, reject);
                    return;
                }
            }
            if (status >= 400) {
                res.destroy();
                reject(new Error(`HTTP ${status}`));
                return;
            }
            let data = "";
            res.setEncoding("utf8");
            res.on("data", (chunk) => {
                data += chunk;
            });
            res.on("end", () => resolve(data));
        });
        req.on("error", reject);
        req.on("timeout", () => {
            req.destroy();
            reject(new Error("Request timed out"));
        });
    });
}
// ============================================================
// 1. Bing RSS 搜索后端
// ============================================================
function parseBingRss(xml) {
    const results = [];
    // 解析 RSS <item> 元素
    const itemRegex = /<item>([\s\S]*?)<\/item>/gi;
    let match;
    while ((match = itemRegex.exec(xml)) !== null) {
        const itemXml = match[1];
        const titleMatch = itemXml.match(/<title>([\s\S]*?)<\/title>/i);
        const linkMatch = itemXml.match(/<link>([\s\S]*?)<\/link>/i);
        const descMatch = itemXml.match(/<description>([\s\S]*?)<\/description>/i);
        if (!linkMatch)
            continue;
        const title = titleMatch
            ? decodeXmlEntities(titleMatch[1]).trim()
            : "";
        const url = decodeXmlEntities(linkMatch[1]).trim();
        const snippet = descMatch
            ? decodeXmlEntities(stripTags(descMatch[1])).trim()
            : "";
        if (url && url.startsWith("http")) {
            results.push({ title: title || url, url, snippet });
        }
    }
    return results;
}
async function searchBing(query, count) {
    const encoded = encodeURIComponent(query);
    // Bing RSS 搜索端点
    const url = `https://www.bing.com/search?format=rss&q=${encoded}&setlang=zh-cn`;
    const xml = await fetchUrl(url);
    // 检查是否被限流
    if (xml.includes("<title>400") || xml.includes("rate limit")) {
        throw new Error("Bing is rate-limiting requests. Please wait a moment.");
    }
    const results = parseBingRss(xml);
    return results.slice(0, count);
}
// ============================================================
// 2. DuckDuckGo HTML 搜索后端
// ============================================================
function parseDuckDuckGoHtml(html) {
    const results = [];
    const positions = [];
    const resultRegex = /<div\s+class="[^"]*\bresult\b[^"]*"/gi;
    let match;
    while ((match = resultRegex.exec(html)) !== null) {
        positions.push(match.index);
    }
    for (let i = 0; i < positions.length; i++) {
        const start = positions[i];
        const end = i + 1 < positions.length ? positions[i + 1] : html.length;
        const block = html.slice(start, end);
        if (block.includes("result--ad") || block.includes('class="badge--ad"')) {
            continue;
        }
        const hrefMatch = block.match(/<a\s+[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]*)"/i);
        if (!hrefMatch)
            continue;
        const url = decodeHtml(hrefMatch[1]);
        if (url.includes("duckduckgo.com") || !url.startsWith("http"))
            continue;
        let title = "";
        const titleMatch = block.match(/<a\s+[^>]*class="[^"]*result__a[^"]*"[^>]*>([\s\S]*?)<\/a>/i);
        if (titleMatch) {
            title = decodeHtml(stripTags(titleMatch[1]));
        }
        let snippet = "";
        const snippetMatch = block.match(/<[^>]*class="[^"]*result__snippet[^"]*"[^>]*>([\s\S]*?)<\//i);
        if (snippetMatch) {
            snippet = decodeHtml(stripTags(snippetMatch[1]));
        }
        if (title && url) {
            results.push({ title, url, snippet });
        }
    }
    return results;
}
async function searchDuckDuckGo(query, count) {
    const encoded = encodeURIComponent(query);
    const url = `https://html.duckduckgo.com/html/?q=${encoded}`;
    const html = await fetchUrl(url);
    if (html.includes("rate limit") || html.includes("captcha")) {
        throw new Error("DuckDuckGo is rate-limiting requests. Please wait a moment.");
    }
    const results = parseDuckDuckGoHtml(html);
    if (results.length === 0) {
        // 备用解析
        return parseDuckDuckGoFallback(html).slice(0, count);
    }
    return results.slice(0, count);
}
function parseDuckDuckGoFallback(html) {
    const results = [];
    const linkRegex = /<a\s+[^>]*href="(https?:\/\/(?!(?:www\.)?duckduckgo\.com)[^"]*)"[^>]*>([\s\S]*?)<\/a>/gi;
    const seen = new Set();
    let match;
    while ((match = linkRegex.exec(html)) !== null) {
        const url = decodeHtml(match[1]);
        const title = decodeHtml(stripTags(match[2]));
        if (!seen.has(url) &&
            title.trim().length > 0 &&
            title.trim().length < 200) {
            seen.add(url);
            results.push({ title: title.trim(), url, snippet: "" });
        }
    }
    return results;
}
// ============================================================
// 搜索调度
// ============================================================
function getBackend() {
    const env = process.env.WEB_SEARCH_BACKEND?.toLowerCase();
    if (env === "duckduckgo" || env === "ddg")
        return "duckduckgo";
    if (env === "bing")
        return "bing";
    // 默认使用 Bing RSS（全球最可靠）
    return "bing";
}
async function search(query, count) {
    const backend = getBackend();
    if (backend === "duckduckgo") {
        return {
            results: await searchDuckDuckGo(query, count),
            backend: "DuckDuckGo",
        };
    }
    try {
        return {
            results: await searchBing(query, count),
            backend: "Bing (RSS)",
        };
    }
    catch (bingErr) {
        const bingMsg = bingErr instanceof Error ? bingErr.message : "Unknown";
        log(`Bing search failed: ${bingMsg}. Falling back to DuckDuckGo...`);
        try {
            return {
                results: await searchDuckDuckGo(query, count),
                backend: "DuckDuckGo (fallback)",
            };
        }
        catch {
            throw new Error(`All search backends failed. Bing: ${bingMsg}. DuckDuckGo: unreachable.`);
        }
    }
}
// ============================================================
// 格式化输出
// ============================================================
function formatResults(results, backend) {
    if (results.length === 0) {
        return "No search results found. Try rephrasing your query or using different keywords.";
    }
    const lines = [
        `Found ${results.length} results via ${backend}:\n`,
    ];
    for (let i = 0; i < results.length; i++) {
        const r = results[i];
        lines.push(`## ${i + 1}. ${r.title}`);
        lines.push(`- **URL**: ${r.url}`);
        if (r.snippet) {
            lines.push(`- ${truncate(r.snippet, 300)}`);
        }
        lines.push("");
    }
    return lines.join("\n");
}
// ============================================================
// MCP 协议处理
// ============================================================
const SERVER_NAME = "web-search";
const SERVER_VERSION = "1.0.0";
function log(msg) {
    process.stderr.write(`[${SERVER_NAME}] ${msg}\n`);
}
function send(response) {
    process.stdout.write(JSON.stringify(response) + "\n");
}
function buildToolList() {
    return {
        tools: [
            {
                name: "web_search",
                description: "Search the web and get structured results (titles, URLs, snippets). " +
                    "Uses Bing RSS by default (free, no API key required). " +
                    "Set env var WEB_SEARCH_BACKEND=duckduckgo to switch backend. " +
                    "Use for: finding current information, researching topics, looking up documentation.",
                inputSchema: {
                    type: "object",
                    properties: {
                        query: {
                            type: "string",
                            description: "The search query to execute",
                        },
                        count: {
                            type: "number",
                            description: "Maximum number of results to return (default: 10, max: 30)",
                            default: 10,
                        },
                    },
                    required: ["query"],
                },
            },
        ],
    };
}
async function handleRequest(req) {
    const { method, id } = req;
    // 通知类型不需要回复
    if (id === undefined || id === null) {
        if (method === "notifications/initialized") {
            log("Client initialized. Ready to serve search requests.");
        }
        return;
    }
    switch (method) {
        case "initialize": {
            send({
                jsonrpc: "2.0",
                id,
                result: {
                    protocolVersion: "2024-11-05",
                    capabilities: { tools: {} },
                    serverInfo: {
                        name: SERVER_NAME,
                        version: SERVER_VERSION,
                    },
                },
            });
            break;
        }
        case "tools/list": {
            send({
                jsonrpc: "2.0",
                id,
                result: buildToolList(),
            });
            break;
        }
        case "tools/call": {
            const params = req.params;
            const toolName = params?.name;
            const args = params?.arguments ?? {};
            if (toolName !== "web_search") {
                send({
                    jsonrpc: "2.0",
                    id,
                    error: { code: -32601, message: `Unknown tool: ${toolName}` },
                });
                return;
            }
            const query = args.query;
            if (!query || typeof query !== "string" || query.trim().length === 0) {
                send({
                    jsonrpc: "2.0",
                    id,
                    error: {
                        code: -32602,
                        message: 'Missing or invalid required parameter: "query"',
                    },
                });
                return;
            }
            const count = Math.min(Math.max(1, args.count ?? 10), 30);
            try {
                log(`Searching: "${query}" (count: ${count}, backend: ${getBackend()})`);
                const { results, backend } = await search(query.trim(), count);
                log(`Found ${results.length} result(s) via ${backend}`);
                send({
                    jsonrpc: "2.0",
                    id,
                    result: {
                        content: [
                            {
                                type: "text",
                                text: formatResults(results, backend),
                            },
                        ],
                    },
                });
            }
            catch (err) {
                const message = err instanceof Error ? err.message : "Unknown search error";
                log(`Search error: ${message}`);
                send({
                    jsonrpc: "2.0",
                    id,
                    result: {
                        content: [
                            {
                                type: "text",
                                text: `Search failed: ${message}\n\nPlease try again later or rephrase your query.`,
                            },
                        ],
                        isError: true,
                    },
                });
            }
            break;
        }
        default: {
            send({
                jsonrpc: "2.0",
                id,
                error: {
                    code: -32601,
                    message: `Method not found: ${method}`,
                },
            });
        }
    }
}
// ============================================================
// 主入口
// ============================================================
async function main() {
    const backend = getBackend();
    log(`Starting ${SERVER_NAME} v${SERVER_VERSION}`);
    log(`Transport: stdio (MCP)`);
    log(`Search backend: ${backend}`);
    const rl = createInterface({
        input: process.stdin,
        output: process.stdout,
        terminal: false,
    });
    for await (const line of rl) {
        const trimmed = line.trim();
        if (!trimmed)
            continue;
        try {
            const req = JSON.parse(trimmed);
            await handleRequest(req);
        }
        catch {
            log(`Failed to parse message: ${trimmed.slice(0, 200)}`);
            try {
                const partial = JSON.parse(trimmed);
                if (partial.id) {
                    send({
                        jsonrpc: "2.0",
                        id: partial.id,
                        error: { code: -32700, message: "Parse error" },
                    });
                }
            }
            catch {
                // 完全无法解析，丢弃
            }
        }
    }
    log("Stopping — stdin closed");
}
main().catch((err) => {
    const msg = err instanceof Error ? err.message : "Unknown error";
    log(`Fatal: ${msg}`);
    process.exit(1);
});
//# sourceMappingURL=index.js.map