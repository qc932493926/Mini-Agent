大家好，我是小松鼠。

一名AI时代的学习者，专注探索个体在新时代的生存模式。

这是我的第 15 篇 AIGC 文章。

---

最近 AI Agent 这个词又火了。

各种 Agent 框架满天飞，LangChain、AutoGPT、MetaGPT...

一个比一个复杂，一个比一个能吹。

说实话，看完我更懵了。

直到我翻到 MiniMax 官方开源的这个项目——Mini-Agent。

500 行代码，把 Agent 的核心逻辑写得清清楚楚。

孩子们，今天就带你们从源码角度，彻底搞懂 AI Agent 到底是什么。

真没那么玄乎。

---

一、为什么要读这个项目？

市面上 Agent 项目太多了，为什么偏偏推荐这个？

三个理由：

1️⃣ 官方出品，思路正统

这是 MiniMax 为自家 M2.5 大模型写的最佳实践。

不是某个博主自嗨的 demo，是经过生产验证的。

2️⃣ 麻雀虽小，五脏俱全

500 行代码里包含了：

- Agent Loop（核心循环）
- Tool Calling（工具调用）
- MCP 协议（扩展接入）
- 重试机制（容错处理）
- 消息摘要（上下文管理）

你在 Claude、豆包、元宝这些产品里看到的功能，这里全有。

3️⃣ 代码干净，注释友好

没有那些花里胡哨的抽象，每个模块职责单一。

读完你能说出为什么这样设计。

---

二、先记住一句话

Agent = LLM + Plan + Tools + Memory

这是我们今天的核心公式。

所有 Agent 框架，不管它吹得多高级，本质都是这四个东西的组合。

翻译一下：

- LLM（大模型）：大脑，负责思考和决策
- Plan（规划）：执行计划，决定先做什么后做什么
- Tools（工具）：手和脚，实际干活的
- Memory（记忆）：笔记本，记住之前聊了什么

```
┌─────────────────────────────────────────────────────────────┐
│                    Mini-Agent 架构图                         │
│                                                             │
│    ┌─────────┐      Agent = LLM + Plan + Tools + Memory     │
│    │   CLI   │                                              │
│    └────┬────┘                                              │
│         │                                                   │
│         ▼                                                   │
│    ┌─────────────────────────────────────────────────┐     │
│    │                    Agent                         │     │
│    │  ┌───────────┐  ┌───────────┐  ┌───────────┐   │     │
│    │  │    LLM    │  │   Plan    │  │   Tools   │   │     │
│    │  │ (大脑)    │  │  (循环)   │  │ (工具)    │   │     │
│    │  └───────────┘  └───────────┘  └───────────┘   │     │
│    │                 ┌───────────┐                   │     │
│    │                 │  Memory   │                   │     │
│    │                 │ (记忆)    │                   │     │
│    │                 └───────────┘                   │     │
│    └─────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

看懂这张图，你就懂了 80%。

---

三、跟着一个请求走一遍

理论讲完了，我们来走一遍代码。

假设用户输入：「创建一个 hello.py 文件」

代码执行流程是这样的：

```
① main()                    cli.py:844
   │
   ▼
② run_agent()               cli.py:486
   ├─ 加载配置
   ├─ 创建 LLMClient
   ├─ 初始化工具
   ├─ 创建 Agent
   │
   ▼
③ 交互循环                   cli.py:679
   ├─ 获取用户输入
   ├─ agent.add_user_message()
   │
   ▼
④ agent.run()               agent.py:321   ← 核心！

   这里是重点，展开讲。
```

agent.run() 的核心逻辑：

```python
while step < self.max_steps:
    # 1. 调用 LLM，让大模型思考
    response = await self.llm.generate(messages, tools)

    # 2. 没有工具调用 → 任务完成，返回结果
    if not response.tool_calls:
        return response.content

    # 3. 有工具调用 → 执行工具，继续循环
    for tool_call in response.tool_calls:
        result = await tool.execute(**arguments)
        self.messages.append(tool_result_message)

    step += 1
```

就这么简单。

Agent 的本质就是：**循环调用 LLM，LLM 决定要不要用工具，用完工具继续问 LLM，直到 LLM 说"我做完了"**。

这就是所谓的 Agent Loop。

AI 圈最擅长造词营销了，什么 ReAct、CoT、自主决策...

说白了就是个 while 循环。

---

四、两层循环，别搞混了

Mini-Agent 有两层循环，很多人会搞混。

外层循环：用户输入循环

```
cli.py:679
while True:
    user_input = await prompt_async()  # 等用户输入
    agent.add_user_message(user_input)
    result = await agent.run()         # 执行内层循环
    print(result)                      # 输出结果
```

内层循环：Agent 执行循环

```
agent.py:343
while step < max_steps:
    response = llm.generate()    # LLM 思考
    if no tool_calls: break      # 完成了
    execute tools...             # 执行工具
    step += 1                    # 继续
```

关键结论：

**内层循环执行期间，用户无法插入信息。**

只能按 Esc 取消，或者等它跑完。

想补充信息？等下一轮输入。

这和我们想象的"随时打断"不太一样。

但也合理，毕竟 Agent 正在"工作"，你突然插一句，它容易懵。

---

五、Mini-Agent 有 TodoList 吗？没有！

用过 Claude Code 的同学知道，它有一个 TodoList 功能。

你能看到 Claude 的任务列表，它做完一个划掉一个。

但 Mini-Agent 没有。

这是两种不同的设计思路：

| 特性 | Mini-Agent | Claude Code |
|------|------------|-------------|
| 规划机制 | 隐式 - LLM 自己想 | 显式 - TodoList 工具 |
| 用户可见 | 只能看执行过程 | 能看到任务列表 |
| 动态调整 | LLM 自己决定 | 用户可以改 |
| 实现复杂度 | 简单 | 更复杂 |

Mini-Agent 的 Plan 是隐式的：

LLM 在 thinking 里自己规划步骤，用户看不到显式列表。

```
# LLM 心里想的（用户看不到）
"""
用户要创建 hello.py，我需要：
1. 检查当前目录
2. 用 Write 工具创建文件
3. 返回结果
"""
```

每一步都是 LLM 即时决策，没有显式存储的计划。

Mini-Agent 选择了更简单的方案：让 LLM 自己管理计划。

这对 LLM 的能力要求更高，但代码量少了一半。

孩子们，现在你知道为什么 DeepSeek、Claude 这些顶级模型比那些小模型贵了吧~~

不是笨不笨的问题，是能不能做到的问题。

---

五点五、Tools 不只是文件操作

公式里的 Tools，不只是读写文件这么简单。

Mini-Agent 的工具系统有三层：

```
Tools 工具系统
├─ 内置工具：Read、Write、Edit、Bash...（直接执行）
├─ MCP 工具：通过 MCP 协议调用外部服务（独立进程）
└─ Skill 工具：封装好的复杂能力（如 Claude Skills）
```

MCP 是 Anthropic 推的标准协议。

好处是：工具可以用任何语言写（Node.js、Go、Rust），只要遵循协议就能被 Agent 调用。

比如你配置一个 MCP 服务：

```json
{
  "mcpServers": {
    "memory": {
      "command": "npx",
      "args": ["-y", "@anthropic/memory-mcp"]
    }
  }
}
```

重启 Agent，它就多了一个"长期记忆"的能力。

Skill 更高级，可以把一整套流程封装成一个技能。

比如"代码审查"技能：读代码 → 分析问题 → 给出建议 → 生成报告。

说人话：MCP 是插件系统，Skill 是技能包。

---

六、三个让我眼前一亮的设计

读完源码，有三个设计让我觉得"写得漂亮"。

设计1：LLM 抽象层（策略模式 + 工厂模式）

Mini-Agent 支持 Anthropic 和 OpenAI 两种协议。

怎么做到的？

```
┌───────────────────┐
│    LLMClient      │  ← 工厂 + 门面
│   (Wrapper)       │
└─────────┬─────────┘
          │ 根据 provider 创建
          ▼
┌───────────────────┐
│  LLMClientBase    │  ← 策略接口
│     (ABC)         │
└─────────┬─────────┘
          │
    ┌─────┴─────┐
    │           │
    ▼           ▼
┌─────────┐ ┌─────────┐
│Anthropic│ │ OpenAI  │
│ Client  │ │ Client  │
└─────────┘ └─────────┘
```

工厂模式负责创建，策略模式负责执行。

新增一个 Provider？加一个类就行，不改现有代码。

这就是开闭原则的教科书示范。

设计2：双重 Token 检测触发摘要

LLM 有上下文长度限制，聊太久会爆。

怎么办？摘要。

但什么时候触发摘要？

Mini-Agent 用了双重检测：

```python
estimated_tokens = self._estimate_tokens()  # 本地估算
should_summarize = (
    estimated_tokens > self.token_limit or    # 本地超限
    self.api_total_tokens > self.token_limit  # API 返回超限
)
```

| 检测方式 | 优点 | 缺点 |
|----------|------|------|
| 本地估算 | 快速、预判 | 不够精确 |
| API 返回 | 精确 | 有滞后性 |
| 两者结合 | 既及时又准确 | - |

这个设计很细节，但很实用。

设计3：指数退避重试（装饰器模式）

API 调用失败怎么办？重试。

但怎么重试？

傻傻地立刻重试？服务器会更崩。

Mini-Agent 用指数退避：

```
第1次失败 → 等待 1s → 重试
第2次失败 → 等待 2s → 重试
第3次失败 → 等待 4s → 重试
第4次失败 → 抛出异常
```

等待时间指数增长，给服务器喘息的机会。

而且用装饰器实现，不污染业务代码：

```python
@async_retry(RetryConfig(max_retries=3, initial_delay=1.0))
async def call_api():
    # API 调用代码
    pass
```

一行注解搞定，优雅。

---

七、如果让我从零写一个？

看完源码，我想了一个问题：

如果从零写一个 Agent，最小可用版本需要什么？

必须有的：

1. Agent Loop（while 循环）
2. LLM 调用（能和大模型对话）
3. 1 个 Tool（至少能干一件事）

建议有的：

1. 重试机制（API 不稳定是常态）
2. 消息历史（不然每次都是新对话）

锦上添花的：

1. 消息摘要（长对话才需要）
2. MCP 协议（扩展工具才需要）
3. Skills（高级玩法）
4. 配置系统（多环境才需要）

说人话就是：

先把核心循环跑通，其他的慢慢加。

别一上来就想做一个"完美"的框架。

用起来，用着用着就懂了。

---

八、总结

今天我们从源码角度拆解了 Mini-Agent。

核心公式记住：

Agent = LLM + Plan + Tools + Memory

核心机制理解：

1. Agent Loop：while 循环 + LLM 决策 + 工具执行
2. 两层循环：外层等输入，内层跑任务
3. 隐式规划：LLM 自己管理计划
4. 双重检测：本地估算 + API 返回
5. 指数退避：1s → 2s → 4s → 失败

设计模式学到：

- 策略模式（LLM 抽象层）
- 工厂模式（Provider 切换）
- 装饰器模式（重试机制）
- 命令模式（Tool 基类）

我还是那句话：

先跑通，再理解，最后自己写一遍。

收藏着，关注着。

后续我会继续更新 Agent 相关的深度内容。

---

看到这里了，如果觉得不错：

✓ 点个「赞」，让我知道你在看
✓ 点个「在看」，分享给更多朋友
✓ 点个「转发」，帮助更多人
✓ 加个「星标⭐」，第一时间收到推送

也可以加我的个人微信围观学习：archerqc

小松鼠爱你们！
