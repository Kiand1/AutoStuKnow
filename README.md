# AutoStuKnow

面向 NAS 的自托管视频知识库 V1：通过带登录的 Web 页面批量提交 YouTube 视频，系统优先读取人工/自动字幕，没有可用字幕时才下载音频并用 Faster Whisper 本地识别；随后调用 OpenAI 兼容模型生成中文摘要，最后把 Markdown 知识笔记导入 AnythingLLM 的 RAG workspace。n8n 继续提供自动化编排入口。

## 组成

```text
Web 批量页面 / n8n Webhook
    │
    ▼
n8n Webhook ──► Ingestor ──► yt-dlp + Deno
                    │
                    ├──► YouTube 字幕（人工优先、自动字幕次之）
                    ├──► Faster Whisper（无字幕时本地识别）
                    ├──► OpenAI 兼容接口（可选总结/标签）
                    └──► AnythingLLM API（入库并向量化）
```

- AnythingLLM：知识库、向量检索和问答入口，默认使用内置 LanceDB。
- n8n：接收链接和后续扩展编排。仓库附带一个可导入的初始工作流。
- hwdsl2/whisper-server：支持 amd64/arm64 的 Faster Whisper OpenAI 兼容服务。
- Ingestor：提供带登录的批量 Web 页面，并负责限流、持久化任务、字幕/音频处理、总结和入库。
- yt-dlp + Deno：适配 YouTube 当前的 JavaScript challenge。

不配置总结模型也能运行：系统会生成并导入带时间戳的完整转录，只跳过“摘要/标签”部分。

## NAS 要求

- Docker Engine 和 Docker Compose v2。
- CPU 模式支持常见的 x86_64 与 ARM64 NAS。
- 建议至少 4 GB 可用内存；`base` Whisper 模型适合多数 CPU NAS。AnythingLLM 官方建议自身至少预留 2 GB 内存、10 GB 磁盘。
- 首次启动需要联网拉取镜像和 Whisper 模型。
- GPU 覆盖文件只适用于装好 NVIDIA 驱动及 NVIDIA Container Toolkit 的 x86_64 主机。

## 首次部署

从 GitHub 克隆项目：

```bash
git clone https://github.com/Kiand1/AutoStuKnow.git
cd AutoStuKnow
```

然后在项目目录执行初始化：

```bash
chmod +x scripts/init-nas.sh
sudo ./scripts/init-nas.sh
```

编辑 `.env`，至少检查：

```dotenv
DATA_ROOT=/volume1/docker/autostuknow
N8N_PUBLIC_URL=http://你的NAS局域网IP:5678/
PUID=1000
PGID=1000
```

不同 NAS 的数据目录示例：

- Synology：`/volume1/docker/autostuknow`
- QNAP：`/share/Container/autostuknow`
- Unraid：`/mnt/user/appdata/autostuknow`

启动：

```bash
docker compose pull
docker compose up -d --build
docker compose ps
```

默认入口：

- AnythingLLM：`http://NAS_IP:3001`
- n8n：`http://NAS_IP:5678`
- 批量提交页面：`http://NAS_IP:8090/ui`
- Ingestor API 文档：`http://NAS_IP:8090/docs`

Whisper 端口不映射到宿主机，只允许 Compose 内部服务访问。

### 目录权限

应用容器以 `.env` 中的 `PUID:PGID` 访问绑定目录。初始化脚本以 `sudo` 运行时会设置顶层目录所有者。若日志中仍出现 `Permission denied`，确认 NAS 上的目标目录属于该 UID/GID：

```bash
sudo chown -R 1000:1000 /你的/DATA_ROOT
```

请把示例中的路径和 UID/GID 换成自己的实际值，不要对 NAS 根目录运行递归 `chown`。

## AnythingLLM 首次配置

1. 打开 `http://NAS_IP:3001`，完成管理员和模型引导。
2. DeepSeek 部署可在 `.env` 设置 `ANYTHINGLLM_LLM_PROVIDER=deepseek`，复用自动总结的 Key 和模型。
3. 中文知识库建议设置 `ANYTHINGLLM_EMBEDDING_MODEL=MintplexLabs/multilingual-e5-small`；首次使用会下载约 487MB。
4. 可以先创建一个 workspace，也可以部署完成后直接在 8090 页面创建任意名称的知识库。
5. 在 AnythingLLM 的开发者/API 设置中创建 API Key。
6. 写入 `.env`：

```dotenv
ANYTHINGLLM_LLM_PROVIDER=deepseek
ANYTHINGLLM_EMBEDDING_MODEL=MintplexLabs/multilingual-e5-small
ANYTHINGLLM_API_KEY=你的AnythingLLM_API_Key
ANYTHINGLLM_WORKSPACE_SLUG=视频知识库的slug
ANYTHINGLLM_AUTO_SYNC=true
ANYTHINGLLM_SYNC_TIMEOUT_SECONDS=1800
```

`ANYTHINGLLM_WORKSPACE_SLUG` 是直接调用 API 或 n8n 时的默认目标。8090 页面会实时读取 AnythingLLM 的全部 workspace，每批内容必须自行选择目标知识库，也可以在页面创建新知识库，不使用任何写死的分类名称。

AnythingLLM 的 workspace 本身是一级结构，因此 AutoStuKnow 在每个 workspace 内维护一棵独立的虚拟目录树，例如 `投资/虚拟币/BTC/技术分析`。目录层级和名称完全由用户创建；路径会写入 Markdown 来源信息和 AnythingLLM 文档描述，但 RAG 检索范围仍是整个 workspace。需要严格隔离检索范围时，应创建不同的 AnythingLLM workspace。

让导入服务重新读取配置：

```bash
docker compose up -d --force-recreate ingestor
```

AnythingLLM API Key 只在 Docker 内网中使用，不会写入生成的 Markdown。

## 配置自动总结

任何支持 `POST /chat/completions` 的 OpenAI 兼容接口均可使用。DeepSeek 推荐配置如下：

```dotenv
LLM_BASE_URL=https://api.deepseek.com
LLM_API_KEY=你的DeepSeek_API_Key
LLM_MODEL=deepseek-v4-pro
LLM_THINKING_MODE=disabled
LLM_JSON_MODE=true
SUMMARY_LANGUAGE=zh-CN
```

`deepseek-v4-pro` 质量优先；需要更低延迟和费用时可改为 `deepseek-v4-flash`。摘要流程显式关闭思考模式并启用 JSON Output，用于稳定解析摘要、要点和标签。其他 OpenAI 兼容服务不支持这两个参数时，把 `LLM_THINKING_MODE` 留空并设置 `LLM_JSON_MODE=false`。模型名和接口变化请以 [DeepSeek API 文档](https://api-docs.deepseek.com/) 为准。

如果 Ollama 运行在 NAS 宿主机：

```dotenv
LLM_BASE_URL=http://host.docker.internal:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=qwen3:8b
```

修改后重建 ingestor 容器配置：

```bash
docker compose up -d --force-recreate ingestor
```

长转录会先分段总结，再做分层合并，避免一次把整段视频塞进模型上下文。

## 提交视频

### 通过 Web 页面（推荐）

打开 `http://NAS_IP:8090/ui`，先选择已有知识库或输入任意名称创建新知识库，再选择根目录或创建任意多级目录，然后在文本框中每行粘贴一个 YouTube 地址，单次最多 50 条。页面会自动合并同批重复链接，并持续显示字幕读取、Whisper、DeepSeek 总结、目标知识库、目录和 AnythingLLM 入库进度。

页面支持删除 AutoStuKnow 导入的单条知识，也支持递归删除整个目录。单条删除前会展示知识位置并二次确认；目录删除前会统计子目录、处理记录和已入库文档数量，并要求输入完整目录路径。删除已入库知识时，系统会先从 AnythingLLM workspace 删除向量，再永久删除 AnythingLLM 源文档，最后清理 AutoStuKnow 本地处理记录。任一远端步骤失败时会保留本地记录以便重试。正在处理的任务不能删除，目录内存在运行中任务时也不会执行递归删除。

目标知识库旁也提供“删除知识库”入口。删除前会读取 AnythingLLM 的文档关联数量，统计 AutoStuKnow 管理的目录和任务，并要求输入完整知识库名称。确认后会永久删除该 workspace 的向量、会话、配置、AutoStuKnow 处理记录及 AutoStuKnow 上传的源文档。手工上传或其他工具导入的共享源文件只解除该 workspace 的向量关联，不从 AnythingLLM 公共文档池永久清理，以免影响其他 workspace。

每条已完成或失败的处理记录提供“移动”入口，可移动到同一知识库的其他目录，也可移动到另一个 AnythingLLM workspace。目录内移动只更新 AutoStuKnow 的目录归属和本地 Markdown；跨 workspace 移动会先把文档加入目标 workspace 并确认嵌入成功，再从原 workspace 移除。原 workspace 移除失败时会自动撤销目标端的添加，任务记录仍保留在原位置。正在处理的任务不能移动，目标多级目录需要先在页面创建。

页面中的“知识库预览”会按当前 workspace 展示根目录、任意层级子目录、每层直接包含的知识数量、子树总数、文件大小和处理状态。点击知识可查看完整 Markdown 内容或下载单个文件；也可以下载当前目录及全部子目录，或把整个知识库导出为 ZIP。压缩包保留目录结构和空目录，并附带 `知识库信息.json`，记录来源地址、目录、文件路径、更新时间等清单信息。为兼容 Windows 自带压缩文件夹，过长的中文目录或标题会在 ZIP 内自动缩短并附加稳定哈希，完整原始标题仍保留在清单和 Markdown 中。所有预览和下载接口都要求有效的 Web 登录会话。

预览和导出范围是由 AutoStuKnow 生成并管理的本地 Markdown。若同一 AnythingLLM workspace 还包含手工上传或由其他工具导入的文件，这些外部文件不会出现在 AutoStuKnow 的压缩包中。

目录只是 AutoStuKnow 对自己导入内容的组织层，不会删除同一 AnythingLLM workspace 中手工上传或由其他工具导入的文档。空目录保存在 `${DATA_ROOT}/ingestor/catalog.json`，已有未分类任务会继续显示在根目录。

Web 用户名默认为 `admin`，密码由 `scripts/init-nas.sh` 随机生成。只在自己的 NAS 终端查看：

```bash
sudo sed -n 's/^WEB_UI_USERNAME=//p; s/^WEB_UI_PASSWORD=//p' /volume2/docker/autoStuKnow/.env
```

初始化密码只用于首次登录。第一次登录后必须在页面设置至少 8 位的新密码，完成后才能提交和查看任务；以后也可以通过右上角的“修改密码”入口更换。新密码只以 PBKDF2 哈希保存在 `${DATA_ROOT}/ingestor/auth/web-credentials.json`，不会回写 `.env`，也不会以明文保存。

登录状态使用 HttpOnly、SameSite=Strict 的签名 Cookie；修改密码后旧会话会自动失效。直接使用局域网 HTTP 时 `WEB_UI_SECURE_COOKIE=false`；配置 HTTPS 反向代理后改为 `true` 并重建 Ingestor。

### 通过 n8n 自动化接口

n8n 启动时会导入并发布带 Header Auth 的 `AutoStuKnow - YouTube to RAG` 工作流。

若没有自动出现，在 n8n UI 中手动导入：

```text
n8n/workflows/youtube-to-rag.json
```

激活后调用生产 Webhook：

```bash
curl -X POST 'http://NAS_IP:5678/webhook/youtube-to-rag' \
  -H 'Content-Type: application/json' \
  -H 'X-Webhook-Key: 你的WEBHOOK_API_KEY' \
  -d '{"url":"https://www.youtube.com/watch?v=VIDEO_ID","language":"auto"}'
```

`WEBHOOK_API_KEY` 由 `scripts/init-nas.sh` 自动生成并保存在 NAS 的 `.env` 中。Webhook 使用 n8n 原生 Header Auth：缺少或填写错误的 `X-Webhook-Key` 会在工作流启动前被拒绝。返回 `job_id` 后，用 Ingestor 查询进度。即使已有请求头鉴权，若要暴露到互联网，仍建议在反向代理上增加 HTTPS、IP 限制和限流。

在自己的 NAS 终端查看密钥（不要粘贴到聊天或公开记录）：

```bash
sudo sed -n 's/^WEBHOOK_API_KEY=//p' /volume2/docker/autoStuKnow/.env
```

### 直接调用 Ingestor

把 `.env` 中的 `INGESTOR_API_KEY` 放到请求头：

```bash
curl -X POST 'http://NAS_IP:8090/jobs' \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: 你的INGESTOR_API_KEY' \
  -d '{"url":"https://youtu.be/VIDEO_ID","language":"auto","workspace_slug":"目标slug","category_path":"投资/虚拟币/BTC"}'
```

查看状态：

```bash
curl 'http://NAS_IP:8090/jobs/JOB_ID' \
  -H 'X-API-Key: 你的INGESTOR_API_KEY'
```

任务阶段依次包括 `fetching_metadata`、`fetching_subtitles`；没有字幕时还会经过 `downloading_audio`、`transcribing`，之后是 `summarizing`、`writing_document`、`syncing_anythingllm`、`completed`。同一视频成功处理后再次提交会直接返回原任务；需要重新处理时传入 `"force": true`。

如果第一次处理时还没有配置 AnythingLLM，可以稍后手动同步：

```bash
curl -X POST 'http://NAS_IP:8090/jobs/JOB_ID/sync?workspace_slug=你的slug' \
  -H 'X-API-Key: 你的INGESTOR_API_KEY'
```

## 字幕与 Whisper 策略

默认配置为：

1. 优先选择请求语言的 YouTube 人工字幕。
2. 没有匹配的人工字幕时，使用请求语言的 YouTube 自动字幕。
3. `language=auto` 时优先跟随视频原语言，避免误用自动翻译轨道。
4. 没有可用字幕、字幕下载失败或内容为空时，下载音频并调用 NAS 上的 Faster Whisper。

最终 Markdown 的“来源信息”会记录转录来源，任务接口和 Web 页面也会显示 `人工字幕`、`自动字幕` 或 `Whisper`。可通过以下变量关闭字幕优先或禁用自动字幕：

```dotenv
PREFER_YOUTUBE_SUBTITLES=true
ALLOW_AUTOMATIC_SUBTITLES=true
```

## YouTube cookies

公开内容通常不需要 cookies。登录、会员或年龄限制内容需要浏览器导出的 Netscape 格式 cookies：

```bash
cp config/yt-cookies.txt.example config/yt-cookies.txt
```

把 cookies 写入 `config/yt-cookies.txt`，再设置：

```dotenv
YTDLP_COOKIES_FILE=/config/yt-cookies.txt
```

然后重建 ingestor 容器配置。真实 cookies 已被 `.gitignore` 排除；它相当于登录凭据，必须限制文件权限并定期更新。

## NVIDIA GPU 模式

确认宿主机 `nvidia-smi` 和 `docker run --gpus all ...` 均可用后：

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

GPU 模式使用 `float16` 和 CUDA 镜像。普通 Synology/QNAP 的 Intel 核显不能使用这个覆盖文件，继续用 CPU 配置即可。

## 数据与备份

所有持久数据都在 `DATA_ROOT`：

```text
DATA_ROOT/
├── anythingllm/   # 配置、数据库、文档、向量
├── anythingllm-hotdir/  # AnythingLLM 文档处理临时目录
├── anythingllm-outputs/ # AnythingLLM 文档处理输出
├── ingestor/      # job.json、转录、最终 Markdown（KEEP_AUDIO=true 时也保留音频）
├── n8n/           # n8n SQLite 数据库与凭据
└── whisper/       # Whisper 模型缓存
```

备份前建议短暂停止写入：

```bash
docker compose stop
# 用 NAS 快照或备份工具备份 DATA_ROOT 和项目内的 .env
docker compose start
```

`.env` 与 n8n 数据库包含密钥，应加密备份。

## 日常维护

查看日志：

```bash
docker compose logs -f --tail=200 ingestor whisper anythingllm n8n
```

拉取最新代码、更新镜像并重建适配层（NAS 没有安装 Git 时会临时使用 `alpine/git` 容器）：

```bash
sh scripts/update-nas.sh
```

也可以手动执行：

```bash
git pull --ff-only
docker compose pull
docker compose up -d --build
```

健康检查：

```bash
curl http://NAS_IP:8090/readyz
```

常见问题：

- Whisper 第一次长时间 `starting`：正在下载模型，查看 `docker compose logs -f whisper`。
- YouTube 提示需要登录：配置 cookies，并确认 Deno/yt-dlp 镜像已重新构建。
- `sync_status=failed`：检查 AnythingLLM API Key、workspace slug 和 AnythingLLM 的 Embedding 配置。
- CPU 太慢或内存不足：把 `WHISPER_MODEL` 改成 `tiny`/`base`，保持 `WHISPER_COMPUTE_TYPE=int8`，并维持 `MAX_CONCURRENT_JOBS=1`。
- 外部 LLM 无法连接：容器内的 `localhost` 指向容器自己；宿主机服务应使用 `host.docker.internal`。

## V1 边界与后续路线

当前 V1 聚焦“单个 YouTube 视频 → 可检索知识笔记”的稳定闭环，已经有基于规范 URL 的基础去重。后续建议按顺序增加：

1. V2：语义去重、自动主题体系、失败重试和通知。
2. V3：跨视频知识融合、来源冲突提示、知识版本管理。
3. V4：在数据质量稳定后再做多 Agent 协作。
