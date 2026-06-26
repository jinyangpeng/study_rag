/**
 * ModelConfigDialog — Embedder / Reranker / Parser 配置新增/编辑对话框
 *
 * 复用同一个组件，根据 kind 切换字段：
 *  - embedder:  provider / model_name / dimension / batch_size / extra / description
 *  - reranker:  provider / protocol / model_name / top_k / extra / description
 *  - parser:    strategy / chunk_size / chunk_overlap / paragraph_separator / buffer_size / breakpoint_percentile_threshold / extra
 *
 * extra 用 JSON 文本框编辑（保留灵活性，支持任意键值）。
 *
 * 表单 UI 约定（避免长说明挤爆 label）：
 *  - Field 组件封装「Label / input / hint」三段式，label 只放字段名
 *  - 详细说明放 hint 行（短文本 + ⓘ tooltip 全文）
 *  - 必填字段用 `required` 标记
 */
import { useEffect, useState } from "react";
import { Loader2, Info as InfoIcon } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectItemText,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useApi } from "@/api/client";
import type {
  EmbedderConfigItem,
  ParserConfigItem,
  RerankerConfigItem,
} from "@/api/types";
import { toast } from "sonner";

type Kind = "embedder" | "reranker" | "parser";

type AnyConfigItem = EmbedderConfigItem | RerankerConfigItem | ParserConfigItem;

interface Props {
  open: boolean;
  onOpenChange: (v: boolean) => void;
  kind: Kind;
  /** 编辑时传现有配置；新增时传 null */
  initial: AnyConfigItem | null;
  onSaved: () => void;
}

// ===== 带描述的选项列表 =====

const EMBEDDER_PROVIDERS = [
  { value: "mock", desc: "占位实现，不依赖外部服务" },
  { value: "openai", desc: "OpenAI Embeddings API" },
  { value: "bge", desc: "BGE 本地模型（需 sentence-transformers）" },
  { value: "bge_zh", desc: "BGE 中文模型（自动加 query 前缀）" },
  { value: "fastembed", desc: "FastEmbed 本地推理（需 fastembed 库）" },
  { value: "azure_openai", desc: "Azure OpenAI Embeddings API" },
];

const RERANKER_PROVIDERS = [
  { value: "none", desc: "不做重排，仅截断到 top_k" },
  { value: "mock", desc: "占位实现，不依赖外部服务" },
  { value: "http", desc: "HTTP 远程服务（TEI / Jina 等）" },
  { value: "bge", desc: "BGE 本地重排模型（需 sentence-transformers）" },
  { value: "cohere", desc: "Cohere Rerank API" },
];

const RERANKER_PROTOCOLS = [
  { value: "tei", desc: "HuggingFace TEI 推理服务" },
  { value: "jina", desc: "Jina Reranker API" },
  { value: "cohere_compat", desc: "Cohere 兼容接口" },
  { value: "openai", desc: "OpenAI 兼容接口" },
];

const PARSER_STRATEGIES = [
  { value: "whole", desc: "整篇不切，适合短文档" },
  { value: "sentence", desc: "按句子切，适合大多数场景" },
  { value: "semantic", desc: "按语义切（需 embed_model，更智能）" },
  { value: "token", desc: "按 token 数切，适合精确控制大小" },
];

// ===== 动态 placeholder / helper =====

function embedderModelNameHint(provider: string) {
  switch (provider) {
    case "openai":
      return { placeholder: "text-embedding-3-small", hint: "API 模型 ID" };
    case "azure_openai":
      return { placeholder: "text-embedding-3-small", hint: "Azure 部署的模型 ID" };
    case "bge":
      return { placeholder: "BAAI/bge-m3", hint: "本地模型路径或 HuggingFace ID" };
    case "bge_zh":
      return { placeholder: "BAAI/bge-large-zh-v1.5", hint: "本地中文模型路径或 HuggingFace ID" };
    case "fastembed":
      return { placeholder: "BAAI/bge-small-en-v1.5", hint: "FastEmbed 支持的模型名" };
    default:
      return { placeholder: "（mock 模型，可留空）", hint: "模型标识" };
  }
}

function rerankerModelNameHint(provider: string) {
  switch (provider) {
    case "http":
      return { placeholder: "（HTTP 服务自动识别，可留空）", hint: "HTTP 服务使用的模型标识" };
    case "bge":
      return { placeholder: "BAAI/bge-reranker-base", hint: "本地重排模型路径或 HuggingFace ID" };
    case "cohere":
      return { placeholder: "rerank-v3.5", hint: "Cohere API 模型 ID" };
    default:
      return { placeholder: "（可留空）", hint: "模型标识" };
  }
}

function chunkSizeUnit(strategy: string) {
  return strategy === "token" ? "token" : "字符";
}

/**
 * Field — 标准化的「Label / input / hint」三段式表单字段。
 *
 * 设计要点：
 * - Label 只放字段名（必要时加 required `*` 标记），不再拼接长说明
 * - 详细说明放 hint 行（短文本 + ⓘ tooltip 全文），永远不换行撑爆 label
 * - hint 与 input 同行右对齐（输入框下方），不占额外列宽
 *
 * 用法：
 *   <Field label="Chunk Size" hint="字符数，越大粒度越粗" detail="建议 256~1024；过大影响检索粒度" required>
 *     <Input ... />
 *   </Field>
 */
function Field({
  label,
  hint,
  detail,
  required = false,
  warn,
  children,
}: {
  label: string;
  /** 短提示（一行内说明） */
  hint?: React.ReactNode;
  /** 完整说明（hover ⓘ 看到） */
  detail?: string;
  required?: boolean;
  /** 橙色警告文字（如「维度修改会破坏数据」），单独显示 */
  warn?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-1">
      <div className="flex items-center gap-1">
        <Label className="text-xs font-medium">
          {label}
          {required && <span className="ml-0.5 text-danger">*</span>}
        </Label>
        {detail && (
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <InfoIcon className="size-3 cursor-help text-fg-muted" />
              </TooltipTrigger>
              <TooltipContent className="max-w-xs">
                <p className="text-xs">{detail}</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        )}
      </div>
      {children}
      {hint && <p className="text-[10px] leading-tight text-fg-muted">{hint}</p>}
      {warn && <p className="text-[10px] leading-tight text-orange-500">{warn}</p>}
    </div>
  );
}

export default function ModelConfigDialog({
  open,
  onOpenChange,
  kind,
  initial,
  onSaved,
}: Props) {
  const { client } = useApi();
  const isEdit = initial !== null;

  // 通用字段
  const [name, setName] = useState("");
  const [extraText, setExtraText] = useState("{}");
  const [saving, setSaving] = useState(false);

  // embedder 字段
  const [provider, setProvider] = useState("");
  const [modelName, setModelName] = useState("");
  const [dimension, setDimension] = useState("1024");
  const [batchSize, setBatchSize] = useState("32");
  const [description, setDescription] = useState("");

  // reranker 字段
  const [protocol, setProtocol] = useState("tei");
  const [topK, setTopK] = useState("5");

  // parser 字段
  const [strategy, setStrategy] = useState("sentence");
  const [chunkSize, setChunkSize] = useState("512");
  const [chunkOverlap, setChunkOverlap] = useState("50");
  const [paragraphSeparator, setParagraphSeparator] = useState("\\n\\n");
  const [bufferSize, setBufferSize] = useState("");
  const [breakpointPercentile, setBreakpointPercentile] = useState("");

  useEffect(() => {
    if (!open) return;
    if (initial) {
      setName(initial.name);
      setExtraText(JSON.stringify(initial.extra ?? {}, null, 2));
      if (kind === "embedder") {
        const e = initial as EmbedderConfigItem;
        setProvider(e.provider);
        setModelName(e.model_name);
        setDimension(String(e.dimension));
        setBatchSize(String(e.batch_size));
        setDescription(e.description);
      } else if (kind === "reranker") {
        const r = initial as RerankerConfigItem;
        setProvider(r.provider);
        setProtocol(r.protocol);
        setModelName(r.model_name);
        setTopK(String(r.top_k));
        setDescription(r.description);
      } else {
        const p = initial as ParserConfigItem;
        setStrategy(p.strategy);
        setChunkSize(String(p.chunk_size));
        setChunkOverlap(String(p.chunk_overlap));
        // paragraph_separator 里 \n 在 JSON 里是字面值，UI 显示转义形式
        setParagraphSeparator(JSON.stringify(p.paragraph_separator));
        setBufferSize(p.buffer_size != null ? String(p.buffer_size) : "");
        setBreakpointPercentile(
          p.breakpoint_percentile_threshold != null
            ? String(p.breakpoint_percentile_threshold)
            : ""
        );
      }
    } else {
      // 默认值
      setName("");
      setExtraText("{}");
      if (kind === "embedder") {
        setProvider("openai");
        setModelName("");
        setDimension("1024");
        setBatchSize("32");
        setDescription("");
      } else if (kind === "reranker") {
        setProvider("http");
        setProtocol("tei");
        setModelName("");
        setTopK("5");
        setDescription("");
      } else {
        setStrategy("sentence");
        setChunkSize("512");
        setChunkOverlap("50");
        setParagraphSeparator('"\\n\\n"');
        setBufferSize("");
        setBreakpointPercentile("");
      }
    }
  }, [open, initial, kind]);

  // 当 provider 变化时，自动调整 protocol（protocol 仅 provider=http 时生效）
  useEffect(() => {
    if (kind === "reranker") {
      if (provider === "http" && !protocol) {
        setProtocol("tei");
      } else if (provider !== "http") {
        setProtocol("");
      }
    }
  }, [provider]); // eslint-disable-line react-hooks/exhaustive-deps

  function parseExtra(): Record<string, unknown> | null {
    try {
      const parsed = JSON.parse(extraText);
      if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
        toast.error("extra 必须是 JSON 对象");
        return null;
      }
      return parsed as Record<string, unknown>;
    } catch {
      toast.error("extra 不是合法 JSON");
      return null;
    }
  }

  function parseParagraphSeparator(): string | null {
    try {
      // 用户输入的是 JSON 字符串形式（如 "\n\n"），解析成实际字符
      return JSON.parse(paragraphSeparator);
    } catch {
      toast.error('段落分隔符需为 JSON 字符串，如 "\\n\\n"');
      return null;
    }
  }

  async function onSave() {
    if (!name.trim()) {
      toast.warning("请填写配置名");
      return;
    }
    const extra = parseExtra();
    if (extra === null) return;

    setSaving(true);
    try {
      if (kind === "embedder") {
        if (!provider) {
          toast.warning("请选择 provider");
          setSaving(false);
          return;
        }
        if (isEdit) {
          await client.updateEmbedderConfig(name, {
            provider,
            model_name: modelName,
            dimension: Number(dimension) || 0,
            batch_size: Number(batchSize) || 32,
            description,
            extra,
          });
        } else {
          await client.createEmbedderConfig({
            name: name.trim(),
            provider,
            model_name: modelName,
            dimension: Number(dimension) || 0,
            batch_size: Number(batchSize) || 32,
            description,
            extra,
          });
        }
      } else if (kind === "reranker") {
        if (!provider) {
          toast.warning("请选择 provider");
          setSaving(false);
          return;
        }
        // protocol 仅 provider=http 时有效，其余清空
        const effectiveProtocol = provider === "http" ? protocol : "";
        if (isEdit) {
          await client.updateRerankerConfig(name, {
            provider,
            protocol: effectiveProtocol,
            model_name: modelName,
            top_k: Number(topK) || 5,
            description,
            extra,
          });
        } else {
          await client.createRerankerConfig({
            name: name.trim(),
            provider,
            protocol: effectiveProtocol,
            model_name: modelName,
            top_k: Number(topK) || 5,
            description,
            extra,
          });
        }
      } else {
        // parser
        const ps = parseParagraphSeparator();
        if (ps === null) {
          setSaving(false);
          return;
        }
        const payload: Record<string, unknown> = {
          strategy,
          chunk_size: Number(chunkSize) || 512,
          chunk_overlap: Number(chunkOverlap) || 0,
          paragraph_separator: ps,
          extra,
        };
        if (bufferSize !== "") payload.buffer_size = Number(bufferSize);
        if (breakpointPercentile !== "")
          payload.breakpoint_percentile_threshold = Number(breakpointPercentile);

        if (isEdit) {
          await client.updateParserConfig(name, payload as any);
        } else {
          await client.createParserConfig({ name: name.trim(), ...payload } as any);
        }
      }
      toast.success(isEdit ? "已更新，配置已热加载生效" : "已创建，配置已热加载生效");
      onSaved();
      onOpenChange(false);
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setSaving(false);
    }
  }

  const providers = kind === "embedder" ? EMBEDDER_PROVIDERS : RERANKER_PROVIDERS;
  const yamlFile =
    kind === "embedder"
      ? "embeddings.yaml"
      : kind === "reranker"
        ? "reranker.yaml"
        : "llamaindex.yaml";
  const roleDesc =
    kind === "embedder"
      ? "Embedding 模型：把文本转成向量，用于文档入库向量化与检索召回。"
      : kind === "reranker"
        ? "Reranker 模型：对向量召回的候选做二次重排，提升检索精度。"
        : "Parser 分块配置：把文档切成多个 chunk 入库，影响检索粒度。";

  const modelNameHint = kind === "embedder"
    ? embedderModelNameHint(provider)
    : rerankerModelNameHint(provider);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>
            {isEdit ? "编辑" : "新建"}
            {kind === "embedder" ? " Embedder" : kind === "reranker" ? " Reranker" : " Parser"}{" "}
            配置
          </DialogTitle>
          <DialogDescription>
            {roleDesc}写入{" "}
            <code className="rounded bg-bg-tertiary px-1 font-mono text-[10px]">
              {yamlFile}
            </code>
            ，修改后立即热加载生效。
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-3">
          {/* ---- 配置名 ---- */}
          <Field
            label="配置名"
            hint="字母开头，仅字母/数字/下划线；编辑时不可修改"
            detail="配置的唯一标识，KB 通过此名引用。命名建议：<provider>_<model>_<size>，如 local_bge_m3 / sentence_512"
            required
          >
            <Input
              value={name}
              onChange={(e) => setName(e.target.value)}
              disabled={isEdit}
              placeholder={
                kind === "embedder"
                  ? "local_bge_m3"
                  : kind === "reranker"
                    ? "local_bge_reranker_base"
                    : "sentence_512"
              }
              className="font-mono text-sm"
            />
          </Field>

          {/* ---- Embedder / Reranker 公共字段 ---- */}
          {kind !== "parser" && (
            <>
              <div className="grid grid-cols-2 gap-3">
                {/* Provider */}
                <Field
                  label="Provider"
                  detail="Embedder/Reranker 的实现后端。openai/bge_zh/cohere 等需对应 API key 或本地模型"
                  required
                >
                  <Select value={provider} onValueChange={setProvider}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {providers.map((p) => (
                        <SelectItem key={p.value} value={p.value}>
                          {/* 主值：trigger 显示 */}
                          <SelectItemText>
                            <span className="font-mono text-xs">{p.value}</span>
                          </SelectItemText>
                          {/* 描述：仅展开时显示 */}
                          <span className="ml-1 text-[10px] text-fg-muted">— {p.desc}</span>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </Field>

                {/* Protocol (reranker) 或 Dimension (embedder) */}
                {kind === "reranker" ? (
                  <Field
                    label="Protocol"
                    hint={provider !== "http" ? "仅 provider=http 时生效" : "HTTP 服务的协议约定"}
                    detail="TEI = HuggingFace TEI 推理服务；Jina = Jina Reranker API；cohere_compat = Cohere 兼容接口；openai = OpenAI 兼容接口"
                  >
                    {provider !== "http" ? (
                      <Input
                        value="不适用"
                        disabled
                        className="font-mono text-sm text-fg-muted"
                      />
                    ) : (
                      <Select value={protocol} onValueChange={setProtocol}>
                        <SelectTrigger>
                          <SelectValue placeholder="选择协议" />
                        </SelectTrigger>
                        <SelectContent>
                          {RERANKER_PROTOCOLS.map((p) => (
                            <SelectItem key={p.value} value={p.value}>
                              {/* 主值：trigger 显示 */}
                              <SelectItemText>
                                <span className="font-mono text-xs">{p.value}</span>
                              </SelectItemText>
                              {/* 描述：仅展开时显示 */}
                              <span className="ml-1 text-[10px] text-fg-muted">— {p.desc}</span>
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    )}
                  </Field>
                ) : (
                  <Field
                    label="Dimension"
                    hint="向量维度"
                    detail="Embedding 输出向量的维度。bge-m3=1024, bge-large=1024, openai-small=1536, openai-large=3072。维度必须与向量库 collection 一致"
                    warn="⚠️ 维度必须与向量库 collection 一致，修改会破坏已有数据"
                  >
                    <Input
                      type="number"
                      value={dimension}
                      onChange={(e) => setDimension(e.target.value)}
                      className="font-mono text-sm"
                      min={0}
                    />
                  </Field>
                )}
              </div>

              {/* 模型名称 */}
              <Field
                label="模型名称"
                hint={modelNameHint.hint}
                detail="具体模型标识。远程 API 填模型 ID（text-embedding-3-small）；本地模型填 HuggingFace ID 或绝对路径（BAAI/bge-m3）"
              >
                <Input
                  value={modelName}
                  onChange={(e) => setModelName(e.target.value)}
                  placeholder={modelNameHint.placeholder}
                  className="font-mono text-sm"
                />
              </Field>

              {/* Batch Size / Top K + 备注 */}
              <div className="grid grid-cols-2 gap-3">
                {kind === "embedder" ? (
                  <Field
                    label="Batch Size"
                    hint="每次推理处理的文本条数"
                    detail="越大吞吐越高但占用更多内存。GPU 推荐 32~64，CPU 推荐 8~16"
                  >
                    <Input
                      type="number"
                      value={batchSize}
                      onChange={(e) => setBatchSize(e.target.value)}
                      className="font-mono text-sm"
                      min={1}
                    />
                  </Field>
                ) : (
                  <Field
                    label="Top K"
                    hint="重排后最终返回条数；向量库召回数 = Top K × over_fetch"
                    detail="Reranker 精排后返回的最终条数。Top K 越大，召回信息越全但 LLM prompt 越长、成本越高。常见值 3~5"
                  >
                    <Input
                      type="number"
                      value={topK}
                      onChange={(e) => setTopK(e.target.value)}
                      className="font-mono text-sm"
                      min={1}
                    />
                  </Field>
                )}
                <Field label="备注" hint="管理员备注，便于团队理解配置用途">
                  <Input
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    placeholder="管理员备注"
                    className="text-sm"
                  />
                </Field>
              </div>
            </>
          )}

          {/* ---- Parser 字段 ---- */}
          {kind === "parser" && (
            <>
              <div className="grid grid-cols-2 gap-3">
                {/* Strategy */}
                <Field
                  label="Strategy"
                  hint="切块策略"
                  detail="whole = 整篇不切（短文档）；sentence = 按句子切（最常用）；semantic = 按语义切（最智能但需 embed_model）；token = 按 token 数切（精确控制大小）"
                  required
                >
                  <Select value={strategy} onValueChange={setStrategy}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {PARSER_STRATEGIES.map((s) => (
                        <SelectItem key={s.value} value={s.value}>
                          {/* 主值：trigger 显示 */}
                          <SelectItemText>
                            <span className="font-mono text-xs">{s.value}</span>
                          </SelectItemText>
                          {/* 描述：仅展开时显示 */}
                          <span className="ml-1 text-[10px] text-fg-muted">— {s.desc}</span>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </Field>
                {/* Chunk Size */}
                <Field
                  label="Chunk Size"
                  hint={`${chunkSizeUnit(strategy)}数；越大粒度越粗、召回更多上下文`}
                  detail="每个 chunk 的目标大小。sentence 策略按字符；token 策略按 token 数。推荐 256~1024。过小粒度细但召回上下文少；过大易超出 LLM 上下文窗口"
                >
                  <Input
                    type="number"
                    value={chunkSize}
                    onChange={(e) => setChunkSize(e.target.value)}
                    className="font-mono text-sm"
                    min={1}
                  />
                </Field>
              </div>

              <div className="grid grid-cols-2 gap-3">
                {/* Chunk Overlap */}
                <Field
                  label="Chunk Overlap"
                  hint={`${chunkSizeUnit(strategy)}数重叠；建议约为 chunk_size 的 10%`}
                  detail="相邻 chunk 之间的重叠量，避免句子被切断后丢失上下文。常见 10%~20%。中文文档建议保持 ≥50 字符重叠"
                >
                  <Input
                    type="number"
                    value={chunkOverlap}
                    onChange={(e) => setChunkOverlap(e.target.value)}
                    className="font-mono text-sm"
                    min={0}
                  />
                </Field>
                {/* Paragraph Separator */}
                <Field
                  label="Paragraph Separator"
                  hint={'JSON 格式字符串，如 "\\n\\n" 代表双换行'}
                  detail='段落分隔符，sentence splitter 按此切分大段。JSON 字符串格式，支持 \n（换行）、\t（制表符）、\\n\\n（双换行）等。常见 "\n\n" 或 "\n"'
                >
                  <Input
                    value={paragraphSeparator}
                    onChange={(e) => setParagraphSeparator(e.target.value)}
                    placeholder='"\\n\\n"'
                    className="font-mono text-sm"
                  />
                </Field>
              </div>

              {/* semantic 策略额外参数 */}
              {strategy === "semantic" && (
                <>
                  <p className="text-[10px] leading-tight text-fg-muted">
                    semantic 策略需要以下额外参数来控制语义切分行为：
                  </p>
                  <div className="grid grid-cols-2 gap-3">
                    <Field
                      label="Buffer Size"
                      hint="合并缓冲句子数；1=保守，5=激进；可空（用默认）"
                      detail="合并相邻句子的窗口大小。越大块越连贯但粒度越粗。中文推荐 1~3，英文推荐 1~5"
                    >
                      <Input
                        type="number"
                        value={bufferSize}
                        onChange={(e) => setBufferSize(e.target.value)}
                        placeholder="留空用默认"
                        className="font-mono text-sm"
                        min={0}
                      />
                    </Field>
                    <Field
                      label="Breakpoint Percentile"
                      hint="语义跳变阈值；60=激进，95=默认，99=极保守；可空"
                      detail="超过此百分位相似度跳变的位置会被切断。越大切得越少块、越保守。中文推荐 95，英文推荐 90"
                    >
                      <Input
                        type="number"
                        value={breakpointPercentile}
                        onChange={(e) => setBreakpointPercentile(e.target.value)}
                        placeholder="留空用默认"
                        className="font-mono text-sm"
                        min={0}
                        max={100}
                      />
                    </Field>
                  </div>
                </>
              )}
            </>
          )}

          {/* ---- Extra 参数 ---- */}
          <Field
            label="Extra 参数（JSON）"
            hint={
              kind === "embedder"
                ? "base_url / api_key / timeout / use_fp16 等，支持 ${VAR} 环境变量"
                : kind === "reranker"
                  ? "base_url / api_key / truncate_input_tokens / batch_size 等，支持 ${VAR} 环境变量"
                  : "separator / use_chinese_splitter（中文建议开启）等"
            }
            detail="任何 Provider/Protocol 特定的配置项都放这里。值支持 ${VAR_NAME} 占位符，从环境变量读取（如 ${OPENAI_API_KEY}）。常见键：base_url、api_key、timeout、use_fp16、truncate_input_tokens"
          >
            <Textarea
              value={extraText}
              onChange={(e) => setExtraText(e.target.value)}
              className="font-mono text-xs"
              rows={5}
              spellCheck={false}
            />
          </Field>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            取消
          </Button>
          <Button onClick={onSave} disabled={saving}>
            {saving && <Loader2 className="size-3.5 animate-spin" />}
            {isEdit ? "保存" : "创建"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
