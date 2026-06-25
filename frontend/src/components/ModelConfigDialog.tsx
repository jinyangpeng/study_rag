/**
 * ModelConfigDialog — Embedder / Reranker / Parser 配置新增/编辑对话框
 *
 * 复用同一个组件，根据 kind 切换字段：
 *  - embedder:  provider / model_name / dimension / batch_size / extra / description
 *  - reranker:  provider / protocol / model_name / top_k / extra / description
 *  - parser:    strategy / chunk_size / chunk_overlap / paragraph_separator / buffer_size / breakpoint_percentile_threshold / extra
 *
 * extra 用 JSON 文本框编辑（保留灵活性，支持任意键值）。
 */
import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";
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
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
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

const EMBEDDER_PROVIDERS = [
  "mock",
  "openai",
  "bge",
  "bge_zh",
  "fastembed",
  "azure_openai",
];
const RERANKER_PROVIDERS = ["none", "mock", "http", "bge", "cohere"];
const RERANKER_PROTOCOLS = ["tei", "jina", "cohere_compat", "openai"];
const PARSER_STRATEGIES = ["whole", "sentence", "semantic", "token"];

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
        setProtocol(r.protocol || "tei");
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
      toast.error("段落分隔符需为 JSON 字符串，如 \"\\n\\n\"");
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
        if (isEdit) {
          await client.updateRerankerConfig(name, {
            provider,
            protocol,
            model_name: modelName,
            top_k: Number(topK) || 5,
            description,
            extra,
          });
        } else {
          await client.createRerankerConfig({
            name: name.trim(),
            provider,
            protocol,
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
      toast.success(isEdit ? "已更新" : "已创建");
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
            ，修改后需重启服务生效。
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-3">
          <div className="space-y-1">
            <Label className="text-xs">配置名</Label>
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
          </div>

          {/* Embedder / Reranker 公共字段 */}
          {kind !== "parser" && (
            <>
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <Label className="text-xs">
                    Provider
                    <span className="ml-1 font-normal text-fg-muted">（实现类型）</span>
                  </Label>
                  <Select value={provider} onValueChange={setProvider}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {providers.map((p) => (
                        <SelectItem key={p} value={p}>
                          <span className="font-mono text-xs">{p}</span>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {kind === "reranker" ? (
                  <div className="space-y-1">
                    <Label className="text-xs">
                      Protocol
                      <span className="ml-1 font-normal text-fg-muted">（仅 http）</span>
                    </Label>
                    <Select
                      value={protocol}
                      onValueChange={setProtocol}
                      disabled={provider !== "http"}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {RERANKER_PROTOCOLS.map((p) => (
                          <SelectItem key={p} value={p}>
                            <span className="font-mono text-xs">{p}</span>
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                ) : (
                  <div className="space-y-1">
                    <Label className="text-xs">
                      Dimension
                      <span className="ml-1 font-normal text-fg-muted">（向量维度）</span>
                    </Label>
                    <Input
                      type="number"
                      value={dimension}
                      onChange={(e) => setDimension(e.target.value)}
                      className="font-mono text-sm"
                    />
                  </div>
                )}
              </div>

              <div className="space-y-1">
                <Label className="text-xs">模型名称</Label>
                <Input
                  value={modelName}
                  onChange={(e) => setModelName(e.target.value)}
                  placeholder="BAAI/bge-reranker-base"
                  className="font-mono text-sm"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                {kind === "embedder" ? (
                  <div className="space-y-1">
                    <Label className="text-xs">
                      Batch Size
                      <span className="ml-1 font-normal text-fg-muted">（批推理数）</span>
                    </Label>
                    <Input
                      type="number"
                      value={batchSize}
                      onChange={(e) => setBatchSize(e.target.value)}
                      className="font-mono text-sm"
                    />
                  </div>
                ) : (
                  <div className="space-y-1">
                    <Label className="text-xs">
                      Top K
                      <span className="ml-1 font-normal text-fg-muted">（重排后保留数）</span>
                    </Label>
                    <Input
                      type="number"
                      value={topK}
                      onChange={(e) => setTopK(e.target.value)}
                      className="font-mono text-sm"
                    />
                  </div>
                )}
                <div className="space-y-1">
                  <Label className="text-xs">描述</Label>
                  <Input
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    placeholder="管理员备注"
                    className="text-sm"
                  />
                </div>
              </div>
            </>
          )}

          {/* Parser 字段 */}
          {kind === "parser" && (
            <>
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <Label className="text-xs">
                    Strategy
                    <span className="ml-1 font-normal text-fg-muted">（切块策略）</span>
                  </Label>
                  <Select value={strategy} onValueChange={setStrategy}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {PARSER_STRATEGIES.map((s) => (
                        <SelectItem key={s} value={s}>
                          <span className="font-mono text-xs">{s}</span>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1">
                  <Label className="text-xs">
                    Chunk Size
                    <span className="ml-1 font-normal text-fg-muted">（块大小）</span>
                  </Label>
                  <Input
                    type="number"
                    value={chunkSize}
                    onChange={(e) => setChunkSize(e.target.value)}
                    className="font-mono text-sm"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <Label className="text-xs">
                    Chunk Overlap
                    <span className="ml-1 font-normal text-fg-muted">（块重叠）</span>
                  </Label>
                  <Input
                    type="number"
                    value={chunkOverlap}
                    onChange={(e) => setChunkOverlap(e.target.value)}
                    className="font-mono text-sm"
                  />
                </div>
                <div className="space-y-1">
                  <Label className="text-xs">
                    Paragraph Separator
                    <span className="ml-1 font-normal text-fg-muted">（JSON 字符串）</span>
                  </Label>
                  <Input
                    value={paragraphSeparator}
                    onChange={(e) => setParagraphSeparator(e.target.value)}
                    placeholder='"\\n\\n"'
                    className="font-mono text-sm"
                  />
                </div>
              </div>

              {strategy === "semantic" && (
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-1">
                    <Label className="text-xs">
                      Buffer Size
                      <span className="ml-1 font-normal text-fg-muted">（滑动窗口，可空）</span>
                    </Label>
                    <Input
                      type="number"
                      value={bufferSize}
                      onChange={(e) => setBufferSize(e.target.value)}
                      placeholder="留空用默认"
                      className="font-mono text-sm"
                    />
                  </div>
                  <div className="space-y-1">
                    <Label className="text-xs">
                      Breakpoint Percentile
                      <span className="ml-1 font-normal text-fg-muted">（0-100，可空）</span>
                    </Label>
                    <Input
                      type="number"
                      value={breakpointPercentile}
                      onChange={(e) => setBreakpointPercentile(e.target.value)}
                      placeholder="留空用默认"
                      className="font-mono text-sm"
                    />
                  </div>
                </div>
              )}
            </>
          )}

          <div className="space-y-1">
            <Label className="text-xs">
              Extra 参数（JSON）
              <span className="ml-1 font-normal text-fg-muted">
                {kind === "parser"
                  ? "separator / use_chinese_splitter 等"
                  : "base_url / api_key / timeout 等"}
              </span>
            </Label>
            <Textarea
              value={extraText}
              onChange={(e) => setExtraText(e.target.value)}
              className="font-mono text-xs"
              rows={5}
              spellCheck={false}
            />
          </div>
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
