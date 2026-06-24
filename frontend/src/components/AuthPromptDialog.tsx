/**
 * AuthPromptDialog — 鉴权引导弹窗
 *
 * 当服务端启用了 STUDY_RAG_ADMIN_TOKEN，但客户端没配 token 时，
 * 通过 ApiProvider 的 promptAuth() 触发此弹窗，引导用户去填写 token。
 *
 * 替代原 antd Modal.confirm + Input.Password 组合。
 */
import { useEffect, useState } from "react";
import { KeyRound, ExternalLink } from "lucide-react";
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

export interface AuthPromptState {
  /** Dialog 打开状态 */
  open: boolean;
  /** 用户输入的 token（受控） */
  value: string;
  /** 用户保存了 token（resolve(true)） */
  onSave: () => void;
  /** 用户点了取消或关闭（resolve(false)） */
  onCancel: () => void;
  /** 跳到设置页（同时关闭此弹窗） */
  onGoToSettings: () => void;
}

interface Props {
  state: AuthPromptState;
}

export function AuthPromptDialog({ state }: Props) {
  const [local, setLocal] = useState(state.value);

  useEffect(() => {
    setLocal(state.value);
  }, [state.value, state.open]);

  return (
    <Dialog
      open={state.open}
      onOpenChange={(o) => {
        if (!o) state.onCancel();
      }}
    >
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <KeyRound className="size-3.5 text-warning" />
            需要 Admin Token
          </DialogTitle>
          <DialogDescription>
            服务端启用了 <code className="rounded bg-bg-tertiary px-1 text-[10px]">STUDY_RAG_ADMIN_TOKEN</code>，
            当前前端未配置 Token。输入后保存，或先去「设置」页面填写。
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-1">
          <Label className="text-xs">Bearer token</Label>
          <Input
            type="password"
            value={local}
            onChange={(e) => {
              setLocal(e.target.value);
              state.value = e.target.value;
            }}
            placeholder="sk-..."
            autoFocus
            className="font-mono"
            onKeyDown={(e) => {
              if (e.key === "Enter" && state.value) {
                state.onSave();
              }
            }}
          />
        </div>
        <DialogFooter className="gap-2">
          <Button variant="ghost" onClick={state.onCancel}>
            取消
          </Button>
          <Button variant="outline" onClick={state.onGoToSettings}>
            <ExternalLink className="size-3.5" />
            去设置
          </Button>
          <Button onClick={state.onSave} disabled={!state.value}>
            保存
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
