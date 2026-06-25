import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

/** 字符数：< 1000 直接显示，>= 1000 切到 K 单位。 */
export function formatChars(chars: number): string {
  if (chars < 1000) return `${chars}`;
  if (chars < 10_000) return `${(chars / 1000).toFixed(1)}K`;
  return `${Math.round(chars / 1000)}K`;
}

export function formatRelativeTime(date: string | Date): string {
  const now = Date.now();
  const ts = new Date(date).getTime();
  const diff = (now - ts) / 1000;
  if (diff < 60) return `${Math.floor(diff)}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return new Date(date).toLocaleDateString();
}
