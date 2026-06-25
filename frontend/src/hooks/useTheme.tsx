/**
 * useTheme — 系统主题（dark / light）状态管理
 *
 * 设计：
 *  - 主题值存到 localStorage（key: study_rag_theme），刷新后保留
 *  - 用 React Context 让 Topbar、Settings 共享同一个状态
 *  - 应用主题通过 <html> 上的 .dark / .light className 切换（见 tailwind.config.ts）
 *  - 解析优先级：localStorage > 系统 prefers-color-scheme > 默认 dark
 *
 * 配套：index.html 的 <head> 有同步内联脚本，在 React 加载前就把主题 class
 * 应用到 <html>，避免首屏闪烁。
 */
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

export type Theme = "dark" | "light";

export const THEME_STORAGE_KEY = "study_rag_theme";

/** SSR-safe：从 localStorage 读已存主题，没有就跟随系统，没有就 dark。 */
export function getStoredTheme(): Theme {
  if (typeof window === "undefined") return "dark";
  try {
    const v = localStorage.getItem(THEME_STORAGE_KEY);
    if (v === "dark" || v === "light") return v;
  } catch {
    // localStorage 可能被禁用（隐私模式 / 配额满），fall through
  }
  if (window.matchMedia?.("(prefers-color-scheme: light)").matches) {
    return "light";
  }
  return "dark";
}

/** 把主题写到 <html> 的 className（同时维护互斥的 .dark / .light）。 */
export function applyTheme(theme: Theme): void {
  if (typeof document === "undefined") return;
  const el = document.documentElement;
  el.classList.toggle("dark", theme === "dark");
  el.classList.toggle("light", theme === "light");
}

interface ThemeContextValue {
  theme: Theme;
  setTheme: (t: Theme) => void;
  toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextValue | null>(null);

export function ThemeProvider({ children }: { children: ReactNode }) {
  // 初始化：直接读 localStorage（index.html 同步脚本保证 <html> class 已对）
  const [theme, setThemeState] = useState<Theme>(() => getStoredTheme());

  // mount 时再 apply 一次兜底（防止有人手动改了 <html> class）
  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  const setTheme = useCallback((t: Theme) => {
    setThemeState(t);
    try {
      localStorage.setItem(THEME_STORAGE_KEY, t);
    } catch {
      // 忽略：隐私模式 / 配额满等情况
    }
  }, []);

  const toggleTheme = useCallback(() => {
    setTheme(theme === "dark" ? "light" : "dark");
  }, [theme, setTheme]);

  const value = useMemo<ThemeContextValue>(
    () => ({ theme, setTheme, toggleTheme }),
    [theme, setTheme, toggleTheme]
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme(): ThemeContextValue {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme must be used within ThemeProvider");
  return ctx;
}
