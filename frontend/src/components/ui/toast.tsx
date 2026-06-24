"use client";

import * as ToastPrimitives from "@radix-ui/react-toast";
import { cva, type VariantProps } from "class-variance-authority";
import { X } from "lucide-react";
import {
  forwardRef,
  type ComponentPropsWithoutRef,
  type ElementRef,
  type HTMLAttributes,
} from "react";
import { cn } from "@/lib/utils";

export const ToastProvider = ToastPrimitives.Provider;

export const ToastViewport = forwardRef<
  ElementRef<typeof ToastPrimitives.Viewport>,
  ComponentPropsWithoutRef<typeof ToastPrimitives.Viewport>
>(({ className, ...props }, ref) => (
  <ToastPrimitives.Viewport
    ref={ref}
    className={cn(
      "fixed bottom-0 right-0 z-[100] flex max-h-screen w-full flex-col-reverse gap-2 p-4 sm:max-w-[420px]",
      className
    )}
    {...props}
  />
));
ToastViewport.displayName = "ToastViewport";

const toastVariants = cva(
  "group pointer-events-auto relative flex w-full items-center justify-between gap-2 overflow-hidden rounded border border-border bg-bg-secondary p-3 pr-6 shadow-lg transition-all",
  {
    variants: {
      variant: {
        default: "border-border",
        success: "border-success/40",
        warning: "border-warning/40",
        danger: "border-danger/40",
      },
    },
    defaultVariants: { variant: "default" },
  }
);

export const Toast = forwardRef<
  ElementRef<typeof ToastPrimitives.Root>,
  ComponentPropsWithoutRef<typeof ToastPrimitives.Root> & VariantProps<typeof toastVariants>
>(({ className, variant, ...props }, ref) => (
  <ToastPrimitives.Root
    ref={ref}
    className={cn(toastVariants({ variant }), className)}
    {...props}
  />
));
Toast.displayName = "Toast";

export const ToastTitle = forwardRef<
  ElementRef<typeof ToastPrimitives.Title>,
  ComponentPropsWithoutRef<typeof ToastPrimitives.Title>
>(({ className, ...props }, ref) => (
  <ToastPrimitives.Title
    ref={ref}
    className={cn("text-sm font-medium text-fg", className)}
    {...props}
  />
));
ToastTitle.displayName = "ToastTitle";

export const ToastDescription = forwardRef<
  ElementRef<typeof ToastPrimitives.Description>,
  ComponentPropsWithoutRef<typeof ToastPrimitives.Description>
>(({ className, ...props }, ref) => (
  <ToastPrimitives.Description
    ref={ref}
    className={cn("text-xs text-fg-muted", className)}
    {...props}
  />
));
ToastDescription.displayName = "ToastDescription";

export const ToastClose = forwardRef<
  ElementRef<typeof ToastPrimitives.Close>,
  ComponentPropsWithoutRef<typeof ToastPrimitives.Close>
>(({ className, ...props }, ref) => (
  <ToastPrimitives.Close
    ref={ref}
    className={cn(
      "absolute right-1 top-1 rounded p-1 text-fg-muted hover:text-fg",
      className
    )}
    toast-close=""
    {...props}
  >
    <X className="size-3.5" />
  </ToastPrimitives.Close>
));
ToastClose.displayName = "ToastClose";

export type ToastProps = ComponentPropsWithoutRef<typeof Toast>;
export type ToastActionElement = React.ReactElement;

export function ToasterPortal({ children }: HTMLAttributes<HTMLDivElement>) {
  return <>{children}</>;
}
