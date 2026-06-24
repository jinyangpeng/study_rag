import { cva, type VariantProps } from "class-variance-authority";
import { type HTMLAttributes, forwardRef } from "react";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded-sm border px-1.5 py-0.5 text-[10px] font-medium transition-colors focus:outline-none",
  {
    variants: {
      variant: {
        default: "border-transparent bg-accent text-accent-fg",
        secondary: "border-transparent bg-bg-tertiary text-fg",
        outline: "border-border text-fg",
        success: "border-transparent bg-success/15 text-success",
        warning: "border-transparent bg-warning/15 text-warning",
        danger: "border-transparent bg-danger/15 text-danger",
        muted: "border-transparent bg-bg-tertiary text-fg-muted",
      },
    },
    defaultVariants: { variant: "default" },
  }
);

export interface BadgeProps
  extends HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

export const Badge = forwardRef<HTMLSpanElement, BadgeProps>(
  ({ className, variant, ...props }, ref) => (
    <span
      ref={ref}
      className={cn(badgeVariants({ variant }), className)}
      {...props}
    />
  )
);
Badge.displayName = "Badge";

export { badgeVariants };
