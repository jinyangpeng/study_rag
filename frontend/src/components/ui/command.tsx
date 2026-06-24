import { Command as CommandPrimitive } from "cmdk";
import { Search } from "lucide-react";
import { type ComponentPropsWithoutRef, type ElementRef, forwardRef } from "react";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { cn } from "@/lib/utils";

export const Command = forwardRef<
  ElementRef<typeof CommandPrimitive>,
  ComponentPropsWithoutRef<typeof CommandPrimitive>
>(({ className, ...props }, ref) => (
  <CommandPrimitive
    ref={ref}
    className={cn(
      "flex h-full w-full flex-col overflow-hidden rounded-lg bg-bg-secondary text-fg",
      className
    )}
    {...props}
  />
));
Command.displayName = "Command";

export interface CommandDialogProps extends ComponentPropsWithoutRef<typeof Dialog> {
  children: React.ReactNode;
}
export function CommandDialog({ children, ...props }: CommandDialogProps) {
  return (
    <Dialog {...props}>
      <DialogContent className="overflow-hidden p-0 shadow-2xl">
        <Command className="[&_[cmdk-group-heading]]:px-3 [&_[cmdk-group-heading]]:font-medium [&_[cmdk-group-heading]]:text-fg-muted [&_[cmdk-group-heading]]:text-[10px] [&_[cmdk-group]]:px-1 [&_[cmdk-input-wrapper]_svg]:size-4 [&_[cmdk-input-wrapper]_svg]:text-fg-muted [&_[cmdk-input]]:h-10 [&_[cmdk-item]]:px-2 [&_[cmdk-item]]:py-2 [&_[cmdk-item]_svg]:size-4">
          {children}
        </Command>
      </DialogContent>
    </Dialog>
  );
}

export const CommandInput = forwardRef<
  ElementRef<typeof CommandPrimitive.Input>,
  ComponentPropsWithoutRef<typeof CommandPrimitive.Input>
>(({ className, ...props }, ref) => (
  <div className="flex items-center border-b border-border px-3" cmdk-input-wrapper="">
    <Search className="mr-2 size-4 shrink-0 text-fg-muted" />
    <CommandPrimitive.Input
      ref={ref}
      className={cn(
        "flex h-9 w-full rounded-md bg-transparent py-2 text-sm outline-none placeholder:text-fg-muted disabled:cursor-not-allowed disabled:opacity-50",
        className
      )}
      {...props}
    />
  </div>
));
CommandInput.displayName = "CommandInput";

export const CommandList = forwardRef<
  ElementRef<typeof CommandPrimitive.List>,
  ComponentPropsWithoutRef<typeof CommandPrimitive.List>
>(({ className, ...props }, ref) => (
  <CommandPrimitive.List
    ref={ref}
    className={cn("max-h-[300px] overflow-y-auto overflow-x-hidden p-1", className)}
    {...props}
  />
));
CommandList.displayName = "CommandList";

export const CommandEmpty = forwardRef<
  ElementRef<typeof CommandPrimitive.Empty>,
  ComponentPropsWithoutRef<typeof CommandPrimitive.Empty>
>((props, ref) => (
  <CommandPrimitive.Empty
    ref={ref}
    className="py-4 text-center text-xs text-fg-muted"
    {...props}
  />
));
CommandEmpty.displayName = "CommandEmpty";

export const CommandGroup = forwardRef<
  ElementRef<typeof CommandPrimitive.Group>,
  ComponentPropsWithoutRef<typeof CommandPrimitive.Group>
>(({ className, ...props }, ref) => (
  <CommandPrimitive.Group
    ref={ref}
    className={cn("overflow-hidden p-1 text-fg", className)}
    {...props}
  />
));
CommandGroup.displayName = "CommandGroup";

export const CommandSeparator = forwardRef<
  ElementRef<typeof CommandPrimitive.Separator>,
  ComponentPropsWithoutRef<typeof CommandPrimitive.Separator>
>(({ className, ...props }, ref) => (
  <CommandPrimitive.Separator
    ref={ref}
    className={cn("-mx-1 h-px bg-border", className)}
    {...props}
  />
));
CommandSeparator.displayName = "CommandSeparator";

export const CommandItem = forwardRef<
  ElementRef<typeof CommandPrimitive.Item>,
  ComponentPropsWithoutRef<typeof CommandPrimitive.Item>
>(({ className, ...props }, ref) => (
  <CommandPrimitive.Item
    ref={ref}
    className={cn(
      "relative flex cursor-default select-none items-center gap-2 rounded px-2 py-1.5 text-sm outline-none aria-selected:bg-bg-tertiary aria-selected:text-fg data-[disabled=true]:pointer-events-none data-[disabled=true]:opacity-50",
      className
    )}
    {...props}
  />
));
CommandItem.displayName = "CommandItem";

export const CommandShortcut = ({ className, ...props }: React.HTMLAttributes<HTMLSpanElement>) => (
  <span
    className={cn("ml-auto text-xs tracking-widest text-fg-muted", className)}
    {...props}
  />
);
CommandShortcut.displayName = "CommandShortcut";
