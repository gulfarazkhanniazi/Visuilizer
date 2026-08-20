export function LoadingOverlay({ message, sub }: { message: string; sub?: string }) {
  return (
    <div className="absolute inset-0 z-20 flex flex-col items-center justify-center gap-4 bg-white/80 backdrop-blur-sm dark:bg-black/70">
      <div className="relative h-12 w-12">
        <div className="absolute inset-0 animate-spin rounded-full border-2 border-zinc-200 border-t-zinc-900 dark:border-zinc-700 dark:border-t-zinc-100" />
      </div>
      <div className="flex flex-col items-center gap-1 text-center">
        <p className="text-sm font-medium text-zinc-900 dark:text-zinc-100">{message}</p>
        {sub && <p className="text-xs text-zinc-500 dark:text-zinc-400">{sub}</p>}
      </div>
    </div>
  );
}
