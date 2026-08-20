"use client";

import { useMemo, useState } from "react";
import Image from "next/image";
import { CatalogItem } from "@/lib/types";

export function DesignPicker({
  title,
  items,
  selectedId,
  onSelect,
}: {
  title: string;
  items: CatalogItem[];
  selectedId?: string;
  onSelect: (id: string) => void;
}) {
  const [query, setQuery] = useState("");

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return items;
    return items.filter((item) => item.label.toLowerCase().includes(q));
  }, [items, query]);

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">{title}</h3>
        <span className="text-xs text-zinc-400">
          {filtered.length} design{filtered.length === 1 ? "" : "s"}
        </span>
      </div>

      {items.length > 8 && (
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search designs…"
          className="w-full rounded-lg border border-zinc-200 bg-white px-3 py-1.5 text-sm text-zinc-900 outline-none focus:border-zinc-400 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-100 dark:focus:border-zinc-500"
        />
      )}

      <div className="grid max-h-80 grid-cols-3 gap-2 overflow-y-auto pr-1">
        {filtered.map((item) => {
          const active = item.id === selectedId;
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => onSelect(item.id)}
              title={item.label}
              className={`group relative aspect-square overflow-hidden rounded-lg border-2 text-left transition-colors ${
                active
                  ? "border-zinc-900 dark:border-zinc-100"
                  : "border-transparent hover:border-zinc-300 dark:hover:border-zinc-700"
              }`}
            >
              <Image
                src={item.src}
                alt={item.label}
                fill
                sizes="140px"
                loading="lazy"
                className="object-cover"
              />
              {active && (
                <span className="absolute right-1 top-1 flex h-5 w-5 items-center justify-center rounded-full bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900">
                  <svg viewBox="0 0 20 20" fill="currentColor" className="h-3.5 w-3.5">
                    <path
                      fillRule="evenodd"
                      d="M16.704 5.29a1 1 0 0 1 0 1.415l-7.5 7.5a1 1 0 0 1-1.414 0l-3.5-3.5a1 1 0 1 1 1.414-1.414L8.5 12.086l6.79-6.796a1 1 0 0 1 1.414 0Z"
                      clipRule="evenodd"
                    />
                  </svg>
                </span>
              )}
              <span className="absolute inset-x-0 bottom-0 truncate bg-black/55 px-1.5 py-1 text-[10px] font-medium text-white opacity-0 transition-opacity group-hover:opacity-100">
                {item.label}
              </span>
            </button>
          );
        })}
        {filtered.length === 0 && (
          <p className="col-span-3 py-6 text-center text-sm text-zinc-400">
            No designs match &ldquo;{query}&rdquo;
          </p>
        )}
      </div>
    </div>
  );
}
