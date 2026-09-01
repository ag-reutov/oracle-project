import Link from "next/link";

export function AppHeader() {
  return (
    <header className="border-b border-line">
      <div className="mx-auto flex h-9 max-w-[1100px] items-center justify-between px-4">
        <Link
          href="/"
          className="text-[12px] font-medium tracking-[0.2em] text-foreground"
        >
          PREDICTOR
        </Link>
        <p className="text-[10px] tracking-[0.16em] text-dim uppercase">
          Mock feed · v0.1
        </p>
      </div>
    </header>
  );
}
