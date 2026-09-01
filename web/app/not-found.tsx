import Link from "next/link";

export default function NotFound() {
  return (
    <main className="mx-auto max-w-5xl px-4 py-16">
      <p className="text-[11px] tracking-[0.22em] text-dim uppercase">404</p>
      <h1 className="mt-3 text-[22px] text-foreground">Match not found</h1>
      <p className="mt-2 text-[13px] text-muted">
        That id is not in the mock feed.
      </p>
      <Link
        href="/"
        className="mt-6 inline-block text-[12px] tracking-[0.14em] text-muted uppercase hover:text-foreground"
      >
        ← Board
      </Link>
    </main>
  );
}
