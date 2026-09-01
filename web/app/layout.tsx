import type { Metadata } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans } from "next/font/google";
import { AppHeader } from "@/components/AppHeader";
import "./globals.css";

const plexSans = IBM_Plex_Sans({
  variable: "--font-plex-sans",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  display: "swap",
});

const plexMono = IBM_Plex_Mono({
  variable: "--font-plex-mono",
  subsets: ["latin"],
  weight: ["400", "500"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Predictor",
  description: "Dota 2 betting match terminal",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${plexSans.variable} ${plexMono.variable} h-full antialiased`}
    >
      <body className="min-h-full font-sans">
        <AppHeader />
        {children}
      </body>
    </html>
  );
}
