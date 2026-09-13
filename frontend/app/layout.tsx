import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI MADAC — Autonomous Data Science Analyst",
  description:
    "AI-powered multi-agent platform for enterprise dataset analysis, visualization, and predictions.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
