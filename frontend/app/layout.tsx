import type { Metadata } from "next";
import Script from "next/script";
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
      <head>
        <Script
          src="https://cdn.plot.ly/plotly-2.35.2.min.js"
          strategy="afterInteractive"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
