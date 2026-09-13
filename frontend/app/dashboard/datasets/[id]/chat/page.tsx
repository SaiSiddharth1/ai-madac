"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { apiGetDataset } from "@/lib/api";
import ChatInterface from "@/components/ChatInterface";
import type { Dataset } from "@/types";

export default function ChatPage() {
  const params = useParams();
  const datasetId = params.id as string;
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadDataset();
  }, [datasetId]);

  const loadDataset = async () => {
    try {
      const data = await apiGetDataset(datasetId);
      setDataset(data);
    } catch (err) {
      console.error("Failed to load dataset:", err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div style={{ display: "flex", justifyContent: "center", padding: "60px" }}>
        <div className="spinner" />
      </div>
    );
  }

  return (
    <div className="animate-fade-in" style={{ maxWidth: "1000px" }}>
      <div style={{ marginBottom: "20px" }}>
        <h1 style={{ fontSize: "1.5rem", fontWeight: 800, marginBottom: "4px" }}>
          💬 Ask AI Analyst
        </h1>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem" }}>
          Dataset: <strong style={{ color: "var(--primary-light)" }}>{dataset?.name}</strong> ({dataset?.row_count.toLocaleString()} rows)
        </p>
      </div>

      <ChatInterface datasetId={datasetId} datasetName={dataset?.name || "Dataset"} />
    </div>
  );
}
