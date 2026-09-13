"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { apiGetDatasets, apiDeleteDataset } from "@/lib/api";
import type { Dataset } from "@/types";

export default function DatasetsPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadDatasets();
  }, []);

  const loadDatasets = async () => {
    try {
      const data = await apiGetDatasets();
      setDatasets(data.datasets);
    } catch (err) {
      console.error("Failed to load datasets:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id: string, name: string) => {
    if (!confirm(`Delete dataset "${name}"? This cannot be undone.`)) return;
    try {
      await apiDeleteDataset(id);
      setDatasets((prev) => prev.filter((d) => d.id !== id));
    } catch (err: any) {
      alert(err.message);
    }
  };

  return (
    <div className="animate-fade-in">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "32px" }}>
        <div>
          <h1 style={{ fontSize: "1.5rem", fontWeight: 800, marginBottom: "4px" }}>Datasets</h1>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem" }}>
            Manage your uploaded datasets
          </p>
        </div>
        <Link href="/dashboard/datasets/upload" className="btn-primary" style={{ textDecoration: "none", fontSize: "0.85rem" }}>
          + Upload New
        </Link>
      </div>

      {loading ? (
        <div style={{ display: "flex", justifyContent: "center", padding: "60px" }}>
          <div className="spinner" />
        </div>
      ) : datasets.length === 0 ? (
        <div className="glass-card" style={{ textAlign: "center", padding: "60px 20px" }}>
          <div style={{ fontSize: "3rem", marginBottom: "16px" }}>📂</div>
          <h3 style={{ fontWeight: 600, marginBottom: "8px" }}>No datasets uploaded</h3>
          <p style={{ color: "var(--text-muted)", marginBottom: "20px" }}>Upload a CSV or XLSX file to get started</p>
          <Link href="/dashboard/datasets/upload" className="btn-primary" style={{ textDecoration: "none" }}>
            Upload Dataset
          </Link>
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
          {datasets.map((ds) => (
            <div key={ds.id} className="glass-card" style={{ padding: "20px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <Link href={`/dashboard/datasets/${ds.id}`} style={{ textDecoration: "none", color: "inherit", flex: 1 }}>
                <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
                  <div style={{ width: "44px", height: "44px", borderRadius: "12px", background: "rgba(99, 102, 241, 0.15)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "1.2rem" }}>
                    {ds.file_type === "csv" ? "📄" : "📊"}
                  </div>
                  <div>
                    <div style={{ fontWeight: 600, marginBottom: "2px" }}>{ds.name}</div>
                    <div style={{ color: "var(--text-muted)", fontSize: "0.8rem" }}>
                      {ds.row_count.toLocaleString()} rows • {ds.column_count} columns • {ds.file_type.toUpperCase()} • {new Date(ds.created_at).toLocaleDateString()}
                    </div>
                  </div>
                </div>
              </Link>
              <div style={{ display: "flex", gap: "8px" }}>
                <Link href={`/dashboard/datasets/${ds.id}`} style={{ padding: "8px 16px", borderRadius: "8px", border: "1px solid var(--border)", color: "var(--primary-light)", textDecoration: "none", fontSize: "0.8rem", fontWeight: 500 }}>
                  View
                </Link>
                <button onClick={() => handleDelete(ds.id, ds.name)} style={{ padding: "8px 16px", borderRadius: "8px", border: "1px solid rgba(239, 68, 68, 0.3)", background: "rgba(239, 68, 68, 0.1)", color: "#fca5a5", fontSize: "0.8rem", fontWeight: 500, cursor: "pointer" }}>
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
