"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { apiGetDatasets } from "@/lib/api";
import type { Dataset } from "@/types";

export default function DashboardPage() {
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

  return (
    <div className="animate-fade-in">
      {/* Page Header */}
      <div style={{ marginBottom: "32px" }}>
        <h1 style={{ fontSize: "1.75rem", fontWeight: 800, marginBottom: "8px" }}>
          Dashboard
        </h1>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem" }}>
          Welcome to your AI-powered data analysis workspace
        </p>
      </div>

      {/* Quick Stats */}
      <div className="stats-grid">
        <div className="stat-card glass-card">
          <div className="stat-icon" style={{ background: "rgba(99, 102, 241, 0.15)" }}>📁</div>
          <div>
            <div className="stat-value">{datasets.length}</div>
            <div className="stat-label">Datasets</div>
          </div>
        </div>
        <div className="stat-card glass-card">
          <div className="stat-icon" style={{ background: "rgba(6, 182, 212, 0.15)" }}>💬</div>
          <div>
            <div className="stat-value">—</div>
            <div className="stat-label">Queries Today</div>
          </div>
        </div>
        <div className="stat-card glass-card">
          <div className="stat-icon" style={{ background: "rgba(34, 197, 94, 0.15)" }}>🤖</div>
          <div>
            <div className="stat-value">—</div>
            <div className="stat-label">ML Models</div>
          </div>
        </div>
        <div className="stat-card glass-card">
          <div className="stat-icon" style={{ background: "rgba(245, 158, 11, 0.15)" }}>📋</div>
          <div>
            <div className="stat-value">—</div>
            <div className="stat-label">Reports</div>
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div style={{ marginTop: "32px", marginBottom: "24px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h2 style={{ fontSize: "1.15rem", fontWeight: 600 }}>Your Datasets</h2>
        <Link href="/dashboard/datasets/upload" className="btn-primary" style={{ fontSize: "0.85rem", padding: "10px 20px", textDecoration: "none" }}>
          + Upload Dataset
        </Link>
      </div>

      {/* Datasets List */}
      {loading ? (
        <div style={{ display: "flex", justifyContent: "center", padding: "60px 0" }}>
          <div className="spinner" />
        </div>
      ) : datasets.length === 0 ? (
        <div className="empty-state glass-card">
          <div style={{ fontSize: "3rem", marginBottom: "16px" }}>📊</div>
          <h3 style={{ fontSize: "1.1rem", fontWeight: 600, marginBottom: "8px" }}>
            No datasets yet
          </h3>
          <p style={{ color: "var(--text-muted)", fontSize: "0.9rem", marginBottom: "20px" }}>
            Upload your first CSV or Excel file to start analyzing
          </p>
          <Link href="/dashboard/datasets/upload" className="btn-primary" style={{ textDecoration: "none" }}>
            Upload Dataset
          </Link>
        </div>
      ) : (
        <div className="datasets-grid">
          {datasets.map((ds) => (
            <Link
              key={ds.id}
              href={`/dashboard/datasets/${ds.id}`}
              className="dataset-card glass-card"
              style={{ textDecoration: "none", color: "inherit" }}
            >
              <div className="dataset-card-header">
                <div className="dataset-type-badge">
                  {ds.file_type.toUpperCase()}
                </div>
                <span style={{ color: "var(--text-muted)", fontSize: "0.75rem" }}>
                  {new Date(ds.created_at).toLocaleDateString()}
                </span>
              </div>
              <h3 style={{ fontSize: "1rem", fontWeight: 600, marginBottom: "8px" }}>
                {ds.name}
              </h3>
              <div className="dataset-meta">
                <span>{ds.row_count.toLocaleString()} rows</span>
                <span>•</span>
                <span>{ds.column_count} columns</span>
              </div>
            </Link>
          ))}
        </div>
      )}

      <style jsx>{`
        .stats-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 16px;
        }
        .stat-card {
          padding: 20px;
          display: flex;
          align-items: center;
          gap: 16px;
        }
        .stat-icon {
          width: 48px;
          height: 48px;
          border-radius: 12px;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 1.4rem;
          flex-shrink: 0;
        }
        .stat-value {
          font-size: 1.5rem;
          font-weight: 800;
          color: var(--text-primary);
        }
        .stat-label {
          font-size: 0.8rem;
          color: var(--text-muted);
          font-weight: 500;
        }
        .empty-state {
          text-align: center;
          padding: 60px 20px;
        }
        .datasets-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
          gap: 16px;
        }
        .dataset-card {
          padding: 20px;
          cursor: pointer;
        }
        .dataset-card-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 12px;
        }
        .dataset-type-badge {
          background: rgba(99, 102, 241, 0.15);
          color: var(--primary-light);
          padding: 4px 10px;
          border-radius: 6px;
          font-size: 0.7rem;
          font-weight: 700;
          letter-spacing: 0.5px;
        }
        .dataset-meta {
          display: flex;
          gap: 8px;
          font-size: 0.8rem;
          color: var(--text-muted);
        }
      `}</style>
    </div>
  );
}
