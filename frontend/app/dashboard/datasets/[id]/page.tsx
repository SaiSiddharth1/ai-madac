"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { apiDownloadCleanedDataset, apiGetDataset, apiGetDatasetProfile } from "@/lib/api";
import ProfileSummary from "@/components/ProfileSummary";
import type { Dataset, DatasetProfile } from "@/types";

export default function DatasetDetailPage() {


  const params = useParams();
  const datasetId = params.id as string;
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [profile, setProfile] = useState<DatasetProfile | null>(null);
  const [loading, setLoading] = useState(true);
  const [downloadError, setDownloadError] = useState("");

  useEffect(() => {
    loadData();
  }, [datasetId]);

  const loadData = async () => {
    try {
      const [ds, prof] = await Promise.allSettled([
        apiGetDataset(datasetId),
        apiGetDatasetProfile(datasetId),
      ]);
      if (ds.status === "fulfilled") setDataset(ds.value);
      if (prof.status === "fulfilled") setProfile(prof.value);
    } catch (err) {
      console.error(err);
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

  if (!dataset) {
    return <div style={{ color: "var(--error)" }}>Dataset not found</div>;
  }

  const cleaning = dataset.schema_info?.cleaning;

  const downloadCleanedFile = async () => {
    try {
      setDownloadError("");
      await apiDownloadCleanedDataset(datasetId, dataset.file_name);
    } catch (err: any) {
      setDownloadError(err.message || "Could not download the cleaned file.");
    }
  };

  return (
    <div className="animate-fade-in">
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "32px" }}>
        <div>
          <h1 style={{ fontSize: "1.5rem", fontWeight: 800, marginBottom: "4px" }}>
            {dataset.name}
          </h1>
          <p style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>
            {dataset.file_name} • {dataset.file_type.toUpperCase()} • Uploaded {new Date(dataset.created_at).toLocaleDateString()}
          </p>
        </div>
        <Link
          href={`/dashboard/datasets/${datasetId}/chat`}
          className="btn-primary"
          style={{ textDecoration: "none", fontSize: "0.85rem" }}
        >
          💬 Ask Questions
        </Link>
      </div>

      {cleaning && (
        <div className="glass-card" style={{ padding: "16px", marginBottom: "24px", border: "1px solid rgba(34, 211, 238, .35)" }}>
          <div style={{ fontWeight: 700, color: "var(--accent)", marginBottom: "5px" }}>✓ Data quality cleanup applied</div>
          <div style={{ color: "var(--text-secondary)", fontSize: ".84rem", marginBottom: "10px" }}>
            {cleaning.clean_rows?.toLocaleString()} usable rows retained from {cleaning.original_rows?.toLocaleString()} source rows. The analysis uses a normalized UTF-8 copy.
          </div>
          <button className="btn-secondary" onClick={downloadCleanedFile} style={{ fontSize: ".8rem" }}>Download cleaned CSV</button>
          {downloadError && <div style={{ color: "var(--error)", fontSize: ".8rem", marginTop: "8px" }}>{downloadError}</div>}
        </div>
      )}

      {/* Stats */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: "12px", marginBottom: "32px" }}>
        <div className="glass-card" style={{ padding: "16px", textAlign: "center" }}>
          <div style={{ fontSize: "1.5rem", fontWeight: 800 }}>{dataset.row_count.toLocaleString()}</div>
          <div style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>Rows</div>
        </div>
        <div className="glass-card" style={{ padding: "16px", textAlign: "center" }}>
          <div style={{ fontSize: "1.5rem", fontWeight: 800 }}>{dataset.column_count}</div>
          <div style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>Columns</div>
        </div>
        <div className="glass-card" style={{ padding: "16px", textAlign: "center" }}>
          <div style={{ fontSize: "1.5rem", fontWeight: 800 }}>{dataset.file_type.toUpperCase()}</div>
          <div style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>Format</div>
        </div>
      </div>

      {/* Schema */}
      {dataset.schema_info?.columns && (
        <div className="glass-card" style={{ padding: "24px", marginBottom: "24px" }}>
          <h2 style={{ fontSize: "1.1rem", fontWeight: 700, marginBottom: "16px" }}>📋 Schema</h2>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr>
                  <th style={thStyle}>Column</th>
                  <th style={thStyle}>Type</th>
                  <th style={thStyle}>Unique</th>
                  <th style={thStyle}>Nullable</th>
                  <th style={thStyle}>Sample Values</th>
                </tr>
              </thead>
              <tbody>
                {dataset.schema_info.columns.map((col: any, i: number) => (
                  <tr key={i}>
                    <td style={tdStyle}><code style={{ color: "var(--primary-light)" }}>{col.name}</code></td>
                    <td style={tdStyle}>{col.dtype}</td>
                    <td style={tdStyle}>{col.unique_count}</td>
                    <td style={tdStyle}>{col.nullable ? "Yes" : "No"}</td>
                    <td style={{ ...tdStyle, color: "var(--text-muted)", fontSize: "0.8rem" }}>
                      {col.sample_values?.slice(0, 3).join(", ")}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Profile */}
      {profile && (
        <div style={{ marginTop: "24px" }}>
          <h2 style={{ fontSize: "1.2rem", fontWeight: 700, marginBottom: "16px" }}>📊 Dataset Profile & Analysis</h2>
          <ProfileSummary profile={profile.profile_json} />
        </div>
      )}

    </div>
  );
}

const thStyle: React.CSSProperties = {
  textAlign: "left",
  padding: "10px 12px",
  fontSize: "0.8rem",
  fontWeight: 600,
  color: "var(--text-muted)",
  borderBottom: "1px solid var(--border)",
};

const tdStyle: React.CSSProperties = {
  padding: "10px 12px",
  fontSize: "0.85rem",
  borderBottom: "1px solid rgba(45, 45, 94, 0.5)",
};
