"use client";

import { useState, useRef } from "react";
import { useRouter } from "next/navigation";
import { apiUploadDataset } from "@/lib/api";

export default function UploadDatasetPage() {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [name, setName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") setDragActive(true);
    if (e.type === "dragleave") setDragActive(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    const dropped = e.dataTransfer.files?.[0];
    if (dropped) validateAndSetFile(dropped);
  };

  const validateAndSetFile = (f: File) => {
    const ext = f.name.split(".").pop()?.toLowerCase();
    if (ext !== "csv" && ext !== "xlsx") {
      setError("Only CSV and XLSX files are supported");
      return;
    }
    if (f.size > 50 * 1024 * 1024) {
      setError("File exceeds 50MB limit");
      return;
    }
    setError("");
    setFile(f);
    if (!name) setName(f.name.replace(/\.[^/.]+$/, ""));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    setLoading(true);
    setError("");

    try {
      const result = await apiUploadDataset(file, name);
      router.push(`/dashboard/datasets/${result.dataset.id}`);
    } catch (err: any) {
      setError(err.message || "Upload failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="animate-fade-in" style={{ maxWidth: "640px" }}>
      <h1 style={{ fontSize: "1.5rem", fontWeight: 800, marginBottom: "4px" }}>Upload Dataset</h1>
      <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem", marginBottom: "32px" }}>
        Upload a CSV or Excel file to begin analysis
      </p>

      <form onSubmit={handleSubmit}>
        {error && (
          <div style={{ background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.3)", color: "#fca5a5", padding: "10px 14px", borderRadius: "10px", fontSize: "0.85rem", marginBottom: "16px" }}>
            {error}
          </div>
        )}

        {/* Drop Zone */}
        <div
          className={`drop-zone glass-card ${dragActive ? "active" : ""} ${file ? "has-file" : ""}`}
          onDragEnter={handleDrag}
          onDragOver={handleDrag}
          onDragLeave={handleDrag}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,.xlsx"
            style={{ display: "none" }}
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) validateAndSetFile(f);
            }}
          />
          {file ? (
            <div style={{ textAlign: "center" }}>
              <div style={{ fontSize: "2.5rem", marginBottom: "12px" }}>✅</div>
              <div style={{ fontWeight: 600, marginBottom: "4px" }}>{file.name}</div>
              <div style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>
                {(file.size / 1024).toFixed(1)} KB • Click to change
              </div>
            </div>
          ) : (
            <div style={{ textAlign: "center" }}>
              <div style={{ fontSize: "2.5rem", marginBottom: "12px" }}>📁</div>
              <div style={{ fontWeight: 600, marginBottom: "4px" }}>
                Drop your file here or click to browse
              </div>
              <div style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>
                Supports CSV and XLSX • Max 50MB
              </div>
            </div>
          )}
        </div>

        {/* Dataset Name */}
        <div style={{ marginTop: "20px" }}>
          <label style={{ display: "block", fontSize: "0.85rem", fontWeight: 500, color: "var(--text-secondary)", marginBottom: "6px" }}>
            Dataset Name
          </label>
          <input
            type="text"
            className="input-field"
            placeholder="e.g., Finance Data Q3 2024"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
        </div>

        {/* Submit */}
        <button type="submit" className="btn-primary" style={{ width: "100%", marginTop: "24px" }} disabled={!file || loading}>
          {loading ? (
            <>
              <span className="spinner" style={{ width: 20, height: 20 }} /> Uploading & Processing...
            </>
          ) : (
            "Upload & Analyze"
          )}
        </button>
      </form>

      <style jsx>{`
        .drop-zone {
          padding: 48px 24px;
          cursor: pointer;
          text-align: center;
          transition: all 0.3s ease;
          border-style: dashed !important;
        }
        .drop-zone:hover, .drop-zone.active {
          border-color: var(--primary) !important;
          background: rgba(99, 102, 241, 0.05);
        }
        .drop-zone.has-file {
          border-color: var(--success) !important;
          border-style: solid !important;
        }
      `}</style>
    </div>
  );
}
