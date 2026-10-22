"use client";

import React, { useMemo, useState } from "react";

interface InteractiveTableProps {
  columns: string[];
  rows: Record<string, any>[];
  totalRowCount?: number;
  explanation?: {
    what_this_shows?: string;
    parameters_used?: string[];
    key_insights?: string[];
  } | null;
  onToggleView?: () => void;
  isChartViewAvailable?: boolean;
  isChartViewActive?: boolean;
}

export default function InteractiveTable({
  columns,
  rows,
  totalRowCount,
  explanation,
  onToggleView,
  isChartViewAvailable = false,
  isChartViewActive = false,
}: InteractiveTableProps) {
  const [searchTerm, setSearchTerm] = useState("");
  const [sortCol, setSortCol] = useState<string | null>(null);
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);

  // Filter rows based on search term
  const filteredRows = useMemo(() => {
    if (!searchTerm.trim()) return rows;
    const term = searchTerm.toLowerCase();
    return rows.filter((r) =>
      columns.some((col) => String(r[col] ?? "").toLowerCase().includes(term))
    );
  }, [rows, columns, searchTerm]);

  // Sort rows
  const sortedRows = useMemo(() => {
    if (!sortCol) return filteredRows;
    return [...filteredRows].sort((a, b) => {
      const valA = a[sortCol];
      const valB = b[sortCol];

      // Handle nulls
      if (valA === null || valA === undefined) return 1;
      if (valB === null || valB === undefined) return -1;

      // Handle numbers
      const numA = Number(valA);
      const numB = Number(valB);
      if (!isNaN(numA) && !isNaN(numB)) {
        return sortDir === "asc" ? numA - numB : numB - numA;
      }

      // Strings
      const strA = String(valA).toLowerCase();
      const strB = String(valB).toLowerCase();
      if (strA < strB) return sortDir === "asc" ? -1 : 1;
      if (strA > strB) return sortDir === "asc" ? 1 : -1;
      return 0;
    });
  }, [filteredRows, sortCol, sortDir]);

  // Pagination
  const totalPages = Math.ceil(sortedRows.length / pageSize) || 1;
  const paginatedRows = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return sortedRows.slice(start, start + pageSize);
  }, [sortedRows, currentPage, pageSize]);

  const handleSort = (col: string) => {
    if (sortCol === col) {
      setSortDir(sortDir === "asc" ? "desc" : "asc");
    } else {
      setSortCol(col);
      setSortDir("desc");
    }
    setCurrentPage(1);
  };

  const exportToCsv = () => {
    if (rows.length === 0) return;
    const header = columns.join(",");
    const csvRows = sortedRows.map((r) =>
      columns.map((col) => `"${String(r[col] ?? "").replace(/"/g, '""')}"`).join(",")
    );
    const csvContent = "data:text/csv;charset=utf-8," + [header, ...csvRows].join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "query_results.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const formatCell = (val: any) => {
    if (val === null || val === undefined) return <span style={{ color: "#64748b" }}>null</span>;
    if (typeof val === "number") {
      return Number.isInteger(val) ? val.toLocaleString() : val.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    }
    return String(val);
  };

  return (
    <div
      style={{
        margin: "16px 0",
        borderRadius: "14px",
        background: "#0f1326",
        border: "1px solid var(--border)",
        overflow: "hidden",
      }}
    >
      {/* Table Toolbar */}
      <div
        style={{
          padding: "12px 16px",
          background: "rgba(16, 20, 38, 0.8)",
          borderBottom: "1px solid rgba(255, 255, 255, 0.08)",
          display: "flex",
          flexWrap: "wrap",
          justifyContent: "space-between",
          alignItems: "center",
          gap: "10px",
        }}
      >
        {/* Left: Search input */}
        <div style={{ display: "flex", alignItems: "center", gap: "8px", flex: 1, minWidth: "220px", maxWidth: "340px" }}>
          <span style={{ fontSize: "0.85rem", color: "#94a3b8" }}>🔍</span>
          <input
            type="text"
            placeholder="Search in table..."
            value={searchTerm}
            onChange={(e) => {
              setSearchTerm(e.target.value);
              setCurrentPage(1);
            }}
            style={{
              width: "100%",
              padding: "6px 12px",
              background: "rgba(255, 255, 255, 0.05)",
              border: "1px solid rgba(255, 255, 255, 0.12)",
              borderRadius: "8px",
              color: "#f1f5f9",
              fontSize: "0.8rem",
              outline: "none",
            }}
          />
        </div>

        {/* Right: Actions (Chart/Table Toggle + CSV Export) */}
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          {isChartViewAvailable && onToggleView && (
            <button
              onClick={onToggleView}
              style={{
                padding: "6px 12px",
                borderRadius: "8px",
                background: isChartViewActive ? "rgba(99, 102, 241, 0.2)" : "rgba(255, 255, 255, 0.06)",
                border: "1px solid " + (isChartViewActive ? "var(--primary)" : "rgba(255, 255, 255, 0.1)"),
                color: isChartViewActive ? "var(--primary-light)" : "#e2e8f0",
                fontSize: "0.78rem",
                fontWeight: 600,
                cursor: "pointer",
              }}
            >
              {isChartViewActive ? "📋 Show Table" : "📊 Show Chart"}
            </button>
          )}

          <button
            onClick={exportToCsv}
            style={{
              padding: "6px 12px",
              borderRadius: "8px",
              background: "rgba(255, 255, 255, 0.06)",
              border: "1px solid rgba(255, 255, 255, 0.1)",
              color: "#cbd5e1",
              fontSize: "0.78rem",
              fontWeight: 500,
              cursor: "pointer",
            }}
            title="Export full table to CSV"
          >
            📥 Export CSV
          </button>
        </div>
      </div>

      {/* Table Data Area */}
      <div style={{ overflowX: "auto", maxHeight: "400px" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.82rem", textAlign: "left" }}>
          <thead>
            <tr style={{ background: "rgba(16, 20, 38, 0.95)", position: "sticky", top: 0, zIndex: 1 }}>
              {columns.map((col) => {
                const isSorted = sortCol === col;
                return (
                  <th
                    key={col}
                    onClick={() => handleSort(col)}
                    style={{
                      padding: "10px 14px",
                      color: isSorted ? "var(--accent)" : "#94a3b8",
                      borderBottom: "1px solid rgba(255, 255, 255, 0.12)",
                      fontWeight: 600,
                      cursor: "pointer",
                      userSelect: "none",
                      whiteSpace: "nowrap",
                    }}
                  >
                    <span style={{ display: "inline-flex", alignItems: "center", gap: "6px" }}>
                      {col.replace("_", " ").title ? col.replace("_", " ").toUpperCase() : col}
                      <span style={{ fontSize: "0.7rem", opacity: isSorted ? 1 : 0.4 }}>
                        {isSorted ? (sortDir === "asc" ? "▲" : "▼") : "⇅"}
                      </span>
                    </span>
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {paginatedRows.length === 0 ? (
              <tr>
                <td colSpan={columns.length} style={{ padding: "24px", textAlign: "center", color: "#64748b" }}>
                  No matching records found.
                </td>
              </tr>
            ) : (
              paginatedRows.map((row, rIdx) => (
                <tr
                  key={rIdx}
                  style={{
                    borderBottom: "1px solid rgba(255, 255, 255, 0.05)",
                    background: rIdx % 2 === 0 ? "transparent" : "rgba(255, 255, 255, 0.015)",
                  }}
                >
                  {columns.map((col) => (
                    <td
                      key={col}
                      style={{
                        padding: "8px 14px",
                        color: "#e2e8f0",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {formatCell(row[col])}
                    </td>
                  ))}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination Footer */}
      <div
        style={{
          padding: "10px 16px",
          background: "rgba(16, 20, 38, 0.8)",
          borderTop: "1px solid rgba(255, 255, 255, 0.06)",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          fontSize: "0.78rem",
          color: "#94a3b8",
        }}
      >
        <div>
          Showing {paginatedRows.length > 0 ? (currentPage - 1) * pageSize + 1 : 0} to{" "}
          {Math.min(currentPage * pageSize, sortedRows.length)} of {sortedRows.length} record(s)
          {totalRowCount && totalRowCount > sortedRows.length ? ` (from ${totalRowCount} total in dataset)` : ""}
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          {/* Page size select */}
          <select
            value={pageSize}
            onChange={(e) => {
              setPageSize(Number(e.target.value));
              setCurrentPage(1);
            }}
            style={{
              background: "rgba(255, 255, 255, 0.06)",
              border: "1px solid rgba(255, 255, 255, 0.1)",
              borderRadius: "6px",
              color: "#e2e8f0",
              padding: "3px 6px",
              fontSize: "0.75rem",
              outline: "none",
            }}
          >
            <option value={10}>10 per page</option>
            <option value={25}>25 per page</option>
            <option value={50}>50 per page</option>
          </select>

          {/* Prev / Next buttons */}
          <button
            onClick={() => setCurrentPage((p) => Math.max(p - 1, 1))}
            disabled={currentPage <= 1}
            style={{
              padding: "4px 8px",
              borderRadius: "6px",
              background: "rgba(255, 255, 255, 0.06)",
              border: "1px solid rgba(255, 255, 255, 0.1)",
              color: currentPage <= 1 ? "#475569" : "#e2e8f0",
              cursor: currentPage <= 1 ? "not-allowed" : "pointer",
              fontSize: "0.75rem",
            }}
          >
            ◀ Prev
          </button>
          <span>
            {currentPage} / {totalPages}
          </span>
          <button
            onClick={() => setCurrentPage((p) => Math.min(p + 1, totalPages))}
            disabled={currentPage >= totalPages}
            style={{
              padding: "4px 8px",
              borderRadius: "6px",
              background: "rgba(255, 255, 255, 0.06)",
              border: "1px solid rgba(255, 255, 255, 0.1)",
              color: currentPage >= totalPages ? "#475569" : "#e2e8f0",
              cursor: currentPage >= totalPages ? "not-allowed" : "pointer",
              fontSize: "0.75rem",
            }}
          >
            Next ▶
          </button>
        </div>
      </div>
    </div>
  );
}
