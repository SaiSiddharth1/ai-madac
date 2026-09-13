"use client";

import { useEffect, useState } from "react";
import { getUser } from "@/lib/api";

export default function SettingsPage() {
  const [user, setUser] = useState<any>(null);

  useEffect(() => {
    setUser(getUser());
  }, []);

  return (
    <div className="animate-fade-in">
      <h1 style={{ fontSize: "1.5rem", fontWeight: 800, marginBottom: "4px" }}>Settings</h1>
      <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem", marginBottom: "32px" }}>
        Manage your account settings
      </p>

      <div className="glass-card" style={{ padding: "24px", maxWidth: "500px" }}>
        <h2 style={{ fontSize: "1.1rem", fontWeight: 700, marginBottom: "16px" }}>Account</h2>
        {user && (
          <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
            <div>
              <div style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginBottom: "4px" }}>Name</div>
              <div style={{ fontWeight: 500 }}>{user.name}</div>
            </div>
            <div>
              <div style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginBottom: "4px" }}>Email</div>
              <div style={{ fontWeight: 500 }}>{user.email}</div>
            </div>
            <div>
              <div style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginBottom: "4px" }}>Account Created</div>
              <div style={{ fontWeight: 500 }}>{new Date(user.created_at).toLocaleDateString()}</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
