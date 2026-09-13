/**
 * API client for communicating with the FastAPI backend.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/**
 * Get the stored JWT token from localStorage.
 */
function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("madac_token");
}

/**
 * Store JWT token in localStorage.
 */
export function setToken(token: string): void {
  localStorage.setItem("madac_token", token);
}

/**
 * Remove stored JWT token.
 */
export function clearToken(): void {
  localStorage.removeItem("madac_token");
  localStorage.removeItem("madac_user");
}

/**
 * Store user data in localStorage.
 */
export function setUser(user: any): void {
  localStorage.setItem("madac_user", JSON.stringify(user));
}

/**
 * Get stored user data.
 */
export function getUser(): any | null {
  if (typeof window === "undefined") return null;
  const data = localStorage.getItem("madac_user");
  return data ? JSON.parse(data) : null;
}

/**
 * Core fetch wrapper with auth headers and error handling.
 */
async function apiFetch(
  endpoint: string,
  options: RequestInit = {}
): Promise<any> {
  const token = getToken();
  const headers: Record<string, string> = {
    ...(options.headers as Record<string, string>),
  };

  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  // Don't set Content-Type for FormData (browser sets it with boundary)
  if (!(options.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }

  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers,
  });

  if (response.status === 401) {
    if (endpoint !== "/api/auth/login" && endpoint !== "/api/auth/register") {
      clearToken();
      if (typeof window !== "undefined") {
        window.location.href = "/login";
      }
      throw new Error("Unauthorized");
    }
  }


  if (response.status === 204) {
    return null;
  }

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.detail || "An error occurred");
  }

  return data;
}

// ---- Auth API ----

export async function apiRegister(name: string, email: string, password: string) {
  return apiFetch("/api/auth/register", {
    method: "POST",
    body: JSON.stringify({ name, email, password }),
  });
}

export async function apiLogin(email: string, password: string) {
  return apiFetch("/api/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
}

export async function apiForgotPassword(email: string) {
  return apiFetch("/api/auth/forgot-password", {
    method: "POST",
    body: JSON.stringify({ email }),
  });
}

export async function apiResetPassword(email: string, otp_code: string, new_password: string) {
  return apiFetch("/api/auth/reset-password", {
    method: "POST",
    body: JSON.stringify({ email, otp_code, new_password }),
  });
}


// ---- Dataset API ----

export async function apiUploadDataset(file: File, name?: string) {
  const formData = new FormData();
  formData.append("file", file);
  if (name) formData.append("name", name);

  return apiFetch("/api/datasets/upload", {
    method: "POST",
    body: formData,
  });
}

export async function apiGetDatasets() {
  return apiFetch("/api/datasets");
}

export async function apiGetDataset(id: string) {
  return apiFetch(`/api/datasets/${id}`);
}

export async function apiDeleteDataset(id: string) {
  return apiFetch(`/api/datasets/${id}`, { method: "DELETE" });
}

export async function apiGetDatasetProfile(id: string) {
  return apiFetch(`/api/datasets/${id}/profile`);
}

export async function apiDownloadCleanedDataset(id: string, originalName: string) {
  const token = getToken();
  const response = await fetch(`${API_BASE}/api/datasets/${id}/cleaned-file`, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.detail || "Cleaned file is unavailable.");
  }
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `cleaned_${originalName.replace(/\.[^.]+$/, "")}.csv`;
  link.click();
  URL.revokeObjectURL(url);
}

export async function apiQueryDataset(datasetId: string, question: string) {
  return apiFetch(`/api/datasets/${datasetId}/query`, {
    method: "POST",
    body: JSON.stringify({ question }),
  });
}

export async function apiGetDatasetQueries(datasetId: string) {
  return apiFetch(`/api/datasets/${datasetId}/queries`);
}

// ---- Machine Learning API ----

export async function apiTrainModel(
  datasetId: string,
  target_column: string,
  feature_columns?: string[]
) {
  return apiFetch(`/api/datasets/${datasetId}/train`, {
    method: "POST",
    body: JSON.stringify({ target_column, feature_columns }),
  });
}

export async function apiGetDatasetModels(datasetId: string) {
  return apiFetch(`/api/datasets/${datasetId}/models`);
}

export async function apiPredictWithModel(
  datasetId: string,
  model_id: string,
  input_features: Record<string, unknown>
) {
  return apiFetch(`/api/datasets/${datasetId}/predict`, {
    method: "POST",
    body: JSON.stringify({ model_id, input_features }),
  });
}

// ---- Reports API ----

export async function apiGetReports() {
  return apiFetch("/api/reports");
}

export function getReportPdfUrl(reportId: string): string {
  return `${API_BASE}/api/reports/${reportId}/export/pdf`;
}

export function getReportJsonUrl(reportId: string): string {
  return `${API_BASE}/api/reports/${reportId}/export/json`;
}



// ---- Health ----

export async function apiHealthCheck() {
  return apiFetch("/api/health");
}
