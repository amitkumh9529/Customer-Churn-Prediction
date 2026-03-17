import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export const apiClient = axios.create({
  baseURL: BASE_URL,
  headers: { "Content-Type": "application/json" },
  timeout: 30_000,
});

// ── Types ──────────────────────────────────────────────────────────────────

export interface CustomerFeatures {
  gender: "Male" | "Female";
  SeniorCitizen: 0 | 1;
  Partner: "Yes" | "No";
  Dependents: "Yes" | "No";
  tenure: number;
  PhoneService: "Yes" | "No";
  MultipleLines: "Yes" | "No" | "No phone service";
  InternetService: "DSL" | "Fiber optic" | "No";
  OnlineSecurity: "Yes" | "No" | "No internet service";
  OnlineBackup: "Yes" | "No" | "No internet service";
  DeviceProtection: "Yes" | "No" | "No internet service";
  TechSupport: "Yes" | "No" | "No internet service";
  StreamingTV: "Yes" | "No" | "No internet service";
  StreamingMovies: "Yes" | "No" | "No internet service";
  Contract: "Month-to-month" | "One year" | "Two year";
  PaperlessBilling: "Yes" | "No";
  PaymentMethod:
    | "Electronic check"
    | "Mailed check"
    | "Bank transfer (automatic)"
    | "Credit card (automatic)";
  MonthlyCharges: number;
  TotalCharges: number;
}

export interface PredictionResponse {
  prediction: 0 | 1;
  prediction_label: string;
  churn_probability: number;
  no_churn_probability: number;
  model_version: string;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  version: string;
}

// ── API calls ──────────────────────────────────────────────────────────────

export async function predictChurn(
  customer: CustomerFeatures
): Promise<PredictionResponse> {
  const res = await apiClient.post<PredictionResponse>(
    "/api/v1/predict",
    customer
  );
  return res.data;
}

export async function checkHealth(): Promise<HealthResponse> {
  const res = await apiClient.get<HealthResponse>("/api/v1/health");
  return res.data;
}

export async function predictBatch(
  records: CustomerFeatures[]
): Promise<{ total: number; results: PredictionResponse[]; churn_rate: number }> {
  const res = await apiClient.post("/api/v1/predict/batch", { records });
  return res.data;
}
