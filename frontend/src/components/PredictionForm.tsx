import { useState } from "react";
import type { CustomerFeatures } from "../services/api";

interface FieldConfig {
  key: keyof CustomerFeatures;
  label: string;
  type: "select" | "number";
  options?: string[];
  min?: number;
  max?: number;
  step?: number;
  placeholder?: string;
}

const FIELDS: FieldConfig[] = [
  { key: "gender", label: "Gender", type: "select", options: ["Male", "Female"] },
  { key: "SeniorCitizen", label: "Senior Citizen", type: "select", options: ["0", "1"] },
  { key: "Partner", label: "Partner", type: "select", options: ["Yes", "No"] },
  { key: "Dependents", label: "Dependents", type: "select", options: ["Yes", "No"] },
  { key: "tenure", label: "Tenure (months)", type: "number", min: 0, max: 72, step: 1, placeholder: "0–72" },
  { key: "PhoneService", label: "Phone Service", type: "select", options: ["Yes", "No"] },
  {
    key: "MultipleLines",
    label: "Multiple Lines",
    type: "select",
    options: ["Yes", "No", "No phone service"],
  },
  {
    key: "InternetService",
    label: "Internet Service",
    type: "select",
    options: ["DSL", "Fiber optic", "No"],
  },
  {
    key: "OnlineSecurity",
    label: "Online Security",
    type: "select",
    options: ["Yes", "No", "No internet service"],
  },
  {
    key: "OnlineBackup",
    label: "Online Backup",
    type: "select",
    options: ["Yes", "No", "No internet service"],
  },
  {
    key: "DeviceProtection",
    label: "Device Protection",
    type: "select",
    options: ["Yes", "No", "No internet service"],
  },
  {
    key: "TechSupport",
    label: "Tech Support",
    type: "select",
    options: ["Yes", "No", "No internet service"],
  },
  {
    key: "StreamingTV",
    label: "Streaming TV",
    type: "select",
    options: ["Yes", "No", "No internet service"],
  },
  {
    key: "StreamingMovies",
    label: "Streaming Movies",
    type: "select",
    options: ["Yes", "No", "No internet service"],
  },
  {
    key: "Contract",
    label: "Contract Type",
    type: "select",
    options: ["Month-to-month", "One year", "Two year"],
  },
  { key: "PaperlessBilling", label: "Paperless Billing", type: "select", options: ["Yes", "No"] },
  {
    key: "PaymentMethod",
    label: "Payment Method",
    type: "select",
    options: [
      "Electronic check",
      "Mailed check",
      "Bank transfer (automatic)",
      "Credit card (automatic)",
    ],
  },
  {
    key: "MonthlyCharges",
    label: "Monthly Charges ($)",
    type: "number",
    min: 0,
    max: 200,
    step: 0.01,
    placeholder: "e.g. 29.85",
  },
  {
    key: "TotalCharges",
    label: "Total Charges ($)",
    type: "number",
    min: 0,
    step: 0.01,
    placeholder: "e.g. 1200.00",
  },
];

const DEFAULT_VALUES: CustomerFeatures = {
  gender: "Female",
  SeniorCitizen: 0,
  Partner: "Yes",
  Dependents: "No",
  tenure: 1,
  PhoneService: "No",
  MultipleLines: "No phone service",
  InternetService: "DSL",
  OnlineSecurity: "No",
  OnlineBackup: "Yes",
  DeviceProtection: "No",
  TechSupport: "No",
  StreamingTV: "No",
  StreamingMovies: "No",
  Contract: "Month-to-month",
  PaperlessBilling: "Yes",
  PaymentMethod: "Electronic check",
  MonthlyCharges: 29.85,
  TotalCharges: 29.85,
};

const SECTIONS = [
  {
    title: "Demographics",
    keys: ["gender", "SeniorCitizen", "Partner", "Dependents"],
  },
  {
    title: "Account",
    keys: ["tenure", "Contract", "PaperlessBilling", "PaymentMethod"],
  },
  {
    title: "Services",
    keys: [
      "PhoneService",
      "MultipleLines",
      "InternetService",
      "OnlineSecurity",
      "OnlineBackup",
      "DeviceProtection",
      "TechSupport",
      "StreamingTV",
      "StreamingMovies",
    ],
  },
  {
    title: "Charges",
    keys: ["MonthlyCharges", "TotalCharges"],
  },
];

interface Props {
  onSubmit: (data: CustomerFeatures) => void;
  loading: boolean;
}

export default function PredictionForm({ onSubmit, loading }: Props) {
  const [values, setValues] = useState<CustomerFeatures>(DEFAULT_VALUES);

  const handleChange = (key: keyof CustomerFeatures, raw: string) => {
    const field = FIELDS.find((f) => f.key === key)!;
    if (field.type === "number") {
      setValues((prev) => ({ ...prev, [key]: parseFloat(raw) || 0 }));
    } else if (key === "SeniorCitizen") {
      setValues((prev) => ({ ...prev, [key]: parseInt(raw) as 0 | 1 }));
    } else {
      setValues((prev) => ({ ...prev, [key]: raw as never }));
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(values);
  };

  const fieldByKey = Object.fromEntries(FIELDS.map((f) => [f.key, f]));

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {SECTIONS.map((section) => (
        <div key={section.title} className="card p-5">
          <p className="section-title">{section.title}</p>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
            {section.keys.map((key) => {
              const field = fieldByKey[key as keyof CustomerFeatures];
              const value = values[field.key];
              return (
                <div key={key}>
                  <label className="form-label">{field.label}</label>
                  {field.type === "select" ? (
                    <div className="relative">
                      <select
                        className="select-field pr-8"
                        value={String(value)}
                        onChange={(e) =>
                          handleChange(field.key, e.target.value)
                        }
                      >
                        {field.options!.map((opt) => (
                          <option key={opt} value={opt}>
                            {opt}
                          </option>
                        ))}
                      </select>
                      <div className="pointer-events-none absolute inset-y-0 right-2 flex items-center">
                        <svg className="w-3.5 h-3.5 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                        </svg>
                      </div>
                    </div>
                  ) : (
                    <input
                      type="number"
                      className="input-field"
                      value={value as number}
                      min={field.min}
                      max={field.max}
                      step={field.step}
                      placeholder={field.placeholder}
                      onChange={(e) =>
                        handleChange(field.key, e.target.value)
                      }
                    />
                  )}
                </div>
              );
            })}
          </div>
        </div>
      ))}

      <button type="submit" disabled={loading} className="btn-primary w-full text-base">
        {loading ? (
          <>
            <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            Analyzing…
          </>
        ) : (
          <>
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            Predict Churn Risk
          </>
        )}
      </button>
    </form>
  );
}
