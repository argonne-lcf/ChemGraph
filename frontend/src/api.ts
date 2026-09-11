export interface Artifact {
  id: string;
  name: string;
  kind: string;
  size: number;
  url: string;
  preview_url: string | null;
  run_id: string | null;
}
export interface Session {
  id: string;
  title: string;
  model: string;
  workflow: string;
  created?: string;
}
export interface Capabilities {
  user: string;
  models: string[];
  workflows: string[];
  calculators: string[];
  upload_limit: number;
  demo: boolean;
}
export interface Run {
  id: string;
  session_id: string;
  query: string;
  status: string;
  final_text: string;
  error: string | null;
  question_id: string | null;
  question: string | null;
  artifacts: Artifact[];
  attachments: Artifact[];
  responses: { question: string; answer: string }[];
}
export interface SessionDetail extends Session {
  runs: Run[];
}
export const terminal = (status: string) =>
  ["completed", "failed", "interrupted"].includes(status);

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}
export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api/v1${path}`, init);
  if (
    response.ok &&
    !response.headers.get("Content-Type")?.includes("application/json")
  ) {
    throw new ApiError(
      "Your sign-in may have expired. Refresh this page to sign in again.",
      401,
    );
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new ApiError(
      body?.error?.message || `Request failed (${response.status}).`,
      response.status,
    );
  }
  return response.json();
}
export const post = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});
