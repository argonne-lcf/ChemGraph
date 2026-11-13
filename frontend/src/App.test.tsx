import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import App from "./App";

vi.mock("./MoleculeViewer", () => ({ default: () => <div>Viewer</div> }));
const caps = {
  user: "alice",
  models: ["Lab model"],
  workflows: ["single_agent", "multi_agent"],
  calculators: ["emt"],
  upload_limit: 25000000,
  demo: false,
};
const session = {
  id: "session-a",
  title: "Copper",
  model: "Lab model",
  workflow: "single_agent",
  runs: [],
};
const response = (value: unknown) =>
  new Response(JSON.stringify(value), {
    headers: { "Content-Type": "application/json" },
  });

beforeEach(() => {
  window.history.replaceState(null, "", "/");
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

test("restored conversations retain their model and workflow", async () => {
  window.history.replaceState(null, "", "/?session=session-a");
  vi.stubGlobal(
    "fetch",
    vi.fn(async (path: string) =>
      response(
        path.endsWith("/capabilities")
          ? caps
          : path.endsWith("/sessions")
            ? [session]
            : session,
      ),
    ),
  );
  render(<App />);
  await waitFor(() =>
    expect(screen.getByLabelText("Model")).toHaveValue("Lab model"),
  );
  expect(screen.getByLabelText("Model")).toBeDisabled();
  expect(screen.getByLabelText("Workflow")).toBeDisabled();
  fireEvent.click(screen.getByRole("button", { name: "New conversation" }));
  expect(screen.getByLabelText("Model")).toBeEnabled();
});

test("uncertain submissions retry the same request ID and upload only once", async () => {
  const submissions: string[] = [];
  let uploads = 0;
  const fetch = vi.fn(async (path: string, init?: RequestInit) => {
    if (path.endsWith("/capabilities")) return response(caps);
    if (path.endsWith("/uploads?filename=copper.xyz")) {
      uploads++;
      return response({
        id: "file-a",
        name: "copper.xyz",
        url: "/file",
        preview_url: null,
      });
    }
    if (path.endsWith("/runs")) {
      submissions.push(String(init?.body));
      if (submissions.length === 1) throw new Error("Connection lost");
      return response({ id: "run-a" });
    }
    if (path.endsWith("/sessions"))
      return response(init?.method === "POST" ? session : []);
    return response(session);
  });
  vi.stubGlobal("fetch", fetch);
  render(<App />);
  await screen.findByText("alice");
  fireEvent.change(screen.getByLabelText("Message"), {
    target: { value: "Optimize copper" },
  });
  fireEvent.change(screen.getByLabelText("Attach structure or data files"), {
    target: { files: [new File(["2\n\nCu 0 0 0\nCu 0 0 2.5"], "copper.xyz")] },
  });
  fireEvent.click(screen.getByRole("button", { name: "Send message" }));
  await screen.findByRole("alert");
  expect(
    screen.getByRole("button", { name: "New conversation" }),
  ).toBeDisabled();
  fireEvent.click(screen.getByRole("button", { name: "Retry send" }));
  await waitFor(() => expect(submissions).toHaveLength(2));
  expect(submissions[0]).toEqual(submissions[1]);
  expect(uploads).toBe(1);
  await waitFor(() => expect(screen.getByLabelText("Message")).toHaveValue(""));
});

test("unavailable models are disabled and the first configured model is selected", async () => {
  const readiness = {
    ...caps,
    models: ["Missing", "Lab model"],
    model_status: {
      Missing: {
        configured: false,
        code: "missing_credentials",
        message: "Credentials are missing.",
      },
      "Lab model": {
        configured: true,
        code: "configured",
        message: "Configured; connectivity has not been verified.",
      },
    },
  };
  vi.stubGlobal(
    "fetch",
    vi.fn(async (path: string) =>
      response(path.endsWith("/capabilities") ? readiness : []),
    ),
  );
  render(<App />);
  await waitFor(() =>
    expect(screen.getByLabelText("Model")).toHaveValue("Lab model"),
  );
  expect(
    screen.getByRole("option", { name: "Missing (unavailable)" }),
  ).toBeDisabled();
  expect(
    screen.getByText("Configured; connectivity has not been verified."),
  ).toBeVisible();
});

test("a rejected provider request preserves attachments and allows recovery", async () => {
  const submissions: string[] = [];
  let uploads = 0;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (path: string, init?: RequestInit) => {
      if (path.endsWith("/capabilities")) return response(caps);
      if (path.includes("/uploads?")) {
        uploads++;
        return response({
          id: "file-a",
          name: "copper.xyz",
          url: "/file",
          preview_url: null,
        });
      }
      if (path.endsWith("/runs")) {
        submissions.push(String(init?.body));
        if (submissions.length === 1)
          return new Response(
            JSON.stringify({
              error: {
                code: "missing_credentials",
                message: "Provider unavailable",
                submission_rejected: true,
              },
            }),
            { status: 503, headers: { "Content-Type": "application/json" } },
          );
        return response({ id: "run-a" });
      }
      if (path.endsWith("/sessions"))
        return response(init?.method === "POST" ? session : [session]);
      return response(session);
    }),
  );
  render(<App />);
  await screen.findByText("alice");
  fireEvent.change(screen.getByLabelText("Message"), {
    target: { value: "Optimize copper" },
  });
  fireEvent.change(screen.getByLabelText("Attach structure or data files"), {
    target: { files: [new File(["Cu"], "copper.xyz")] },
  });
  fireEvent.click(screen.getByRole("button", { name: "Send message" }));
  await screen.findByText("Provider unavailable");
  expect(
    screen.getByRole("button", { name: "New conversation" }),
  ).toBeEnabled();
  expect(screen.getByLabelText("Message")).toBeEnabled();
  fireEvent.click(screen.getByRole("button", { name: "Send message" }));
  await waitFor(() => expect(submissions).toHaveLength(2));
  expect(uploads).toBe(1);
  expect(JSON.parse(submissions[1]).attachments).toEqual(["file-a"]);
  expect(JSON.parse(submissions[0]).request_id).not.toEqual(
    JSON.parse(submissions[1]).request_id,
  );
});

test("archived conversations stay readable and cannot submit", async () => {
  window.history.replaceState(null, "", "/?session=session-a");
  const archived = {
    ...session,
    model_status: {
      configured: false,
      code: "model_changed",
      message: "Start a new conversation.",
    },
  };
  vi.stubGlobal(
    "fetch",
    vi.fn(async (path: string) =>
      response(
        path.endsWith("/capabilities")
          ? caps
          : path.endsWith("/sessions")
            ? [session]
            : archived,
      ),
    ),
  );
  render(<App />);
  await screen.findByText("Start a new conversation.");
  expect(screen.getByLabelText("Message")).toBeDisabled();
  expect(
    screen.getByRole("button", { name: "New conversation" }),
  ).toBeEnabled();
});
