import { afterEach, expect, test, vi } from "vitest";
import { api, ApiError } from "./api";

afterEach(() => vi.unstubAllGlobals());

test("an SSO login redirect gives a useful sign-in error", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(
      async () =>
        new Response("<html>Sign in</html>", {
          headers: { "Content-Type": "text/html" },
        }),
    ),
  );
  await expect(api("/sessions")).rejects.toMatchObject({
    status: 401,
    message: expect.stringContaining("sign-in may have expired"),
  });
});

test("structured API errors preserve the server message and status", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(
      async () =>
        new Response(
          JSON.stringify({
            error: { message: "Conversation already has an unfinished run." },
          }),
          { status: 409, headers: { "Content-Type": "application/json" } },
        ),
    ),
  );
  await expect(api("/sessions/id/runs")).rejects.toBeInstanceOf(ApiError);
  await expect(api("/sessions/id/runs")).rejects.toMatchObject({
    status: 409,
    message: "Conversation already has an unfinished run.",
  });
});
