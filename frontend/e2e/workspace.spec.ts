import { expect, test } from "@playwright/test";

const real = process.env.WEB_TEST_REAL === "true";
const resultText = real
  ? "Calculation complete."
  : "Demo calculation complete.";

test("upload, clarify, reconnect, inspect a structure and replay a trajectory", async ({
  page,
}, testInfo) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await expect(
    page.getByText(
      real
        ? "Configured; connectivity has not been verified."
        : "Local demo workspace",
    ),
  ).toBeVisible();
  await page.screenshot({
    path: testInfo.outputPath("workspace.png"),
    fullPage: true,
  });
  await page.getByLabel("Attach structure or data files").setInputFiles({
    name: "copper.xyz",
    mimeType: "text/plain",
    buffer: Buffer.from("2\nCopper dimer\nCu 0 0 0\nCu 0 0 2.5\n"),
  });
  await page
    .getByRole("textbox", { name: "Message", exact: true })
    .fill("Confirm optimization of the attached copper dimer");
  await page.getByRole("button", { name: "Send message" }).click();
  await expect(page.getByText("A quick clarification")).toBeVisible();
  const sessionURL = page.url();
  await page.reload();
  await expect(page.getByText("A quick clarification")).toBeVisible();
  await page
    .getByRole("textbox", { name: "Response to agent" })
    .fill("Yes, continue");
  await page.getByRole("button", { name: "Send response" }).click();
  await expect(page.getByText(resultText, { exact: false })).toBeVisible();
  await expect(page.locator(".molecule-canvas canvas")).toBeVisible();
  const selector = page.getByLabel("Structure or trajectory");
  const trajectory = await selector
    .locator("option")
    .filter({ hasText: "copper_opt.traj" })
    .getAttribute("value");
  await selector.selectOption(trajectory!);
  await expect(
    page.getByRole("button", { name: "Play trajectory" }),
  ).toBeEnabled();
  await page.getByRole("button", { name: "Play trajectory" }).click();
  await expect(
    page.getByRole("button", { name: "Pause trajectory" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Pause trajectory" }).click();
  const downloadPromise = page.waitForEvent("download");
  const downloadName = real ? "copper-result.json" : "copper.xyz";
  await page.locator(".artifact").filter({ hasText: downloadName }).click();
  expect((await downloadPromise).suggestedFilename()).toBe(downloadName);
  await page.screenshot({
    path: testInfo.outputPath("results.png"),
    fullPage: true,
  });
  await page.getByRole("button", { name: "New conversation" }).click();
  await page.goto(sessionURL);
  await expect(page.getByText(resultText, { exact: false })).toBeVisible();
  expect(errors).toEqual([]);
});

test("real multi-agent results, unavailable models, and provider errors", async ({
  page,
}) => {
  test.skip(!real, "Requires the deterministic non-demo endpoint");
  await page.goto("/");
  await expect(
    page.getByRole("option", { name: "Unavailable (unavailable)" }),
  ).toHaveJSProperty("disabled", true);
  await page.getByLabel("Workflow").selectOption("multi_agent");
  await page
    .getByLabel("Attach structure or data files")
    .setInputFiles({
      name: "copper.xyz",
      mimeType: "text/plain",
      buffer: Buffer.from("2\nCopper\nCu 0 0 0\nCu 0 0 2.5\n"),
    });
  await page
    .getByRole("textbox", { name: "Message", exact: true })
    .fill("Optimize the attached copper with EMT");
  await page.getByRole("button", { name: "Send message" }).click();
  await expect(
    page.getByText("Calculation complete.", { exact: false }),
  ).toBeVisible();
  await page
    .getByRole("textbox", { name: "Message", exact: true })
    .fill("Follow-up: recall our work");
  await page.getByRole("button", { name: "Send message" }).click();
  await expect(
    page.getByText("Previous context retained", { exact: false }),
  ).toBeVisible();
  await page.getByRole("button", { name: "New conversation" }).click();
  await page.getByLabel("Model").selectOption("Auth failure");
  await page
    .getByRole("textbox", { name: "Message", exact: true })
    .fill("Test authentication failure");
  await page.getByRole("button", { name: "Send message" }).click();
  await expect(
    page.getByText("The provider rejected the shared credentials.", {
      exact: false,
    }),
  ).toBeVisible();
  await expect(page.getByText("PRIVATE_PROVIDER_RESPONSE_MARKER")).toHaveCount(
    0,
  );
  await page.getByRole("button", { name: "New conversation" }).click();
  await expect(page.getByLabel("Model")).toBeEnabled();
});

test("small screens keep the composer reachable and panels toggle without overflow", async ({
  page,
}, testInfo) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(
    page.getByRole("textbox", { name: "Message", exact: true }),
  ).toBeVisible();
  await expect(page.getByRole("button", { name: "Close results" })).toHaveCount(
    0,
  );
  await page.getByRole("button", { name: "Open conversations" }).click();
  await expect(
    page.getByRole("button", { name: "New conversation" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Close conversations" }).click();
  await page.getByRole("button", { name: "Results", exact: true }).click();
  await page.getByRole("button", { name: "Close results" }).click();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: testInfo.outputPath("mobile.png"),
    fullPage: true,
  });
});
