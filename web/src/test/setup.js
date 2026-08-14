/**
 * Vitest setup.
 *
 * jest-dom gives the DOM matchers; the ResizeObserver stub is needed because
 * Recharts' ResponsiveContainer looks for it and jsdom does not implement it.
 */
import "@testing-library/jest-dom/vitest";

if (!globalThis.ResizeObserver) {
  globalThis.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
}
