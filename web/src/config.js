const rawBase = import.meta.env.VITE_API_BASE_URL ?? "";
export const API_BASE_URL = rawBase.replace(/\/$/, "");
export function wsUrl(path) {
  if (API_BASE_URL) {
    return API_BASE_URL.replace(/^http/, "ws") + path;
  }
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}${path}`;
}
