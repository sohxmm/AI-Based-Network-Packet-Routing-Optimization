/**
 * Turn a failed fetch Response into a sentence a human can act on.
 *
 * Two failure modes this fixes:
 *
 * 1. The backend returns `detail` as EITHER a string OR an object
 *    ({ message, missing_nodes, available_nodes }). The old code did
 *    `new Error(result.detail || ...)`, so the object case rendered as the
 *    literal text "[object Object]".
 *
 * 2. Both hooks called `await response.json()` on the failure path. A 502 from
 *    nginx returns HTML, so the parse threw and the user saw a JSON syntax
 *    error instead of the actual problem.
 */
export async function extractApiError(response, fallback = "Request failed.") {
  let payload = null;
  try {
    payload = await response.json();
  } catch {
    // Not JSON — a proxy error page, a timeout, an empty body.
    return `${fallback} (HTTP ${response.status} ${response.statusText || ""})`.trim();
  }

  const detail = payload?.detail;

  if (typeof detail === "string") return detail;

  if (detail && typeof detail === "object") {
    if (detail.message && Array.isArray(detail.missing_nodes)) {
      return `${detail.message} Unknown node(s): ${detail.missing_nodes.join(", ")}.`;
    }
    if (detail.message) return detail.message;
    return JSON.stringify(detail);
  }

  // FastAPI validation errors arrive as an array of {loc, msg, type}.
  if (Array.isArray(detail)) {
    const first = detail[0];
    if (first?.msg) {
      const field = Array.isArray(first.loc) ? first.loc.slice(-1)[0] : "input";
      return `${field}: ${first.msg}`;
    }
  }

  return `${fallback} (HTTP ${response.status})`;
}

export default extractApiError;
