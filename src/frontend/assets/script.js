/* ─────────────────────────────────────────────────────────────────────────────
   IndiNeuroFM — Shared layout helpers
───────────────────────────────────────────────────────────────────────────── */

const API_BASE = 'http://localhost:8000';

/* ── Auth ── */
function requireAuth() {
  if (sessionStorage.getItem('loggedIn') !== 'true') {
    window.location.replace('login.html');
  }
}

function logout() {
  sessionStorage.clear();
  window.location.replace('login.html');
}

/* ── Header injection ── */
function buildHeader() {
  const loggedIn = sessionStorage.getItem('loggedIn') === 'true';
  const showLogout = document.body.dataset.showLogout === 'true';
  const currentPage = document.body.dataset.page || '';

  const navLinks = loggedIn ? `
    <a href="upload.html" class="${currentPage === 'upload' ? 'active' : ''}">Upload</a>
    <a href="results.html" class="${currentPage === 'results' ? 'active' : ''}">Results</a>
  ` : '';

  const logoutBtn = (loggedIn && showLogout) ? `
    <button class="btn-logout" onclick="logout()">
      <i class="fas fa-sign-out-alt"></i> Sign out
    </button>
  ` : '';

  return `
    <header class="site-header">
      <a href="index.html" class="header-brand" style="text-decoration:none;">
        <i class="fas fa-brain"></i>
        Dicom Anonymization Tool
      </a>
      <nav class="header-nav">
        ${navLinks}
        ${logoutBtn}
      </nav>
    </header>
  `;
}

function buildFooter() {
  return `
    <footer class="site-footer">
      Dicom Anonymization Tool
    </footer>
  `;
}

/* ── Mount shared elements ── */
document.addEventListener('DOMContentLoaded', () => {
  const hdr = document.getElementById('sharedHeader');
  const ftr = document.getElementById('sharedFooter');
  if (hdr) hdr.outerHTML = buildHeader();
  if (ftr) ftr.outerHTML = buildFooter();
});

/* ── Utility: format bytes ── */
function fmtBytes(bytes) {
  if (!bytes && bytes !== 0) return '—';
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

/* ── Utility: escape HTML ── */
function escapeHtml(s) {
  return String(s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

/* ── Utility: format timestamp ── */
function fmtTimestamp(iso) {
  try { return new Date(iso).toLocaleString('en-IN', { dateStyle: 'medium', timeStyle: 'short' }); }
  catch { return iso || '—'; }
}
