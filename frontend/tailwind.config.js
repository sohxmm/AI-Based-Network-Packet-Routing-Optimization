// TODO: implement
export default {
  darkMode: 'class',
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        app: {
          bg: 'var(--color-bg)',
          panel: 'var(--color-panel)',
          border: 'var(--color-border)',
          text: 'var(--color-text-main)',
          muted: 'var(--color-text-muted)',
          accent: 'var(--color-accent)',
          'accent-text': 'var(--color-accent-text)',
          'input-bg': 'var(--color-input-bg)',
        }
      }
    }
  },
  plugins: []
};
