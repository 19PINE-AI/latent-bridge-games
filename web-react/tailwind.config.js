/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#0a0b10",
        panel: "#13151c",
        "panel-2": "#1a1d27",
        border: "#262a36",
        accent: "#ffb84d",
        good: "#5fd991",
        bad: "#ff6b6b",
        ink: "#e6e7eb",
        muted: "#a6abba",
        "muted-2": "#bfc3d1",
        link: "#7bb5ff",
      },
      fontFamily: {
        sans: ['-apple-system', 'BlinkMacSystemFont', '"Segoe UI"', 'Inter', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
      boxShadow: {
        soft: "0 8px 28px rgba(0,0,0,0.35)",
      },
    },
  },
  plugins: [],
};
