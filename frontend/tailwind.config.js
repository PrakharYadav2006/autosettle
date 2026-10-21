/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["'JetBrains Mono'", "ui-monospace", "monospace"],
      },
      colors: {
        ink: { 950: "#060b18", 900: "#0b1324", 850: "#0f1a30", 800: "#14213b", 700: "#1e2e4f", 600: "#2b3f66" },
      },
    },
  },
  plugins: [],
};
