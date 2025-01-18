/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: "var(--primary)",
        },
        secondary: {
          DEFAULT: "var(--secondary)",
        },
        background: "var(--background)",
        foreground: "var(--foreground)",
      },
      fontFamily: {
        merriweather: ["var(--font-merriweather)", "sans-serif"],
      },
    },
  },
  plugins: [],
};
