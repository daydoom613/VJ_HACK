import type { Config } from "tailwindcss";

export default {
  darkMode: ["class"],
  content: ["./pages/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./app/**/*.{ts,tsx}", "./src/**/*.{ts,tsx}"],
  prefix: "",
  theme: {
    container: {
      center: true,
      padding: "2rem",
      screens: {
        "2xl": "1400px",
      },
    },
    extend: {
      fontFamily: {
        'inter': ['Inter', 'sans-serif'],
        'roboto': ['Roboto', 'sans-serif'],
        'montserrat': ['Montserrat', 'sans-serif'],
        'lato': ['Lato', 'sans-serif'],
        'open-sans': ['Open Sans', 'sans-serif'],
        'plus-jakarta': ['Plus Jakarta Sans', 'sans-serif'],
      },
      fontSize: {
        'xs': ['0.875rem', { lineHeight: '1.25rem' }], // 14px, was 12px
        'sm': ['1rem', { lineHeight: '1.5rem' }], // 16px, was 14px
        'base': ['1.125rem', { lineHeight: '1.75rem' }], // 18px, was 16px
        'lg': ['1.25rem', { lineHeight: '1.75rem' }], // 20px, was 18px
        'xl': ['1.5rem', { lineHeight: '2rem' }], // 24px, was 20px
        '2xl': ['1.75rem', { lineHeight: '2rem' }], // 28px, was 30px (reduced)
        '3xl': ['2rem', { lineHeight: '2.25rem' }], // 32px, was 36px (reduced)
        '4xl': ['2.5rem', { lineHeight: '2.5rem' }], // 40px, was 48px (reduced)
        '5xl': ['3rem', { lineHeight: '1' }], // 48px, was 60px (reduced)
        '6xl': ['3.75rem', { lineHeight: '1' }], // 60px, was 72px (reduced)
        '7xl': ['4.5rem', { lineHeight: '1' }], // 72px, was 96px (reduced)
        '8xl': ['6rem', { lineHeight: '1' }], // 96px, was 128px (reduced)
        '9xl': ['8rem', { lineHeight: '1' }], // 128px, was 160px (reduced)
      },
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
        sidebar: {
          DEFAULT: "hsl(var(--sidebar-background))",
          foreground: "hsl(var(--sidebar-foreground))",
          primary: "hsl(var(--sidebar-primary))",
          "primary-foreground": "hsl(var(--sidebar-primary-foreground))",
          accent: "hsl(var(--sidebar-accent))",
          "accent-foreground": "hsl(var(--sidebar-accent-foreground))",
          border: "hsl(var(--sidebar-border))",
          ring: "hsl(var(--sidebar-ring))",
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      keyframes: {
        "accordion-down": {
          from: {
            height: "0",
          },
          to: {
            height: "var(--radix-accordion-content-height)",
          },
        },
        "accordion-up": {
          from: {
            height: "var(--radix-accordion-content-height)",
          },
          to: {
            height: "0",
          },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
} satisfies Config;
