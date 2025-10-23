/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{vue,js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // 深蓝色科技风格主题
        primary: {
          50: '#e6f0ff',
          100: '#b3d1ff',
          200: '#80b3ff',
          300: '#4d94ff',
          400: '#1a75ff',
          500: '#0056e0',  // 主色调
          600: '#0044ad',
          700: '#00337a',
          800: '#002247',
          900: '#001114',
        },
        tech: {
          dark: '#0a0e27',
          darker: '#050812',
          blue: '#0066ff',
          cyan: '#00d4ff',
          purple: '#7c3aed',
        }
      },
      backgroundImage: {
        'tech-gradient': 'linear-gradient(135deg, #0a0e27 0%, #001447 50%, #003366 100%)',
        'card-gradient': 'linear-gradient(135deg, rgba(10, 14, 39, 0.8) 0%, rgba(0, 51, 102, 0.6) 100%)',
      },
      boxShadow: {
        'tech': '0 0 20px rgba(0, 102, 255, 0.3)',
        'tech-lg': '0 0 40px rgba(0, 102, 255, 0.5)',
        'glow': '0 0 15px rgba(0, 212, 255, 0.6)',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'float': 'float 3s ease-in-out infinite',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-10px)' },
        }
      }
    },
  },
  plugins: [],
}
