import js from "@eslint/js";
import globals from "globals";
import { defineConfig } from "eslint/config";

export default defineConfig([
    {
        files: ["**/*.{js,mjs,cjs}"],
        plugins: { js },
        extends: ["js/recommended"],
    },
    {
        files: ["**/*.{js,mjs,cjs}"],
        languageOptions: { globals: globals.browser },
        rules: {
            "no-var": "error",
            "no-unused-vars": "error",
            "semi": ["error", "always"],
            "no-unassigned-vars": "error",
            "no-use-before-define": "error"
        }
    },
]);
