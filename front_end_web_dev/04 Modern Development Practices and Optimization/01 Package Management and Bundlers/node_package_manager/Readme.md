# Node Package Manager

## **Summary**

In this exercise, you will install and configure a `package.json` file with dependencies.

---

## **Instructions**

### **1. Initialize a Package.json**

- **Initialize**: Open the terminal and start by initializing a `package.json` file with `npm init`.
- **Name**: Give your package a name that reflects its purpose.
- **Version**: Set the version to `1.0.0`.
- **Description**: Describe your package as "Introduction to npm".
- **Main File**: Set the main entry point to `index.js`.
- **Test Command**: Press Enter to skip the test command setup for now.
- **Repository**: Press Enter to skip setting up the Git repository field.
- **Keywords**: Press Enter to skip adding keywords.
- **Author**: Enter your name.
- **License**: Set the license to `MIT`.
- Take a moment to examine your `package.json`.

### **2. Observe the Help Menu**

- In the terminal, enter `npm help` and answer the following question:  
  **Q: What is the command for running a script?**

### **3. Install a Dependency**

- Use `npm install` to install the following dependencies: mocha chai webpack webpack-cli
- Take a moment to examine your `node_modules` directory and `package-lock.json` file.

### **4. Remove a Dependency**

- Remove `mocha` and `chai` using the command:

  ```
    npm remove mocha chai
  ```

### **5. Add Mocha and Chai as Development Dependencies**

- Re-add `mocha` and `chai` as development dependencies with:
  ```
  npm install mocha chai --save-dev
  ```
- Verify that `mocha` and `chai` are listed under the `devDependencies` section in your `package.json`.
