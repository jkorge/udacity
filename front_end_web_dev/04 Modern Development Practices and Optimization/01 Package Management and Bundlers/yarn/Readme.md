# Yarn Package Manager

## **Summary**

In this exercise, you will install Yarn, set up a `package.json` file, and manage dependencies using Yarn.

---

## **Instructions**

### **1. Install Yarn**

- If Yarn is not already installed on your system enter the following in the terminal:
- **Using npm**:
  ```
  npm install -g yarn
  ```
- Verify the installation by checking the Yarn version:
  ```
  yarn --version
  ```

### **2. Initialize a `package.json`**

- **Initialize**: In the terminal and start by initializing a `package.json` file with:
  ```
  yarn init
  ```
- Follow the prompts:
  - **Name**: Give your package a name that reflects its purpose.
  - **Version**: Set the version to `1.0.0`.
  - **Description**: Describe your package as "Introduction to Yarn".
  - **Entry Point**: Set the main entry point to `index.js`.
  - **Repository URL**: Press Enter to skip this for now.
  - **Author**: Enter your name.
  - **License**: Set the license to `MIT`.
  - **Private**: Press Enter to skip.
- Take a moment to examine your `package.json`.

---

### **3. Observe the Help Menu**

- In the following in the terminal and obser the command list.:
  ```
  yarn help
  ```
- use `yarn help COMMAND <command name> to learn about the commands

- Answer the following question:  
  **Q: What is the command for pretting out every cached package?**
  **A:yarn cache.**

---

### **4. Add Dependencies**

- Use the following command to add the specified dependencies:
  ```
  yarn add mocha chai
  ```
- Take a moment to examine your `node_modules` directory and `yarn.lock` file.

---

### **5. Remove Dependencies**

- Use the following command to remove `mocha` and `chai`:
  ```
  yarn remove mocha chai
  ```

### **6. Add Mocha and Chai as Development Dependencies**

- Re-add `mocha` and `chai` as development dependencies with:
  ```
  yarn add mocha chai --dev
  ```
- Verify that `mocha` and `chai` are listed under the `devDependencies` section in your `package.json`.

---
